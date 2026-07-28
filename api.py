import json
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import unquote, urljoin, urlparse

import requests
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import StreamingResponse

# --- Prometheus Metrics ---
_chat_requests = Counter("rag_chat_requests_total", "Total chat requests", ["type"])
_ingest_operations = Counter("rag_ingest_operations_total", "Total ingest operations", ["source_type"])
_file_operations = Counter("rag_file_operations_total", "Total file operations", ["operation"])
_indexed_documents = Gauge("rag_indexed_documents_total", "Total indexed document chunks in Qdrant")
_active_sessions = Gauge("rag_active_sessions", "Number of active chat sessions")
_ingest_running = Gauge("rag_ingest_running", "1 if an ingest job is currently running")
_request_duration = Histogram(
    "rag_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

from src.api_auth import verify_api_key
from src.api_models import (
    ChatRequest,
    ChatResponse,
    FileItem,
    FileListResponse,
    IngestPathRequest,
    IngestStatusResponse,
    IngestUrlRequest,
    IngestWebRequest,
    OperationResponse,
    RetrieveRequest,
    RetrievedChunk,
    RetrieveResponse,
    TokenUsage,
    UploadFileItem,
    UploadFileListResponse,
)
from src.chat import (
    SYSTEM_PROMPT_TEMPLATE,
    answer_with_web_fallback,
    estimate_tokens,
    get_chat_chain,
    retrieve_documents,
    stream_chat_response,
)
from src.citation import build_source_items, format_display
from src.config import config
from src.namespace import ApiClient, resolve_write_namespace
from src.file_index import (
    decode_source_id,
    encode_source_id,
    get_source_detail,
    list_indexed_sources,
)
from src.ingestion import get_text_splitter, load_local_document, parse_web_url, process_directory
from src.s3_storage import (
    is_s3_enabled,
    upload_to_s3,
    download_from_s3,
    delete_from_s3,
    get_s3_key_for_source,
    file_exists_in_s3,
    list_s3_files,
)
from src.vector_store import (
    delete_by_source,
    get_db_stats,
    ingest_documents,
    initialize_vector_store,
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Warm the reranker off-thread at startup so the first chat query isn't
    slowed by the model cold-load. Keeps startup/health-check immediate."""

    def _warm():
        from src.reranker import warm_reranker

        warm_reranker()

    threading.Thread(target=_warm, daemon=True, name="warm-reranker").start()
    yield


app = FastAPI(title="Universal RAG API", version="1.0.0", lifespan=_lifespan)

_ingest_lock = threading.Lock()
_ingest_status_lock = threading.Lock()
_chain_lock = threading.Lock()
_vector_store = None
_chat_chain = None
_session_histories: dict[str, list] = {}
_ingest_status = {
    "running": False,
    "current_task": None,
    "current_source": None,
    "started_at": None,
    "finished_at": None,
    "last_message": None,
}


def _parse_cors_origins() -> list[str]:
    raw = config.API_CORS_ORIGINS.strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _get_ingest_base_dir() -> str:
    base = (config.INGEST_BASE_DIR or config.UPLOADS_DIR or "uploads").strip() or "uploads"
    return os.path.abspath(base)


def _is_within_base_dir(path: str, base_dir: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(base_dir)]) == os.path.abspath(
            base_dir
        )
    except ValueError:
        return False


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_or_create_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = initialize_vector_store()
    return _vector_store


def _get_or_create_chain():
    global _chat_chain
    if _chat_chain is None:
        with _chain_lock:
            if _chat_chain is None:
                _chat_chain = get_chat_chain(_get_or_create_vector_store())
    return _chat_chain


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _enrich_docs_metadata(
    docs: list,
    source: str | None = None,
    source_type: str | None = None,
    namespace: str | None = None,
):
    ingested_at = _iso_now()
    for doc in docs:
        if source is not None:
            doc.metadata["source"] = source
        if source_type is not None:
            doc.metadata["source_type"] = source_type
        doc.metadata["ingested_at"] = ingested_at
        if namespace:
            doc.metadata["namespace"] = namespace


def _namespace_for_client(client: ApiClient | None) -> str:
    if client is None:
        return (config.DEFAULT_WRITE_NAMESPACE or "").strip()
    return resolve_write_namespace(client, config.DEFAULT_WRITE_NAMESPACE)


def _set_ingest_status_start(task: str):
    with _ingest_status_lock:
        _ingest_status["running"] = True
        _ingest_status["current_task"] = task
        _ingest_status["current_source"] = None
        _ingest_status["started_at"] = _iso_now()
        _ingest_status["finished_at"] = None
        _ingest_status["last_message"] = None


def _set_ingest_status_current_source(source: str | None):
    with _ingest_status_lock:
        _ingest_status["current_source"] = source


def _set_ingest_status_finish(message: str):
    with _ingest_status_lock:
        _ingest_status["running"] = False
        _ingest_status["current_task"] = None
        _ingest_status["current_source"] = None
        _ingest_status["finished_at"] = _iso_now()
        _ingest_status["last_message"] = message


def _run_ingest_path(
    path: str, client: ApiClient | None = None
) -> tuple[int, int, int]:
    docs, changed_sources = process_directory(path, on_file_start=_set_ingest_status_current_source)
    if not docs:
        return 0, 0, 0

    namespace = _namespace_for_client(client)
    _enrich_docs_metadata(docs, namespace=namespace)

    deleted_chunks = 0
    for source in changed_sources:
        deleted_chunks += delete_by_source(source, namespace=namespace or None)

    ingest_documents(docs, _get_or_create_vector_store())
    return len(changed_sources), deleted_chunks, len(docs)


def _ingest_single_file(
    filepath: str,
    source_type: str = "local",
    client: ApiClient | None = None,
    source: str | None = None,
) -> tuple[int, int]:
    abs_path = os.path.abspath(filepath)
    source_key = source or abs_path
    try:
        # OCR_GATEWAY_URL enables OCR during PDF/PPTX parse (see src.ocr_client).
        chunks = load_local_document(abs_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not chunks:
        return 0, 0

    namespace = _namespace_for_client(client)
    # load_local_document already returns final chunks; just stamp metadata.
    _enrich_docs_metadata(
        chunks, source=source_key, source_type=source_type, namespace=namespace
    )

    deleted_chunks = delete_by_source(source_key, namespace=namespace or None)
    ingest_documents(chunks, _get_or_create_vector_store())
    return deleted_chunks, len(chunks)


_REMOTE_INGEST_EXTS = frozenset(
    {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx"}
)
_REMOTE_INGEST_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "application/csv",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        # Presigned MinIO/S3 often omits a precise type.
        "application/octet-stream",
        "binary/octet-stream",
    }
)
_CONTENT_DISPOSITION_FILENAME_RE = re.compile(
    r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
    re.IGNORECASE,
)


def _validate_remote_ingest_url(url: str) -> str:
    """Allow http/https only. Skips public-IP SSRF checks so private MinIO
    presigned URLs work for Phase 3 PPID ingest."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError("url cannot be empty")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    if not (parsed.hostname or "").strip():
        raise ValueError("URL hostname is required")
    return cleaned


def _filename_from_content_disposition(header: str | None) -> str | None:
    if not header:
        return None
    match = _CONTENT_DISPOSITION_FILENAME_RE.search(header)
    if not match:
        return None
    return unquote(match.group(1).strip().strip("'"))


def _guess_remote_filename(url: str, content_disposition: str | None) -> str:
    from_header = _filename_from_content_disposition(content_disposition)
    if from_header:
        return os.path.basename(from_header) or from_header
    path = unquote(urlparse(url).path or "")
    name = os.path.basename(path)
    return name or "remote.bin"


def _remote_ingest_type_allowed(filename: str, content_type: str | None) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    if ext in _REMOTE_INGEST_EXTS:
        return True
    if not content_type:
        return False
    media = content_type.split(";", 1)[0].strip().lower()
    return media in _REMOTE_INGEST_CONTENT_TYPES and media not in {
        "application/octet-stream",
        "binary/octet-stream",
    }


def _download_remote_file(url: str) -> tuple[str, str]:
    """Download URL to a temp file. Returns (temp_path, filename).

    Enforces UPLOAD_MAX_BYTES via Content-Length and/or streamed byte cap.
    Does not use validate_public_http_url (private MinIO hosts are expected).
    """
    max_bytes = config.UPLOAD_MAX_BYTES if config.UPLOAD_MAX_BYTES > 0 else 104857600
    max_redirects = 5
    current_url = url
    last_error: Exception | None = None

    for _ in range(max_redirects + 1):
        current_url = _validate_remote_ingest_url(current_url)
        try:
            with requests.get(
                current_url,
                timeout=60,
                allow_redirects=False,
                stream=True,
            ) as response:
                if 300 <= response.status_code < 400 and response.headers.get(
                    "Location"
                ):
                    current_url = urljoin(current_url, response.headers["Location"])
                    continue

                if response.status_code >= 400:
                    raise ValueError(
                        f"Failed to download URL (HTTP {response.status_code})"
                    )

                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared = int(content_length)
                    except ValueError as exc:
                        raise ValueError("Invalid Content-Length header") from exc
                    if declared > max_bytes:
                        raise ValueError(
                            f"Remote file exceeds max allowed size ({max_bytes} bytes)"
                        )

                filename = _guess_remote_filename(
                    current_url, response.headers.get("Content-Disposition")
                )
                content_type = response.headers.get("Content-Type")
                if not _remote_ingest_type_allowed(filename, content_type):
                    raise ValueError(
                        "Unsupported remote file type; allowed: "
                        + ", ".join(sorted(_REMOTE_INGEST_EXTS))
                    )

                ext = os.path.splitext(filename)[1].lower()
                if ext not in _REMOTE_INGEST_EXTS:
                    # Content-Type allowed but no usable extension — invent one.
                    media = (content_type or "").split(";", 1)[0].strip().lower()
                    ext_map = {
                        "application/pdf": ".pdf",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                        "application/msword": ".doc",
                        "text/plain": ".txt",
                        "text/markdown": ".md",
                        "text/x-markdown": ".md",
                        "text/csv": ".csv",
                        "application/csv": ".csv",
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                    }
                    ext = ext_map.get(media, ".bin")
                    filename = f"{filename}{ext}" if not filename.endswith(ext) else filename

                fd, temp_path = tempfile.mkstemp(
                    suffix=os.path.splitext(filename)[1].lower() or ".bin",
                    prefix="ingest_url_",
                )
                total = 0
                try:
                    with os.fdopen(fd, "wb") as out:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > max_bytes:
                                raise ValueError(
                                    f"Remote file exceeds max allowed size ({max_bytes} bytes)"
                                )
                            out.write(chunk)
                except Exception:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise

                if total == 0:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise ValueError("Remote file is empty")

                return temp_path, filename
        except requests.RequestException as exc:
            last_error = exc
            raise ValueError(f"Failed to download URL: {exc}") from exc

    if last_error:
        raise ValueError(f"Failed to download URL: {last_error}") from last_error
    raise ValueError("Too many redirects while downloading URL")


def _is_web_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _reingest_source(
    source: str, client: ApiClient | None = None
) -> OperationResponse:
    namespace = _namespace_for_client(client)
    if _is_web_source(source):
        try:
            docs, changed = parse_web_url(source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not changed:
            return OperationResponse(
                success=True,
                message="Content unchanged. Skipped re-ingest.",
                skipped=True,
                deleted_chunks=0,
                added_chunks=0,
            )
        if not docs:
            raise HTTPException(status_code=500, detail="Failed to parse web content")
        _enrich_docs_metadata(
            docs, source=source, source_type="web", namespace=namespace
        )
        deleted_chunks = delete_by_source(source, namespace=namespace or None)
        ingest_documents(docs, _get_or_create_vector_store())
        return OperationResponse(
            success=True,
            message="Web source re-ingested",
            deleted_chunks=deleted_chunks,
            added_chunks=len(docs),
        )

    # If local file missing, try downloading from S3
    s3_temp_path = None
    actual_source = source
    if not os.path.exists(source) or not os.path.isfile(source):
        if is_s3_enabled():
            s3_key = get_s3_key_for_source(source)
            if file_exists_in_s3(s3_key):
                s3_temp_path = download_from_s3(s3_key)
                actual_source = s3_temp_path
            else:
                raise HTTPException(status_code=404, detail="source file not found locally or in S3")
        else:
            raise HTTPException(status_code=404, detail="local source file not found")

    try:
        deleted_chunks, added_chunks = _ingest_single_file(
            actual_source, source_type="local", client=client
        )
        if added_chunks == 0:
            raise HTTPException(status_code=400, detail="source cannot be ingested")
        return OperationResponse(
            success=True,
            message="Local source re-ingested" + (" (downloaded from S3)" if s3_temp_path else ""),
            deleted_chunks=deleted_chunks,
            added_chunks=added_chunks,
        )
    finally:
        # Clean up temp file downloaded from S3
        if s3_temp_path and os.path.exists(s3_temp_path):
            os.remove(s3_temp_path)
            temp_dir = os.path.dirname(s3_temp_path)
            if temp_dir and not os.listdir(temp_dir):
                os.rmdir(temp_dir)


def _calculate_token_usage(context_docs, history, question: str, answer: str) -> TokenUsage:
    context_text = "\n".join(doc.page_content for doc in context_docs) if context_docs else ""
    history_text = " ".join(msg.content for msg in history) if history else ""
    t_input = (
        estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
        + estimate_tokens(context_text)
        + estimate_tokens(history_text)
        + estimate_tokens(question)
    )
    t_output = estimate_tokens(answer)
    return TokenUsage(
        input_estimate=t_input,
        output_estimate=t_output,
        total_estimate=t_input + t_output,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-qdrant-api"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint (public, no auth required)."""
    # Update live gauges
    _active_sessions.set(len(_session_histories))
    with _ingest_status_lock:
        _ingest_running.set(1 if _ingest_status["running"] else 0)
    try:
        stats = get_db_stats()
        if "vectors_count" in stats:
            _indexed_documents.set(stats["vectors_count"])
    except Exception:
        pass
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])


@api_router.get("/status")
def status_endpoint():
    stats = get_db_stats()
    if "error" in stats:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=stats["error"],
        )
    return stats


@api_router.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status_endpoint():
    with _ingest_status_lock:
        return IngestStatusResponse(
            running=bool(_ingest_status["running"]),
            current_task=_ingest_status["current_task"],
            current_source=_ingest_status["current_source"],
            started_at=_ingest_status["started_at"],
            finished_at=_ingest_status["finished_at"],
            last_message=_ingest_status["last_message"],
        )


@api_router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    _chat_requests.labels(type="sync").inc()
    session_id = (payload.session_id or "default").strip() or "default"
    history = _session_histories.setdefault(session_id, [])
    web_active = bool(payload.enable_web_search and config.WEB_SEARCH_ENABLED)

    start = time.perf_counter()
    with _request_duration.labels(endpoint="/chat").time():
        answer, sources, web_used, context_docs = answer_with_web_fallback(
            payload.question,
            history,
            _get_or_create_vector_store(),
            (payload.system_prompt or "").strip(),
            enable_web_search=web_active,
        )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    token_usage = _calculate_token_usage(context_docs, history, payload.question, answer)

    history.extend([HumanMessage(content=payload.question), AIMessage(content=answer)])
    if len(history) > config.MEMORY_WINDOW_SIZE * 2:
        _session_histories[session_id] = history[-config.MEMORY_WINDOW_SIZE * 2 :]

    return ChatResponse(
        answer=answer,
        sources=sources,
        token_usage=token_usage,
        elapsed_ms=elapsed_ms,
        web_search_used=web_used,
    )


@api_router.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    _chat_requests.labels(type="stream").inc()
    session_id = (payload.session_id or "default").strip() or "default"
    history = _session_histories.setdefault(session_id, [])
    vector_store = _get_or_create_vector_store()
    web_active = bool(payload.enable_web_search and config.WEB_SEARCH_ENABLED)

    async def event_generator():
        async for data, event_type in stream_chat_response(
            payload.question, session_id, vector_store, history,
            (payload.system_prompt or "").strip(),
            web_active,
        ):
            if event_type == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
            elif event_type == "web_search":
                yield f"data: {json.dumps({'type': 'web_search', 'used': data['used']})}\n\n"
            elif event_type == "sources":
                yield f"data: {json.dumps({'type': 'sources', 'sources': data})}\n\n"
            elif event_type == "token_usage":
                yield f"data: {json.dumps({'type': 'token_usage', **data})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api_router.delete("/chat/{session_id}", response_model=OperationResponse)
def reset_chat_session(session_id: str):
    """Clear an existing session's conversation history in place.

    Starting a brand-new session_id is already a clean slate; use this only to
    reset history for an id the client wants to keep reusing.
    """
    existed = _session_histories.pop(session_id, None) is not None
    _active_sessions.set(len(_session_histories))
    return OperationResponse(
        success=True,
        message=(
            f"Riwayat sesi '{session_id}' direset."
            if existed
            else f"Sesi '{session_id}' tidak punya riwayat."
        ),
    )


@api_router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(
    payload: RetrieveRequest,
    client: ApiClient = Depends(verify_api_key),
):
    """Retrieve context chunks + citations without calling the LLM.

    Knowledge-space filter comes from the bearer token scope, never from the
    request body — so clients cannot widen their read access.
    """
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    start = time.perf_counter()
    previous_k = config.MAX_SEARCH_RESULTS
    if payload.top_k is not None:
        config.MAX_SEARCH_RESULTS = payload.top_k
    try:
        with _request_duration.labels(endpoint="/retrieve").time():
            docs = retrieve_documents(
                payload.question,
                _get_or_create_vector_store(),
                read_namespaces=client.read_namespaces,
            )
    finally:
        config.MAX_SEARCH_RESULTS = previous_k

    chunks: list[RetrievedChunk] = []
    for doc in docs:
        meta = dict(doc.metadata or {})
        source = str(meta.get("source") or "Unknown")
        source_type = str(meta.get("source_type") or "local")
        heading = meta.get("heading_path")
        if heading is not None and not isinstance(heading, list):
            heading = None
        filename = None
        if source_type != "web" and not source.startswith(("http://", "https://")):
            filename = os.path.basename(source) or None
        chunks.append(
            RetrievedChunk(
                text=doc.page_content or "",
                source=source,
                source_type=source_type,
                filename=filename,
                page=meta.get("page"),
                heading_path=heading,
                chunk_kind=meta.get("chunk_kind"),
                namespace=meta.get("namespace"),
                score=meta.get("score") or meta.get("rerank_score"),
                display=format_display(meta, source_type),
            )
        )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return RetrieveResponse(
        chunks=chunks,
        sources=build_source_items(docs),
        elapsed_ms=elapsed_ms,
    )


@api_router.post("/ingest/web", response_model=OperationResponse)
def ingest_web(
    payload: IngestWebRequest,
    client: ApiClient = Depends(verify_api_key),
):
    if not payload.url or not payload.url.strip():
        raise HTTPException(status_code=400, detail="url cannot be empty")

    namespace = _namespace_for_client(client)
    _ingest_operations.labels(source_type="web").inc()
    with _ingest_lock:
        _set_ingest_status_start("ingest_web")
        status_message = "Web ingestion completed"
        try:
            try:
                docs, changed = parse_web_url(payload.url.strip())
            except ValueError as exc:
                status_message = str(exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not changed:
                result = OperationResponse(
                    success=True,
                    message="Content unchanged. Skipped ingestion.",
                    skipped=True,
                    deleted_chunks=0,
                    added_chunks=0,
                )
                status_message = result.message
                return result
            if not docs:
                status_message = "Failed to parse web content"
                raise HTTPException(status_code=500, detail="Failed to parse web content")

            _enrich_docs_metadata(
                docs,
                source=payload.url.strip(),
                source_type="web",
                namespace=namespace,
            )
            deleted_chunks = delete_by_source(
                payload.url.strip(), namespace=namespace or None
            )
            ingest_documents(docs, _get_or_create_vector_store())
            result = OperationResponse(
                success=True,
                message="Web ingestion completed",
                skipped=False,
                deleted_chunks=deleted_chunks,
                added_chunks=len(docs),
            )
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


@api_router.post("/ingest/url", response_model=OperationResponse)
def ingest_url(
    payload: IngestUrlRequest,
    client: ApiClient = Depends(verify_api_key),
):
    """Download a remote file (presigned S3/MinIO OK) and ingest into write_namespace."""
    try:
        url = _validate_remote_ingest_url(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    source = (payload.source or "").strip() or url
    source_type = (payload.source_type or "remote").strip() or "remote"
    _ingest_operations.labels(source_type="url").inc()

    with _ingest_lock:
        _set_ingest_status_start("ingest_url")
        _set_ingest_status_current_source(source)
        status_message = "URL ingestion completed"
        temp_path: str | None = None
        try:
            try:
                temp_path, _filename = _download_remote_file(url)
            except ValueError as exc:
                status_message = str(exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            deleted_chunks, added_chunks = _ingest_single_file(
                temp_path,
                source_type=source_type,
                client=client,
                source=source,
            )
            if added_chunks == 0:
                result = OperationResponse(
                    success=True,
                    message="No supported content found in remote file",
                    processed_files=0,
                    deleted_chunks=deleted_chunks,
                    added_chunks=0,
                )
                status_message = result.message
                return result

            result = OperationResponse(
                success=True,
                message="URL ingestion completed",
                processed_files=1,
                deleted_chunks=deleted_chunks,
                added_chunks=added_chunks,
            )
            status_message = result.message
            return result
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            _set_ingest_status_finish(status_message)


@api_router.post("/ingest/file-path", response_model=OperationResponse)
def ingest_file_path(
    payload: IngestPathRequest,
    client: ApiClient = Depends(verify_api_key),
):
    _ingest_operations.labels(source_type="file").inc()
    path_raw = (payload.path or "").strip()
    if not path_raw:
        raise HTTPException(status_code=400, detail="path cannot be empty")
    path = os.path.abspath(path_raw)
    ingest_base_dir = _get_ingest_base_dir()
    if not _is_within_base_dir(path, ingest_base_dir):
        raise HTTPException(
            status_code=403,
            detail=f"path must be inside ingest base directory: {ingest_base_dir}",
        )
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="path not found")

    with _ingest_lock:
        _set_ingest_status_start("ingest_file_path")
        status_message = "Path ingestion completed"
        try:
            if os.path.isfile(path):
                deleted_chunks, added_chunks = _ingest_single_file(
                    path, source_type="local", client=client
                )
                if added_chunks == 0:
                    result = OperationResponse(
                        success=True,
                        message="No supported content found in file",
                        processed_files=0,
                        deleted_chunks=deleted_chunks,
                        added_chunks=0,
                    )
                    status_message = result.message
                    return result
                result = OperationResponse(
                    success=True,
                    message="File ingestion completed",
                    processed_files=1,
                    deleted_chunks=deleted_chunks,
                    added_chunks=added_chunks,
                )
                status_message = result.message
                return result

            processed_files, deleted_chunks, added_chunks = _run_ingest_path(
                path, client=client
            )
            result = OperationResponse(
                success=True,
                message="Path ingestion completed",
                processed_files=processed_files,
                deleted_chunks=deleted_chunks,
                added_chunks=added_chunks,
            )
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


@api_router.post("/ingest/uploads", response_model=OperationResponse)
def ingest_uploads(client: ApiClient = Depends(verify_api_key)):
    _ingest_operations.labels(source_type="uploads").inc()
    uploads_dir = config.UPLOADS_DIR.strip() or "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    with _ingest_lock:
        _set_ingest_status_start("ingest_uploads")
        status_message = "Uploads ingestion completed"
        try:
            processed_files, deleted_chunks, added_chunks = _run_ingest_path(
                uploads_dir, client=client
            )

            # After successful ingest, move files to S3 and clean up local
            s3_moved = 0
            if is_s3_enabled() and processed_files > 0:
                for root, _, filenames in os.walk(uploads_dir):
                    for filename in filenames:
                        filepath = os.path.join(root, filename)
                        try:
                            upload_to_s3(filepath)
                            os.remove(filepath)
                            s3_moved += 1
                        except Exception:
                            pass  # keep local if S3 upload fails

            message = "Uploads ingestion completed"
            if s3_moved > 0:
                message += f" ({s3_moved} file(s) moved to S3)"
            result = OperationResponse(
                success=True,
                message=message,
                uploads_dir=uploads_dir,
                processed_files=processed_files,
                deleted_chunks=deleted_chunks,
                added_chunks=added_chunks,
            )
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


def _filter_sort_files(
    items: list[dict],
    *,
    search: str | None,
    source_type: str | None,
    in_s3: bool | None,
    sort_by: str,
    sort_dir: str,
) -> list[dict]:
    """Filter + sort indexed-source dicts. Pure (no mutation); newest-first by default.

    Each item must have: source, source_type, chunk_count, last_seen, in_s3.
    `last_seen` may be None — it sorts to the bottom under the default desc order.
    """
    out = items
    if search:
        q = search.lower()
        out = [it for it in out if q in (it.get("source") or "").lower()]
    if source_type:
        out = [it for it in out if it.get("source_type") == source_type]
    if in_s3 is not None:
        out = [it for it in out if bool(it.get("in_s3")) == in_s3]

    def key(it: dict):
        if sort_by == "chunk_count":
            return it.get("chunk_count") or 0
        if sort_by == "filename":
            return os.path.basename(it.get("source") or "").lower()
        if sort_by == "source_type":
            return (it.get("source_type") or "").lower()
        return it.get("last_seen") or ""  # last_seen: None -> "" sinks under desc

    return sorted(out, key=key, reverse=(sort_dir == "desc"))


@api_router.get("/files", response_model=FileListResponse)
def list_files(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    search: str | None = Query(default=None, description="Substring match on source/filename"),
    source_type: str | None = Query(default=None, description="Filter: e.g. 'local' or 'web'"),
    in_s3: bool | None = Query(default=None, description="Filter by S3 presence"),
    sort_by: Literal["last_seen", "chunk_count", "filename", "source_type"] = Query(
        default="last_seen"
    ),
    sort_dir: Literal["asc", "desc"] = Query(default="desc"),
    client: ApiClient = Depends(verify_api_key),
):
    all_sources = list_indexed_sources(
        _get_or_create_vector_store(),
        read_namespaces=client.read_namespaces,
    )

    # Check S3 status for each source
    s3_keys: set[str] = set()
    if is_s3_enabled():
        try:
            s3_keys = {f["filename"] for f in list_s3_files()}
        except Exception:
            pass  # S3 unavailable — treat all as not in S3

    enriched = [
        {**item, "in_s3": os.path.basename(item["source"]) in s3_keys} for item in all_sources
    ]
    filtered = _filter_sort_files(
        enriched,
        search=search,
        source_type=source_type,
        in_s3=in_s3,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )

    total = len(filtered)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size
    return FileListResponse(
        items=[FileItem(**item) for item in filtered[start:end]],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@api_router.get("/uploads", response_model=UploadFileListResponse)
def list_uploads(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=10),
    client: ApiClient = Depends(verify_api_key),
):
    uploads_dir = os.path.abspath(config.UPLOADS_DIR.strip() or "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    indexed_items = list_indexed_sources(
        _get_or_create_vector_store(),
        read_namespaces=client.read_namespaces,
    )
    indexed_by_source = {item["source"]: item for item in indexed_items if item.get("source")}

    all_files: list[dict] = []
    for root, _, filenames in os.walk(uploads_dir):
        for filename in filenames:
            path = os.path.abspath(os.path.join(root, filename))
            try:
                stat = os.stat(path)
            except OSError:
                continue

            indexed = indexed_by_source.get(path)
            ingested = indexed is not None
            all_files.append(
                {
                    "upload_id": encode_source_id(path),
                    "filename": filename,
                    "path": path,
                    "size_bytes": stat.st_size,
                    "modified_at": _iso_from_timestamp(stat.st_mtime),
                    "ingested": ingested,
                    "ingest_status": "ingested" if ingested else "not_ingested",
                    "source_id": indexed.get("source_id") if indexed else None,
                    "chunk_count": indexed.get("chunk_count") if indexed else None,
                    "last_seen": indexed.get("last_seen") if indexed else None,
                    "_mtime": stat.st_mtime,
                }
            )

    all_files.sort(key=lambda item: item["_mtime"], reverse=True)
    total = len(all_files)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_files[start:end]

    return UploadFileListResponse(
        items=[UploadFileItem(**{k: v for k, v in item.items() if k != "_mtime"}) for item in page_items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        uploads_dir=uploads_dir,
    )


@api_router.delete("/uploads/{upload_id}", response_model=OperationResponse)
def delete_upload(
    upload_id: str,
    client: ApiClient = Depends(verify_api_key),
):
    try:
        upload_path = os.path.abspath(decode_source_id(upload_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads_dir = os.path.abspath(config.UPLOADS_DIR.strip() or "uploads")
    if not _is_within_base_dir(upload_path, uploads_dir):
        raise HTTPException(status_code=403, detail="upload path is outside uploads directory")
    if not os.path.exists(upload_path) or not os.path.isfile(upload_path):
        raise HTTPException(status_code=404, detail="upload file not found")

    namespace = _namespace_for_client(client) or None
    with _ingest_lock:
        deleted_chunks = delete_by_source(upload_path, namespace=namespace)
        os.remove(upload_path)

        # Also delete from S3 if enabled
        if is_s3_enabled():
            s3_key = get_s3_key_for_source(upload_path)
            delete_from_s3(s3_key)

    return OperationResponse(
        success=True,
        message="Upload file deleted",
        deleted_chunks=deleted_chunks,
        processed_files=1,
    )


@api_router.get("/files/{source_id}", response_model=FileItem)
def file_detail(source_id: str):
    try:
        source = decode_source_id(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detail = get_source_detail(_get_or_create_vector_store(), source)
    if not detail:
        raise HTTPException(status_code=404, detail="source not found")
    return FileItem(**detail)


@api_router.post("/files/upload", response_model=OperationResponse)
def upload_file(file: UploadFile = File(...)):
    uploads_dir = os.path.abspath(config.UPLOADS_DIR.strip() or "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    clean_name = os.path.basename(file.filename or "upload.bin")
    target = os.path.abspath(os.path.join(uploads_dir, f"{timestamp}_{clean_name}"))
    max_upload_bytes = config.UPLOAD_MAX_BYTES if config.UPLOAD_MAX_BYTES > 0 else 104857600

    total_bytes = 0
    try:
        with open(target, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"uploaded file exceeds max allowed size ({max_upload_bytes} bytes)",
                    )
                out.write(chunk)
    except HTTPException:
        if os.path.exists(target):
            os.remove(target)
        raise

    return OperationResponse(
        success=True,
        message="Upload completed. File not ingested yet.",
        uploads_dir=uploads_dir,
        processed_files=1,
        skipped=True,
    )


@api_router.put("/files/{source_id}", response_model=OperationResponse)
def reingest_source(
    source_id: str,
    client: ApiClient = Depends(verify_api_key),
):
    _file_operations.labels(operation="reingest").inc()
    try:
        source = decode_source_id(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _ingest_lock:
        _set_ingest_status_start("reingest_source")
        status_message = "Source re-ingest completed"
        try:
            result = _reingest_source(source, client=client)
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


@api_router.post("/files/reingest-all", response_model=OperationResponse)
def reingest_all_sources(client: ApiClient = Depends(verify_api_key)):
    with _ingest_lock:
        _set_ingest_status_start("reingest_all_sources")
        status_message = "Re-ingest all completed"
        try:
            # Scope to the caller's readable namespaces so a PPID token cannot
            # pull tabalong-umum (or other) sources into its write_namespace.
            sources = list_indexed_sources(
                _get_or_create_vector_store(),
                read_namespaces=client.read_namespaces,
            )
            if not sources:
                result = OperationResponse(
                    success=True,
                    message="No indexed sources found",
                    processed_files=0,
                    deleted_chunks=0,
                    added_chunks=0,
                    skipped=True,
                )
                status_message = result.message
                return result

            total_deleted = 0
            total_added = 0
            processed = 0
            skipped_count = 0
            failed_sources: list[str] = []

            for item in sources:
                source = item.get("source", "")
                if not source:
                    continue
                try:
                    result = _reingest_source(source, client=client)
                    processed += 1
                    total_deleted += result.deleted_chunks or 0
                    total_added += result.added_chunks or 0
                    if result.skipped:
                        skipped_count += 1
                except HTTPException:
                    failed_sources.append(source)

            failed_count = len(failed_sources)
            message = (
                "Re-ingest all completed"
                if failed_count == 0
                else (
                    f"Re-ingest all completed with {failed_count} failure(s): "
                    + ", ".join(failed_sources[:5])
                )
            )
            result = OperationResponse(
                success=failed_count == 0,
                message=message,
                processed_files=processed,
                deleted_chunks=total_deleted,
                added_chunks=total_added,
                skipped=processed > 0 and skipped_count == processed and failed_count == 0,
            )
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


@api_router.post("/uploads/migrate-to-s3", response_model=OperationResponse)
def migrate_uploads_to_s3():
    """Move all local upload files to S3 and delete local copies."""
    if not is_s3_enabled():
        raise HTTPException(status_code=400, detail="S3 is not configured. Set S3_BUCKET env variable.")

    uploads_dir = os.path.abspath(config.UPLOADS_DIR.strip() or "uploads")
    if not os.path.isdir(uploads_dir):
        return OperationResponse(success=True, message="No uploads directory found", processed_files=0)

    moved = 0
    failed = 0
    errors: list[str] = []
    for root, _, filenames in os.walk(uploads_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            try:
                upload_to_s3(filepath)
                os.remove(filepath)
                moved += 1
            except Exception as exc:
                failed += 1
                if len(errors) < 3:
                    errors.append(f"{filename}: {exc}")

    message = f"Migrated {moved} file(s) to S3"
    if failed > 0:
        message += f", {failed} failed"
        if errors:
            message += " — " + "; ".join(errors)
    return OperationResponse(success=failed == 0, message=message, processed_files=moved)


@api_router.delete("/files/{source_id}", response_model=OperationResponse)
def delete_source(
    source_id: str,
    client: ApiClient = Depends(verify_api_key),
):
    _file_operations.labels(operation="delete").inc()
    try:
        source = decode_source_id(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    namespace = _namespace_for_client(client) or None
    with _ingest_lock:
        deleted_chunks = delete_by_source(source, namespace=namespace)
    return OperationResponse(
        success=True,
        message="Source deleted",
        deleted_chunks=deleted_chunks,
    )


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False)
