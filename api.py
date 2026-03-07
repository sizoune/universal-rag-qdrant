import json
import os
import threading
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from starlette.responses import StreamingResponse

from src.api_auth import verify_api_key
from src.api_models import (
    ChatRequest,
    ChatResponse,
    FileItem,
    FileListResponse,
    IngestPathRequest,
    IngestStatusResponse,
    IngestWebRequest,
    OperationResponse,
    TokenUsage,
    UploadFileItem,
    UploadFileListResponse,
)
from src.chat import SYSTEM_PROMPT_TEMPLATE, estimate_tokens, get_chat_chain, stream_chat_response
from src.config import config
from src.file_index import (
    decode_source_id,
    encode_source_id,
    get_source_detail,
    list_indexed_sources,
)
from src.ingestion import get_text_splitter, load_local_document, parse_web_url, process_directory
from src.vector_store import (
    delete_by_source,
    get_db_stats,
    ingest_documents,
    initialize_vector_store,
)

app = FastAPI(title="Universal RAG API", version="1.0.0")

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


def _enrich_docs_metadata(docs: list, source: str | None = None, source_type: str | None = None):
    ingested_at = _iso_now()
    for doc in docs:
        if source is not None:
            doc.metadata["source"] = source
        if source_type is not None:
            doc.metadata["source_type"] = source_type
        doc.metadata["ingested_at"] = ingested_at


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


def _run_ingest_path(path: str) -> tuple[int, int, int]:
    docs, changed_sources = process_directory(path, on_file_start=_set_ingest_status_current_source)
    if not docs:
        return 0, 0, 0

    _enrich_docs_metadata(docs)

    deleted_chunks = 0
    for source in changed_sources:
        deleted_chunks += delete_by_source(source)

    ingest_documents(docs, _get_or_create_vector_store())
    return len(changed_sources), deleted_chunks, len(docs)


def _ingest_single_file(filepath: str, source_type: str = "local") -> tuple[int, int]:
    abs_path = os.path.abspath(filepath)
    try:
        docs = load_local_document(abs_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not docs:
        return 0, 0

    for doc in docs:
        doc.metadata["source"] = abs_path
        doc.metadata["source_type"] = source_type
    chunks = get_text_splitter().split_documents(docs)
    _enrich_docs_metadata(chunks, source=abs_path, source_type=source_type)

    deleted_chunks = delete_by_source(abs_path)
    ingest_documents(chunks, _get_or_create_vector_store())
    return deleted_chunks, len(chunks)


def _is_web_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _reingest_source(source: str) -> OperationResponse:
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
        _enrich_docs_metadata(docs, source=source, source_type="web")
        deleted_chunks = delete_by_source(source)
        ingest_documents(docs, _get_or_create_vector_store())
        return OperationResponse(
            success=True,
            message="Web source re-ingested",
            deleted_chunks=deleted_chunks,
            added_chunks=len(docs),
        )

    if not os.path.exists(source) or not os.path.isfile(source):
        raise HTTPException(status_code=404, detail="local source file not found")

    deleted_chunks, added_chunks = _ingest_single_file(source, source_type="local")
    if added_chunks == 0:
        raise HTTPException(status_code=400, detail="source cannot be ingested")
    return OperationResponse(
        success=True,
        message="Local source re-ingested",
        deleted_chunks=deleted_chunks,
        added_chunks=added_chunks,
    )


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

    session_id = (payload.session_id or "default").strip() or "default"
    history = _session_histories.setdefault(session_id, [])
    chain = _get_or_create_chain()

    response = chain.invoke({"input": payload.question, "chat_history": history})
    answer = response.get("answer", "No answer generated.")
    context_docs = response.get("context", [])
    sources = list(dict.fromkeys(doc.metadata.get("source", "Unknown") for doc in context_docs))
    token_usage = _calculate_token_usage(context_docs, history, payload.question, answer)

    history.extend([HumanMessage(content=payload.question), AIMessage(content=answer)])
    if len(history) > config.MEMORY_WINDOW_SIZE * 2:
        _session_histories[session_id] = history[-config.MEMORY_WINDOW_SIZE * 2 :]

    return ChatResponse(answer=answer, sources=sources, token_usage=token_usage)


@api_router.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    session_id = (payload.session_id or "default").strip() or "default"
    history = _session_histories.setdefault(session_id, [])
    vector_store = _get_or_create_vector_store()

    async def event_generator():
        async for data, event_type in stream_chat_response(
            payload.question, session_id, vector_store, history
        ):
            if event_type == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
            elif event_type == "sources":
                yield f"data: {json.dumps({'type': 'sources', 'sources': data})}\n\n"
            elif event_type == "token_usage":
                yield f"data: {json.dumps({'type': 'token_usage', **data})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api_router.post("/ingest/web", response_model=OperationResponse)
def ingest_web(payload: IngestWebRequest):
    if not payload.url or not payload.url.strip():
        raise HTTPException(status_code=400, detail="url cannot be empty")

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

            _enrich_docs_metadata(docs, source=payload.url.strip(), source_type="web")
            deleted_chunks = delete_by_source(payload.url.strip())
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


@api_router.post("/ingest/file-path", response_model=OperationResponse)
def ingest_file_path(payload: IngestPathRequest):
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
                deleted_chunks, added_chunks = _ingest_single_file(path, source_type="local")
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

            processed_files, deleted_chunks, added_chunks = _run_ingest_path(path)
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
def ingest_uploads():
    uploads_dir = config.UPLOADS_DIR.strip() or "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    with _ingest_lock:
        _set_ingest_status_start("ingest_uploads")
        status_message = "Uploads ingestion completed"
        try:
            processed_files, deleted_chunks, added_chunks = _run_ingest_path(uploads_dir)
            result = OperationResponse(
                success=True,
                message="Uploads ingestion completed",
                uploads_dir=uploads_dir,
                processed_files=processed_files,
                deleted_chunks=deleted_chunks,
                added_chunks=added_chunks,
            )
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


@api_router.get("/files", response_model=FileListResponse)
def list_files():
    items = [FileItem(**item) for item in list_indexed_sources(_get_or_create_vector_store())]
    return FileListResponse(items=items, total=len(items))


@api_router.get("/uploads", response_model=UploadFileListResponse)
def list_uploads(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=10),
):
    uploads_dir = os.path.abspath(config.UPLOADS_DIR.strip() or "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    indexed_items = list_indexed_sources(_get_or_create_vector_store())
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
def delete_upload(upload_id: str):
    try:
        upload_path = os.path.abspath(decode_source_id(upload_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads_dir = os.path.abspath(config.UPLOADS_DIR.strip() or "uploads")
    if not _is_within_base_dir(upload_path, uploads_dir):
        raise HTTPException(status_code=403, detail="upload path is outside uploads directory")
    if not os.path.exists(upload_path) or not os.path.isfile(upload_path):
        raise HTTPException(status_code=404, detail="upload file not found")

    with _ingest_lock:
        deleted_chunks = delete_by_source(upload_path)
        os.remove(upload_path)

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
def reingest_source(source_id: str):
    try:
        source = decode_source_id(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _ingest_lock:
        _set_ingest_status_start("reingest_source")
        status_message = "Source re-ingest completed"
        try:
            result = _reingest_source(source)
            status_message = result.message
            return result
        finally:
            _set_ingest_status_finish(status_message)


@api_router.post("/files/reingest-all", response_model=OperationResponse)
def reingest_all_sources():
    with _ingest_lock:
        _set_ingest_status_start("reingest_all_sources")
        status_message = "Re-ingest all completed"
        try:
            sources = list_indexed_sources(_get_or_create_vector_store())
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
                    result = _reingest_source(source)
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


@api_router.delete("/files/{source_id}", response_model=OperationResponse)
def delete_source(source_id: str):
    try:
        source = decode_source_id(source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _ingest_lock:
        deleted_chunks = delete_by_source(source)
    return OperationResponse(
        success=True,
        message="Source deleted",
        deleted_chunks=deleted_chunks,
    )


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False)
