from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from src.citation import build_source_items
from src.config import config
from src.reranker import is_reranker_enabled, rerank_with_scores
from src.web_search import search_web, web_context_text, web_results_to_documents
from datetime import date
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


def _date_guidance() -> str:
    """Year-preference instruction, evaluated per-invoke so cached chains stay current.

    When the user asks without naming a year, prefer the running year; fall back to
    the closest year present in the context (a ranking pure vector search can't do).
    """
    today = date.today()
    return (
        f"Tanggal hari ini: {today.isoformat()}. "
        f"Jika pertanyaan pengguna tidak menyebutkan tahun secara eksplisit, "
        f"utamakan data dari tahun berjalan ({today.year}); jika tahun itu tidak ada "
        f"di konteks, gunakan data dari tahun yang paling dekat dengan {today.year}."
    )

# Basic In-Memory Chat History
chat_history = []

# Condense prompt: rewrite a follow-up question into a standalone one using
# chat history, so retrieval works for anaphora ("berapa harganya?" -> "..").
CONDENSE_SYSTEM_PROMPT = (
    "Diberikan riwayat percakapan dan pertanyaan terakhir pengguna yang mungkin "
    "merujuk konteks dalam riwayat, susun ulang menjadi sebuah pertanyaan mandiri "
    "yang bisa dipahami tanpa riwayat. JANGAN dijawab — cukup susun ulang bila "
    "perlu, jika tidak kembalikan apa adanya. Pertahankan bahasa aslinya."
)

# System prompt (defined once for token counting)
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant connected to a knowledge base.\n"
    "Use the following pieces of retrieved context to answer the user's question.\n"
    "If the answer is not in the context, just say that you don't know based on the provided documents. "
    "Do not make up information that isn't supported by the context.\n\n"
    "Context:\n{context}"
)

NO_ANSWER_SENTINEL = "NO_ANSWER"

NOT_FOUND_MSG = "Maaf, jawaban tidak ditemukan di dokumen."

NOT_FOUND_WEB_MSG = (
    "Maaf, jawaban tidak ditemukan di dokumen maupun hasil pencarian web."
)

# Varian prompt untuk jalur web-fallback: ganti klausa "say you don't know"
# dengan instruksi sentinel agar deteksi deterministik. Tetap SATU system message.
SENTINEL_SYSTEM_TEMPLATE = (
    "You are a helpful AI assistant connected to a knowledge base.\n"
    "Use the following pieces of retrieved context to answer the user's question.\n"
    f"If the answer is NOT in the context, reply with EXACTLY `{NO_ANSWER_SENTINEL}` "
    "and nothing else (no other words, no punctuation, no explanation). "
    "If the answer IS in the context, answer normally and never output that token.\n"
    "Do not make up information that isn't supported by the context.\n\n"
    "Context:\n{context}"
)

# Panduan recency KHUSUS jalur web-fallback (tidak dipakai jalur RAG dokumen).
# Hasil web kerap mencampur artikel lama & baru; tanpa ini LLM mengutip yang usang.
WEB_RECENCY_GUIDANCE = (
    "Konteks di atas berasal dari pencarian web dan bisa memuat sumber usang. "
    "Utamakan sumber RESMI (mis. situs pemerintah) dan informasi PALING BARU. "
    "Perhatikan tanggal yang tertera di dalam teks; bila beberapa sumber menyebut "
    "nama berbeda untuk jabatan/posisi yang sama, pilih yang paling baru dan sebutkan "
    "perkiraan periodenya. Jangan mengarang — nyatakan bila suatu informasi tidak ada "
    "dalam konteks."
)


class DenseThresholdFallbackRetriever(BaseRetriever):
    """Dense retriever with score-threshold first, then similarity fallback."""

    threshold_retriever: object
    similarity_retriever: object

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs):
        docs = self.threshold_retriever.invoke(query)
        if docs:
            return docs
        logger.info(
            "No docs passed score_threshold=%.3f. Falling back to top-k similarity.",
            config.SEARCH_SCORE_THRESHOLD,
        )
        return self.similarity_retriever.invoke(query)


def estimate_tokens(text: str) -> int:
    """Estimate token count. ~1 token per 3 chars for Indonesian, ~4 for English."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def print_token_usage(context_docs, history, question, answer):
    """Print estimated token usage breakdown."""
    context_text = (
        "\n".join(doc.page_content for doc in context_docs) if context_docs else ""
    )
    history_text = " ".join(msg.content for msg in history) if history else ""

    t_system = estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
    t_context = estimate_tokens(context_text)
    t_history = estimate_tokens(history_text)
    t_question = estimate_tokens(question)
    t_answer = estimate_tokens(answer)
    t_input = t_system + t_context + t_history + t_question
    t_total = t_input + t_answer

    print(f"\n[Token Usage (estimated)]:")
    print(
        f"  Context   : ~{t_context:,} tokens ({len(context_docs) if context_docs else 0} chunks)"
    )
    print(f"  History   : ~{t_history:,} tokens")
    print(f"  Question  : ~{t_question:,} tokens")
    print(f"  System    : ~{t_system:,} tokens")
    print(f"  ─────────────────────────")
    print(f"  Input     : ~{t_input:,} tokens")
    print(f"  Output    : ~{t_answer:,} tokens")
    print(f"  TOTAL     : ~{t_total:,} tokens")


def get_llm():
    """Factory function for Chat LLMs using separate LLM_* config."""
    base_url = config.LLM_BASE_URL
    model_name = config.LLM_MODEL
    api_key = config.LLM_API_KEY

    # Heuristic to determine LLM provider
    if base_url and "api.openai.com" in base_url:
        logger.info(f"Using OpenAI Chat with model {model_name}")
        return ChatOpenAI(model=model_name, api_key=api_key)
    elif api_key and api_key.startswith(("AIza", "AQ.")):
        # Gemini API keys: legacy "AIza..." and newer "AQ...." format
        logger.info(f"Using Google Gemini Chat with model {model_name}")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    elif base_url and ("ollama" in base_url.lower() or ":11434" in base_url):
        logger.info(f"Using Ollama Chat with model {model_name} at {base_url}")
        return ChatOllama(model=model_name, base_url=base_url)
    elif base_url:
        # Generic OpenAI compatible
        logger.info(
            f"Using OpenAI Compatible Chat at {base_url} with model {model_name}"
        )
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key or "sk-dummy",
            openai_api_base=base_url,
        )
    else:
        # Default Fallback
        logger.info(f"Defaulting to OpenAI Chat with model {model_name}")
        return ChatOpenAI(model=model_name, api_key=api_key)


def _apply_relevance_gate(query: str, docs: list) -> list:
    """Drop retrieved chunks that fail the relevance gate.

    With reranker: cross-encoder score must meet RERANKER_MIN_SCORE.
    Without reranker: dense similarity score must meet SEARCH_SCORE_THRESHOLD
    when scores are present in metadata; otherwise keep the retriever output.
    """
    if not docs:
        return []

    if is_reranker_enabled():
        scored = rerank_with_scores(query, docs, top_k=config.MAX_SEARCH_RESULTS)
        gated = []
        for doc, score in scored:
            if score >= config.RERANKER_MIN_SCORE:
                doc.metadata["rerank_score"] = score
                gated.append(doc)
        if not gated:
            top = scored[0][1] if scored else None
            logger.info(
                "All %d retrieved docs below reranker min score %.3f (top=%s).",
                len(docs),
                config.RERANKER_MIN_SCORE,
                f"{top:.4f}" if top is not None else "n/a",
            )
        return gated

    scored_docs = [d for d in docs if "score" in d.metadata]
    if scored_docs:
        gated = [
            d
            for d in scored_docs
            if d.metadata.get("score", 0) >= config.SEARCH_SCORE_THRESHOLD
        ]
        if not gated:
            logger.info(
                "All %d retrieved docs below search score threshold %.3f.",
                len(docs),
                config.SEARCH_SCORE_THRESHOLD,
            )
        return gated[: config.MAX_SEARCH_RESULTS]

    return docs[: config.MAX_SEARCH_RESULTS]


def _retrieve_gated_docs(question: str, history: list, vector_store) -> list:
    """History-aware retrieval followed by relevance gating."""
    llm = get_llm()
    retriever = build_history_aware_retriever(vector_store, llm)
    docs = retriever.invoke({"input": question, "chat_history": history})
    return _apply_relevance_gate(question, docs)


def _context_text(docs: list) -> str:
    return "\n".join(d.page_content for d in docs) if docs else ""


def _invoke_llm(
    question: str,
    history: list,
    context_text: str,
    extra_system: str,
    *,
    with_sentinel: bool,
    web: bool = False,
) -> str:
    sys_msg = _build_system_message(
        context_text, extra_system, with_sentinel=with_sentinel, web=web
    )
    messages = [sys_msg] + list(history) + [HumanMessage(content=question)]
    return (get_llm().invoke(messages).content or "").strip()


def build_retriever(vector_store):
    """Single source of truth for the active retriever (dense+fallback or hybrid).
    Used by chat, streaming, AND the eval harness so all three retrieve identically."""
    if config.SEARCH_MODE.lower() == "hybrid":
        logger.info("Using HYBRID search mode (dense + sparse BM25)")
        from src.hybrid_retriever import HybridRetriever

        return HybridRetriever(
            vector_store=vector_store,
            score_threshold=config.SEARCH_SCORE_THRESHOLD,
            k=config.MAX_SEARCH_RESULTS,
        )

    logger.info("Using DENSE search mode")
    threshold_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": config.SEARCH_SCORE_THRESHOLD,
            "k": config.MAX_SEARCH_RESULTS,
        },
    )
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.MAX_SEARCH_RESULTS},
    )
    return DenseThresholdFallbackRetriever(
        threshold_retriever=threshold_retriever,
        similarity_retriever=similarity_retriever,
    )


def build_history_aware_retriever(vector_store, llm):
    """Wrap the base retriever so the latest question is condensed against
    chat_history into a standalone query BEFORE retrieval.

    With empty history (a fresh session) LangChain passes the input straight to
    the base retriever — no rewrite, no extra LLM call — so a new session behaves
    exactly like a first turn. History flows in at invoke time, never baked in,
    so one instance safely serves every session.
    """
    condense_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(
        llm, build_retriever(vector_store), condense_prompt
    )


def build_qa_prompt():
    """Build the QA ChatPromptTemplate as a SINGLE system message.

    IMPORTANT: keep everything in ONE system message. Some OpenAI-compatible
    backends (prod LLM "my-combo") only honour the FIRST system message — any
    extra system message (even an empty {extra_system}) makes the model ignore
    the retrieved context and hallucinate. So fold the optional per-request
    {extra_system} and {date_guidance} into the single context system message.
    ({context} is filled by create_stuff_documents_chain.)
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE + "\n\n{extra_system}\n\n{date_guidance}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    ).partial(date_guidance=_date_guidance)  # callable -> re-evaluated each invoke


def get_chat_chain(vector_store):
    """Sets up the retrieval + LLM conversational chain.
    Supports dense (default) and hybrid (dense+sparse) search modes,
    with optional cross-encoder re-ranking.
    """
    llm = get_llm()
    retriever = build_history_aware_retriever(vector_store, llm)

    question_answer_chain = create_stuff_documents_chain(llm, build_qa_prompt())
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def _build_system_message(
    context_text: str, extra_system: str, with_sentinel: bool, web: bool = False
):
    """Bangun SATU SystemMessage (lihat build_qa_prompt: my-combo hanya menghormati
    system message pertama). with_sentinel=True memakai template sentinel.
    web=True menyisipkan panduan recency (HANYA jalur web-fallback call #2)."""
    template = SENTINEL_SYSTEM_TEMPLATE if with_sentinel else SYSTEM_PROMPT_TEMPLATE
    parts = [template.format(context=context_text)]
    if web:
        parts.append(WEB_RECENCY_GUIDANCE)
    if extra_system:
        parts.append(extra_system)
    parts.append(_date_guidance())
    return SystemMessage(content="\n\n".join(parts))


def answer_with_web_fallback(
    question: str,
    history: list,
    vector_store,
    extra_system: str = "",
    enable_web_search: bool = False,
) -> tuple[str, list, bool, list]:
    """Jawab dari RAG dengan sentinel; opsional fallback web search.

    Return (answer, sources, web_used, context_docs_used).
    context_docs_used adalah list Document yang benar-benar dikonsumsi LLM:
      - RAG menjawab -> context_docs dari retrieval
      - Fallback web -> web_results_to_documents(results)
      - Tidak relevan / tidak ada jawaban -> []
    """
    context_docs = _retrieve_gated_docs(question, history, vector_store)
    if not context_docs:
        return NOT_FOUND_MSG, [], False, []

    answer = _invoke_llm(
        question,
        history,
        _context_text(context_docs),
        extra_system,
        with_sentinel=True,
    )

    if answer != NO_ANSWER_SENTINEL:
        return answer, build_source_items(context_docs), False, context_docs

    if not enable_web_search:
        return NOT_FOUND_MSG, [], False, []

    results = search_web(question)
    if not results:
        return NOT_FOUND_WEB_MSG, [], False, []

    web_docs = web_results_to_documents(results)
    web_answer = _invoke_llm(
        question,
        history,
        web_context_text(results),
        extra_system,
        with_sentinel=False,
        web=True,
    )
    return web_answer, build_source_items(web_docs), True, web_docs


async def _stream_llm_with_sentinel(llm, formatted):
    """Stream tokens, buffering while output may still be the NO_ANSWER sentinel."""
    buffer = []
    emitted = False
    full_answer = []
    async for chunk in llm.astream(formatted):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if not token:
            continue
        if emitted:
            full_answer.append(token)
            yield token, "token"
            continue
        buffer.append(token)
        joined = "".join(buffer).strip()
        if NO_ANSWER_SENTINEL.startswith(joined):
            continue
        emitted = True
        flushed = "".join(buffer)
        full_answer.append(flushed)
        yield flushed, "token"

    joined_final = "".join(buffer).strip()
    yield {
        "emitted": emitted,
        "buffer": buffer,
        "full_answer": full_answer,
        "joined_final": joined_final,
    }, "sentinel_state"


async def stream_chat_response(
    question: str,
    session_id: str,
    vector_store,
    history: list,
    extra_system: str = "",
    enable_web_search: bool = False,
):
    """Async generator untuk SSE streaming. Dua fase: retrieval sync + LLM stream.

    Strict RAG: sentinel selalu aktif; chunk tidak relevan dibuang sebelum LLM.
    enable_web_search=True menambah fallback web bila sentinel NO_ANSWER."""
    start = time.perf_counter()
    llm = get_llm()

    context_docs = _retrieve_gated_docs(question, history, vector_store)
    context_text = _context_text(context_docs)

    web_used = False
    sources_docs = context_docs
    token_est_context = context_text
    answer = ""

    if not context_docs:
        answer = NOT_FOUND_MSG
        yield answer, "token"
    else:
        sys_msg = _build_system_message(context_text, extra_system, with_sentinel=True)
        formatted = [sys_msg] + list(history) + [HumanMessage(content=question)]

        sentinel_state = None
        async for data, event_type in _stream_llm_with_sentinel(llm, formatted):
            if event_type == "token":
                yield data, "token"
            else:
                sentinel_state = data

        emitted = sentinel_state["emitted"]
        joined_final = sentinel_state["joined_final"]
        full_answer = sentinel_state["full_answer"]
        buffer = sentinel_state["buffer"]

        if not emitted and joined_final == NO_ANSWER_SENTINEL:
            if enable_web_search:
                results = await asyncio.to_thread(search_web, question)
                if results:
                    web_ctx = web_context_text(results)
                    web_sys = _build_system_message(
                        web_ctx, extra_system, with_sentinel=False, web=True
                    )
                    web_formatted = (
                        [web_sys] + list(history) + [HumanMessage(content=question)]
                    )
                    full_answer = []
                    async for chunk in llm.astream(web_formatted):
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if token:
                            full_answer.append(token)
                            yield token, "token"
                    answer = "".join(full_answer)
                    sources_docs = web_results_to_documents(results)
                    web_used = True
                    token_est_context = web_ctx
                else:
                    answer = NOT_FOUND_WEB_MSG
                    sources_docs = []
                    yield answer, "token"
                    token_est_context = ""
            else:
                answer = NOT_FOUND_MSG
                sources_docs = []
                token_est_context = ""
                yield answer, "token"
        elif not emitted:
            answer = "".join(buffer)
            if answer:
                yield answer, "token"
        else:
            answer = "".join(full_answer)

    # Event status web (sebelum sources).
    yield {"used": web_used}, "web_search"

    # Sources (web atau RAG).
    sources = [item.model_dump() for item in build_source_items(sources_docs)]
    yield sources, "sources"

    # Token usage (estimasi; pakai konteks yang benar-benar dikonsumsi LLM).
    history_text = " ".join(msg.content for msg in history) if history else ""
    t_input = (
        estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
        + estimate_tokens(token_est_context)
        + estimate_tokens(history_text)
        + estimate_tokens(question)
    )
    t_output = estimate_tokens(answer)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    yield {
        "input_estimate": t_input,
        "output_estimate": t_output,
        "total_estimate": t_input + t_output,
        "elapsed_ms": elapsed_ms,
    }, "token_usage"

    # Update history.
    history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    if len(history) > config.MEMORY_WINDOW_SIZE * 2:
        history[:] = history[-config.MEMORY_WINDOW_SIZE * 2 :]


def chat_interface(vector_store):
    """Interactive CLI loop for chatting."""
    print("\n--- Interactive Chat ('/new' sesi baru, '/exit' keluar) ---")

    global chat_history

    while True:
        user_input = input("\nYou: ")
        if user_input.strip() == "/exit":
            break
        if user_input.strip() in ("/new", "/reset"):
            chat_history.clear()
            print("🔄 Sesi baru — riwayat chat dikosongkan.")
            continue

        print("\nThinking...")

        try:
            start = time.perf_counter()
            answer, sources, _web_used, context_docs = answer_with_web_fallback(
                user_input,
                chat_history,
                vector_store,
                enable_web_search=False,
            )
            elapsed = time.perf_counter() - start

            print(f"AI: {answer}")
            print(f"\n⏱️  Waktu respons: {elapsed:.1f}s")

            if context_docs:
                print("\n📚 Sumber:")
                for i, src in enumerate(sources, start=1):
                    label = src.filename or src.source
                    print(f"  {i}. {label}")
                    for loc in src.locations:
                        print(f"     • {loc.display}")
                        if loc.chunk_preview:
                            print(f'       "{loc.chunk_preview}"')

            print_token_usage(context_docs, chat_history, user_input, answer)

            chat_history.extend(
                [HumanMessage(content=user_input), AIMessage(content=answer)]
            )

            if len(chat_history) > config.MEMORY_WINDOW_SIZE * 2:
                chat_history = chat_history[-config.MEMORY_WINDOW_SIZE * 2 :]

        except Exception as e:
            logger.error(f"Chat Error: {e}")
            print(f"Error generating response: {e}")
