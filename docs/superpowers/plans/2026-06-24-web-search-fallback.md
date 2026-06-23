# Web Search Fallback (9router) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Saat RAG tidak menemukan jawaban dari embedding, lakukan web search via 9router lalu jawab ulang — opt-in per request (`enable_web_search`) dengan kill-switch global.

**Architecture:** Fitur aditif & opt-in. Deteksi "tak ada jawaban" pakai sentinel `NO_ANSWER` dari LLM (call #1); jika sentinel terdeteksi dan web aktif, panggil `POST {LLM_BASE_URL}/search` (9router), bangun konteks web, lalu LLM call #2 menjawab atas hasil web. Hasil web mengalir ke pipeline sitasi yang sudah ada (`source_type="web"`). Jalur default (`enable_web_search=false`) tidak berubah.

**Tech Stack:** Python, FastAPI, LangChain, `requests` (sudah dipakai `src/ingestion.py`), pytest.

## Global Constraints

- **Satu system message saja** di setiap panggilan LLM — backend prod my-combo hanya menghormati system message pertama; system message kedua membuat model mengabaikan konteks. Semua prompt dibangun sebagai satu `SystemMessage`.
- Jalur `enable_web_search=false` **tidak boleh berubah** (byte-for-byte seperti sekarang).
- `web_active = request.enable_web_search and config.WEB_SEARCH_ENABLED`. Global OFF → param diabaikan.
- `search_web` **tidak pernah melempar exception** ke caller — gagal → `[]`.
- Sentinel persis: `NO_ANSWER_SENTINEL = "NO_ANSWER"`.
- Bahasa pesan ke pengguna: Indonesia.
- PEP 8, type hints di semua signature, gunakan `logging` (bukan `print`) di kode non-CLI.

---

## File Structure

- `src/config.py` (modify) — 6 env var `WEB_SEARCH_*`.
- `src/web_search.py` (create) — klien 9router `/v1/search` + mapper Document/konteks.
- `src/chat.py` (modify) — sentinel const/template, `_build_system_message`, `answer_with_web_fallback`, param `enable_web_search` di `stream_chat_response`.
- `src/api_models.py` (modify) — `ChatRequest.enable_web_search`, `ChatResponse.web_search_used`.
- `api.py` (modify) — wiring `/chat` & `/chat/stream`.
- `requirements.txt` (modify) — pin `requests`.
- `.env.example` (modify) — dokumentasi var baru.
- `tests/test_web_search.py` (create), `tests/test_chat_web_fallback.py` (create), `tests/test_api.py` (modify).

---

## Task 1: Config + web search client

**Files:**
- Modify: `src/config.py` (tambah blok `=== Web Search ===` setelah blok Hybrid Search, sekitar baris 80-84)
- Modify: `requirements.txt` (tambah `requests`)
- Modify: `.env.example` (tambah blok WEB SEARCH)
- Create: `src/web_search.py`
- Test: `tests/test_web_search.py`

**Interfaces:**
- Consumes: `config` dari `src/config.py`; `Document` dari `langchain_core.documents`.
- Produces:
  - `WebResult(title: str, url: str, snippet: str, score: float | None = None)` — frozen dataclass.
  - `search_web(query: str) -> list[WebResult]`
  - `web_results_to_documents(results: list[WebResult]) -> list[Document]`
  - `web_context_text(results: list[WebResult]) -> str`
  - Config: `WEB_SEARCH_ENABLED: bool`, `WEB_SEARCH_URL: str`, `WEB_SEARCH_API_KEY: str`, `WEB_SEARCH_PROVIDER: str`, `WEB_SEARCH_MAX_RESULTS: int`, `WEB_SEARCH_TIMEOUT: int`.

- [ ] **Step 1: Tambah config var di `src/config.py`**

Setelah baris `RERANKER_MODEL = ...` (akhir blok Hybrid Search), tambahkan:

```python
    # === Web Search Fallback (9router /v1/search) ===
    WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Endpoint POST penuh. Kosong -> diturunkan dari LLM_BASE_URL + "/search".
    WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "")
    # Bearer untuk search. Kosong -> fallback ke LLM_API_KEY.
    WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", "")
    WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "search-combo")
    try:
        WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
        WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
    except ValueError:
        WEB_SEARCH_MAX_RESULTS = 5
        WEB_SEARCH_TIMEOUT = 10
```

- [ ] **Step 2: Pin `requests` di `requirements.txt`**

Tambahkan baris (mis. setelah `beautifulsoup4>=4.15.0`):

```
requests>=2.32.0
```

- [ ] **Step 3: Dokumentasikan env di `.env.example`**

Tambahkan blok:

```bash
# === WEB SEARCH FALLBACK (9router /v1/search) ===
# Aktifkan fallback web search global (kill-switch admin). Tetap perlu enable_web_search=true per request.
WEB_SEARCH_ENABLED="false"
# Endpoint POST penuh. Kosongkan untuk pakai LLM_BASE_URL + "/search".
WEB_SEARCH_URL=""
# Bearer token. Kosongkan untuk pakai LLM_API_KEY.
WEB_SEARCH_API_KEY=""
WEB_SEARCH_PROVIDER="search-combo"
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=10
```

- [ ] **Step 4: Tulis test yang gagal — `tests/test_web_search.py`**

```python
import importlib
from types import SimpleNamespace

import pytest

import src.web_search as web_search


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_search_web_parses_results(monkeypatch):
    payload = {
        "results": [
            {"title": "T1", "url": "https://a.test", "snippet": "snip1", "score": 0.9},
            {"title": "T2", "url": "https://b.test", "content": "snip2"},
        ],
        "errors": [],
    }
    monkeypatch.setattr(
        web_search.requests, "post", lambda *a, **k: _FakeResp(payload)
    )
    results = web_search.search_web("apa kabar")
    assert [r.url for r in results] == ["https://a.test", "https://b.test"]
    assert results[0].snippet == "snip1"
    assert results[1].snippet == "snip2"  # fallback ke "content"


def test_search_web_timeout_returns_empty(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(web_search.requests, "post", _boom)
    assert web_search.search_web("x") == []


def test_search_web_errors_field_returns_empty(monkeypatch):
    payload = {"results": [{"url": "https://a.test"}], "errors": ["quota"]}
    monkeypatch.setattr(
        web_search.requests, "post", lambda *a, **k: _FakeResp(payload)
    )
    assert web_search.search_web("x") == []


def test_search_url_derives_from_llm_base_url(monkeypatch):
    monkeypatch.setattr(web_search.config, "WEB_SEARCH_URL", "")
    monkeypatch.setattr(web_search.config, "LLM_BASE_URL", "http://host:9/v1")
    assert web_search._search_url() == "http://host:9/v1/search"


def test_web_results_to_documents_metadata():
    results = [web_search.WebResult(title="T", url="https://a.test", snippet="s", score=0.5)]
    docs = web_search.web_results_to_documents(results)
    assert docs[0].metadata["source_type"] == "web"
    assert docs[0].metadata["source"] == "https://a.test"
    assert docs[0].page_content == "s"
```

- [ ] **Step 5: Jalankan test — pastikan GAGAL**

Run: `python -m pytest tests/test_web_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.web_search'`.

- [ ] **Step 6: Implementasi `src/web_search.py`**

```python
"""Klien web search 9router (POST /v1/search), dipakai sebagai fallback RAG.

Tanpa state, config-driven. Tidak pernah melempar exception ke caller:
kegagalan apa pun -> list kosong (fallback yang gagal cukup berujung
pesan "tidak ditemukan").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    score: float | None = None


def _search_url() -> str:
    """Endpoint /v1/search. WEB_SEARCH_URL eksplisit menang; jika kosong
    diturunkan dari LLM_BASE_URL (mis. http://h/v1 -> http://h/v1/search)."""
    if config.WEB_SEARCH_URL:
        return config.WEB_SEARCH_URL
    base = (config.LLM_BASE_URL or "").rstrip("/")
    return f"{base}/search" if base else ""


def _api_key() -> str:
    return config.WEB_SEARCH_API_KEY or config.LLM_API_KEY or ""


def search_web(query: str) -> list[WebResult]:
    url = _search_url()
    if not url:
        logger.warning("Web search dilewati: WEB_SEARCH_URL/LLM_BASE_URL kosong.")
        return []

    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": config.WEB_SEARCH_PROVIDER,
        "query": query,
        "max_results": config.WEB_SEARCH_MAX_RESULTS,
    }

    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=config.WEB_SEARCH_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("Web search gagal: %s", e)
        return []

    if data.get("errors"):
        logger.warning("Web search mengembalikan error: %s", data["errors"])
        return []

    results: list[WebResult] = []
    for item in data.get("results") or []:
        item_url = item.get("url")
        if not item_url:
            continue
        results.append(
            WebResult(
                title=item.get("title") or item_url,
                url=item_url,
                snippet=item.get("snippet") or item.get("content") or "",
                score=item.get("score"),
            )
        )
    return results


def web_results_to_documents(results: list[WebResult]) -> list[Document]:
    """Map ke Document agar mengalir ke build_source_items (source_type='web')."""
    return [
        Document(
            page_content=r.snippet,
            metadata={"source": r.url, "source_type": "web", "title": r.title},
        )
        for r in results
    ]


def web_context_text(results: list[WebResult]) -> str:
    """Blok konteks untuk LLM call #2 (judul + url + snippet)."""
    return "\n\n".join(f"[{r.title}]({r.url})\n{r.snippet}" for r in results)
```

- [ ] **Step 7: Jalankan test — pastikan LULUS**

Run: `python -m pytest tests/test_web_search.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add src/config.py src/web_search.py tests/test_web_search.py requirements.txt .env.example
git commit -m "feat(web-search): klien 9router /v1/search + config WEB_SEARCH_*"
```

---

## Task 2: Sentinel + orkestrasi sync (`answer_with_web_fallback`)

**Files:**
- Modify: `src/chat.py` (tambah konstanta & template di dekat `SYSTEM_PROMPT_TEMPLATE` ~baris 48-54; tambah helper setelah `stream_chat_response` atau setelah `get_chat_chain`)
- Test: `tests/test_chat_web_fallback.py`

**Interfaces:**
- Consumes: `search_web`, `web_results_to_documents`, `web_context_text` (Task 1); `build_retriever`/`build_history_aware_retriever`, `get_llm`, `SYSTEM_PROMPT_TEMPLATE`, `_date_guidance` (existing `src/chat.py`); `build_source_items` (existing `src/citation.py`).
- Produces:
  - `NO_ANSWER_SENTINEL = "NO_ANSWER"` (str)
  - `NOT_FOUND_MSG` (str)
  - `SENTINEL_SYSTEM_TEMPLATE` (str)
  - `_build_system_message(context_text: str, extra_system: str, with_sentinel: bool) -> SystemMessage`
  - `answer_with_web_fallback(question: str, history: list, vector_store, extra_system: str = "") -> tuple[str, list, bool]` — return `(answer, sources, web_used)`.

- [ ] **Step 1: Tambah import top-level di `src/chat.py`**

Setelah baris `from src.citation import build_source_items` (baris 12), tambahkan:

```python
from src.web_search import search_web, web_context_text, web_results_to_documents
```

(Tidak ada import siklik: `web_search` hanya mengimpor `src.config` + langchain.)

- [ ] **Step 2: Tambah konstanta & template sentinel di `src/chat.py`**

Setelah blok `SYSTEM_PROMPT_TEMPLATE = (...)` (setelah baris 54), tambahkan:

```python
NO_ANSWER_SENTINEL = "NO_ANSWER"

NOT_FOUND_MSG = (
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
```

- [ ] **Step 3: Tulis test yang gagal — `tests/test_chat_web_fallback.py`**

```python
from types import SimpleNamespace

import src.chat as chat
from src.web_search import WebResult
from langchain_core.documents import Document


class _FakeLLM:
    """invoke() mengembalikan konten berikutnya dari antrean."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def invoke(self, messages):
        content = self._contents[self.calls]
        self.calls += 1
        return SimpleNamespace(content=content)


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, _payload):
        return self._docs


def _patch_common(monkeypatch, llm, docs):
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(
        chat, "build_history_aware_retriever", lambda vs, _llm: _FakeRetriever(docs)
    )


def test_no_fallback_when_rag_answers(monkeypatch):
    llm = _FakeLLM(["Jawaban dari dokumen."])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    _patch_common(monkeypatch, llm, docs)

    called = {"web": False}
    monkeypatch.setattr(chat, "search_web", lambda q: called.__setitem__("web", True) or [])

    answer, sources, web_used = chat.answer_with_web_fallback("q", [], object(), "")
    assert answer == "Jawaban dari dokumen."
    assert web_used is False
    assert called["web"] is False  # search tidak dipanggil
    assert sources and sources[0].source_type == "local"


def test_fallback_triggers_on_sentinel(monkeypatch):
    llm = _FakeLLM([chat.NO_ANSWER_SENTINEL, "Jawaban dari web."])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    _patch_common(monkeypatch, llm, docs)
    monkeypatch.setattr(
        chat, "search_web", lambda q: [WebResult("T", "https://x.test", "snip", 0.9)]
    )

    answer, sources, web_used = chat.answer_with_web_fallback("q", [], object(), "")
    assert answer == "Jawaban dari web."
    assert web_used is True
    assert sources and sources[0].source_type == "web"
    assert llm.calls == 2


def test_fallback_no_web_results_returns_not_found(monkeypatch):
    llm = _FakeLLM([chat.NO_ANSWER_SENTINEL])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    _patch_common(monkeypatch, llm, docs)
    monkeypatch.setattr(chat, "search_web", lambda q: [])

    answer, sources, web_used = chat.answer_with_web_fallback("q", [], object(), "")
    assert answer == chat.NOT_FOUND_MSG
    assert web_used is False
    assert sources == []
```

- [ ] **Step 4: Jalankan test — pastikan GAGAL**

Run: `python -m pytest tests/test_chat_web_fallback.py -v`
Expected: FAIL — `AttributeError: module 'src.chat' has no attribute 'answer_with_web_fallback'`.

- [ ] **Step 5: Implementasi helper di `src/chat.py`**

Tambahkan setelah fungsi `get_chat_chain` (sekitar baris 227):

```python
def _build_system_message(context_text: str, extra_system: str, with_sentinel: bool):
    """Bangun SATU SystemMessage (lihat build_qa_prompt: my-combo hanya menghormati
    system message pertama). with_sentinel=True memakai template sentinel."""
    from langchain_core.messages import SystemMessage

    template = SENTINEL_SYSTEM_TEMPLATE if with_sentinel else SYSTEM_PROMPT_TEMPLATE
    parts = [template.format(context=context_text)]
    if extra_system:
        parts.append(extra_system)
    parts.append(_date_guidance())
    return SystemMessage(content="\n\n".join(parts))


def answer_with_web_fallback(
    question: str, history: list, vector_store, extra_system: str = ""
) -> tuple[str, list, bool]:
    """Jawab dari RAG; jika model tak bisa menjawab dari konteks (sentinel),
    fallback ke web search lalu jawab ulang. Return (answer, sources, web_used).
    Caller wajib menggerbang ini pada web aktif (request + global enabled)."""
    llm = get_llm()
    retriever = build_history_aware_retriever(vector_store, llm)
    context_docs = retriever.invoke({"input": question, "chat_history": history})
    context_text = (
        "\n".join(d.page_content for d in context_docs) if context_docs else ""
    )

    sys_msg = _build_system_message(context_text, extra_system, with_sentinel=True)
    messages = [sys_msg] + list(history) + [HumanMessage(content=question)]
    answer = (llm.invoke(messages).content or "").strip()

    if answer != NO_ANSWER_SENTINEL:
        return answer, build_source_items(context_docs), False

    # ponytail: query web pakai pertanyaan mentah; follow-up anaforik bisa
    # kurang presisi. Upgrade = condense pakai history (panggilan LLM ke-3).
    results = search_web(question)
    if not results:
        return NOT_FOUND_MSG, [], False

    web_sys = _build_system_message(
        web_context_text(results), extra_system, with_sentinel=False
    )
    web_messages = [web_sys] + list(history) + [HumanMessage(content=question)]
    web_answer = (llm.invoke(web_messages).content or "").strip()
    return web_answer, build_source_items(web_results_to_documents(results)), True
```

- [ ] **Step 6: Jalankan test — pastikan LULUS**

Run: `python -m pytest tests/test_chat_web_fallback.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add src/chat.py tests/test_chat_web_fallback.py
git commit -m "feat(web-search): sentinel + answer_with_web_fallback (jalur sync)"
```

---

## Task 3: Streaming web fallback (`stream_chat_response`)

**Files:**
- Modify: `src/chat.py` (fungsi `stream_chat_response`, baris ~230-292)
- Test: `tests/test_chat_web_fallback.py` (tambah test streaming)

**Interfaces:**
- Consumes: `_build_system_message`, `NO_ANSWER_SENTINEL`, `NOT_FOUND_MSG`, `search_web`, `web_context_text`, `web_results_to_documents` (Task 1-2).
- Produces: `stream_chat_response(question, session_id, vector_store, history, extra_system="", enable_web_search=False)` — async generator yang kini juga meng-yield event `(payload_dict, "web_search")` dengan `payload_dict = {"used": bool}` sebelum event `"sources"`.

- [ ] **Step 1: Tulis test streaming yang gagal — tambahkan ke `tests/test_chat_web_fallback.py`**

```python
import asyncio


class _FakeStreamLLM:
    """astream() mengembalikan token batch berikutnya per panggilan."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0

    async def astream(self, messages):
        tokens = self._batches[self.calls]
        self.calls += 1
        for t in tokens:
            yield SimpleNamespace(content=t)


def _drain(agen):
    async def _run():
        out = []
        async for item in agen:
            out.append(item)
        return out

    return asyncio.run(_run())


def _collect_tokens(events):
    return "".join(d for d, et in events if et == "token")


def _event(events, name):
    return next(d for d, et in events if et == name)


def test_stream_no_fallback_when_rag_answers(monkeypatch):
    llm = _FakeStreamLLM([["Doc ", "answer"]])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(
        chat, "build_history_aware_retriever", lambda vs, _llm: _FakeRetriever(docs)
    )
    called = {"web": False}
    monkeypatch.setattr(chat, "search_web", lambda q: called.__setitem__("web", True) or [])

    events = _drain(
        chat.stream_chat_response("q", "s", object(), [], "", enable_web_search=True)
    )
    assert _collect_tokens(events) == "Doc answer"
    assert _event(events, "web_search") == {"used": False}
    assert called["web"] is False


def test_stream_fallback_on_sentinel(monkeypatch):
    # call #1 -> sentinel; call #2 -> jawaban web
    llm = _FakeStreamLLM([[chat.NO_ANSWER_SENTINEL], ["Web ", "answer"]])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(
        chat, "build_history_aware_retriever", lambda vs, _llm: _FakeRetriever(docs)
    )
    monkeypatch.setattr(
        chat, "search_web", lambda q: [WebResult("T", "https://x.test", "snip", 0.9)]
    )

    events = _drain(
        chat.stream_chat_response("q", "s", object(), [], "", enable_web_search=True)
    )
    assert _collect_tokens(events) == "Web answer"
    assert _event(events, "web_search") == {"used": True}
    sources = _event(events, "sources")
    assert sources and sources[0]["source_type"] == "web"
```

- [ ] **Step 2: Jalankan test — pastikan GAGAL**

Run: `python -m pytest tests/test_chat_web_fallback.py -k stream -v`
Expected: FAIL — `TypeError: stream_chat_response() got an unexpected keyword argument 'enable_web_search'`.

- [ ] **Step 3: Implementasi — ganti isi `stream_chat_response` di `src/chat.py`**

Ganti seluruh fungsi `stream_chat_response` (baris ~230-292) dengan:

```python
async def stream_chat_response(
    question: str,
    session_id: str,
    vector_store,
    history: list,
    extra_system: str = "",
    enable_web_search: bool = False,
):
    """Async generator untuk SSE streaming. Dua fase: retrieval sync + LLM stream.
    enable_web_search=True menambah deteksi sentinel + fallback web."""
    import asyncio

    start = time.perf_counter()
    llm = get_llm()
    retriever = build_history_aware_retriever(vector_store, llm)

    # Phase A: retrieval history-aware (kosong -> input lewat apa adanya).
    context_docs = retriever.invoke({"input": question, "chat_history": history})
    context_text = (
        "\n".join(doc.page_content for doc in context_docs) if context_docs else ""
    )

    web_used = False
    sources_docs = context_docs

    if not enable_web_search:
        # Jalur lama — tidak berubah.
        sys_msg = _build_system_message(context_text, extra_system, with_sentinel=False)
        formatted = [sys_msg] + list(history) + [HumanMessage(content=question)]
        full_answer = []
        async for chunk in llm.astream(formatted):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_answer.append(token)
                yield token, "token"
        answer = "".join(full_answer)
    else:
        # Jalur web-fallback: stream call #1 dengan supresi sentinel berbuffer.
        # ponytail: buffer ditahan selama akumulasi masih prefiks "NO_ANSWER";
        # begitu menyimpang -> flush. Bisa menunda <=1 token di kasus jawaban
        # yang kebetulan diawali "N". Trade-off untuk mencegah sentinel bocor.
        sys_msg = _build_system_message(context_text, extra_system, with_sentinel=True)
        formatted = [sys_msg] + list(history) + [HumanMessage(content=question)]
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
                continue  # masih mungkin sentinel; tahan
            # menyimpang -> jawaban nyata; flush buffer sekaligus
            emitted = True
            flushed = "".join(buffer)
            full_answer.append(flushed)
            yield flushed, "token"

        joined_final = "".join(buffer).strip()
        if not emitted and joined_final == NO_ANSWER_SENTINEL:
            results = await asyncio.to_thread(search_web, question)
            if results:
                web_sys = _build_system_message(
                    web_context_text(results), extra_system, with_sentinel=False
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
            else:
                answer = NOT_FOUND_MSG
                sources_docs = []
                yield answer, "token"
        elif not emitted:
            # buffer tertahan tapi bukan sentinel penuh (jawaban pendek) -> flush
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

    # Token usage (estimasi).
    history_text = " ".join(msg.content for msg in history) if history else ""
    t_input = (
        estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
        + estimate_tokens(context_text)
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
```

Catatan: import `SystemMessage` kini lewat `_build_system_message`, jadi baris `from langchain_core.messages import SystemMessage` di dalam fungsi lama dihapus (sudah tidak dipakai).

- [ ] **Step 4: Jalankan test — pastikan LULUS**

Run: `python -m pytest tests/test_chat_web_fallback.py -v`
Expected: PASS (5 tests: 3 sync + 2 stream).

- [ ] **Step 5: Regresi — pastikan test streaming/lain lama tetap hijau**

Run: `python -m pytest tests/test_chat_single_system_message.py tests/test_chat_date_guidance.py tests/test_history_aware.py -v`
Expected: PASS (perilaku default tak berubah).

- [ ] **Step 6: Commit**

```bash
git add src/chat.py tests/test_chat_web_fallback.py
git commit -m "feat(web-search): fallback streaming + supresi sentinel berbuffer"
```

---

## Task 4: API wiring (models + endpoints)

**Files:**
- Modify: `src/api_models.py` (`ChatRequest`, `ChatResponse`)
- Modify: `api.py` (`chat_endpoint` ~350-381, `chat_stream_endpoint` ~384-407, import)
- Test: `tests/test_api.py` (tambah test)

**Interfaces:**
- Consumes: `answer_with_web_fallback`, `stream_chat_response` (Task 2-3); `config.WEB_SEARCH_ENABLED` (Task 1).
- Produces: `ChatRequest.enable_web_search: bool = False`; `ChatResponse.web_search_used: bool = False`; endpoint `/chat` & `/chat/stream` yang menggerbang `web_active`.

- [ ] **Step 1: Tambah field di `src/api_models.py`**

Di `ChatRequest` (setelah `system_prompt`):

```python
    enable_web_search: bool = Field(
        default=False,
        description="Aktifkan fallback web search bila jawaban tak ada di dokumen",
    )
```

Di `ChatResponse` (setelah `elapsed_ms`):

```python
    web_search_used: bool = False
```

- [ ] **Step 2: Tulis test API yang gagal — tambahkan ke `tests/test_api.py`**

```python
def test_chat_uses_web_fallback_when_enabled(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)
    monkeypatch.setattr(api.config, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(api, "_get_or_create_vector_store", lambda: object())
    monkeypatch.setattr(
        api,
        "answer_with_web_fallback",
        lambda q, h, vs, extra: ("Jawaban web", [], True),
    )

    resp = client.post(
        "/api/v1/chat",
        headers=_auth_header(),
        json={"question": "apa itu X?", "enable_web_search": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Jawaban web"
    assert body["web_search_used"] is True


def test_chat_ignores_web_when_global_disabled(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)
    monkeypatch.setattr(api.config, "WEB_SEARCH_ENABLED", False)

    class _FakeChain:
        def invoke(self, _payload):
            return {"answer": "Jawaban RAG", "context": []}

    monkeypatch.setattr(api, "_get_or_create_chain", lambda: _FakeChain())

    sentinel = {"web": False}
    monkeypatch.setattr(
        api,
        "answer_with_web_fallback",
        lambda *a, **k: sentinel.__setitem__("web", True) or ("x", [], True),
    )

    resp = client.post(
        "/api/v1/chat",
        headers=_auth_header(),
        json={"question": "apa itu X?", "enable_web_search": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Jawaban RAG"
    assert body["web_search_used"] is False
    assert sentinel["web"] is False  # helper web TIDAK dipanggil
```

- [ ] **Step 3: Jalankan test — pastikan GAGAL**

Run: `python -m pytest tests/test_api.py -k web -v`
Expected: FAIL — `answer_with_web_fallback` belum diimpor di `api`, dan respons belum punya `web_search_used`.

- [ ] **Step 4: Tambah import di `api.py`**

Ubah baris 44:

```python
from src.chat import SYSTEM_PROMPT_TEMPLATE, estimate_tokens, get_chat_chain, stream_chat_response
```

menjadi:

```python
from src.chat import (
    SYSTEM_PROMPT_TEMPLATE,
    answer_with_web_fallback,
    estimate_tokens,
    get_chat_chain,
    stream_chat_response,
)
```

- [ ] **Step 5: Ganti isi `chat_endpoint` di `api.py` (baris ~350-381)**

```python
@api_router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    _chat_requests.labels(type="sync").inc()
    session_id = (payload.session_id or "default").strip() or "default"
    history = _session_histories.setdefault(session_id, [])
    web_active = bool(payload.enable_web_search and config.WEB_SEARCH_ENABLED)

    start = time.perf_counter()
    if web_active:
        with _request_duration.labels(endpoint="/chat").time():
            answer, sources, web_used = answer_with_web_fallback(
                payload.question,
                history,
                _get_or_create_vector_store(),
                (payload.system_prompt or "").strip(),
            )
        context_docs = []
    else:
        chain = _get_or_create_chain()
        with _request_duration.labels(endpoint="/chat").time():
            response = chain.invoke(
                {
                    "input": payload.question,
                    "chat_history": history,
                    "extra_system": (payload.system_prompt or "").strip(),
                }
            )
        answer = response.get("answer", "No answer generated.")
        context_docs = response.get("context", [])
        sources = build_source_items(context_docs)
        web_used = False

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
```

- [ ] **Step 6: Update `chat_stream_endpoint` di `api.py` (baris ~384-407)**

```python
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
```

- [ ] **Step 7: Jalankan test — pastikan LULUS**

Run: `python -m pytest tests/test_api.py -k web -v`
Expected: PASS (2 tests).

- [ ] **Step 8: Suite penuh + regresi**

Run: `python -m pytest tests/ -q`
Expected: PASS semua (termasuk test_api lama, perilaku default tak berubah).

- [ ] **Step 9: Commit**

```bash
git add src/api_models.py api.py tests/test_api.py
git commit -m "feat(web-search): wiring API /chat & /chat/stream (enable_web_search)"
```

---

## Self-Review (sudah dijalankan saat penulisan plan)

- **Spec coverage:** B→Task 2/3 (sentinel); C→Task 1 (web_search.py); D→Task 2/3 (orkestrasi); E→Task 1 (config); F→Task 4 (API); G→tests tersebar di Task 1-4. ✔
- **Placeholder scan:** tak ada TBD/TODO; semua step memuat kode konkret. ✔
- **Type consistency:** `answer_with_web_fallback(question, history, vector_store, extra_system)` dipanggil identik di Task 4; `stream_chat_response(..., enable_web_search)` konsisten Task 3↔4; event `{"used": bool}` konsisten Task 3↔4; `WebResult`/`search_web`/`web_results_to_documents`/`web_context_text` konsisten Task 1↔2↔3. ✔
