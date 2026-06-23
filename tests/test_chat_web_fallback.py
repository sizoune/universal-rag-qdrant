from types import SimpleNamespace

import src.chat as chat
from src.web_search import WebResult
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage


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

    answer, sources, web_used, ctx = chat.answer_with_web_fallback("q", [], object(), "")
    assert answer == "Jawaban dari dokumen."
    assert web_used is False
    assert called["web"] is False  # search tidak dipanggil
    assert sources and sources[0].source_type == "local"
    # ctx harus berisi RAG docs yang dikirim ke LLM
    assert ctx == docs


def test_fallback_triggers_on_sentinel(monkeypatch):
    llm = _FakeLLM([chat.NO_ANSWER_SENTINEL, "Jawaban dari web."])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    _patch_common(monkeypatch, llm, docs)
    web_results = [WebResult("T", "https://x.test", "snip", 0.9)]
    monkeypatch.setattr(chat, "search_web", lambda q: web_results)

    answer, sources, web_used, ctx = chat.answer_with_web_fallback("q", [], object(), "")
    assert answer == "Jawaban dari web."
    assert web_used is True
    assert sources and sources[0].source_type == "web"
    assert llm.calls == 2
    # ctx harus berisi web docs (len sesuai jumlah hasil web)
    assert len(ctx) == len(web_results)
    assert ctx[0].metadata["source_type"] == "web"


def test_fallback_no_web_results_returns_not_found(monkeypatch):
    llm = _FakeLLM([chat.NO_ANSWER_SENTINEL])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    _patch_common(monkeypatch, llm, docs)
    monkeypatch.setattr(chat, "search_web", lambda q: [])

    answer, sources, web_used, ctx = chat.answer_with_web_fallback("q", [], object(), "")
    assert answer == chat.NOT_FOUND_MSG
    assert web_used is False
    assert sources == []
    # ctx harus kosong ketika tidak ada hasil web
    assert ctx == []


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


# ---------------------------------------------------------------------------
# FIX 2: invariant — _build_system_message harus menghasilkan SATU SystemMessage
# (lihat mycombo-llm-quirk: beberapa system message -> halusinasi di prod)
# ---------------------------------------------------------------------------

def test_build_system_message_is_single_and_folds_parts():
    msg = chat._build_system_message("CTXDATA", "EXTRADATA", with_sentinel=False)
    assert isinstance(msg, SystemMessage)
    assert "CTXDATA" in msg.content
    assert "EXTRADATA" in msg.content
    assert "Tanggal hari ini" in msg.content  # date guidance folded in
    assert chat.NO_ANSWER_SENTINEL not in msg.content  # no sentinel when with_sentinel=False


def test_build_system_message_sentinel_variant_includes_sentinel():
    msg = chat._build_system_message("CTXDATA", "", with_sentinel=True)
    assert isinstance(msg, SystemMessage)
    assert chat.NO_ANSWER_SENTINEL in msg.content
    assert "CTXDATA" in msg.content


# ---------------------------------------------------------------------------
# FIX 3b: default path (enable_web_search=False) harus emit web_search used=False
# ---------------------------------------------------------------------------

def test_stream_default_path_emits_web_search_false(monkeypatch):
    llm = _FakeStreamLLM([["Hello ", "world"]])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt", "source_type": "local"})]
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(chat, "build_history_aware_retriever", lambda vs, _llm: _FakeRetriever(docs))
    events = _drain(chat.stream_chat_response("q", "s", object(), [], "", enable_web_search=False))
    assert _collect_tokens(events) == "Hello world"
    assert _event(events, "web_search") == {"used": False}
