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
