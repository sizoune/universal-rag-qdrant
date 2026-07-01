"""Strict RAG: relevance gate + sentinel on all paths."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

import src.chat as chat


class _FakeLLM:
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


def test_relevance_gate_blocks_low_rerank_scores(monkeypatch):
    docs = [Document(page_content="irrelevant", metadata={"source": "a"})]
    monkeypatch.setattr(chat, "is_reranker_enabled", lambda: True)
    monkeypatch.setattr(
        chat,
        "rerank_with_scores",
        lambda q, d, top_k=None: [(docs[0], -2.5)],
    )

    gated = chat._apply_relevance_gate("buatkan config nginx", docs)
    assert gated == []


def test_relevance_gate_keeps_high_rerank_scores(monkeypatch):
    docs = [Document(page_content="relevant", metadata={"source": "a"})]
    monkeypatch.setattr(chat, "is_reranker_enabled", lambda: True)
    monkeypatch.setattr(
        chat,
        "rerank_with_scores",
        lambda q, d, top_k=None: [(docs[0], 1.5)],
    )

    gated = chat._apply_relevance_gate("siapa ketua dprd", docs)
    assert len(gated) == 1
    assert gated[0].metadata["rerank_score"] == 1.5


def test_off_topic_returns_not_found_without_llm(monkeypatch):
    llm = _FakeLLM(["should not be called"])
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(
        chat,
        "build_history_aware_retriever",
        lambda vs, _llm: _FakeRetriever(
            [Document(page_content="wifi desa", metadata={"source": "x"})]
        ),
    )
    monkeypatch.setattr(chat, "is_reranker_enabled", lambda: True)
    monkeypatch.setattr(
        chat,
        "rerank_with_scores",
        lambda q, d, top_k=None: [(d[0], -3.0)],
    )

    answer, sources, web_used, ctx = chat.answer_with_web_fallback(
        "buatkan config nginx", [], object(), enable_web_search=False
    )
    assert answer == chat.NOT_FOUND_MSG
    assert sources == []
    assert ctx == []
    assert web_used is False
    assert llm.calls == 0


def test_sentinel_without_web_returns_not_found(monkeypatch):
    llm = _FakeLLM([chat.NO_ANSWER_SENTINEL])
    docs = [Document(page_content="ctx", metadata={"source": "/a.txt"})]
    monkeypatch.setattr(chat, "get_llm", lambda: llm)
    monkeypatch.setattr(
        chat, "build_history_aware_retriever", lambda vs, _llm: _FakeRetriever(docs)
    )
    monkeypatch.setattr(chat, "is_reranker_enabled", lambda: False)

    answer, sources, web_used, ctx = chat.answer_with_web_fallback(
        "q", [], object(), enable_web_search=False
    )
    assert answer == chat.NOT_FOUND_MSG
    assert sources == []
    assert ctx == []
    assert web_used is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
