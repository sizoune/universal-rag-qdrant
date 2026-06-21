"""History-aware retrieval: a fresh session must NOT rewrite the question,
a continued session MUST condense the follow-up before retrieval.

No network/LLM: a FakeListChatModel supplies the condensed query and a recorder
stands in for the base retriever to capture exactly what query reached it.
"""
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

import src.chat as chat


class _Recorder:
    """Base-retriever stand-in that records the query it was invoked with."""

    last_query = None

    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query, **kwargs):
        type(self).last_query = query if isinstance(query, str) else getattr(query, "content", query)
        return self._docs


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs

    def as_retriever(self, **kwargs):
        return _Recorder(self._docs)


def _build(monkeypatch, condensed):
    monkeypatch.setattr(chat.config, "SEARCH_MODE", "dense")
    _Recorder.last_query = None
    docs = [Document(page_content="hasil", metadata={"source": "a.pdf"})]
    llm = FakeListChatModel(responses=[condensed])
    retriever = chat.build_history_aware_retriever(_FakeVectorStore(docs), llm)
    return retriever, docs


def test_empty_history_passes_question_through(monkeypatch):
    retriever, docs = _build(monkeypatch, condensed="UNUSED")
    out = retriever.invoke({"input": "apa itu X?", "chat_history": []})
    assert out == docs
    assert _Recorder.last_query == "apa itu X?"  # fresh session => no rewrite


def test_with_history_condenses_before_retrieval(monkeypatch):
    retriever, docs = _build(monkeypatch, condensed="berapa harga produk X?")
    history = [HumanMessage(content="apa itu produk X?"), AIMessage(content="penjelasan")]
    out = retriever.invoke({"input": "berapa harganya?", "chat_history": history})
    assert out == docs
    assert _Recorder.last_query == "berapa harga produk X?"  # condensed, not raw follow-up
