"""Regression test for the truncate-before-rerank bug.

Symptom seen in prod: query "siapa ketua dprd tabalong" returned a hallucinated
name because the two chunks naming the real Ketua DPRD sat at RRF positions 13
and 20, and the fallback truncated the fused pool to top-k BEFORE the reranker
ran — so the reranker (which ranks the right chunk #1) never saw them.

This pins the contract by driving the REAL HybridRetriever with a mocked Qdrant:
when the reranker is enabled, it must receive the FULL fused candidate pool.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.hybrid_retriever import HybridRetriever


def _fake_points(n: int):
    # Low RRF scores (< threshold) so the hybrid path takes the fallback branch.
    return [
        SimpleNamespace(
            payload={"page_content": f"doc {i}", "metadata": {"source": f"s{i}"}},
            score=1.0 / (i + 2),
        )
        for i in range(n)
    ]


def _retriever_with_pool(n: int, k: int) -> HybridRetriever:
    vs = MagicMock()
    vs.embeddings.embed_query.return_value = [0.0] * 8
    vs.vector_name = "dense"
    # Collection reports a configured sparse vector so the true-hybrid path runs.
    vs.client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(sparse_vectors={"sparse": {}}))
    )
    vs.client.query_points.return_value = SimpleNamespace(points=_fake_points(n))
    return HybridRetriever(vector_store=vs, score_threshold=0.7, k=k)


@patch("src.hybrid_retriever.encode_sparse", return_value=[{"indices": [1], "values": [1.0]}])
def test_reranker_sees_full_pool_not_truncated(_mock_sparse):
    """With reranker enabled, rerank() must get all 20 candidates, not just top-k."""
    k = 10
    retriever = _retriever_with_pool(20, k)
    captured = {}

    def fake_rerank(query, documents, top_k=None):
        captured["n_in"] = len(documents)
        return documents[:top_k]

    with patch("src.hybrid_retriever.is_reranker_enabled", return_value=True), \
         patch("src.hybrid_retriever.rerank", side_effect=fake_rerank):
        docs = retriever._get_relevant_documents("siapa ketua dprd tabalong")

    assert captured["n_in"] == 20, "reranker must see the full fused pool, not a truncated slice"
    assert len(docs) == k


@patch("src.hybrid_retriever.encode_sparse", return_value=[{"indices": [1], "values": [1.0]}])
def test_rrf_top_score_must_not_collapse_pool(_mock_sparse):
    """Rank-1 RRF score can be 1.0 (>= 0.7) — must not filter down to a single doc."""
    k = 10
    points = [
        SimpleNamespace(
            payload={"page_content": f"doc {i}", "metadata": {"source": f"s{i}"}},
            score=1.0 if i == 0 else 1.0 / (i + 2),
        )
        for i in range(20)
    ]
    vs = MagicMock()
    vs.embeddings.embed_query.return_value = [0.0] * 8
    vs.vector_name = "dense"
    vs.client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(sparse_vectors={"sparse": {}}))
    )
    vs.client.query_points.return_value = SimpleNamespace(points=points)
    retriever = HybridRetriever(vector_store=vs, score_threshold=0.7, k=k)
    captured = {}

    def fake_rerank(query, documents, top_k=None):
        captured["n_in"] = len(documents)
        return documents[:top_k]

    with patch("src.hybrid_retriever.is_reranker_enabled", return_value=True), \
         patch("src.hybrid_retriever.rerank", side_effect=fake_rerank):
        docs = retriever._get_relevant_documents("siapa bupati tabalong")

    assert captured["n_in"] == 20
    assert len(docs) == k


@patch("src.hybrid_retriever.encode_sparse", return_value=[{"indices": [1], "values": [1.0]}])
def test_no_reranker_still_caps_at_k(_mock_sparse):
    """Without a reranker, the fallback must still cap the pool at k."""
    k = 10
    retriever = _retriever_with_pool(20, k)
    with patch("src.hybrid_retriever.is_reranker_enabled", return_value=False):
        docs = retriever._get_relevant_documents("q")
    assert len(docs) == k


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
