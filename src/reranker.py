"""
Cross-encoder re-ranking for improved search precision.
Uses fastembed's lightweight reranker model (no GPU required).
"""

import logging
from src.config import config

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    """Get or create a singleton reranker instance."""
    global _reranker
    if _reranker is None:
        model_name = getattr(config, "RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Loading reranker model: {model_name} (first-time download)...")
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            from fastembed import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name=model_name)
        logger.info("Reranker model loaded.")
    return _reranker


def is_reranker_enabled() -> bool:
    """Check if reranker is enabled via config."""
    enabled = getattr(config, "RERANKER_ENABLED", "false")
    return str(enabled).lower() in ("true", "1", "yes")


def warm_reranker() -> None:
    """Preload the reranker at startup so the first real query doesn't pay the
    cold-load (model download + ONNX session init, ~several seconds).
    No-op when reranking is disabled; never raises (best-effort)."""
    if not is_reranker_enabled():
        return
    try:
        reranker = get_reranker()
        # Tiny dummy rerank to force the ONNX inference session to initialise.
        list(reranker.rerank("warmup", ["warmup document"]))
        logger.info("Reranker warmed up.")
    except Exception as e:
        logger.warning("Reranker warm-up failed (continuing lazily): %s", e)


def rerank(query: str, documents: list, top_k: int = None) -> list:
    """Re-rank documents using cross-encoder scoring.

    Args:
        query: The search query
        documents: List of LangChain Documents
        top_k: Optional limit on returned documents

    Returns:
        Re-ranked list of Documents (highest relevance first)
    """
    if not documents:
        return documents

    if not is_reranker_enabled():
        return documents

    reranker = get_reranker()

    # Build query-document pairs
    texts = [doc.page_content for doc in documents]
    raw_scores = list(reranker.rerank(query, texts))

    # Normalize scores: newer fastembed returns floats, older returns objects with .relevance_score
    scores = [
        s.relevance_score if hasattr(s, "relevance_score") else float(s)
        for s in raw_scores
    ]

    # Sort by score descending
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    result = [doc for doc, _ in scored_docs]

    if top_k:
        result = result[:top_k]

    logger.info(
        f"Re-ranked {len(documents)} docs → top score: "
        f"{scored_docs[0][1]:.4f}, "
        f"bottom: {scored_docs[-1][1]:.4f}"
    )

    return result
