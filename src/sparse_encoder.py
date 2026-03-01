"""
BM25 sparse encoder for hybrid search.
Uses fastembed's Bm25SparseEmbedding for efficient keyword-based sparse vectors.
"""

import logging
from fastembed import SparseTextEmbedding

logger = logging.getLogger(__name__)

_model = None


def get_sparse_encoder() -> SparseTextEmbedding:
    """Get or create a singleton sparse encoder instance."""
    global _model
    if _model is None:
        logger.info("Loading BM25 sparse encoder (first-time download)...")
        _model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 sparse encoder loaded.")
    return _model


def encode_sparse(texts: list[str]) -> list[dict]:
    """Encode texts into sparse vectors for Qdrant.

    Returns list of dicts with 'indices' and 'values' keys,
    compatible with Qdrant's SparseVector format.
    """
    encoder = get_sparse_encoder()
    results = list(encoder.embed(texts))
    sparse_vectors = []
    for result in results:
        sparse_vectors.append(
            {
                "indices": result.indices.tolist(),
                "values": result.values.tolist(),
            }
        )
    return sparse_vectors
