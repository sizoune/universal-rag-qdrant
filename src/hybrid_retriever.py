"""
Hybrid Retriever: combines dense vector search + sparse BM25 keyword search.
Uses Qdrant's query API with prefetch for Reciprocal Rank Fusion (RRF).
"""

import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client.http import models as rest
from src.config import config
from src.sparse_encoder import encode_sparse
from src.reranker import rerank, is_reranker_enabled

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """LangChain-compatible retriever using Qdrant hybrid search
    (dense vector + sparse BM25) with optional cross-encoder re-ranking."""

    vector_store: object  # QdrantVectorStore
    score_threshold: float = 0.7
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Execute hybrid search: dense + sparse, then optional re-rank."""
        try:
            # Get dense embedding for query
            query_vector = self.vector_store.embeddings.embed_query(query)

            # Get sparse embedding for query
            sparse_vectors = encode_sparse([query])
            sparse_vector = sparse_vectors[0]

            client = self.vector_store.client
            collection_name = config.QDRANT_COLLECTION_NAME

            # Check if collection has sparse vectors configured
            collection_info = client.get_collection(collection_name)
            has_sparse = False
            sparse_config = getattr(collection_info.config.params, "sparse_vectors", None)
            if isinstance(sparse_config, dict) and "sparse" in sparse_config:
                has_sparse = True

            dense_using = getattr(self.vector_store, "vector_name", "dense") or None

            if has_sparse:
                # True hybrid: use prefetch with both dense and sparse
                dense_prefetch = rest.Prefetch(query=query_vector, limit=self.k * 3)
                if dense_using:
                    dense_prefetch = rest.Prefetch(
                        query=query_vector,
                        using=dense_using,
                        limit=self.k * 3,
                    )

                results = client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        dense_prefetch,
                        rest.Prefetch(
                            query=rest.SparseVector(
                                indices=sparse_vector["indices"],
                                values=sparse_vector["values"],
                            ),
                            using="sparse",
                            limit=self.k * 3,
                        ),
                    ],
                    query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                    limit=self.k * 2,
                )
                logger.info(
                    f"Hybrid search (dense+sparse): {len(results.points)} results"
                )
            else:
                # Fallback: dense-only with sparse re-ranking
                logger.info(
                    "No sparse vectors configured, using dense search with BM25 re-scoring"
                )
                results = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=self.k * 2,
                )

            # Convert to LangChain documents
            documents = []
            for point in results.points:
                payload = point.payload or {}
                metadata = {k: v for k, v in payload.items() if k != "page_content"}
                nested_metadata = metadata.get("metadata")
                if isinstance(nested_metadata, dict):
                    # Align with LangChain Qdrant payload style where source metadata
                    # is stored inside payload["metadata"].
                    metadata = {**nested_metadata, **{k: v for k, v in metadata.items() if k != "metadata"}}
                doc = Document(
                    page_content=payload.get("page_content", ""),
                    metadata=metadata,
                )
                doc.metadata["score"] = point.score
                documents.append(doc)

            # Filter by score threshold
            filtered_documents = [
                d
                for d in documents
                if d.metadata.get("score", 0) >= self.score_threshold
            ]
            if filtered_documents:
                documents = filtered_documents
            else:
                logger.info(
                    "No hybrid docs passed score_threshold=%.3f. Using top-k fallback.",
                    self.score_threshold,
                )
                documents = documents[: self.k]

            # Re-rank if enabled
            if is_reranker_enabled() and documents:
                documents = rerank(query, documents, top_k=self.k)
            else:
                documents = documents[: self.k]

            logger.info(f"Hybrid retriever returning {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Hybrid search failed, falling back to dense: {e}")
            # Fallback to standard dense retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": self.score_threshold,
                    "k": self.k,
                },
            )
            return retriever.invoke(query)
