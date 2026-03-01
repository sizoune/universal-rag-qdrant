from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore
from src.config import config
from src.embedding_manager import get_embedder
import logging

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(url=config.QDRANT_URL)
    return client


def initialize_vector_store() -> QdrantVectorStore:
    """
    Initializes Qdrant, validates dimensions (Strict Dimension Checking),
    and returns a LangChain Qdrant vector store instance.
    """
    client = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME
    expected_dimension = config.EMBEDDER_DIMENSION

    try:
        collection_info = client.get_collection(collection_name)
        # Collection exists, check dimensions
        # Qdrant 1.7+ collection_info structure format
        existing_dim = collection_info.config.params.vectors.size

        if existing_dim != expected_dimension:
            logger.error(f"Dimension Mismatch Alert!")
            logger.error(
                f"Existing collection '{collection_name}' has dimension {existing_dim}."
            )
            logger.error(
                f"Current .env expects dimension {expected_dimension} (Model: {config.EMBEDDER_MODEL})."
            )
            logger.error(
                f"Please use the Re-Index option or Migration tool, or fix .env."
            )
            raise ValueError(
                f"Strict Dimension Checking Failed. Expected {expected_dimension}, got {existing_dim}."
            )

        logger.info(
            f"Collection '{collection_name}' found and dimension {expected_dimension} verified."
        )

    except Exception as e:
        if "Strict Dimension Checking Failed" in str(e) or "Failed. Expected" in str(e):
            raise e  # Re-raise custom validation error

        # Collection doesn't exist or other error, try to create it
        logger.info(
            f"Collection '{collection_name}' not found. Creating new collection with dimension {expected_dimension}."
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=expected_dimension, distance=rest.Distance.COSINE
            ),
        )

    embedder = get_embedder()

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embedder
    )

    return vector_store


def get_db_stats():
    """Helper to get current database statistics."""
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
        count = collection_info.points_count
        dim = collection_info.config.params.vectors.size
        return {
            "collection_name": config.QDRANT_COLLECTION_NAME,
            "document_count": count,
            "dimension": dim,
            "status": "Online",
        }
    except Exception as e:
        return {"error": str(e), "status": "Offline or Collection not found"}


def clear_database():
    """Dangerous option: Re-Index/Clear collection."""
    client = get_qdrant_client()
    try:
        client.delete_collection(config.QDRANT_COLLECTION_NAME)
        logger.info(f"Collection {config.QDRANT_COLLECTION_NAME} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False


def delete_by_source(source: str) -> int:
    """Delete all points in Qdrant that match the given source metadata.
    Returns the number of points deleted."""
    client = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME

    try:
        client.get_collection(collection_name)
    except Exception:
        return 0

    # Scroll through all points matching this source
    point_ids = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.source",
                        match=rest.MatchValue(value=source),
                    )
                ]
            ),
            limit=100,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )

        if not records:
            break

        point_ids.extend([record.id for record in records])
        offset = next_offset
        if offset is None:
            break

    if point_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=rest.PointIdsList(points=point_ids),
        )
        logger.info(f"Deleted {len(point_ids)} old points for source: {source}")

    return len(point_ids)
