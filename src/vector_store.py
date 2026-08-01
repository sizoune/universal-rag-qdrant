from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore
from src.config import config
from src.embedding_manager import get_embedder
from src.namespace import NAMESPACE_PAYLOAD_KEY
import logging

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(url=config.QDRANT_URL)
    return client


def ensure_namespace_payload_index(client: QdrantClient | None = None) -> None:
    """Create a keyword payload index on metadata.namespace (idempotent)."""
    client = client or get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=NAMESPACE_PAYLOAD_KEY,
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        logger.info("Payload index ensured on %s", NAMESPACE_PAYLOAD_KEY)
    except Exception as exc:
        # Already exists or collection missing — both fine at call sites that
        # create the collection first.
        logger.debug("Payload index create skipped: %s", exc)


def initialize_vector_store() -> QdrantVectorStore:
    """
    Initializes Qdrant, validates dimensions (Strict Dimension Checking),
    and returns a LangChain Qdrant vector store instance.
    """
    client = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME
    expected_dimension = config.EMBEDDER_DIMENSION
    vector_name = ""

    try:
        collection_info = client.get_collection(collection_name)
        # Collection exists, check dimensions
        # Qdrant 1.7+ collection_info structure format
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config, dict):
            if "dense" in vectors_config:
                existing_dim = vectors_config["dense"].size
                vector_name = "dense"
            elif "" in vectors_config:
                existing_dim = vectors_config[""].size
                vector_name = ""
            else:
                first_vector_name, first_vector_params = next(iter(vectors_config.items()))
                existing_dim = first_vector_params.size
                vector_name = first_vector_name
        else:
            existing_dim = vectors_config.size
            vector_name = ""

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
            f"Collection '{collection_name}' not found. Creating new collection with "
            f"dense dimension {expected_dimension}."
        )
        if config.SEARCH_MODE.lower() == "hybrid":
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": rest.VectorParams(
                        size=expected_dimension, distance=rest.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": rest.SparseVectorParams(modifier=rest.Modifier.IDF)
                },
            )
            vector_name = "dense"
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=expected_dimension, distance=rest.Distance.COSINE
                ),
            )
            vector_name = ""

    ensure_namespace_payload_index(client)

    embedder = get_embedder()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedder,
        vector_name=vector_name,
    )

    return vector_store


def get_db_stats():
    """Helper to get current database statistics."""
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
        count = collection_info.points_count
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config, dict):
            if "dense" in vectors_config:
                dim = vectors_config["dense"].size
            elif "" in vectors_config:
                dim = vectors_config[""].size
            else:
                dim = next(iter(vectors_config.values())).size
        else:
            dim = vectors_config.size
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


def delete_by_source(source: str, namespace: str | None = None) -> int:
    """Delete all points in Qdrant that match the given source metadata.
    When ``namespace`` is set, only points in that knowledge space are removed.
    Returns the number of points deleted."""
    from src.file_index import invalidate_sources_cache

    client = get_qdrant_client()
    collection_name = config.QDRANT_COLLECTION_NAME

    try:
        client.get_collection(collection_name)
    except Exception:
        return 0

    must = [
        rest.FieldCondition(
            key="metadata.source",
            match=rest.MatchValue(value=source),
        )
    ]
    if namespace:
        must.append(
            rest.FieldCondition(
                key=NAMESPACE_PAYLOAD_KEY,
                match=rest.MatchValue(value=namespace),
            )
        )
    scroll_filter = rest.Filter(must=must)

    try:
        deleted = int(
            client.count(
                collection_name=collection_name,
                count_filter=scroll_filter,
                exact=True,
            ).count
        )
    except Exception:
        deleted = 0

    if deleted <= 0:
        return 0

    # Filter delete avoids holding thousands of point IDs in Python memory.
    client.delete(
        collection_name=collection_name,
        points_selector=rest.FilterSelector(filter=scroll_filter),
    )
    logger.info(f"Deleted {deleted} old points for source: {source}")
    invalidate_sources_cache()
    return deleted


def ingest_documents(documents: list, vector_store) -> None:
    """
    Custom wrapper to ingest documents into Qdrant.
    It computes dense embeddings (via LangChain) and sparse embeddings (via fastembed)
    then upserts them as named vectors.

    Processing is streamed per EMBEDDING_BATCH_SIZE so large PDFs (thousands of
    chunks) never hold all dense+sparse vectors and PointStructs in memory at once.
    """
    if not documents:
        return

    # Drop boilerplate chunks (footer URLs, "Sumber:" stamps, fragments) before
    # spending embeddings on them. Single choke point — covers file/web/dir.
    from src.file_index import invalidate_sources_cache
    from src.ingestion import drop_low_value_chunks

    before = len(documents)
    documents = drop_low_value_chunks(documents)
    if before != len(documents):
        logger.info(
            "Filtered %d boilerplate chunk(s); %d remain.", before - len(documents), len(documents)
        )
    if not documents:
        return

    import uuid
    from datetime import datetime, timezone

    from src.config import config

    # Stamp ingest time and knowledge-space namespace on every chunk that
    # lacks them. Callers (API) should set namespace from the token scope;
    # setdefault covers CLI/dir paths with DEFAULT_WRITE_NAMESPACE.
    now_iso = datetime.now(timezone.utc).isoformat()
    default_ns = (config.DEFAULT_WRITE_NAMESPACE or "").strip()
    for doc in documents:
        doc.metadata.setdefault("ingested_at", now_iso)
        if default_ns:
            doc.metadata.setdefault("namespace", default_ns)

    client = vector_store.client
    collection_name = config.QDRANT_COLLECTION_NAME
    embedder = vector_store.embeddings
    collection_info = client.get_collection(collection_name)
    params = collection_info.config.params

    vectors_config = params.vectors
    dense_vector_name = ""
    dense_is_named = False

    if isinstance(vectors_config, dict):
        dense_is_named = True
        if "dense" in vectors_config:
            dense_vector_name = "dense"
        elif "" in vectors_config:
            dense_vector_name = ""
        else:
            dense_vector_name = next(iter(vectors_config.keys()))

    sparse_vectors_config = getattr(params, "sparse_vectors", None)
    has_sparse = isinstance(sparse_vectors_config, dict) and "sparse" in sparse_vectors_config
    use_sparse = has_sparse and dense_is_named
    if has_sparse and not dense_is_named:
        logger.warning(
            "Sparse vector config detected but dense vector is unnamed; ingesting dense only."
        )

    encode_sparse = None
    if use_sparse:
        from src.sparse_encoder import encode_sparse as _encode_sparse

        encode_sparse = _encode_sparse

    # Stream: embed → sparse → upsert per batch. Avoids O(N) peak RAM on large docs.
    batch_size = max(1, config.EMBEDDING_BATCH_SIZE)
    total = len(documents)
    logger.info(
        "Ingesting %d chunks in batches of %d (%s)...",
        total,
        batch_size,
        "dense+sparse" if use_sparse else "dense only",
    )

    ingested = 0
    for start in range(0, total, batch_size):
        batch_docs = documents[start : start + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        dense_embeddings = embedder.embed_documents(texts)
        sparse_embeddings = encode_sparse(texts) if encode_sparse is not None else None

        points = []
        for i, doc in enumerate(batch_docs):
            point_id = str(uuid.uuid4())
            payload = {"page_content": doc.page_content, "metadata": doc.metadata}

            if dense_is_named:
                vector_payload = {dense_vector_name: dense_embeddings[i]}
                if sparse_embeddings is not None:
                    sparse_vector = sparse_embeddings[i]
                    vector_payload["sparse"] = rest.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    )
            else:
                vector_payload = dense_embeddings[i]

            points.append(
                rest.PointStruct(id=point_id, payload=payload, vector=vector_payload)
            )

        client.upsert(collection_name=collection_name, points=points)
        ingested += len(points)

    invalidate_sources_cache()
    if use_sparse:
        logger.info(
            "Successfully ingested %d points (dense+sparse) to Qdrant.", ingested
        )
    else:
        logger.info(
            "Successfully ingested %d points (dense only) to Qdrant.", ingested
        )
