import os
import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.config import config
from src.embedding_manager import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def migrate_scenario_a(old_client_url: str, old_collection: str):
    """
    Skenario A: Pindah Server Database (Model Embedding Tetap Sama).
    Transfers vectors and payload without re-embedding.
    """
    logger.info(f"--- MIGRATION SCENARIO A ---")

    # 1. Connect to old client
    old_client = QdrantClient(url=old_client_url)

    # 2. Connect to new client (current config)
    new_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    new_collection = config.QDRANT_COLLECTION_NAME

    try:
        old_info = old_client.get_collection(old_collection)
        dim = old_info.config.params.vectors.size
    except Exception as e:
        logger.error(f"Failed to access old collection: {e}")
        return

    # 3. Create new collection
    try:
        new_client.create_collection(
            collection_name=new_collection,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )
        logger.info(f"Created new collection '{new_collection}' with dim {dim}")
    except Exception as e:
        logger.warning(f"New collection might already exist: {e}")

    # 4. Scroll and transfer
    logger.info("Transferring data...")
    offset = None
    total_transferred = 0
    batch_size = 100

    while True:
        records, next_offset = old_client.scroll(
            collection_name=old_collection,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )

        if not records:
            break

        points = [
            rest.PointStruct(id=record.id, vector=record.vector, payload=record.payload)
            for record in records
        ]

        new_client.upsert(collection_name=new_collection, points=points)
        total_transferred += len(records)
        logger.info(f"Transferred {total_transferred} records...")

        offset = next_offset
        if offset is None:
            break

    logger.info(f"Scenario A complete. {total_transferred} records transferred.")


def migrate_scenario_b(old_client_url: str, old_collection: str):
    """
    Skenario B: Ganti Model Embedding (Re-Indexing Wajib).
    Pulls text payload from old DB, re-embeds, and saves to new DB.
    """
    logger.info(f"--- MIGRATION SCENARIO B (Re-Embed) ---")
    old_client = QdrantClient(url=old_client_url)

    # Needs to go through LangChain vectorstore implementation for convenience
    from src.vector_store import initialize_vector_store

    try:
        vector_store = (
            initialize_vector_store()
        )  # This will create the new collection based on ENV
    except Exception as e:
        logger.error(f"Failed to initialize new vector store: {e}")
        return

    logger.info("Fetching payloads from old database...")
    offset = None
    total_reindexed = 0
    batch_size = 50  # Smaller batch for embedding API limits

    from langchain_core.documents import Document

    while True:
        records, next_offset = old_client.scroll(
            collection_name=old_collection,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,  # We don't need old vectors
            offset=offset,
        )

        if not records:
            break

        docs_to_embed = []
        for record in records:
            # Langchain stores text in 'page_content' and rest in 'metadata' usually
            # Qdrant integration puts text inside payload.page_content with metadata dict.
            # Adjust according to how LangChain formats its payloads
            payload = record.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})

            if page_content:
                doc = Document(page_content=page_content, metadata=metadata)
                docs_to_embed.append(doc)

        if docs_to_embed:
            try:
                vector_store.add_documents(docs_to_embed)
                total_reindexed += len(docs_to_embed)
                logger.info(f"Re-embedded {total_reindexed} documents...")
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")

        offset = next_offset
        if offset is None:
            break

    logger.info(f"Scenario B complete. {total_reindexed} documents re-indexed.")


if __name__ == "__main__":
    print("Migration Utilities")
    print("1. Skenario A (Pindah Server, Vector Tetap)")
    print("2. Skenario B (Ganti Model, Re-Embed)")
    choice = input("Pilih skenario (1/2): ")

    if choice in ["1", "2"]:
        old_url = input("URL Qdrant Lama (misal: http://localhost:6333): ").strip()
        old_col = input("Nama Collection Lama: ").strip()

        if choice == "1":
            migrate_scenario_a(old_url, old_col)
        else:
            migrate_scenario_b(old_url, old_col)
    else:
        print("Pilihan dibatalkan.")
