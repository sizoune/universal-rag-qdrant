import base64

from qdrant_client.http import models as rest

from src.config import config


def encode_source_id(source: str) -> str:
    encoded = base64.urlsafe_b64encode(source.encode("utf-8")).decode("utf-8")
    return encoded.rstrip("=")


def decode_source_id(source_id: str) -> str:
    padding = "=" * (-len(source_id) % 4)
    try:
        return base64.urlsafe_b64decode((source_id + padding).encode("utf-8")).decode(
            "utf-8"
        )
    except Exception as exc:
        raise ValueError("Invalid source_id") from exc


def _aggregate_sources(records) -> dict:
    aggregated = {}
    for rec in records:
        payload = rec.payload or {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        source = metadata.get("source")
        if not source:
            continue
        source_type = metadata.get("source_type", "unknown")
        ingested_at = metadata.get("ingested_at")

        if source not in aggregated:
            aggregated[source] = {
                "source_id": encode_source_id(source),
                "source": source,
                "source_type": source_type,
                "chunk_count": 0,
                "last_seen": ingested_at,
            }
        aggregated[source]["chunk_count"] += 1
        if ingested_at and (
            aggregated[source]["last_seen"] is None
            or ingested_at > aggregated[source]["last_seen"]
        ):
            aggregated[source]["last_seen"] = ingested_at
    return aggregated


def list_indexed_sources(vector_store) -> list[dict]:
    client = vector_store.client
    offset = None
    all_records = []

    while True:
        records, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        all_records.extend(records)
        offset = next_offset
        if offset is None:
            break

    aggregated = _aggregate_sources(all_records)
    return sorted(aggregated.values(), key=lambda item: item["source"])


def get_source_detail(vector_store, source: str) -> dict | None:
    client = vector_store.client
    offset = None
    records_all = []
    while True:
        records, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.source",
                        match=rest.MatchValue(value=source),
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        records_all.extend(records)
        offset = next_offset
        if offset is None:
            break

    aggregated = _aggregate_sources(records_all)
    return aggregated.get(source)
