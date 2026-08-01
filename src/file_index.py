import base64
import copy
import logging
import threading
import time

from qdrant_client.http import models as rest

from src.config import config
from src.namespace import build_namespace_filter

logger = logging.getLogger(__name__)

# TTL cache for list_indexed_sources. n8n reconciles by paginating /files; each
# page used to full-scroll ~90k Qdrant points. Cache collapses that to one scroll
# per TTL window (keyed by read_namespaces).
_cache_lock = threading.Lock()
_sources_cache: dict[tuple[str, ...], tuple[float, list[dict]]] = {}


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


def invalidate_sources_cache() -> None:
    """Drop cached /files aggregations after ingest or delete."""
    with _cache_lock:
        _sources_cache.clear()


def _cache_key(read_namespaces: tuple[str, ...] | None) -> tuple[str, ...]:
    if read_namespaces is None:
        return ("*",)
    return tuple(sorted(read_namespaces))


def _accumulate_record(aggregated: dict, rec) -> None:
    payload = rec.payload or {}
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    source = metadata.get("source")
    if not source:
        return
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


def _aggregate_sources(records) -> dict:
    aggregated = {}
    for rec in records:
        _accumulate_record(aggregated, rec)
    return aggregated


def _scroll_aggregate_sources(
    client,
    *,
    scroll_filter,
    limit: int = 256,
) -> list[dict]:
    """Scroll Qdrant and aggregate source metadata without retaining every record."""
    offset = None
    aggregated: dict = {}
    pages = 0

    while True:
        records, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=["metadata"],  # buang page_content; cuma butuh metadata
            with_vectors=False,
        )
        pages += 1
        if not records:
            break
        for rec in records:
            _accumulate_record(aggregated, rec)
        offset = next_offset
        if offset is None:
            break

    logger.debug(
        "list_indexed_sources scrolled %d page(s), %d source(s)",
        pages,
        len(aggregated),
    )
    return sorted(aggregated.values(), key=lambda item: item["source"])


def list_indexed_sources(
    vector_store,
    read_namespaces: tuple[str, ...] | None = None,
) -> list[dict]:
    """List aggregated indexed sources, optionally scoped to read namespaces."""
    ttl = max(0, int(getattr(config, "SOURCES_LIST_CACHE_TTL_SECONDS", 0) or 0))
    key = _cache_key(read_namespaces)
    now = time.monotonic()

    if ttl > 0:
        with _cache_lock:
            hit = _sources_cache.get(key)
            if hit and hit[0] > now:
                return copy.deepcopy(hit[1])

    client = vector_store.client
    scroll_filter = build_namespace_filter(read_namespaces)
    result = _scroll_aggregate_sources(client, scroll_filter=scroll_filter)

    if ttl > 0:
        with _cache_lock:
            _sources_cache[key] = (now + ttl, copy.deepcopy(result))

    return result


def get_source_detail(vector_store, source: str) -> dict | None:
    client = vector_store.client
    result = _scroll_aggregate_sources(
        client,
        scroll_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.source",
                    match=rest.MatchValue(value=source),
                )
            ]
        ),
    )
    for item in result:
        if item["source"] == source:
            return item
    return None
