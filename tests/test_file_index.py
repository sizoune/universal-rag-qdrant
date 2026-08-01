"""Tests for source listing aggregation + TTL cache (memory-sensitive path)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import src.file_index as file_index


def _record(source: str, source_type: str = "local", ingested_at: str | None = None):
    return SimpleNamespace(
        payload={
            "metadata": {
                "source": source,
                "source_type": source_type,
                "ingested_at": ingested_at,
            }
        }
    )


def test_scroll_aggregates_without_retaining_pages(monkeypatch):
    """Each scroll page is folded into aggregates; pages are not kept as one giant list."""
    monkeypatch.setattr(file_index.config, "QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(file_index.config, "SOURCES_LIST_CACHE_TTL_SECONDS", 0)
    file_index.invalidate_sources_cache()

    pages = [
        [_record("/a.pdf", ingested_at="2026-01-01T00:00:00+00:00")],
        [
            _record("/a.pdf", ingested_at="2026-01-02T00:00:00+00:00"),
            _record("/b.pdf", ingested_at="2026-01-01T00:00:00+00:00"),
        ],
        [],
    ]
    offsets = [1, None]
    calls = {"n": 0}

    def scroll(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        records = pages[i]
        next_offset = offsets[i] if i < len(offsets) else None
        # Simulate end when empty page
        if not records:
            return [], None
        return records, next_offset

    client = MagicMock()
    client.scroll.side_effect = scroll
    vector_store = MagicMock()
    vector_store.client = client

    out = file_index.list_indexed_sources(vector_store)
    assert [x["source"] for x in out] == ["/a.pdf", "/b.pdf"]
    assert out[0]["chunk_count"] == 2
    assert out[0]["last_seen"] == "2026-01-02T00:00:00+00:00"
    assert out[1]["chunk_count"] == 1
    assert client.scroll.call_count == 2


def test_sources_list_cache_avoids_repeat_scroll(monkeypatch):
    monkeypatch.setattr(file_index.config, "QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(file_index.config, "SOURCES_LIST_CACHE_TTL_SECONDS", 60)
    file_index.invalidate_sources_cache()

    client = MagicMock()
    client.scroll.return_value = (
        [_record("/cached.pdf", ingested_at="2026-01-01T00:00:00+00:00")],
        None,
    )
    vector_store = MagicMock()
    vector_store.client = client

    first = file_index.list_indexed_sources(vector_store, read_namespaces=("ppid",))
    second = file_index.list_indexed_sources(vector_store, read_namespaces=("ppid",))

    assert client.scroll.call_count == 1
    assert first[0]["source"] == "/cached.pdf"
    assert second[0]["source"] == "/cached.pdf"
    # Caller mutation must not poison the cache
    second[0]["source"] = "mutated"
    third = file_index.list_indexed_sources(vector_store, read_namespaces=("ppid",))
    assert third[0]["source"] == "/cached.pdf"


def test_invalidate_sources_cache_forces_rescroll(monkeypatch):
    monkeypatch.setattr(file_index.config, "QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(file_index.config, "SOURCES_LIST_CACHE_TTL_SECONDS", 60)
    file_index.invalidate_sources_cache()

    client = MagicMock()
    client.scroll.return_value = ([_record("/x.pdf")], None)
    vector_store = MagicMock()
    vector_store.client = client

    file_index.list_indexed_sources(vector_store)
    file_index.invalidate_sources_cache()
    file_index.list_indexed_sources(vector_store)
    assert client.scroll.call_count == 2
