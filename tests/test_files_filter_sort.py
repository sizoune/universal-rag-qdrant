"""Tests for the /files filter+sort helper (pure, no Qdrant needed)."""

import api

_filter_sort_files = api._filter_sort_files

ITEMS = [
    {"source": "/up/alpha.pdf", "source_type": "local", "chunk_count": 5,
     "last_seen": "2026-06-01T00:00:00+00:00", "in_s3": True},
    {"source": "/up/beta.md", "source_type": "local", "chunk_count": 20,
     "last_seen": "2026-06-23T00:00:00+00:00", "in_s3": False},
    {"source": "https://x/news", "source_type": "web", "chunk_count": 1,
     "last_seen": None, "in_s3": False},
]


def _sources(items):
    return [it["source"] for it in items]


def test_default_is_newest_first_none_last():
    out = _filter_sort_files(ITEMS, search=None, source_type=None, in_s3=None,
                             sort_by="last_seen", sort_dir="desc")
    # beta (newest) first, alpha next, web (None last_seen) last
    assert _sources(out) == ["/up/beta.md", "/up/alpha.pdf", "https://x/news"]


def test_search_matches_filename():
    out = _filter_sort_files(ITEMS, search="beta", source_type=None, in_s3=None,
                             sort_by="last_seen", sort_dir="desc")
    assert _sources(out) == ["/up/beta.md"]


def test_filter_source_type_and_in_s3():
    web = _filter_sort_files(ITEMS, search=None, source_type="web", in_s3=None,
                             sort_by="last_seen", sort_dir="desc")
    assert _sources(web) == ["https://x/news"]
    s3 = _filter_sort_files(ITEMS, search=None, source_type=None, in_s3=True,
                            sort_by="last_seen", sort_dir="desc")
    assert _sources(s3) == ["/up/alpha.pdf"]


def test_sort_by_chunk_count():
    out = _filter_sort_files(ITEMS, search=None, source_type=None, in_s3=None,
                             sort_by="chunk_count", sort_dir="desc")
    assert [it["chunk_count"] for it in out] == [20, 5, 1]


def test_no_mutation_of_input():
    before = _sources(ITEMS)
    _filter_sort_files(ITEMS, search=None, source_type=None, in_s3=None,
                       sort_by="chunk_count", sort_dir="asc")
    assert _sources(ITEMS) == before
