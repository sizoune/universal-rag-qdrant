"""Knowledge-space (Ruang Pengetahuan) unit tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models as rest

from src.namespace import (
    ApiClient,
    build_namespace_filter,
    parse_token_scopes,
    resolve_write_namespace,
)


def test_parse_token_scopes_happy_path():
    raw = json.dumps(
        {
            "ppid-token": {
                "write": "ppid",
                "read": ["ppid", "tabalong-umum"],
            }
        }
    )
    scopes = parse_token_scopes(raw)
    assert "ppid-token" in scopes
    client = scopes["ppid-token"]
    assert client.write_namespace == "ppid"
    assert client.read_namespaces == ("ppid", "tabalong-umum")
    assert client.is_scoped is True


def test_parse_token_scopes_rejects_malformed():
    assert parse_token_scopes("") == {}
    assert parse_token_scopes("not-json") == {}
    assert parse_token_scopes('{"t":{"write":"ppid"}}') == {}  # missing read
    assert parse_token_scopes('{"t":{"read":["ppid"]}}') == {}  # missing write


def test_resolve_write_namespace_prefers_client():
    client = ApiClient(token="t", write_namespace="ppid", read_namespaces=("ppid",))
    assert resolve_write_namespace(client, "tabalong-umum") == "ppid"
    legacy = ApiClient(token="t", write_namespace=None, read_namespaces=None)
    assert resolve_write_namespace(legacy, "tabalong-umum") == "tabalong-umum"


def test_build_namespace_filter_none_means_full_access():
    assert build_namespace_filter(None) is None


def test_build_namespace_filter_single_and_any():
    single = build_namespace_filter(("ppid",))
    assert isinstance(single, rest.Filter)
    assert single.must[0].match.value == "ppid"

    multi = build_namespace_filter(("ppid", "tabalong-umum"))
    assert isinstance(multi.must[0].match, rest.MatchAny)
    assert set(multi.must[0].match.any) == {"ppid", "tabalong-umum"}


def test_verify_api_key_legacy_and_scoped(monkeypatch):
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    import src.api_auth as api_auth
    from src import config as config_module

    monkeypatch.setattr(config_module.config, "API_BEARER_TOKEN", "legacy-token")
    monkeypatch.setattr(
        config_module.config,
        "API_TOKEN_SCOPES_RAW",
        json.dumps({"ppid-token": {"write": "ppid", "read": ["ppid", "tabalong-umum"]}}),
    )

    legacy = api_auth.verify_api_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="legacy-token")
    )
    assert legacy.read_namespaces is None
    assert legacy.is_scoped is False

    scoped = api_auth.verify_api_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="ppid-token")
    )
    assert scoped.write_namespace == "ppid"
    assert scoped.read_namespaces == ("ppid", "tabalong-umum")

    with pytest.raises(HTTPException) as exc:
        api_auth.verify_api_key(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials="nope")
        )
    assert exc.value.status_code == 401


@patch("src.hybrid_retriever.encode_sparse", return_value=[{"indices": [1], "values": [1.0]}])
def test_hybrid_retriever_applies_filter_to_both_prefetches(_mock_sparse):
    from src.hybrid_retriever import HybridRetriever

    vs = MagicMock()
    vs.embeddings.embed_query.return_value = [0.0] * 8
    vs.vector_name = "dense"
    vs.client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(sparse_vectors={"sparse": {}}))
    )
    vs.client.query_points.return_value = SimpleNamespace(points=[])

    retriever = HybridRetriever(
        vector_store=vs,
        score_threshold=0.7,
        k=4,
        read_namespaces=("ppid", "tabalong-umum"),
    )
    retriever._get_relevant_documents("berapa anggaran?")

    kwargs = vs.client.query_points.call_args.kwargs
    assert "query_filter" in kwargs
    assert kwargs["query_filter"] is not None
    prefetches = kwargs["prefetch"]
    assert len(prefetches) == 2
    for pf in prefetches:
        assert pf.filter is not None


@patch("src.hybrid_retriever.encode_sparse", return_value=[{"indices": [1], "values": [1.0]}])
def test_hybrid_retriever_no_filter_for_legacy(_mock_sparse):
    from src.hybrid_retriever import HybridRetriever

    vs = MagicMock()
    vs.embeddings.embed_query.return_value = [0.0] * 8
    vs.vector_name = "dense"
    vs.client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(sparse_vectors={"sparse": {}}))
    )
    vs.client.query_points.return_value = SimpleNamespace(points=[])

    retriever = HybridRetriever(
        vector_store=vs,
        score_threshold=0.7,
        k=4,
        read_namespaces=None,
    )
    retriever._get_relevant_documents("siapa bupati?")

    kwargs = vs.client.query_points.call_args.kwargs
    assert kwargs.get("query_filter") is None
    for pf in kwargs["prefetch"]:
        assert pf.filter is None


def test_enrich_stamps_namespace():
    from langchain_core.documents import Document

    import api as api_module

    docs = [Document(page_content="x", metadata={"source": "/a.pdf"})]
    api_module._enrich_docs_metadata(
        docs, source="/a.pdf", source_type="local", namespace="ppid"
    )
    assert docs[0].metadata["namespace"] == "ppid"
    assert docs[0].metadata["source_type"] == "local"


def test_ingest_documents_setdefault_namespace(monkeypatch):
    from langchain_core.documents import Document

    import src.vector_store as vs_mod

    monkeypatch.setattr(vs_mod.config, "DEFAULT_WRITE_NAMESPACE", "tabalong-umum")
    monkeypatch.setattr(vs_mod.config, "EMBEDDING_BATCH_SIZE", 10)
    monkeypatch.setattr(vs_mod.config, "QDRANT_COLLECTION_NAME", "test_collection")

    docs = [Document(page_content="hello world chunk", metadata={"source": "/x.pdf"})]

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.config.params.vectors = {"dense": MagicMock()}
    mock_collection.config.params.sparse_vectors = None
    mock_client.get_collection.return_value = mock_collection

    mock_vs = MagicMock()
    mock_vs.client = mock_client
    mock_vs.embeddings.embed_documents.return_value = [[0.1] * 8]

    monkeypatch.setattr(vs_mod, "drop_low_value_chunks", lambda d: d, raising=False)
    # drop_low_value_chunks is imported inside ingest_documents from ingestion
    with patch("src.ingestion.drop_low_value_chunks", side_effect=lambda d: d):
        vs_mod.ingest_documents(docs, mock_vs)

    assert docs[0].metadata["namespace"] == "tabalong-umum"
    mock_client.upsert.assert_called()
