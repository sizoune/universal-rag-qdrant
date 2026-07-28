import importlib
import os
from unittest.mock import MagicMock

import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.embedding_manager import FallbackEmbeddings, _is_transport_error


# Helper to reload config and embeddings
def reload_components():
    import src.config
    import src.embedding_manager

    importlib.reload(src.config)
    importlib.reload(src.embedding_manager)
    return src.embedding_manager.get_embedder


def _patch_env(monkeypatch, **env):
    # clear=True equivalent for embedding-related keys
    for key in list(os.environ):
        if key.startswith("EMBEDDER_"):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_openai_embedding_factory(monkeypatch):
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="https://api.openai.com/v1",
        EMBEDDER_API_KEY="sk-dummy123",
        EMBEDDER_MODEL="text-embedding-3-small",
    )
    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, OpenAIEmbeddings)
    assert embedder.model == "text-embedding-3-small"


def test_gemini_embedding_factory(monkeypatch):
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="",
        EMBEDDER_API_KEY="AIzaSy_dummy_gemini_key",
        EMBEDDER_MODEL="models/embedding-001",
    )
    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, GoogleGenerativeAIEmbeddings)


def test_ollama_embedding_factory(monkeypatch):
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="http://localhost:11434",
        EMBEDDER_API_KEY="",
        EMBEDDER_MODEL="nomic-embed-text",
    )
    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, OllamaEmbeddings)


def test_ollama_embedding_factory_remote_default_port(monkeypatch):
    """Detect Ollama by default port 11434 even when host is non-localhost."""
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="http://30.10.10.20:11434",
        EMBEDDER_API_KEY="",
        EMBEDDER_MODEL="bge-m3:latest",
    )
    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, OllamaEmbeddings)


def test_openai_compatible_fallback_when_not_ollama(monkeypatch):
    """Non-Ollama hosts with API key fall through to OpenAI-compatible."""
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="http://internal-vllm:8080/v1",
        EMBEDDER_API_KEY="sk-dummy",
        EMBEDDER_MODEL="BAAI/bge-m3",
    )
    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, OpenAIEmbeddings)


def test_fallback_disabled_when_urls_equal(monkeypatch):
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="http://10.91.101.32:11434",
        EMBEDDER_FALLBACK_BASE_URL="http://10.91.101.32:11434/",
        EMBEDDER_API_KEY="",
        EMBEDDER_MODEL="bge-m3:latest",
    )
    import src.embedding_manager as em

    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, OllamaEmbeddings)
    assert not isinstance(embedder, em.FallbackEmbeddings)


def test_fallback_wrapper_enabled(monkeypatch):
    _patch_env(
        monkeypatch,
        EMBEDDER_BASE_URL="http://100.76.102.104:11434",
        EMBEDDER_FALLBACK_BASE_URL="http://10.91.101.32:11434",
        EMBEDDER_API_KEY="",
        EMBEDDER_MODEL="bge-m3:latest",
        EMBEDDER_FALLBACK_RETRY_SECONDS="60",
    )
    import src.embedding_manager as em

    get_embedder = reload_components()
    embedder = get_embedder()
    # Compare against reloaded class — top-level import is stale after importlib.reload.
    assert isinstance(embedder, em.FallbackEmbeddings)
    assert isinstance(embedder.primary, OllamaEmbeddings)
    assert isinstance(embedder.fallback, OllamaEmbeddings)
    assert embedder.retry_primary_after_seconds == 60


def test_is_transport_error_detection():
    assert _is_transport_error(TimeoutError("timed out"))
    assert _is_transport_error(ConnectionError("connection refused"))
    assert _is_transport_error(OSError("Network unreachable"))
    assert _is_transport_error(RuntimeError("HTTPConnectionPool: Max retries exceeded"))
    assert not _is_transport_error(ValueError("bad input"))
    assert not _is_transport_error(RuntimeError("dimension mismatch"))


def test_fallback_embeddings_switches_on_transport_error():
    primary = MagicMock()
    fallback = MagicMock()
    primary.embed_documents.side_effect = ConnectionError("connection refused")
    fallback.embed_documents.return_value = [[0.1, 0.2]]
    primary.embed_query.side_effect = ConnectionError("down")
    fallback.embed_query.return_value = [0.3, 0.4]

    wrap = FallbackEmbeddings(
        primary,
        fallback,
        primary_label="mark-7",
        fallback_label="local",
        retry_primary_after_seconds=300,
    )
    assert wrap.embed_documents(["a"]) == [[0.1, 0.2]]
    # sticky: second call should not hit primary again
    primary.embed_documents.side_effect = AssertionError("primary should be skipped")
    assert wrap.embed_documents(["b"]) == [[0.1, 0.2]]
    assert fallback.embed_documents.call_count == 2

    assert wrap.embed_query("q") == [0.3, 0.4]
    primary.embed_query.assert_not_called()


def test_fallback_embeddings_does_not_mask_logic_errors():
    primary = MagicMock()
    fallback = MagicMock()
    primary.embed_query.side_effect = ValueError("bad dims")
    wrap = FallbackEmbeddings(
        primary,
        fallback,
        primary_label="mark-7",
        fallback_label="local",
        retry_primary_after_seconds=60,
    )
    with pytest.raises(ValueError, match="bad dims"):
        wrap.embed_query("q")
    fallback.embed_query.assert_not_called()


def test_fallback_retries_primary_after_cooldown(monkeypatch):
    primary = MagicMock()
    fallback = MagicMock()
    primary.embed_query.side_effect = [
        ConnectionError("down"),
        [9.0, 8.0],
    ]
    fallback.embed_query.return_value = [1.0, 2.0]

    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "src.embedding_manager.time.monotonic", lambda: clock["t"]
    )

    wrap = FallbackEmbeddings(
        primary,
        fallback,
        primary_label="mark-7",
        fallback_label="local",
        retry_primary_after_seconds=10,
    )
    assert wrap.embed_query("a") == [1.0, 2.0]
    clock["t"] = 1005.0  # still within cooldown
    assert wrap.embed_query("b") == [1.0, 2.0]
    assert primary.embed_query.call_count == 1

    clock["t"] = 1011.0  # cooldown elapsed — retry primary
    assert wrap.embed_query("c") == [9.0, 8.0]
    assert primary.embed_query.call_count == 2
