import os
import importlib
from unittest.mock import patch, MagicMock
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings


# Helper to reload config and embeddings
def reload_components():
    import src.config
    import src.embedding_manager

    importlib.reload(src.config)
    importlib.reload(src.embedding_manager)
    return src.embedding_manager.get_embedder


@patch.dict(
    os.environ,
    {
        "EMBEDDER_BASE_URL": "https://api.openai.com/v1",
        "EMBEDDER_API_KEY": "sk-dummy123",
        "EMBEDDER_MODEL": "text-embedding-3-small",
    },
    clear=True,
)
def test_openai_embedding_factory():
    get_embedder = reload_components()
    embedder = get_embedder()

    # Assert type
    assert isinstance(embedder, OpenAIEmbeddings)
    assert embedder.model == "text-embedding-3-small"


@patch.dict(
    os.environ,
    {
        "EMBEDDER_BASE_URL": "",
        "EMBEDDER_API_KEY": "AIzaSy_dummy_gemini_key",
        "EMBEDDER_MODEL": "models/embedding-001",
    },
    clear=True,
)
def test_gemini_embedding_factory():
    get_embedder = reload_components()
    embedder = get_embedder()

    # Assert type based on heuristic (starts with AIza)
    assert isinstance(embedder, GoogleGenerativeAIEmbeddings)


@patch.dict(
    os.environ,
    {
        "EMBEDDER_BASE_URL": "http://localhost:11434",
        "EMBEDDER_API_KEY": "",
        "EMBEDDER_MODEL": "nomic-embed-text",
    },
    clear=True,
)
def test_ollama_embedding_factory():
    get_embedder = reload_components()
    embedder = get_embedder()

    # Assert it is an instance of Ollama depending on the logic
    assert isinstance(embedder, OllamaEmbeddings)


@patch.dict(
    os.environ,
    {
        # Remote Ollama server (non-localhost IP) on default port 11434
        "EMBEDDER_BASE_URL": "http://30.10.10.20:11434",
        "EMBEDDER_API_KEY": "",
        "EMBEDDER_MODEL": "bge-m3:latest",
    },
    clear=True,
)
def test_ollama_embedding_factory_remote_default_port():
    """Detect Ollama by default port 11434 even when host is non-localhost."""
    get_embedder = reload_components()
    embedder = get_embedder()
    assert isinstance(embedder, OllamaEmbeddings)


@patch.dict(
    os.environ,
    {
        # Generic OpenAI-compatible (e.g. vLLM) on a non-Ollama port
        "EMBEDDER_BASE_URL": "http://internal-vllm:8080/v1",
        "EMBEDDER_API_KEY": "sk-dummy",
        "EMBEDDER_MODEL": "BAAI/bge-m3",
    },
    clear=True,
)
def test_openai_compatible_fallback_when_not_ollama():
    """Non-Ollama hosts with API key fall through to OpenAI-compatible."""
    get_embedder = reload_components()
    embedder = get_embedder()
    # Should not be Ollama — should be OpenAI-compatible
    assert isinstance(embedder, OpenAIEmbeddings)
