import os
import importlib
from unittest.mock import patch


def test_config_loads_env_vars():
    # Environment variables are already mocked by conftest.py
    import src.config

    importlib.reload(src.config)
    from src.config import config

    # Embedding config
    assert config.EMBEDDER_BASE_URL == "https://api.openai.com/v1"
    assert config.EMBEDDER_MODEL == "text-embedding-3-small"
    assert config.EMBEDDER_DIMENSION == 1536
    # Qdrant config
    assert config.QDRANT_URL == "http://localhost:6333"
    assert config.QDRANT_COLLECTION_NAME == "test_collection"
    # LLM Chat config
    assert config.LLM_BASE_URL == "https://api.openai.com/v1"
    assert config.LLM_API_KEY == "test-llm-key-456"
    assert config.LLM_MODEL == "gpt-3.5-turbo"
    # Memory config
    assert config.MEMORY_WINDOW_SIZE == 5


@patch.dict(
    os.environ,
    {"EMBEDDER_DIMENSION": "invalid_string", "SEARCH_SCORE_THRESHOLD": "not_a_float"},
    clear=True,
)
def test_config_handles_invalid_types():
    import src.config

    importlib.reload(src.config)
    from src.config import config

    # Should fallback to defaults if int() or float() casting fails
    assert config.EMBEDDER_DIMENSION == 1536
    assert config.SEARCH_SCORE_THRESHOLD == 0.7


@patch.dict(os.environ, {}, clear=True)
@patch("src.config.load_dotenv")
def test_config_handles_missing_vars(mock_load_dotenv):
    import src.config

    importlib.reload(src.config)
    from src.config import config

    # Defaults applied
    assert config.EMBEDDER_MODEL == "text-embedding-3-small"
    assert config.QDRANT_URL == "http://localhost:6333"
    assert config.EMBEDDING_BATCH_SIZE == 100
    # LLM defaults
    assert config.LLM_BASE_URL == ""
    assert config.LLM_MODEL == "llama3"


@patch.dict(
    os.environ,
    {
        "LLM_BASE_URL": "http://localhost:11434",
        "LLM_API_KEY": "",
        "LLM_MODEL": "llama3",
        "EMBEDDER_BASE_URL": "https://api.openai.com/v1",
        "EMBEDDER_MODEL": "text-embedding-3-small",
    },
    clear=True,
)
@patch("src.config.load_dotenv")
def test_config_llm_independent_from_embedder(mock_load_dotenv):
    """Verify LLM and Embedder configs are truly independent."""
    import src.config

    importlib.reload(src.config)
    from src.config import config

    assert config.LLM_BASE_URL == "http://localhost:11434"
    assert config.LLM_MODEL == "llama3"
    assert config.EMBEDDER_BASE_URL == "https://api.openai.com/v1"
    assert config.EMBEDDER_MODEL == "text-embedding-3-small"
