import os
import importlib
from unittest.mock import patch


def test_config_loads_env_vars():
    # Environment variables are already mocked by conftest.py
    # So we simply import the config module
    import src.config

    # Force reload in case it was imported by another test before the mock took effect
    importlib.reload(src.config)
    from src.config import config

    assert config.EMBEDDER_BASE_URL == "https://api.openai.com/v1"
    assert config.EMBEDDER_MODEL == "text-embedding-3-small"
    assert config.EMBEDDER_DIMENSION == 1536
    assert config.VECTOR_PROVIDER == "qdrant"
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
def test_config_handles_missing_vars():
    import src.config

    importlib.reload(src.config)
    from src.config import config

    # Defaults applied
    assert config.EMBEDDER_MODEL == "text-embedding-3-small"
    assert config.QDRANT_URL == "http://localhost:6333"
    assert config.EMBEDDING_BATCH_SIZE == 100
