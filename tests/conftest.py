import pytest
import os
import importlib
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_env_vars():
    """
    Ensure all tests run with a consistent, isolated environment.
    This prevents local .env files from breaking tests unpredictably.
    We must reload src.config after patching os.environ because
    Config class attributes are evaluated at import time.
    """
    env_vars = {
        "EMBEDDER_BASE_URL": "https://api.openai.com/v1",
        "EMBEDDER_API_KEY": "test-key-123",
        "EMBEDDER_MODEL": "text-embedding-3-small",
        "EMBEDDER_DIMENSION": "1536",
        "VECTOR_PROVIDER": "qdrant",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "QDRANT_COLLECTION_NAME": "test_collection",
        "MEMORY_TYPE": "buffer_window",
        "MEMORY_SESSION_ID": "test_session",
        "MEMORY_WINDOW_SIZE": "5",
        "SEARCH_SCORE_THRESHOLD": "0.7",
        "MAX_SEARCH_RESULTS": "4",
        "EMBEDDING_BATCH_SIZE": "100",
        "MAX_BATCH_SCANNER_RETRIES": "3",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("dotenv.load_dotenv"):
            # Reload config so it picks up the patched env vars
            import src.config

            importlib.reload(src.config)
            # Reload vector_store so it picks up the reloaded config
            import src.vector_store

            importlib.reload(src.vector_store)
            yield
