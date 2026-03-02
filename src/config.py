import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # === Embedding Config ===
    EMBEDDER_BASE_URL = os.getenv("EMBEDDER_BASE_URL")
    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "text-embedding-3-small")
    try:
        EMBEDDER_DIMENSION = int(os.getenv("EMBEDDER_DIMENSION", "1536"))
    except ValueError:
        EMBEDDER_DIMENSION = 1536

    # === Qdrant Config ===
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME = os.getenv(
        "QDRANT_COLLECTION_NAME", "universal_rag_collection"
    )

    # === LLM Chat Config ===
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

    # === Telegram Bot Config ===
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "")

    # === API Config ===
    API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    try:
        API_PORT = int(os.getenv("API_PORT", "8000"))
    except ValueError:
        API_PORT = 8000
    API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "*")
    UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")
    INGEST_BASE_DIR = os.getenv("INGEST_BASE_DIR", "")
    try:
        UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", "104857600"))
    except ValueError:
        UPLOAD_MAX_BYTES = 104857600
    try:
        WEB_MAX_CONTENT_BYTES = int(os.getenv("WEB_MAX_CONTENT_BYTES", "2097152"))
    except ValueError:
        WEB_MAX_CONTENT_BYTES = 2097152

    # === Memory Config ===
    MEMORY_TYPE = os.getenv("MEMORY_TYPE", "buffer_window")
    MEMORY_SESSION_ID = os.getenv("MEMORY_SESSION_ID", "default_session")
    try:
        MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))
    except ValueError:
        MEMORY_WINDOW_SIZE = 5

    # === Advanced Config ===
    try:
        SEARCH_SCORE_THRESHOLD = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.7"))
        MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "4"))
        EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        MAX_BATCH_SCANNER_RETRIES = int(os.getenv("MAX_BATCH_SCANNER_RETRIES", "3"))
    except ValueError:
        SEARCH_SCORE_THRESHOLD = 0.7
        MAX_SEARCH_RESULTS = 4
        EMBEDDING_BATCH_SIZE = 100
        MAX_BATCH_SCANNER_RETRIES = 3

    # === Hybrid Search & Re-ranking ===
    SEARCH_MODE = os.getenv("SEARCH_MODE", "dense")  # "dense" or "hybrid"
    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


config = Config()
