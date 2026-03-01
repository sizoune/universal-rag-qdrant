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


config = Config()
