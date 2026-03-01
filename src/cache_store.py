import hashlib
import json
import os
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "ingestion_hashes.json")


def _ensure_cache_dir():
    """Create .cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_cache() -> dict:
    """Load the persistent hash cache from disk."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load cache, starting fresh: {e}")
    return {}


def save_cache(cache: dict):
    """Save the hash cache to disk."""
    _ensure_cache_dir()
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save cache: {e}")


def get_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content (for web URL dedup)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
