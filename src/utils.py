import hashlib
import os


def get_file_hash(filepath: str) -> str:
    """Computes SHA256 hash of a file for incremental caching."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def is_file_allowed(filepath: str, max_size_mb: int = 1) -> bool:
    """
    Checks if a file is permitted (e.g., skips .git, node_modules,
    binary files, or oversized files).
    """
    # 1. Skip specific directories
    ignore_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv"}
    # Normalize path to handle both Windows (\) and Unix (/) separators uniformly
    normalized_path = filepath.replace("\\", "/")
    path_parts = normalized_path.split("/")
    if any(ignore in path_parts for ignore in ignore_dirs):
        return False

    # 2. Check file size (>1MB skipped)
    try:
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            return False
    except OSError:
        return False

    # 3. Accepted Extensions
    accepted_exts = [".txt", ".pdf", ".csv", ".docx", ".md", ".py", ".js", ".html"]
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in accepted_exts:
        return False

    return True
