import os
import tempfile
from unittest.mock import patch
from src.utils import get_file_hash, is_file_allowed


def test_get_file_hash():
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello world")
        tmp_path = tmp.name

    try:
        # Expected SHA-256 for "hello world"
        expected_hash = (
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
        assert get_file_hash(tmp_path) == expected_hash

        # Consistent hashing
        assert get_file_hash(tmp_path) == get_file_hash(tmp_path)
    finally:
        os.remove(tmp_path)


@patch("os.path.getsize")
def test_is_file_allowed_extensions(mock_getsize):
    mock_getsize.return_value = 100
    assert is_file_allowed("document.pdf") is True
    assert is_file_allowed("data.csv") is True
    assert is_file_allowed("notes.txt") is True
    assert is_file_allowed("script.py") is True
    assert is_file_allowed("image.png") is False
    assert is_file_allowed("app.exe") is False


@patch("os.path.getsize")
def test_is_file_allowed_directories(mock_getsize):
    mock_getsize.return_value = 100
    assert is_file_allowed(os.path.join(".git", "config")) is False
    assert is_file_allowed(os.path.join("node_modules", "package.json")) is False
    assert is_file_allowed(os.path.join("venv", "bin", "python")) is False
    assert is_file_allowed(os.path.join("src", "main.py")) is True


def test_is_file_allowed_size():
    # Create a dummy large file (> 1MB)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"0" * (1024 * 1024 + 100))  # 1MB + 100 bytes
        large_file = tmp.name

    try:
        assert is_file_allowed(large_file, max_size_mb=1) is False
        # Test a smaller file
        assert is_file_allowed(__file__) is True
    finally:
        os.remove(large_file)
