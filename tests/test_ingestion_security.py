from src.ingestion import validate_public_http_url


def test_validate_public_http_url_rejects_localhost():
    try:
        validate_public_http_url("http://localhost:8000/test")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Localhost" in str(exc)


def test_validate_public_http_url_rejects_private_ip_literal():
    try:
        validate_public_http_url("http://10.10.10.10/data")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Private" in str(exc) or "non-routable" in str(exc)


def test_validate_public_http_url_rejects_non_http_scheme():
    try:
        validate_public_http_url("file:///etc/passwd")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "http/https" in str(exc)
