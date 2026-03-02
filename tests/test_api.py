import base64
import importlib

from fastapi.testclient import TestClient


def _load_api():
    import api as api_module

    return importlib.reload(api_module)


def _auth_header():
    return {"Authorization": "Bearer test-api-token"}


def test_health_no_auth_required():
    api = _load_api()
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_status_requires_auth():
    api = _load_api()
    client = TestClient(api.app)
    resp = client.get("/api/v1/status")
    assert resp.status_code == 401


def test_status_success_with_auth(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)
    monkeypatch.setattr(
        api,
        "get_db_stats",
        lambda: {
            "collection_name": "test",
            "document_count": 1,
            "dimension": 1536,
            "status": "Online",
        },
    )
    resp = client.get("/api/v1/status", headers=_auth_header())
    assert resp.status_code == 200
    assert resp.json()["collection_name"] == "test"


def test_chat_requires_question(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)
    monkeypatch.setattr(api, "_get_or_create_chain", lambda: object())
    resp = client.post(
        "/api/v1/chat",
        headers=_auth_header(),
        json={"question": ""},
    )
    assert resp.status_code == 400


def test_ingest_uploads_creates_folder_and_returns_success(monkeypatch, tmp_path):
    api = _load_api()
    client = TestClient(api.app)
    monkeypatch.setattr(api.config, "UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api, "_run_ingest_path", lambda path: (0, 0, 0))

    resp = client.post("/api/v1/ingest/uploads", headers=_auth_header())
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_files_list_returns_items(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)
    monkeypatch.setattr(api, "_get_or_create_vector_store", lambda: object())
    monkeypatch.setattr(
        api,
        "list_indexed_sources",
        lambda _vs: [
            {
                "source_id": "abc",
                "source": "/tmp/a.txt",
                "source_type": "local",
                "chunk_count": 3,
                "last_seen": None,
            }
        ],
    )

    resp = client.get("/api/v1/files", headers=_auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["source_id"] == "abc"


def test_files_delete_by_source_id(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)
    source = "/tmp/a.txt"
    source_id = base64.urlsafe_b64encode(source.encode()).decode().rstrip("=")
    monkeypatch.setattr(api, "delete_by_source", lambda _source: 4)

    resp = client.delete(f"/api/v1/files/{source_id}", headers=_auth_header())
    assert resp.status_code == 200
    assert resp.json()["deleted_chunks"] == 4


def test_reingest_all_sources_success(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)

    monkeypatch.setattr(api, "_get_or_create_vector_store", lambda: object())
    monkeypatch.setattr(
        api,
        "list_indexed_sources",
        lambda _vs: [
            {"source": "https://a.test", "source_id": "1"},
            {"source": "/tmp/a.txt", "source_id": "2"},
        ],
    )

    calls = {"count": 0}

    def _fake_reingest(source):
        calls["count"] += 1
        if source.startswith("https://"):
            return api.OperationResponse(
                success=True,
                message="ok",
                deleted_chunks=1,
                added_chunks=2,
            )
        return api.OperationResponse(
            success=True,
            message="ok",
            deleted_chunks=3,
            added_chunks=4,
        )

    monkeypatch.setattr(api, "_reingest_source", _fake_reingest)

    resp = client.post("/api/v1/files/reingest-all", headers=_auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["processed_files"] == 2
    assert body["deleted_chunks"] == 4
    assert body["added_chunks"] == 6
    assert calls["count"] == 2


def test_reingest_all_sources_partial_failure(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)

    monkeypatch.setattr(api, "_get_or_create_vector_store", lambda: object())
    monkeypatch.setattr(
        api,
        "list_indexed_sources",
        lambda _vs: [
            {"source": "https://ok.test", "source_id": "1"},
            {"source": "/tmp/missing.txt", "source_id": "2"},
        ],
    )

    def _fake_reingest(source):
        if source.endswith("missing.txt"):
            raise api.HTTPException(status_code=404, detail="local source file not found")
        return api.OperationResponse(
            success=True,
            message="ok",
            deleted_chunks=0,
            added_chunks=1,
        )

    monkeypatch.setattr(api, "_reingest_source", _fake_reingest)

    resp = client.post("/api/v1/files/reingest-all", headers=_auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["processed_files"] == 1
    assert body["added_chunks"] == 1
    assert "failure" in body["message"]


def test_ingest_file_path_rejects_outside_ingest_base_dir(monkeypatch, tmp_path):
    api = _load_api()
    client = TestClient(api.app)

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")

    monkeypatch.setattr(api.config, "INGEST_BASE_DIR", str(allowed_dir))

    resp = client.post(
        "/api/v1/ingest/file-path",
        headers=_auth_header(),
        json={"path": str(outside_file)},
    )
    assert resp.status_code == 403
    assert "ingest base directory" in resp.json()["detail"]


def test_upload_file_rejects_when_too_large(monkeypatch, tmp_path):
    api = _load_api()
    client = TestClient(api.app)

    monkeypatch.setattr(api.config, "UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api.config, "UPLOAD_MAX_BYTES", 10)

    resp = client.post(
        "/api/v1/files/upload",
        headers=_auth_header(),
        files={"file": ("big.txt", b"12345678901", "text/plain")},
    )
    assert resp.status_code == 413
    assert "max allowed size" in resp.json()["detail"]


def test_ingest_web_invalid_url_returns_400(monkeypatch):
    api = _load_api()
    client = TestClient(api.app)

    def _raise_invalid(_url):
        raise ValueError("invalid web URL: Localhost URLs are not allowed")

    monkeypatch.setattr(api, "parse_web_url", _raise_invalid)

    resp = client.post(
        "/api/v1/ingest/web",
        headers=_auth_header(),
        json={"url": "http://localhost:8000"},
    )
    assert resp.status_code == 400
    assert "invalid web URL" in resp.json()["detail"]


def test_upload_file_returns_400_when_pdf_dependency_missing(monkeypatch, tmp_path):
    api = _load_api()
    client = TestClient(api.app, raise_server_exceptions=False)

    monkeypatch.setattr(api.config, "UPLOADS_DIR", str(tmp_path / "uploads"))

    def _raise_missing_pdf_dep(_filepath):
        raise RuntimeError("PDF ingestion requires `pypdf`. Install dependency and restart service.")

    monkeypatch.setattr(api, "load_local_document", _raise_missing_pdf_dep)

    resp = client.post(
        "/api/v1/files/upload",
        headers=_auth_header(),
        files={"file": ("sample.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert resp.status_code == 400
    assert "pypdf" in resp.json()["detail"]
