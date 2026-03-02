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
