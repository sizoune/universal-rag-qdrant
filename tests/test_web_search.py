import importlib
from types import SimpleNamespace

import pytest

import src.web_search as web_search


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_search_web_parses_results(monkeypatch):
    payload = {
        "results": [
            {"title": "T1", "url": "https://a.test", "snippet": "snip1", "score": 0.9},
            {"title": "T2", "url": "https://b.test", "content": "snip2"},
        ],
        "errors": [],
    }
    monkeypatch.setattr(
        web_search.requests, "post", lambda *a, **k: _FakeResp(payload)
    )
    monkeypatch.setattr(web_search.config, "LLM_BASE_URL", "http://host:9/v1")
    results = web_search.search_web("apa kabar")
    assert [r.url for r in results] == ["https://a.test", "https://b.test"]
    assert results[0].snippet == "snip1"
    assert results[1].snippet == "snip2"  # fallback ke "content"


def test_search_web_timeout_returns_empty(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(web_search.requests, "post", _boom)
    assert web_search.search_web("x") == []


def test_search_web_errors_field_returns_empty(monkeypatch):
    payload = {"results": [{"url": "https://a.test"}], "errors": ["quota"]}
    monkeypatch.setattr(
        web_search.requests, "post", lambda *a, **k: _FakeResp(payload)
    )
    assert web_search.search_web("x") == []


def test_search_url_derives_from_llm_base_url(monkeypatch):
    monkeypatch.setattr(web_search.config, "WEB_SEARCH_URL", "")
    monkeypatch.setattr(web_search.config, "LLM_BASE_URL", "http://host:9/v1")
    assert web_search._search_url() == "http://host:9/v1/search"


def test_web_results_to_documents_metadata():
    results = [web_search.WebResult(title="T", url="https://a.test", snippet="s", score=0.5)]
    docs = web_search.web_results_to_documents(results)
    assert docs[0].metadata["source_type"] == "web"
    assert docs[0].metadata["source"] == "https://a.test"
    assert docs[0].page_content == "s"
