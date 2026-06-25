"""Klien web search 9router (POST /v1/search), dipakai sebagai fallback RAG.

Tanpa state, config-driven. Tidak pernah melempar exception ke caller:
kegagalan apa pun -> list kosong (fallback yang gagal cukup berujung
pesan "tidak ditemukan").
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)

# (a) isi penuh halaman: snippet search pendek & tanpa tanggal -> LLM mengutip
# artikel usang. Ambil teks penuh top-N (memuat tanggal di badan artikel) via
# 9router /v1/web/fetch. (c) sosmed = boilerplate login-wall, dibuang dari hasil.
_FETCH_TOP_N = 3
_FETCH_MAX_CHARS = 2000
_FETCH_PROVIDER = "exa"
_SOCIAL_RE = re.compile(
    r"(instagram\.com|youtube\.com|youtu\.be|facebook\.com|tiktok\.com|x\.com|twitter\.com)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    score: float | None = None


def _search_url() -> str:
    """Endpoint /v1/search. WEB_SEARCH_URL eksplisit menang; jika kosong
    diturunkan dari LLM_BASE_URL (mis. http://h/v1 -> http://h/v1/search)."""
    if config.WEB_SEARCH_URL:
        return config.WEB_SEARCH_URL
    base = (config.LLM_BASE_URL or "").rstrip("/")
    return f"{base}/search" if base else ""


def _fetch_url() -> str:
    """Endpoint /v1/web/fetch (URL -> teks). Diturunkan dari LLM_BASE_URL."""
    base = (config.LLM_BASE_URL or "").rstrip("/")
    return f"{base}/web/fetch" if base else ""


def _api_key() -> str:
    return config.WEB_SEARCH_API_KEY or config.LLM_API_KEY or ""


def search_web(query: str) -> list[WebResult]:
    url = _search_url()
    if not url:
        logger.warning("Web search dilewati: WEB_SEARCH_URL/LLM_BASE_URL kosong.")
        return []

    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": config.WEB_SEARCH_PROVIDER,
        "query": query,
        "max_results": config.WEB_SEARCH_MAX_RESULTS,
    }

    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=config.WEB_SEARCH_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("Web search gagal: %s", e)
        return []

    if data.get("errors"):
        logger.warning("Web search mengembalikan error: %s", data["errors"])
        return []

    results: list[WebResult] = []
    for item in data.get("results") or []:
        item_url = item.get("url")
        if not item_url:
            continue
        results.append(
            WebResult(
                title=item.get("title") or item_url,
                url=item_url,
                snippet=item.get("snippet") or item.get("content") or "",
                score=item.get("score"),
            )
        )
    return _enrich_results(results)


def _fetch_page_content(url: str) -> str:
    """Teks penuh satu halaman via 9router /v1/web/fetch. Best-effort -> ""."""
    endpoint = _fetch_url()
    if not endpoint:
        return ""
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        resp = requests.post(
            endpoint,
            json={
                "model": _FETCH_PROVIDER,
                "url": url,
                "format": "text",
                "max_characters": _FETCH_MAX_CHARS,
            },
            headers=headers,
            timeout=config.WEB_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        node = resp.json()
        node = node.get("data") or node
        content = node.get("content")
        text = content if isinstance(content, str) else (content or {}).get("text")
        return (text or "").strip()
    except (requests.RequestException, ValueError) as e:  # best-effort
        logger.warning("Web fetch gagal untuk %s: %s", url, e)
        return ""


def _enrich_results(results: list[WebResult]) -> list[WebResult]:
    """(c) buang hasil sosmed (boilerplate), lalu (a) ganti snippet top-N dengan
    isi halaman penuh (memuat tanggal) agar LLM bisa menilai recency. Best-effort:
    fetch yang gagal -> snippet asli dipertahankan."""
    non_social = [r for r in results if not _SOCIAL_RE.search(r.url)]
    cleaned = non_social or results  # jangan sampai kosong total

    targets = cleaned[:_FETCH_TOP_N]
    if not config.WEB_SEARCH_FETCH_CONTENT or not targets:
        return cleaned

    with ThreadPoolExecutor(max_workers=len(targets)) as ex:
        texts = list(ex.map(lambda r: _fetch_page_content(r.url), targets))
    fulltext = {r.url: t for r, t in zip(targets, texts) if t}
    if not fulltext:
        return cleaned

    return [
        WebResult(title=r.title, url=r.url, snippet=fulltext[r.url], score=r.score)
        if r.url in fulltext
        else r
        for r in cleaned
    ]


def web_results_to_documents(results: list[WebResult]) -> list[Document]:
    """Map ke Document agar mengalir ke build_source_items (source_type='web')."""
    return [
        Document(
            page_content=r.snippet,
            metadata={"source": r.url, "source_type": "web", "title": r.title},
        )
        for r in results
    ]


def web_context_text(results: list[WebResult]) -> str:
    """Blok konteks untuk LLM call #2 (judul + url + snippet)."""
    return "\n\n".join(f"[{r.title}]({r.url})\n{r.snippet}" for r in results)
