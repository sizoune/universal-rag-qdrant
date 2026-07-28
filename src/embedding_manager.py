from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_transport_error(exc: BaseException) -> bool:
    """True for connection/timeout failures that justify trying a fallback host."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    name = type(exc).__name__.lower()
    if any(
        token in name
        for token in ("timeout", "connection", "connect", "httpx", "request", "remote")
    ):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "timed out",
            "timeout",
            "connection refused",
            "connection reset",
            "connect error",
            "temporarily unavailable",
            "name or service not known",
            "nodename nor servname",
            "failed to establish",
            "max retries exceeded",
            "all connection attempts failed",
        )
    )


class FallbackEmbeddings(Embeddings):
    """Try primary embedder; on transport errors, stick to fallback for a while.

    Used so ingest can prefer a GPU Ollama host (mark-7) but keep indexing on
    local Ollama when that host is unreachable. Same model/dimension required.
    """

    def __init__(
        self,
        primary: Embeddings,
        fallback: Embeddings,
        *,
        primary_label: str,
        fallback_label: str,
        retry_primary_after_seconds: float = 300.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_label = primary_label
        self.fallback_label = fallback_label
        self.retry_primary_after_seconds = max(0.0, float(retry_primary_after_seconds))
        self._prefer_fallback_until: float = 0.0

    def _using_fallback(self) -> bool:
        return time.monotonic() < self._prefer_fallback_until

    def _stick_to_fallback(self) -> None:
        if self.retry_primary_after_seconds <= 0:
            # 0 = sticky until process restart
            self._prefer_fallback_until = time.monotonic() + 10**9
        else:
            self._prefer_fallback_until = (
                time.monotonic() + self.retry_primary_after_seconds
            )

    def _with_fallback(self, call: Callable[[Embeddings], T]) -> T:
        if not self._using_fallback():
            try:
                return call(self.primary)
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise
                logger.warning(
                    "Primary embedder %s failed (%s: %s); using fallback %s for %.0fs",
                    self.primary_label,
                    type(exc).__name__,
                    exc,
                    self.fallback_label,
                    self.retry_primary_after_seconds
                    if self.retry_primary_after_seconds > 0
                    else float("inf"),
                )
                self._stick_to_fallback()
        try:
            return call(self.fallback)
        except Exception:
            # Fallback also failed — clear sticky so next call retries primary.
            self._prefer_fallback_until = 0.0
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._with_fallback(lambda e: e.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._with_fallback(lambda e: e.embed_query(text))


def _normalize_base_url(url: str | None) -> str:
    return (url or "").strip().rstrip("/")


def _build_embedder(
    base_url: str | None,
    api_key: str | None,
    model_name: str,
) -> Embeddings:
    """Return a concrete LangChain Embeddings client for one endpoint."""
    base_url = _normalize_base_url(base_url)
    api_key = (api_key or "").strip()

    if base_url and "api.openai.com" in base_url:
        logger.info("Using OpenAI Embeddings with model %s", model_name)
        return OpenAIEmbeddings(
            openai_api_key=api_key, model=model_name, openai_api_base=base_url
        )
    if api_key and api_key.startswith(("AIza", "AQ.")):
        logger.info("Using Google Generative AI Embeddings with model %s", model_name)
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    if (
        base_url
        and (
            "localhost" in base_url
            or "127.0.0.1" in base_url
            or "ollama" in base_url.lower()
            or ":11434" in base_url
        )
        and not api_key
    ):
        logger.info("Using Ollama Embeddings with model %s at %s", model_name, base_url)
        return OllamaEmbeddings(base_url=base_url, model=model_name)
    if base_url:
        logger.info(
            "Using OpenAI Compatible Embeddings at %s with model %s",
            base_url,
            model_name,
        )
        return OpenAIEmbeddings(
            openai_api_key=api_key or "sk-dummy",
            model=model_name,
            openai_api_base=base_url,
        )
    logger.info("Defaulting to OpenAI Embeddings with model %s", model_name)
    return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)


def get_embedder() -> Embeddings:
    """
    Factory for the active embedder.

    When EMBEDDER_FALLBACK_BASE_URL is set and differs from EMBEDDER_BASE_URL,
    wrap primary+fallback so transport failures (e.g. mark-7 down) continue on
    the fallback host (local Ollama). Model/dimension must match on both hosts.
    """
    primary = _build_embedder(
        config.EMBEDDER_BASE_URL,
        config.EMBEDDER_API_KEY,
        config.EMBEDDER_MODEL,
    )
    fallback_url = _normalize_base_url(config.EMBEDDER_FALLBACK_BASE_URL)
    primary_url = _normalize_base_url(config.EMBEDDER_BASE_URL)
    if not fallback_url or fallback_url == primary_url:
        return primary

    fallback = _build_embedder(
        fallback_url,
        config.EMBEDDER_FALLBACK_API_KEY,
        config.EMBEDDER_MODEL,
    )
    logger.info(
        "Embedder fallback enabled: primary=%s fallback=%s retry_after=%ss",
        primary_url or "(default)",
        fallback_url,
        config.EMBEDDER_FALLBACK_RETRY_SECONDS,
    )
    return FallbackEmbeddings(
        primary,
        fallback,
        primary_label=primary_url or "primary",
        fallback_label=fallback_url,
        retry_primary_after_seconds=config.EMBEDDER_FALLBACK_RETRY_SECONDS,
    )
