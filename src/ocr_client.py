"""Thin client for an external ocr-gateway (PaddleOCR) service.

OCR is best-effort and fully optional: when OCR_GATEWAY_URL is unset, ocr is
disabled and callers fall back to native text only. Any gateway/network error
returns "" so ingestion never fails because OCR is down.

See: https://github.com/hafiznugrahadev/ocr-gateway  (POST /extract, Bearer auth)
"""

import logging

import requests

from src.config import config

logger = logging.getLogger(__name__)


def ocr_enabled() -> bool:
    """True when an OCR gateway is configured."""
    return bool(config.OCR_GATEWAY_URL)


def ocr_file_bytes(data: bytes, filename: str) -> str:
    """OCR raw file bytes (PDF/image) via the configured gateway.

    Returns "" on any failure.
    """
    if not config.OCR_GATEWAY_URL:
        return ""
    headers = {}
    if config.OCR_API_KEY:
        headers["Authorization"] = f"Bearer {config.OCR_API_KEY}"
    try:
        resp = requests.post(
            f"{config.OCR_GATEWAY_URL}/extract",
            headers=headers,
            files={"file": (filename, data)},
            data={"language": config.OCR_LANGUAGE},
            timeout=config.OCR_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        # ocr-gateway envelope: {"success": true, "result": {"full_text": "..."}}
        result = body.get("result") or {}
        return result.get("full_text") or body.get("text") or ""
    except Exception as exc:  # network, timeout, non-2xx, bad JSON — all best-effort
        logger.warning("OCR failed for %s: %s", filename, exc)
        return ""


def ocr_image_bytes(data: bytes, filename: str) -> str:
    """OCR raw image bytes via the configured gateway. Returns "" on any failure."""
    return ocr_file_bytes(data, filename)
