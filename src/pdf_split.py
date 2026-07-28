"""Helpers for splitting large PDFs before OCR.

When ``OCR_GATEWAY_URL`` is set, PDF/PPTX parsers may call the OCR gateway.
Documents with more than ``max_pages`` pages should be processed in ranges
yielded by :func:`iter_page_ranges` so a single OCR request stays bounded.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def iter_page_ranges(
    total_pages: int, max_pages: int = 500
) -> Iterator[tuple[int, int]]:
    """Yield 1-indexed inclusive ``(start, end)`` page ranges.

    Example: ``total_pages=1200, max_pages=500`` →
    ``(1, 500), (501, 1000), (1001, 1200)``.
    """
    if total_pages <= 0:
        return
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    start = 1
    while start <= total_pages:
        end = min(start + max_pages - 1, total_pages)
        yield start, end
        start = end + 1


def extract_page_range_pdf(
    src_path: str | Path,
    start: int,
    end: int,
    dest_path: str | Path,
) -> Path:
    """Write inclusive 1-indexed pages ``[start, end]`` from ``src_path`` to ``dest_path``.

    Returns the destination path. Out-of-range page indices are skipped.
    """
    if start < 1 or end < start:
        raise ValueError(f"invalid page range: start={start}, end={end}")

    src = Path(src_path)
    dest = Path(dest_path)
    reader = PdfReader(str(src))
    writer = PdfWriter()
    n_pages = len(reader.pages)

    for page_idx in range(start - 1, min(end, n_pages)):
        writer.add_page(reader.pages[page_idx])

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        writer.write(fh)
    return dest
