"""Pure-function helpers for building citation payloads from retrieval context.

Functions here have no I/O — they take Document metadata and return formatted
labels or grouped SourceItem objects. Display labels are in Bahasa Indonesia.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from langchain_core.documents import Document

from src.api_models import LocationItem, SourceItem

DEFAULT_PREVIEW_CHARS = 200
HEADING_SEPARATOR = " § "


def truncate_preview(text: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Trim chunk text for citation preview.

    Collapses internal whitespace to single spaces (so multi-line chunks read
    as a single line) and appends an ellipsis when truncated.
    """
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def format_display(metadata: dict, source_type: str) -> str:
    """Build an Indonesian display label for a chunk's citation entry.

    Dispatches by source_type and metadata signals (page / node_type / row).
    """
    if source_type == "web":
        return _format_web(metadata)

    # Local-ish (local file upload, telegram_upload, etc.) — pick best signal.
    if "node_type" in metadata:
        return _format_code(metadata)
    if metadata.get("page") is not None:
        return _format_pdf(metadata)
    if "row" in metadata:
        return _format_csv(metadata)

    # Default: DOCX-like (heading/table). Returns "Dokumen" when nothing set.
    return _format_docx(metadata)


def build_source_items(context_docs: Iterable[Document]) -> list[SourceItem]:
    """Group context Documents by source path and build SourceItem list.

    Locations within a SourceItem are deduplicated by (display, chunk_preview)
    so identical chunks returned twice by hybrid retrieval don't show up twice.
    First-seen order is preserved at both source-level and location-level.
    """
    grouped: dict[str, dict] = {}
    order: list[str] = []

    for doc in context_docs:
        meta = dict(doc.metadata or {})
        source = meta.get("source") or "Unknown"
        source_type = meta.get("source_type") or "local"

        loc = LocationItem(
            display=format_display(meta, source_type),
            chunk_preview=truncate_preview(doc.page_content or ""),
            page=meta.get("page"),
            url_fragment=meta.get("url_fragment"),
            line_range=meta.get("line_range"),
        )

        if source not in grouped:
            grouped[source] = {
                "source_type": source_type,
                "filename": _filename_for(source, source_type),
                "locations": [],
                "_seen": set(),
            }
            order.append(source)

        bucket = grouped[source]
        dedup_key = (loc.display, loc.chunk_preview)
        if dedup_key in bucket["_seen"]:
            continue
        bucket["_seen"].add(dedup_key)
        bucket["locations"].append(loc)

    return [
        SourceItem(
            source=source,
            source_type=grouped[source]["source_type"],
            filename=grouped[source]["filename"],
            locations=grouped[source]["locations"],
        )
        for source in order
    ]


# ---------------------------------------------------------------------------
# Internal formatters per source kind
# ---------------------------------------------------------------------------


def _format_pdf(metadata: dict) -> str:
    page = metadata["page"]
    base = f"Halaman {page}"
    table = metadata.get("table_caption")
    heading = metadata.get("heading_path") or []

    if table and heading:
        return f"{base} — Tabel: {table} ({HEADING_SEPARATOR.join(heading)})"
    if table:
        return f"{base} — Tabel: {table}"
    if heading:
        return f"{base} — {HEADING_SEPARATOR.join(heading)}"
    return base


def _format_docx(metadata: dict) -> str:
    table = metadata.get("table_caption")
    heading = metadata.get("heading_path") or []

    if table and heading:
        return f"Tabel: {table} ({HEADING_SEPARATOR.join(heading)})"
    if table:
        return f"Tabel: {table}"
    if heading:
        return HEADING_SEPARATOR.join(heading)
    return "Dokumen"


def _format_web(metadata: dict) -> str:
    heading = metadata.get("heading_path") or []
    if heading:
        return f"Bagian: {HEADING_SEPARATOR.join(heading)}"
    return "Halaman web"


def _format_code(metadata: dict) -> str:
    node_type = metadata.get("node_type", "")
    node_name = metadata.get("node_name") or ""

    # Python (Tree-sitter Python grammar)
    if node_type == "function_definition":
        return f"Fungsi {node_name}".strip()
    if node_type == "class_definition":
        return f"Class {node_name}".strip()
    if node_type == "decorated_definition":
        return f"Fungsi/Class {node_name}".strip()

    # JavaScript (Tree-sitter JS grammar)
    if node_type == "function_declaration":
        return f"Fungsi {node_name}".strip()
    if node_type == "class_declaration":
        return f"Class {node_name}".strip()
    if node_type in ("export_statement", "lexical_declaration"):
        return f"Export {node_name}".strip()

    if node_type == "module_scope":
        return "Module-level (imports/konstanta)"

    fallback = f"{node_type} {node_name}".strip()
    return fallback or "Kode"


def _format_csv(metadata: dict) -> str:
    row = metadata.get("row")
    if isinstance(row, int):
        # CSVLoader uses 0-indexed row — display 1-indexed for humans.
        return f"Baris {row + 1}"
    if row is not None:
        return f"Baris {row}"
    return "CSV"


def _filename_for(source: str, source_type: str) -> str | None:
    if source_type == "web" or source.startswith(("http://", "https://")):
        return None
    return os.path.basename(source) or None
