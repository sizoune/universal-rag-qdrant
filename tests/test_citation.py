"""Tests for citation module — pure functions for formatting source citations."""

import pytest
from langchain_core.documents import Document

from src.citation import build_source_items, format_display, truncate_preview


# ---------------------------------------------------------------------------
# truncate_preview
# ---------------------------------------------------------------------------


class TestTruncatePreview:
    def test_empty_string(self):
        assert truncate_preview("") == ""

    def test_short_text_unchanged(self):
        assert truncate_preview("Halo dunia") == "Halo dunia"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 250
        result = truncate_preview(text, max_chars=200)
        assert len(result) <= 203  # 200 + "..."
        assert result.endswith("...")

    def test_normalizes_whitespace(self):
        text = "Halo\n\ndunia\t baru"
        assert truncate_preview(text) == "Halo dunia baru"

    def test_custom_max_chars(self):
        result = truncate_preview("a" * 100, max_chars=50)
        assert len(result) <= 53
        assert result.startswith("a" * 50)
        assert result.endswith("...")

    def test_default_max_chars_is_200(self):
        text = "x" * 199
        # 199 chars: under default 200, returned as-is
        assert truncate_preview(text) == text

    def test_strips_leading_trailing_whitespace(self):
        assert truncate_preview("   hello   ") == "hello"


# ---------------------------------------------------------------------------
# format_display — PDF
# ---------------------------------------------------------------------------


class TestFormatDisplayPdf:
    def test_page_only(self):
        assert format_display({"page": 5}, "local") == "Halaman 5"

    def test_page_with_heading(self):
        meta = {
            "page": 5,
            "heading_path": ["Bab 2: Implementasi", "2.1 Metodologi"],
        }
        result = format_display(meta, "local")
        assert "Halaman 5" in result
        assert "Bab 2: Implementasi" in result
        assert "2.1 Metodologi" in result

    def test_page_with_table_caption(self):
        meta = {"page": 5, "table_caption": "Anggaran 2024"}
        result = format_display(meta, "local")
        assert "Halaman 5" in result
        assert "Tabel: Anggaran 2024" in result

    def test_page_with_heading_and_table(self):
        meta = {
            "page": 5,
            "heading_path": ["Bab 2"],
            "table_caption": "Anggaran",
        }
        result = format_display(meta, "local")
        assert "Halaman 5" in result
        assert "Tabel: Anggaran" in result
        assert "Bab 2" in result


# ---------------------------------------------------------------------------
# format_display — DOCX (no page)
# ---------------------------------------------------------------------------


class TestFormatDisplayDocx:
    def test_heading_only(self):
        meta = {"heading_path": ["Bab 2", "2.1"]}
        result = format_display(meta, "local")
        assert "Bab 2" in result
        assert "2.1" in result
        # No page hint
        assert "Halaman" not in result

    def test_table_only(self):
        meta = {"table_caption": "Anggaran 2024"}
        assert format_display(meta, "local") == "Tabel: Anggaran 2024"

    def test_table_with_heading_suffix(self):
        meta = {"heading_path": ["Bab 2"], "table_caption": "Anggaran"}
        result = format_display(meta, "local")
        assert "Tabel: Anggaran" in result
        assert "Bab 2" in result

    def test_empty_local_falls_back_to_dokumen(self):
        assert format_display({}, "local") == "Dokumen"


# ---------------------------------------------------------------------------
# format_display — Web
# ---------------------------------------------------------------------------


class TestFormatDisplayWeb:
    def test_with_heading(self):
        result = format_display({"heading_path": ["Pendahuluan"]}, "web")
        assert "Bagian:" in result
        assert "Pendahuluan" in result

    def test_empty_web_falls_back(self):
        assert format_display({}, "web") == "Halaman web"

    def test_nested_heading_joined(self):
        result = format_display(
            {"heading_path": ["Bab 1", "1.2 Latar Belakang"]}, "web"
        )
        assert "Bagian:" in result
        assert "Bab 1" in result
        assert "1.2 Latar Belakang" in result


# ---------------------------------------------------------------------------
# format_display — Code (Tree-sitter)
# ---------------------------------------------------------------------------


class TestFormatDisplayCode:
    def test_python_function(self):
        meta = {"node_type": "function_definition", "node_name": "calculate_total"}
        result = format_display(meta, "local")
        assert "Fungsi" in result
        assert "calculate_total" in result

    def test_python_class(self):
        meta = {"node_type": "class_definition", "node_name": "User"}
        result = format_display(meta, "local")
        assert "Class" in result
        assert "User" in result

    def test_python_decorated(self):
        meta = {"node_type": "decorated_definition", "node_name": "my_handler"}
        result = format_display(meta, "local")
        assert "my_handler" in result

    def test_js_function_declaration(self):
        meta = {"node_type": "function_declaration", "node_name": "fetchData"}
        result = format_display(meta, "local")
        assert "fetchData" in result

    def test_module_scope(self):
        meta = {"node_type": "module_scope", "node_name": "<imports/constants>"}
        result = format_display(meta, "local")
        # Should mention module-level concept
        assert "Module" in result or "imports" in result.lower()

    def test_js_class_declaration(self):
        meta = {"node_type": "class_declaration", "node_name": "MyClass"}
        result = format_display(meta, "local")
        assert "Class" in result and "MyClass" in result

    def test_js_export_statement(self):
        meta = {"node_type": "export_statement", "node_name": "myExport"}
        result = format_display(meta, "local")
        assert "Export" in result and "myExport" in result

    def test_js_lexical_declaration(self):
        meta = {"node_type": "lexical_declaration", "node_name": "constVar"}
        result = format_display(meta, "local")
        assert "Export" in result and "constVar" in result

    def test_unknown_node_type_falls_back(self):
        # Unknown node_type still produces something non-empty
        meta = {"node_type": "weird_node", "node_name": "thing"}
        result = format_display(meta, "local")
        assert result and result != "Kode" and "thing" in result

    def test_unknown_node_type_empty_name_returns_kode(self):
        meta = {"node_type": "", "node_name": ""}
        result = format_display(meta, "local")
        assert result == "Kode"


# ---------------------------------------------------------------------------
# format_display — CSV
# ---------------------------------------------------------------------------


class TestFormatDisplayCsv:
    def test_csv_with_int_row_displays_one_indexed(self):
        meta = {"row": 0}
        assert format_display(meta, "local") == "Baris 1"

    def test_csv_with_int_row_42(self):
        meta = {"row": 42}
        assert format_display(meta, "local") == "Baris 43"

    def test_csv_with_string_row(self):
        meta = {"row": "header"}
        assert format_display(meta, "local") == "Baris header"

    def test_csv_with_none_row(self):
        meta = {"row": None}
        # 'row' key present but None — _format_csv returns generic "CSV"
        assert format_display(meta, "local") == "CSV"


# ---------------------------------------------------------------------------
# format_display — Telegram upload (treated like local)
# ---------------------------------------------------------------------------


class TestFormatDisplayTelegramUpload:
    def test_telegram_with_page(self):
        meta = {"page": 3}
        assert format_display(meta, "telegram_upload") == "Halaman 3"

    def test_telegram_empty(self):
        assert format_display({}, "telegram_upload") == "Dokumen"


# ---------------------------------------------------------------------------
# build_source_items
# ---------------------------------------------------------------------------


class TestBuildSourceItems:
    def test_empty_input(self):
        assert build_source_items([]) == []

    def test_single_doc(self):
        doc = Document(
            page_content="Hello world from page 1",
            metadata={
                "source": "/uploads/a.pdf",
                "source_type": "local",
                "page": 1,
            },
        )
        result = build_source_items([doc])
        assert len(result) == 1
        item = result[0]
        assert item.source == "/uploads/a.pdf"
        assert item.source_type == "local"
        assert item.filename == "a.pdf"
        assert len(item.locations) == 1
        assert item.locations[0].page == 1
        assert "Hello world" in item.locations[0].chunk_preview

    def test_multiple_chunks_same_source_grouped(self):
        docs = [
            Document(
                page_content="page 1 content",
                metadata={"source": "/a.pdf", "source_type": "local", "page": 1},
            ),
            Document(
                page_content="page 2 content",
                metadata={"source": "/a.pdf", "source_type": "local", "page": 2},
            ),
        ]
        result = build_source_items(docs)
        assert len(result) == 1
        assert len(result[0].locations) == 2

    def test_dedup_identical_chunks(self):
        # Two docs from same source, same page, same content -> dedup to 1 location
        docs = [
            Document(
                page_content="same",
                metadata={"source": "/a.pdf", "source_type": "local", "page": 1},
            ),
            Document(
                page_content="same",
                metadata={"source": "/a.pdf", "source_type": "local", "page": 1},
            ),
        ]
        result = build_source_items(docs)
        assert len(result[0].locations) == 1

    def test_different_sources_separate_items(self):
        docs = [
            Document(
                page_content="a",
                metadata={"source": "/a.pdf", "source_type": "local", "page": 1},
            ),
            Document(
                page_content="b",
                metadata={"source": "https://x.com", "source_type": "web"},
            ),
        ]
        result = build_source_items(docs)
        assert len(result) == 2
        sources = {item.source for item in result}
        assert sources == {"/a.pdf", "https://x.com"}

    def test_unknown_source_type_does_not_crash(self):
        doc = Document(
            page_content="x",
            metadata={"source": "/a.txt", "source_type": "exotic_unknown"},
        )
        result = build_source_items([doc])
        assert len(result) == 1

    def test_preserves_first_seen_order(self):
        docs = [
            Document(
                page_content="a",
                metadata={"source": "/b.pdf", "source_type": "local", "page": 1},
            ),
            Document(
                page_content="b",
                metadata={"source": "/a.pdf", "source_type": "local", "page": 1},
            ),
        ]
        result = build_source_items(docs)
        assert result[0].source == "/b.pdf"
        assert result[1].source == "/a.pdf"

    def test_url_fragment_preserved_for_web(self):
        doc = Document(
            page_content="x",
            metadata={
                "source": "https://example.com",
                "source_type": "web",
                "url_fragment": "section1",
            },
        )
        result = build_source_items([doc])
        assert result[0].locations[0].url_fragment == "section1"

    def test_filename_for_url_is_none(self):
        doc = Document(
            page_content="x",
            metadata={"source": "https://x.com/article", "source_type": "web"},
        )
        result = build_source_items([doc])
        assert result[0].filename is None

    def test_filename_for_local_is_basename(self):
        doc = Document(
            page_content="x",
            metadata={
                "source": "/uploads/long/path/laporan.pdf",
                "source_type": "local",
                "page": 1,
            },
        )
        result = build_source_items([doc])
        assert result[0].filename == "laporan.pdf"

    def test_missing_source_uses_unknown(self):
        # Document without 'source' metadata still produces a valid SourceItem
        doc = Document(page_content="x", metadata={"source_type": "local"})
        result = build_source_items([doc])
        assert len(result) == 1
        # Should not crash; source defaults to something reasonable
        assert result[0].source

    def test_long_chunk_truncated_in_preview(self):
        long_text = "x" * 500
        doc = Document(
            page_content=long_text,
            metadata={"source": "/a.pdf", "source_type": "local", "page": 1},
        )
        result = build_source_items([doc])
        preview = result[0].locations[0].chunk_preview
        assert len(preview) <= 203  # 200 + "..."
        assert preview.endswith("...")
