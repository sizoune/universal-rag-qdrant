"""Tests for src.layout_parser — Element dataclass, heading detection,
chunker, and PDF/DOCX parsers (with synthetic fixture files).
"""

from __future__ import annotations

import pytest

from src.layout_parser import (
    Element,
    chunk_elements,
    detect_heading,
    parse_docx,
    parse_pdf,
)
from tests.fixtures.layout_factories import (
    make_corrupt_pdf,
    make_docx_with_heading_and_table,
    make_pdf_with_indonesian_heading,
    make_pdf_with_table,
    make_simple_pdf,
)


# ---------------------------------------------------------------------------
# detect_heading — Indonesian regex patterns
# ---------------------------------------------------------------------------


class TestDetectHeading:
    @pytest.mark.parametrize(
        "text,expected_level",
        [
            ("BAB I PENDAHULUAN", 1),
            ("BAB II", 1),
            ("Bab 3", 1),
            ("BAGIAN KESATU", 1),
            ("BAGIAN KEDUA", 1),
            ("Bagian 2 Tata Cara", 1),
            ("Lampiran A", 1),
            ("Pasal 1", 2),
            ("Pasal 42 ayat (1)", 2),
            ("I. Pendahuluan", 2),
            ("II. Tujuan", 2),
            ("1.1 Latar Belakang", 2),
            ("A. Latar Belakang", 3),
            ("1.1.1 Sub-bagian", 3),
        ],
    )
    def test_indonesian_regex_patterns(self, text, expected_level):
        is_heading, level = detect_heading(text)
        assert is_heading is True, f"{text!r} should be heading"
        assert level == expected_level

    def test_normal_paragraph_not_heading(self):
        is_heading, _ = detect_heading("Ini adalah paragraf biasa.")
        assert is_heading is False

    def test_empty_string_not_heading(self):
        assert detect_heading("") == (False, 0)

    def test_whitespace_only_not_heading(self):
        assert detect_heading("   \n\t   ") == (False, 0)

    def test_font_size_heuristic_level1(self):
        is_heading, level = detect_heading(
            "Some Heading Text", font_size=18, body_font_size=12
        )
        assert is_heading is True
        assert level == 1  # ratio 1.5

    def test_font_size_heuristic_level2(self):
        is_heading, level = detect_heading(
            "Subheading", font_size=15, body_font_size=12
        )
        assert is_heading is True
        assert level == 2

    def test_font_size_heuristic_level3(self):
        is_heading, level = detect_heading(
            "Smaller heading", font_size=13.5, body_font_size=12
        )
        assert is_heading is True
        assert level == 3

    def test_font_size_too_close_not_heading(self):
        is_heading, _ = detect_heading(
            "Same as body", font_size=12, body_font_size=12
        )
        assert is_heading is False

    def test_regex_takes_precedence_over_font_size(self):
        # Even with tiny font, regex match wins
        is_heading, level = detect_heading(
            "BAB I", font_size=8, body_font_size=12
        )
        assert is_heading is True
        assert level == 1

    def test_no_font_info_falls_back_to_regex_only(self):
        # No font info, no regex match -> not heading
        is_heading, _ = detect_heading("Just some text")
        assert is_heading is False

    def test_zero_body_font_size_safe(self):
        # Defensive: dont divide by zero
        is_heading, _ = detect_heading("text", font_size=18, body_font_size=0)
        assert is_heading is False


# ---------------------------------------------------------------------------
# chunk_elements — pure logic
# ---------------------------------------------------------------------------


def _para(text: str, page: int = 1) -> Element:
    return Element(kind="paragraph", level=0, text=text, page=page)


def _heading(text: str, level: int, page: int = 1) -> Element:
    return Element(kind="heading", level=level, text=text, page=page)


def _table(text: str, caption: str | None = None, page: int = 1) -> Element:
    return Element(
        kind="table", level=0, text=text, page=page, table_caption=caption
    )


class TestChunkElements:
    def test_empty_input(self):
        assert chunk_elements([]) == []

    def test_single_paragraph(self):
        result = chunk_elements([_para("Hello", page=3)])
        assert len(result) == 1
        assert "Hello" in result[0].page_content
        assert result[0].metadata["page"] == 3
        assert result[0].metadata["chunk_kind"] == "paragraph"
        assert result[0].metadata["parser_version"] == 2
        assert result[0].metadata["heading_path"] == []

    def test_heading_stored_in_metadata_and_prepended(self):
        elements = [
            _heading("Bab 1", level=1),
            _para("Konten paragraf"),
        ]
        result = chunk_elements(elements)
        assert len(result) == 1
        assert result[0].metadata["heading_path"] == ["Bab 1"]
        assert "Bab 1" in result[0].page_content
        assert "Konten paragraf" in result[0].page_content

    def test_nested_headings_stack(self):
        elements = [
            _heading("Bab 1", level=1),
            _heading("1.1 Sub", level=2),
            _para("content"),
        ]
        result = chunk_elements(elements)
        assert result[0].metadata["heading_path"] == ["Bab 1", "1.1 Sub"]

    def test_sibling_heading_pops_stack(self):
        elements = [
            _heading("Bab 1", level=1),
            _heading("1.1 Sub-A", level=2),
            _heading("1.2 Sub-B", level=2),
            _para("content"),
        ]
        result = chunk_elements(elements)
        assert result[0].metadata["heading_path"] == ["Bab 1", "1.2 Sub-B"]

    def test_higher_level_heading_pops_deeper(self):
        elements = [
            _heading("Bab 1", level=1),
            _heading("1.1", level=2),
            _heading("Bab 2", level=1),
            _para("content"),
        ]
        result = chunk_elements(elements)
        assert result[0].metadata["heading_path"] == ["Bab 2"]

    def test_paragraphs_combined_when_under_max(self):
        elements = [_para("a" * 400), _para("b" * 400)]
        result = chunk_elements(elements, max_chunk_size=1000)
        assert len(result) == 1

    def test_paragraphs_split_when_exceed_max(self):
        elements = [_para("a" * 600), _para("b" * 600)]
        result = chunk_elements(elements, max_chunk_size=1000)
        assert len(result) == 2

    def test_table_emits_separate_chunk(self):
        table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
        elements = [
            _para("paragraf sebelum tabel"),
            _table(table_md, caption="My Table"),
            _para("paragraf setelah tabel"),
        ]
        result = chunk_elements(elements)
        assert len(result) == 3
        kinds = [c.metadata["chunk_kind"] for c in result]
        assert kinds == ["paragraph", "table", "paragraph"]
        assert result[1].metadata["table_caption"] == "My Table"

    def test_large_table_split_with_header_repeated(self):
        header = "| Item | Jumlah | Keterangan |"
        sep = "|---|---|---|"
        rows = [
            f"| Item-{i:02d} | {1000 + i} | Keterangan baris ke-{i} |"
            for i in range(30)
        ]
        table_md = "\n".join([header, sep] + rows)

        result = chunk_elements(
            [_table(table_md, caption="Anggaran")], max_chunk_size=400
        )
        assert len(result) > 1, "table large should be split"
        for chunk in result:
            # Header reprinted on every chunk
            assert "Item" in chunk.page_content
            assert "Jumlah" in chunk.page_content
            assert "Keterangan" in chunk.page_content
            assert chunk.metadata["chunk_kind"] == "table"
            assert chunk.metadata["table_caption"] == "Anggaran"

    def test_heading_carries_to_table(self):
        elements = [
            _heading("Bab 2", level=1),
            _table("| H |\n|---|\n| v |", caption="My Tab"),
        ]
        result = chunk_elements(elements)
        assert result[0].metadata["chunk_kind"] == "table"
        assert result[0].metadata["heading_path"] == ["Bab 2"]

    def test_heading_path_is_list_type(self):
        result = chunk_elements(
            [_heading("Bab", level=1), _para("x")]
        )
        assert isinstance(result[0].metadata["heading_path"], list)

    def test_no_heading_yields_empty_path(self):
        result = chunk_elements([_para("x")])
        assert result[0].metadata["heading_path"] == []

    def test_page_is_first_seen_in_chunk(self):
        elements = [_para("first", page=1), _para("second", page=2)]
        result = chunk_elements(elements, max_chunk_size=1000)
        assert len(result) == 1
        assert result[0].metadata["page"] == 1

    def test_skips_blank_paragraphs(self):
        result = chunk_elements([_para("   "), _para("real content")])
        assert len(result) == 1
        assert "real content" in result[0].page_content


# ---------------------------------------------------------------------------
# parse_pdf — fixture-based
# ---------------------------------------------------------------------------


class TestParsePdf:
    def test_parses_simple_pdf(self, tmp_path):
        pdf = make_simple_pdf(tmp_path / "simple.pdf")
        elements = parse_pdf(str(pdf))
        assert len(elements) >= 1
        assert all(el.kind in ("paragraph", "heading", "table") for el in elements)

    def test_parses_pdf_with_table(self, tmp_path):
        pdf = make_pdf_with_table(tmp_path / "with_table.pdf")
        elements = parse_pdf(str(pdf))
        assert any(el.kind == "table" for el in elements)
        # Table content should mention header keywords
        table_text = "\n".join(el.text for el in elements if el.kind == "table")
        assert "Item" in table_text
        assert "Jumlah" in table_text

    def test_parses_pdf_with_indonesian_heading(self, tmp_path):
        pdf = make_pdf_with_indonesian_heading(tmp_path / "id_heading.pdf")
        elements = parse_pdf(str(pdf))
        headings = [el for el in elements if el.kind == "heading"]
        assert len(headings) >= 1
        # At least one heading is "BAB I" or "Pasal 1"
        heading_texts = " ".join(el.text for el in headings)
        assert "BAB" in heading_texts.upper() or "Pasal" in heading_texts

    def test_pages_are_one_indexed(self, tmp_path):
        pdf = make_simple_pdf(tmp_path / "simple.pdf")
        elements = parse_pdf(str(pdf))
        pages = {el.page for el in elements if el.page is not None}
        assert pages == {1}


# ---------------------------------------------------------------------------
# parse_docx — fixture-based
# ---------------------------------------------------------------------------


class TestParseDocx:
    def test_parses_heading_and_paragraph_in_order(self, tmp_path):
        docx_path = make_docx_with_heading_and_table(tmp_path / "doc.docx")
        elements = parse_docx(str(docx_path))

        kinds = [el.kind for el in elements]
        # First element should be heading "Bab 1"
        assert elements[0].kind == "heading"
        assert "Bab 1" in elements[0].text
        assert elements[0].level == 1
        # Should have at least one paragraph and one table
        assert "paragraph" in kinds
        assert "table" in kinds

    def test_table_content_from_docx(self, tmp_path):
        docx_path = make_docx_with_heading_and_table(tmp_path / "doc.docx")
        elements = parse_docx(str(docx_path))

        tables = [el for el in elements if el.kind == "table"]
        assert len(tables) == 1
        assert "Item" in tables[0].text
        assert "Jumlah" in tables[0].text
        assert "Gaji" in tables[0].text

    def test_heading_levels_preserved(self, tmp_path):
        docx_path = make_docx_with_heading_and_table(tmp_path / "doc.docx")
        elements = parse_docx(str(docx_path))

        headings = [el for el in elements if el.kind == "heading"]
        levels = [el.level for el in headings]
        # Should have level-1 (Bab 1) and level-2 (1.1, 1.2)
        assert 1 in levels
        assert 2 in levels


# ---------------------------------------------------------------------------
# Integration: parse + chunk
# ---------------------------------------------------------------------------


class TestParseAndChunkIntegration:
    def test_pdf_to_chunks_pipeline(self, tmp_path):
        pdf = make_pdf_with_indonesian_heading(tmp_path / "id.pdf")
        chunks = chunk_elements(parse_pdf(str(pdf)))
        assert len(chunks) >= 1
        # Some chunk should have heading_path populated
        with_heading = [c for c in chunks if c.metadata["heading_path"]]
        assert len(with_heading) >= 1

    def test_docx_to_chunks_pipeline(self, tmp_path):
        docx_path = make_docx_with_heading_and_table(tmp_path / "doc.docx")
        chunks = chunk_elements(parse_docx(str(docx_path)))
        assert len(chunks) >= 1
        # Should produce at least one table chunk and one paragraph chunk
        kinds = {c.metadata["chunk_kind"] for c in chunks}
        assert "table" in kinds
        assert "paragraph" in kinds
        # Heading carries through to the table chunk
        table_chunks = [
            c for c in chunks if c.metadata["chunk_kind"] == "table"
        ]
        assert len(table_chunks) == 1
        assert "Bab 1: Pendahuluan" in table_chunks[0].metadata["heading_path"]


# ---------------------------------------------------------------------------
# Failure mode: corrupt PDF
# ---------------------------------------------------------------------------


class TestParseFailures:
    def test_corrupt_pdf_raises(self, tmp_path):
        bad = make_corrupt_pdf(tmp_path / "bad.pdf")
        # Caller (ingestion) is expected to catch and fallback;
        # the parser surfaces the exception rather than swallowing.
        with pytest.raises(Exception):
            parse_pdf(str(bad))
