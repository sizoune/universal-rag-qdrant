"""OCR fallback for scan-like PDFs in src.layout_parser (mocked gateway)."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from src import layout_parser
from src.layout_parser import (
    Element,
    _ocr_pdf_page_ranges,
    _ocr_text_to_elements,
    parse_pdf,
)


def _make_blank_pdf(path: Path, n_pages: int = 2) -> Path:
    """Minimal blank PDF with no extractable text (scan-like)."""
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=612, height=792)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        writer.write(fh)
    return path


def test_parse_pdf_ocr_produces_elements(tmp_path, monkeypatch):
    pdf = _make_blank_pdf(tmp_path / "scan.pdf", n_pages=2)
    monkeypatch.setattr(layout_parser, "ocr_enabled", lambda: True)
    monkeypatch.setattr(
        layout_parser,
        "ocr_file_bytes",
        lambda data, filename: (
            "Halaman satu dari Perda.\n\nPasal 1 Ketentuan Umum."
        ),
    )

    elements = parse_pdf(str(pdf))
    paragraphs = [el for el in elements if el.kind == "paragraph"]
    assert paragraphs, "OCR should yield paragraph Elements for blank PDF"
    joined = " ".join(el.text for el in paragraphs)
    assert "Perda" in joined
    assert "Pasal 1" in joined
    assert all(el.page is not None for el in paragraphs)


def test_parse_pdf_skips_ocr_when_disabled(tmp_path, monkeypatch):
    pdf = _make_blank_pdf(tmp_path / "scan.pdf", n_pages=1)
    monkeypatch.setattr(layout_parser, "ocr_enabled", lambda: False)

    def _boom(*_a, **_k):
        raise AssertionError("OCR must not run when gateway unset")

    monkeypatch.setattr(layout_parser, "ocr_file_bytes", _boom)
    elements = parse_pdf(str(pdf))
    assert elements == []


def test_parse_pdf_ocr_failure_is_best_effort(tmp_path, monkeypatch):
    pdf = _make_blank_pdf(tmp_path / "scan.pdf", n_pages=1)
    monkeypatch.setattr(layout_parser, "ocr_enabled", lambda: True)
    monkeypatch.setattr(layout_parser, "ocr_file_bytes", lambda *_a, **_k: "")
    # Should not raise; blank OCR result → no OCR elements.
    assert parse_pdf(str(pdf)) == []


def test_ocr_text_to_elements_form_feed():
    text = "Page A content\n\nMore A\fPage B content"
    elements = _ocr_text_to_elements(text, start=3, end=4)
    assert [el.page for el in elements] == [3, 3, 4]
    assert elements[0].text.startswith("Page A")
    assert elements[-1].text.startswith("Page B")


def test_large_pdf_invokes_iter_page_ranges(tmp_path, monkeypatch):
    """Documents over 500 pages must be split before OCR."""
    pdf = _make_blank_pdf(tmp_path / "tiny.pdf", n_pages=1)
    seen: list[tuple[int, int]] = []
    real_iter = layout_parser.iter_page_ranges

    def tracking_iter(total_pages, max_pages=500):
        seen.append((total_pages, max_pages))
        return real_iter(total_pages, max_pages)

    ocr_calls: list[str] = []

    def fake_ocr(data, filename):
        ocr_calls.append(filename)
        return f"OCR text for {filename}"

    monkeypatch.setattr(layout_parser, "iter_page_ranges", tracking_iter)
    monkeypatch.setattr(layout_parser, "ocr_file_bytes", fake_ocr)

    # Drive the range helper directly with a 589-page count (no 589-page fixture).
    elements = _ocr_pdf_page_ranges(str(pdf), total_pages=589, skip_pages=set())

    assert seen == [(589, 500)]
    assert len(ocr_calls) == 2  # (1,500) and (501,589)
    assert elements
    assert all(isinstance(el, Element) for el in elements)


def test_extract_subset_used_per_range(tmp_path, monkeypatch):
    pdf = _make_blank_pdf(tmp_path / "multi.pdf", n_pages=3)
    extract_calls: list[tuple[int, int]] = []
    real_extract = layout_parser.extract_page_range_pdf

    def tracking_extract(src_path, start, end, dest_path):
        extract_calls.append((start, end))
        return real_extract(src_path, start, end, dest_path)

    monkeypatch.setattr(layout_parser, "extract_page_range_pdf", tracking_extract)
    monkeypatch.setattr(
        layout_parser, "ocr_file_bytes", lambda data, filename: "chunk text"
    )

    _ocr_pdf_page_ranges(str(pdf), total_pages=3, skip_pages=set())
    assert extract_calls == [(1, 3)]


def test_ocr_skips_ranges_covered_by_good_native(tmp_path, monkeypatch):
    pdf = _make_blank_pdf(tmp_path / "mixed.pdf", n_pages=2)

    def _boom(*_a, **_k):
        raise AssertionError("should not OCR when all pages skipped")

    monkeypatch.setattr(layout_parser, "ocr_file_bytes", _boom)
    elements = _ocr_pdf_page_ranges(
        str(pdf), total_pages=2, skip_pages={1, 2}
    )
    assert elements == []
