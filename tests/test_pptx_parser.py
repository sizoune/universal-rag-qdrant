"""Tests for src.pptx_parser — a .pptx is a zip of XML, so we build a minimal one."""

import zipfile

from src import pptx_parser
from src.ingestion import load_local_document

_SLIDE_XML = (
    '<p:sld xmlns:a="x">'
    "<a:p><a:r><a:t>Program</a:t></a:r><a:r><a:t> Prioritas</a:t></a:r></a:p>"
    "<a:p><a:r><a:t>Tabalong 2026</a:t></a:r></a:p>"
    "</p:sld>"
)
_NOTES_XML = '<p:notes xmlns:a="x"><a:p><a:r><a:t>Catatan penyaji</a:t></a:r></a:p></p:notes>'


def _make_pptx(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ppt/slides/slide1.xml", _SLIDE_XML)
        z.writestr("ppt/notesSlides/notesSlide1.xml", _NOTES_XML)


def test_parse_pptx_extracts_text_and_notes(tmp_path):
    f = tmp_path / "deck.pptx"
    _make_pptx(f)
    text = pptx_parser.parse_pptx(str(f))
    # runs joined per paragraph (no mid-word break), notes included
    assert "Program Prioritas" in text
    assert "Tabalong 2026" in text
    assert "Catatan penyaji" in text


def test_ocr_skipped_when_gateway_unset(tmp_path, monkeypatch):
    # No OCR_GATEWAY_URL -> ocr disabled -> ocr_image_bytes must never be called.
    monkeypatch.setattr(pptx_parser, "ocr_enabled", lambda: False)
    monkeypatch.setattr(
        pptx_parser,
        "ocr_image_bytes",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("OCR must not run")),
    )
    f = tmp_path / "deck.pptx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("ppt/slides/slide1.xml", _SLIDE_XML)
        z.writestr("ppt/media/image1.png", b"x" * 99999)  # would be OCR'd if enabled
    text = pptx_parser.parse_pptx(str(f))
    assert "Program Prioritas" in text


def test_load_local_document_routes_pptx(tmp_path):
    f = tmp_path / "deck.pptx"
    _make_pptx(f)
    chunks = load_local_document(str(f))
    assert chunks, "pptx should produce chunks"
    assert chunks[0].metadata["source_type"] == "local"
    assert "file_hash" in chunks[0].metadata
    assert "Program Prioritas" in chunks[0].page_content
