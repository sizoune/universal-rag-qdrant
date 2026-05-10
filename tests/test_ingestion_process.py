from pathlib import Path

import pytest

from src import ingestion
from tests.fixtures.layout_factories import (
    make_corrupt_pdf,
    make_docx_with_heading_and_table,
    make_pdf_with_indonesian_heading,
)


def test_process_directory_respects_upload_max_bytes(monkeypatch, tmp_path):
    big_txt = tmp_path / "big.txt"
    big_txt.write_text("A" * (2 * 1024 * 1024))

    monkeypatch.setattr(ingestion.config, "UPLOAD_MAX_BYTES", 3 * 1024 * 1024)
    monkeypatch.setattr(ingestion, "load_cache", lambda: {})
    monkeypatch.setattr(ingestion, "save_cache", lambda _cache: None)

    docs, changed_sources = ingestion.process_directory(str(tmp_path))

    assert len(changed_sources) == 1
    assert changed_sources[0] == str(big_txt.resolve())
    assert len(docs) > 0


def test_load_local_document_pdf_uses_layout_parser(tmp_path):
    """PDF loading through load_local_document yields layout-aware chunks
    (parser_version=2) with heading_path / page metadata when applicable.
    """
    pdf = make_pdf_with_indonesian_heading(tmp_path / "id.pdf")

    chunks = ingestion.load_local_document(str(pdf))

    assert chunks, "should produce at least one chunk"
    versions = {c.metadata.get("parser_version") for c in chunks}
    assert 2 in versions, f"expected parser_version=2, got {versions}"
    sources = {c.metadata.get("source") for c in chunks}
    assert sources == {str(pdf)}
    # File hash should be stamped
    assert all("file_hash" in c.metadata for c in chunks)


def test_load_local_document_docx_uses_layout_parser(tmp_path):
    """DOCX is parsed by layout_parser, producing heading-aware chunks."""
    docx_path = make_docx_with_heading_and_table(tmp_path / "doc.docx")

    chunks = ingestion.load_local_document(str(docx_path))

    assert chunks
    versions = {c.metadata.get("parser_version") for c in chunks}
    assert 2 in versions
    # Some chunk has heading_path because of "Bab 1" heading
    with_heading = [c for c in chunks if c.metadata.get("heading_path")]
    assert with_heading, "expected heading_path on at least one chunk"


def test_load_local_document_pdf_falls_back_when_layout_fails(monkeypatch, tmp_path):
    """If layout_parser.parse_pdf raises, fall back to legacy PyPDFLoader and
    stamp parser_version=1 on the resulting chunks.
    """
    pdf = make_pdf_with_indonesian_heading(tmp_path / "id.pdf")

    def boom(_path):
        raise RuntimeError("simulated parser failure")

    monkeypatch.setattr(ingestion, "parse_pdf", boom)

    chunks = ingestion.load_local_document(str(pdf))

    assert chunks
    versions = {c.metadata.get("parser_version") for c in chunks}
    assert 1 in versions, f"expected fallback parser_version=1, got {versions}"


def test_load_local_document_corrupt_pdf_falls_back(tmp_path):
    """A non-PDF file with .pdf extension triggers parser exception -> fallback.
    Legacy loader will likely also fail, returning [] without raising."""
    bad = make_corrupt_pdf(tmp_path / "bad.pdf")

    # Should not raise — fallback happens internally; result may be empty.
    chunks = ingestion.load_local_document(str(bad))
    assert isinstance(chunks, list)


def test_load_local_document_text_uses_legacy_path(tmp_path):
    """Plain text files go through the legacy splitter (parser_version=1)."""
    txt = tmp_path / "note.txt"
    txt.write_text("This is plain text content.\n\nSecond paragraph.")

    chunks = ingestion.load_local_document(str(txt))

    assert chunks
    versions = {c.metadata.get("parser_version") for c in chunks}
    assert 1 in versions


def test_process_directory_does_not_double_split_layout_chunks(monkeypatch, tmp_path):
    """Regression: process_directory must not pass already-final chunks through
    a second character splitter.
    """
    pdf = make_pdf_with_indonesian_heading(tmp_path / "id.pdf")

    monkeypatch.setattr(ingestion, "load_cache", lambda: {})
    monkeypatch.setattr(ingestion, "save_cache", lambda _cache: None)

    docs, changed = ingestion.process_directory(str(tmp_path))

    assert changed
    # Every chunk that came from the PDF retains parser_version=2 from the
    # layout parser — would be lost / overwritten if double-splitting occurred.
    pdf_chunks = [d for d in docs if d.metadata.get("source") == str(pdf)]
    assert pdf_chunks
    assert all(d.metadata.get("parser_version") == 2 for d in pdf_chunks)
