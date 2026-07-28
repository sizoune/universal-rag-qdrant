from pypdf import PdfReader, PdfWriter

from src.pdf_split import extract_page_range_pdf, iter_page_ranges


def test_iter_page_ranges_exact_multiples():
    assert list(iter_page_ranges(1000, max_pages=500)) == [(1, 500), (501, 1000)]


def test_iter_page_ranges_remainder():
    assert list(iter_page_ranges(1200, max_pages=500)) == [
        (1, 500),
        (501, 1000),
        (1001, 1200),
    ]


def test_iter_page_ranges_small_doc():
    assert list(iter_page_ranges(12, max_pages=500)) == [(1, 12)]


def test_iter_page_ranges_empty():
    assert list(iter_page_ranges(0)) == []


def test_extract_page_range_pdf(tmp_path):
    src = tmp_path / "src.pdf"
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=612, height=792)
    with src.open("wb") as fh:
        writer.write(fh)

    dest = tmp_path / "subset.pdf"
    extract_page_range_pdf(src, 2, 4, dest)
    assert dest.is_file()
    assert len(PdfReader(str(dest)).pages) == 3
