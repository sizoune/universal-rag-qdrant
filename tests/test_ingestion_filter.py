from langchain_core.documents import Document

from src.ingestion import drop_low_value_chunks, is_low_value_chunk


def test_drops_bare_footer_url():
    assert is_low_value_chunk("https://tabalongkab.bps.go.id")


def test_drops_short_fragment():
    assert is_low_value_chunk("Lampiran 12")


def test_drops_sumber_stamp():
    assert is_low_value_chunk("Sumber: BPS")


def test_drops_url_with_scrap():
    assert is_low_value_chunk("Lampiran 163 https://tabalongkab.bps.go.id")


def test_keeps_real_paragraph():
    txt = (
        "Pada tahun 2024, hasil pendataan Survei Kerangka Sampel Area (KSA) "
        "menunjukkan luas panen padi di Kabupaten Tabalong mencapai sekian hektar."
    )
    assert not is_low_value_chunk(txt)


def test_keeps_statistical_row_with_context():
    # a real (long enough) row with numbers must survive — it IS the data
    txt = "Kecamatan Kelua memiliki luas wilayah 53,36 km2 dan terbagi menjadi 12 desa/kelurahan."
    assert not is_low_value_chunk(txt)


def test_drop_low_value_chunks_filters_and_preserves_order():
    docs = [
        Document(page_content="https://tabalongkab.bps.go.id", metadata={"source": "a.pdf"}),
        Document(
            page_content="Kecamatan Kelua memiliki luas wilayah 53,36 km2 dan terbagi 12 desa.",
            metadata={"source": "a.pdf"},
        ),
        Document(page_content="Sumber: BPS", metadata={"source": "a.pdf"}),
    ]
    out = drop_low_value_chunks(docs)
    assert len(out) == 1
    assert "Kelua" in out[0].page_content
