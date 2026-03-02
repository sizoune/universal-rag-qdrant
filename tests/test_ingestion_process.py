from pathlib import Path

from src import ingestion


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
