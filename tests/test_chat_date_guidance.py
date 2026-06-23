from datetime import date

from src.chat import _date_guidance


def test_date_guidance_mentions_running_year_and_fallback():
    year = str(date.today().year)
    text = _date_guidance()
    assert year in text
    assert "tahun berjalan" in text
    assert "paling dekat" in text  # closest-year fallback present
