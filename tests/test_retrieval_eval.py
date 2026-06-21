from types import SimpleNamespace

import pytest

from src.retrieval_eval import first_relevant_rank, summarize, validate_entries


def _doc(text="", source=""):
    return SimpleNamespace(page_content=text, metadata={"source": source})


def test_substring_match_returns_rank():
    docs = [_doc("irrelevant"), _doc("the TARIF is Rp 5000"), _doc("more")]
    entry = {"question": "q", "expect_substring": ["tarif", "rp"]}
    assert first_relevant_rank(docs, entry) == 2


def test_source_match_returns_rank():
    docs = [_doc("x", "other.pdf"), _doc("y", "Perda-Parkir.pdf")]
    entry = {"question": "q", "expect_source": ["perda-parkir"]}
    assert first_relevant_rank(docs, entry) == 2


def test_no_match_returns_none():
    docs = [_doc("nothing here", "a.pdf")]
    entry = {"question": "q", "expect_substring": ["missing"]}
    assert first_relevant_rank(docs, entry) is None


def test_summarize_hit_rate_and_mrr():
    results = [{"rank": 1}, {"rank": 2}, {"rank": None}, {"rank": None}]
    s = summarize(results)
    assert s["n"] == 4
    assert s["hits"] == 2
    assert s["hit_rate"] == 0.5
    assert s["mrr"] == pytest.approx((1.0 + 0.5) / 4)


def test_validate_requires_question_and_expectation():
    with pytest.raises(ValueError):
        validate_entries([{"question": "q"}])  # no expect_*
    with pytest.raises(ValueError):
        validate_entries([{"expect_substring": ["x"]}])  # no question
    validate_entries([{"question": "q", "expect_source": ["a.pdf"]}])  # ok, no raise
