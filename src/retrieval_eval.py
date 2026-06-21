"""Pure, dependency-light core for retrieval evaluation.

Kept free of LangChain/Qdrant imports so the metric logic stays unit-testable
without a running vector store. The CLI glue lives in ../eval_retrieval.py.
"""
from __future__ import annotations

import yaml


def load_golden(path: str) -> list[dict]:
    """Load golden entries from a YAML file. Drops entries marked `example: true`."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"Golden file {path} must be a YAML list of entries.")
    return [e for e in data if not (isinstance(e, dict) and e.get("example"))]


def validate_entries(entries: list[dict]) -> None:
    """Fail fast on malformed golden entries (boundary validation)."""
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or not e.get("question"):
            raise ValueError(f"Entry #{i} is missing a 'question'.")
        if not e.get("expect_substring") and not e.get("expect_source"):
            raise ValueError(
                f"Entry #{i} ({str(e.get('question', ''))[:40]!r}) needs at least one "
                f"of 'expect_substring' or 'expect_source'."
            )


def first_relevant_rank(docs, entry: dict):
    """1-based rank of the first retrieved doc that matches the entry, else None.

    A doc matches if its text contains any expect_substring (case-insensitive)
    OR its metadata.source contains any expect_source (case-insensitive).
    """
    subs = [str(s).lower() for s in (entry.get("expect_substring") or [])]
    srcs = [str(s).lower() for s in (entry.get("expect_source") or [])]
    for rank, d in enumerate(docs, start=1):
        text = str(getattr(d, "page_content", "") or "").lower()
        meta = getattr(d, "metadata", {}) or {}
        source = str(meta.get("source", "")).lower()
        if any(s in text for s in subs) or any(s in source for s in srcs):
            return rank
    return None


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results into hit_rate and MRR."""
    n = len(results)
    if n == 0:
        return {"n": 0, "hits": 0, "hit_rate": 0.0, "mrr": 0.0}
    hits = sum(1 for r in results if r.get("rank"))
    mrr = sum(1.0 / r["rank"] for r in results if r.get("rank")) / n
    return {"n": n, "hits": hits, "hit_rate": hits / n, "mrr": mrr}
