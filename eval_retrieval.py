#!/usr/bin/env python3
"""Retrieval eval harness.

Measures hit-rate@k and MRR of the LIVE retriever against a golden Q&A set,
using the exact retriever the app uses (src.chat.build_retriever) so the
numbers reflect production behaviour, not a copy that can drift.

Usage:
    python eval_retrieval.py                     # eval/golden.yaml + current .env config
    python eval_retrieval.py --golden other.yaml
    python eval_retrieval.py --min-hit-rate 0.8  # exit 1 if below (CI gate)

Workflow: tweak SEARCH_MODE / SEARCH_SCORE_THRESHOLD / MAX_SEARCH_RESULTS in
.env, re-run, and watch the number. That's how you stop tuning blind.
"""
import argparse
import sys

from src.config import config
from src.retrieval_eval import (
    first_relevant_rank,
    load_golden,
    summarize,
    validate_entries,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality against a golden set.")
    ap.add_argument("--golden", default="eval/golden.yaml", help="Path to golden YAML.")
    ap.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.0,
        help="Exit 1 if hit_rate is below this (default 0.0 = never fail).",
    )
    args = ap.parse_args()

    try:
        entries = load_golden(args.golden)
    except FileNotFoundError:
        print(f"Golden file not found: {args.golden}", file=sys.stderr)
        return 2
    if not entries:
        print(
            f"No real entries in {args.golden} (only examples?). "
            f"Add your own Q&A — see the comments in that file.",
            file=sys.stderr,
        )
        return 2
    validate_entries(entries)

    # Build the live retriever (requires Qdrant up + collection populated).
    try:
        from src.chat import build_retriever
        from src.vector_store import initialize_vector_store

        vector_store = initialize_vector_store()
        retriever = build_retriever(vector_store)
    except Exception as e:
        print(
            f"Could not init retriever (is Qdrant up at {config.QDRANT_URL}?): {e}",
            file=sys.stderr,
        )
        return 2

    mode = config.SEARCH_MODE.lower()
    reranker_on = str(config.RERANKER_ENABLED).lower() in ("1", "true", "yes")
    print(
        f"[cfg] mode={mode}  threshold={config.SEARCH_SCORE_THRESHOLD}  "
        f"k={config.MAX_SEARCH_RESULTS}  reranker={'on' if reranker_on else 'off'}"
    )
    if reranker_on and mode != "hybrid":
        print("[warn] RERANKER_ENABLED but SEARCH_MODE!=hybrid -> reranker is a no-op here.")
    print(f"[cfg] golden={args.golden}  questions={len(entries)}\n")

    results = []
    for e in entries:
        docs = retriever.invoke(e["question"])
        rank = first_relevant_rank(docs, e)
        results.append({"question": e["question"], "rank": rank})
        mark = f"OK r{rank}" if rank else "MISS -"
        print(f"  {mark:7}  {str(e['question'])[:70]}")

    s = summarize(results)
    print("\n" + "-" * 48)
    print(
        f"hit_rate@{config.MAX_SEARCH_RESULTS}: {s['hit_rate']:.2f} "
        f"({s['hits']}/{s['n']})    MRR: {s['mrr']:.3f}"
    )

    if s["hit_rate"] < args.min_hit_rate:
        print(
            f"FAIL: hit_rate {s['hit_rate']:.2f} < min {args.min_hit_rate:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
