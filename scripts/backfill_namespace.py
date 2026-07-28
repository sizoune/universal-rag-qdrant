#!/usr/bin/env python3
"""Backfill metadata.namespace on existing Qdrant points.

Default target: tabalong-umum (shared public Tabalong corpus).
Only points missing the namespace field (or with empty value) are updated.
Sibling metadata keys are preserved.

Usage:
  python scripts/backfill_namespace.py
  python scripts/backfill_namespace.py --namespace tabalong-umum --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import config
from src.vector_store import ensure_namespace_payload_index, get_qdrant_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_namespace")


def _needs_namespace(payload: dict | None) -> bool:
    if not payload:
        return True
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        return True
    ns = meta.get("namespace")
    return not (isinstance(ns, str) and ns.strip())


def backfill(namespace: str, *, dry_run: bool = False, batch_size: int = 256) -> int:
    client = get_qdrant_client()
    collection = config.QDRANT_COLLECTION_NAME
    ensure_namespace_payload_index(client)

    updated = 0
    scanned = 0
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        scanned += len(records)
        for rec in records:
            if not _needs_namespace(rec.payload):
                continue
            if dry_run:
                updated += 1
                continue
            meta = {}
            if isinstance(rec.payload, dict) and isinstance(rec.payload.get("metadata"), dict):
                meta = dict(rec.payload["metadata"])
            meta["namespace"] = namespace
            client.set_payload(
                collection_name=collection,
                payload={"metadata": meta},
                points=[rec.id],
            )
            updated += 1
        logger.info(
            "scanned=%d updated=%d%s",
            scanned,
            updated,
            " (dry-run)" if dry_run else "",
        )
        offset = next_offset
        if offset is None:
            break
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default=config.DEFAULT_WRITE_NAMESPACE or "tabalong-umum",
        help="Namespace to stamp on unscoped points",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    ns = args.namespace.strip()
    if not ns:
        logger.error("namespace must be non-empty")
        return 2

    total = backfill(ns, dry_run=args.dry_run, batch_size=args.batch_size)
    logger.info(
        "%s %d points → namespace=%s in collection=%s",
        "would update" if args.dry_run else "updated",
        total,
        ns,
        config.QDRANT_COLLECTION_NAME,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
