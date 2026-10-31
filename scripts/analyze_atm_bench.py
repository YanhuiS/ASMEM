#!/usr/bin/env python3
"""ATM-Bench analysis: answer score + evidence-id recall.

    python scripts/analyze_atm_bench.py \
        --predictions data/bench_runs/atm4o_full/predictions.jsonl

ATM-Bench is evidence-grounded: every QA carries ``evidence_ids`` (email ids
or image/video timestamps).  A retrieval miss is not visible in the answer
score alone, so this reports both:

- answer score by ``qtype``;
- whether the gold evidence id(s) appear in the answer-visible evidence
  (hit-any / full recall / first-hit rank).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evidence_text(row: dict[str, Any]) -> str:
    return "\n".join(str(item.get("content") or "")
                     for item in (row.get("evidence") or []))


def evidence_ids(row: dict[str, Any]) -> list[str]:
    gold = row.get("gold") or {}
    meta = gold.get("answer_meta") or {}
    ids = meta.get("evidence_ids")
    if ids is None:
        meta = (row.get("answer_meta") or {})
        ids = meta.get("evidence_ids")
    return [str(item) for item in (ids or []) if str(item).strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--show", type=int, default=10,
                        help="print up to N retrieval misses")
    args = parser.parse_args()

    rows = load(args.predictions)
    print(f"n={len(rows)}")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("answer_type") or "?")].append(row)

    print(f"{'type':12} {'n':>5} {'score':>7} {'hit_any':>8} {'full':>7} "
          f"{'miss':>5}")
    all_hits: list[float] = []
    all_full: list[float] = []
    all_scores: list[float] = []
    misses: list[tuple[str, str, list[str], str]] = []
    for name, items in sorted(by_type.items()):
        scores, hits, full = [], [], []
        for row in items:
            score = (row.get("eval") or {}).get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
            gold = evidence_ids(row)
            if not gold:
                continue
            text = evidence_text(row)
            found = [item for item in gold if item in text]
            hits.append(1.0 if found else 0.0)
            full.append(1.0 if len(found) == len(gold) else 0.0)
            if not found and len(misses) < args.show:
                misses.append((name, str(row.get("query_id")),
                               gold, str(row.get("query"))[:100]))
        mean = statistics.mean(scores) if scores else None
        print(f"{name:12} {len(items):5d} "
              f"{(f'{mean:.4f}' if mean is not None else '-'):>7} "
              f"{(statistics.mean(hits) if hits else 0):8.3f} "
              f"{(statistics.mean(full) if full else 0):7.3f} "
              f"{sum(1 for item in hits if not item):5d}")
        all_scores.extend(scores)
        all_hits.extend(hits)
        all_full.extend(full)

    print(f"{'overall':12} {len(rows):5d} "
          f"{(f'{statistics.mean(all_scores):.4f}' if all_scores else '-'):>7} "
          f"{(statistics.mean(all_hits) if all_hits else 0):8.3f} "
          f"{(statistics.mean(all_full) if all_full else 0):7.3f} "
          f"{sum(1 for item in all_hits if not item):5d}")

    # First-hit rank within the answer-visible evidence.
    ranks: list[int] = []
    for row in rows:
        gold = evidence_ids(row)
        if not gold:
            continue
        for rank, item in enumerate(row.get("evidence") or [], 1):
            content = str(item.get("content") or "")
            if any(gid in content for gid in gold):
                ranks.append(rank)
                break
    if ranks:
        print(f"first gold evidence rank: mean={statistics.mean(ranks):.1f} "
              f"median={statistics.median(ranks):.0f} "
              f"max={max(ranks)} (n={len(ranks)})")

    if misses:
        print("\nretrieval misses:")
        for name, qid, gold, question in misses:
            print(f"  [{name}] {qid} gold={gold} :: {question}")


if __name__ == "__main__":
    main()
