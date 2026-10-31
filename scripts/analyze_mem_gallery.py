#!/usr/bin/env python3
"""Compare two Mem-Gallery runs and split image-question failures by cause.

    python scripts/analyze_mem_gallery.py \
        --a data/bench_runs/mg4o_caption400/predictions.jsonl \
        --b data/bench_runs/mg4o_media400/predictions.jsonl

The second run is the multimodal one.  For VS/TTL/VR/KR questions whose gold
answer is an image id, a failure is classified as:

- ``retrieval_miss``: the gold image id never appears in the final evidence;
- ``image_budget``: it appears, but beyond the images the answerer attaches
  (query image + ``max_images - 1`` evidence images);
- ``model_or_vision``: it was attached (or would be) and the answer is still
  wrong.

That distinction is what tells us whether to invest in retrieval, image
budget, or the vision answerer.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_ID = re.compile(r"D\d+:IMG_\d+", re.IGNORECASE)
IMAGE_TYPES = {"VS", "VR", "TTL", "KR"}


def load(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("query_id"))] = row
    return rows


def score(row: dict[str, Any]) -> float | None:
    value = (row.get("eval") or {}).get("score")
    return float(value) if isinstance(value, (int, float)) else None


def image_ids(value: Any) -> list[str]:
    return [match.upper() for match in IMAGE_ID.findall(str(value or ""))]


def evidence_image_ids(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for record in row.get("evidence") or []:
        for image_id in image_ids(record.get("content")):
            if image_id not in seen:
                seen.add(image_id)
                out.append(image_id)
    return out


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows.values():
        value = score(row)
        if value is not None:
            by_type[str(row.get("answer_type") or "?")].append(value)
    values = [value for item in by_type.values() for value in item]
    return {
        "n": len(values),
        "score": round(sum(values) / len(values), 4) if values else None,
        "by_type": {
            name: {"n": len(items), "score": round(sum(items) / len(items), 4)}
            for name, items in sorted(by_type.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="caption-only predictions.jsonl")
    parser.add_argument("--b", required=True, help="multimodal predictions.jsonl")
    parser.add_argument("--max-images", type=int, default=3,
                        help="answer.media.max_images used by run B")
    parser.add_argument("--show", type=int, default=10,
                        help="print up to N misclassified image questions")
    args = parser.parse_args()

    a = load(args.a)
    b = load(args.b)
    common = sorted(set(a) & set(b))
    print(f"A={args.a} ({len(a)} QAs)")
    print(f"B={args.b} ({len(b)} QAs)")
    print(f"paired={len(common)}")
    print("A:", json.dumps(summarize(a), ensure_ascii=False))
    print("B:", json.dumps(summarize(b), ensure_ascii=False))

    both = {"A_only": 0, "B_only": 0, "both": 0, "neither": 0}
    by_type = defaultdict(Counter)
    for qid in common:
        sa, sb = score(a[qid]) or 0.0, score(b[qid]) or 0.0
        kind = ("both" if sa >= 0.99 and sb >= 0.99 else
                "A_only" if sa >= 0.99 else
                "B_only" if sb >= 0.99 else "neither")
        both[kind] += 1
        by_type[str(b[qid].get("answer_type") or "?")][kind] += 1
    print("paired outcomes:", both)
    for name, counts in sorted(by_type.items()):
        print(f"  {name}: {dict(counts)}")

    # Image-question failure attribution (run B).
    causes = Counter()
    examples: list[tuple[str, str, str, str]] = []
    for qid in common:
        row = b[qid]
        if str(row.get("answer_type") or "").upper() not in IMAGE_TYPES:
            continue
        if (score(row) or 0) >= 0.99:
            continue
        gold = image_ids((row.get("gold") or {}).get("answer"))
        if not gold:
            continue
        evidence_ids = evidence_image_ids(row)
        attached_budget = max(0, args.max_images - (1 if (row.get("gold") or {}).get("answer_meta", {}).get("question_image") else 0))
        missing = [item for item in gold if item not in evidence_ids]
        if missing:
            cause = "retrieval_miss"
        else:
            ranks = [evidence_ids.index(item) + 1 for item in gold]
            cause = "image_budget" if min(ranks) > attached_budget else "model_or_vision"
        causes[cause] += 1
        if len(examples) < args.show:
            examples.append((cause, qid, str(row.get("answer_type")),
                             str(row.get("predicted"))[:120]))

    print("image-question failure causes:", dict(causes))
    for cause, qid, answer_type, predicted in examples:
        print(f"  [{cause}] {answer_type} {qid}: {predicted}")

    # Cost/latency of the answer stage.
    for label, rows in (("A", a), ("B", b)):
        prompt_tokens = []
        seconds = []
        for row in rows.values():
            tokens = ((row.get("tokens") or {}).get("answer") or {})
            if tokens.get("prompt_tokens"):
                prompt_tokens.append(float(tokens["prompt_tokens"]))
            value = (row.get("timings") or {}).get("answer_s")
            if isinstance(value, (int, float)):
                seconds.append(float(value))
        mean_tokens = round(statistics.mean(prompt_tokens), 1) if prompt_tokens else None
        mean_seconds = round(statistics.mean(seconds), 2) if seconds else None
        print(f"{label}: answer prompt tokens mean={mean_tokens} "
              f"n={len(prompt_tokens)}; answer_s mean={mean_seconds}")


if __name__ == "__main__":
    main()
