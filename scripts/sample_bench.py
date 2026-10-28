#!/usr/bin/env python3
"""Build reproducible per-dataset evaluation samples.

The benchmark replay files contain full user event streams.  This script
samples ``--per-dataset`` search events with a fixed seed, keeps every add that
happened before the last sampled search for that user, and drops the rest.
Every sampled search keeps its original ``query_id``, gold and options, so the
same file can be used for a baseline and a candidate run.

Example::

    python scripts/sample_bench.py --per-dataset 500 --seed 20261026
    python scripts/sample_bench.py --datasets locomo_refined,clbench --per-dataset 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATASETS = (
    "locomo_refined",
    "personamem_32k",
    "beam_500k",
    "clbench",
)


def _read(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    candidates = [
        (record_index, event)
        for record_index, record in enumerate(records)
        for event in record.get("events") or []
        if event.get("type") == "search"
    ]
    if len(candidates) <= count:
        return records
    picked = random.Random(seed).sample(candidates, count)
    per_record: dict[int, set[str]] = {}
    for record_index, event in picked:
        per_record.setdefault(record_index, set()).add(str(event.get("query_id") or ""))

    sampled: list[dict] = []
    for record_index, record in enumerate(records):
        wanted = per_record.get(record_index)
        if not wanted:
            continue
        events = record.get("events") or []
        search_steps = [
            int(event.get("step") or 0)
            for event in events
            if event.get("type") == "search"
            and str(event.get("query_id") or "") in wanted
        ]
        if not search_steps:
            continue
        last_step = max(search_steps)
        kept: list[dict] = []
        for event in events:
            step = int(event.get("step") or 0)
            if step > last_step:
                break
            if event.get("type") == "add":
                kept.append(event)
            elif event.get("type") == "search" and str(event.get("query_id") or "") in wanted:
                kept.append(event)
        if not any(event.get("type") == "search" for event in kept):
            continue
        record = dict(record)
        record["events"] = kept
        stats = dict(record.get("stats") or {})
        stats["add_events"] = sum(1 for event in kept if event.get("type") == "add")
        stats["search_events"] = sum(1 for event in kept if event.get("type") == "search")
        stats["streaming"] = any(
            event.get("type") == "search" for event in kept[:-1]
        ) and any(event.get("type") == "add" for event in kept)
        stats["first_search_step"] = min(
            (int(event.get("step") or 0) for event in kept if event.get("type") == "search"),
            default=0,
        )
        stats["last_add_step"] = max(
            (int(event.get("step") or 0) for event in kept if event.get("type") == "add"),
            default=0,
        )
        record["stats"] = stats
        if isinstance(record.get("ground_truth"), dict):
            record["ground_truth"] = {
                key: value
                for key, value in record["ground_truth"].items()
                if key in wanted
            }
        sampled.append(record)
    return sampled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--per-dataset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/eval_samples")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        source = data_dir / f"{dataset}.jsonl"
        if not source.exists():
            print(f"skip {dataset}: {source} does not exist")
            continue
        records = _read(source)
        sampled = _sample(records, args.per_dataset, args.seed)
        searches = sum(
            1 for record in sampled for event in record.get("events") or []
            if event.get("type") == "search"
        )
        target = out_dir / f"{dataset}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for record in sampled:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest[dataset] = {
            "source": str(source),
            "users": len(sampled),
            "search_events": searches,
            "per_dataset": args.per_dataset,
            "seed": args.seed,
            "file": str(target),
        }
        print(f"{dataset}: {len(sampled)} users, {searches} searches -> {target}")
    (out_dir / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "datasets": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
