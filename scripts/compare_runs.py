#!/usr/bin/env python3
"""Compare two benchmark runs on the same query ids.

    python scripts/compare_runs.py --baseline data/bench_runs/baseline500 \
        --candidate data/bench_runs/candidate500

Prints per-dataset mean score, paired wins/losses/ties, and the macro delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    path = run / "predictions.jsonl"
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A live benchmark may be in the middle of writing its last line.
            # Ignore that partial tail; completed lines remain comparable.
            continue
        query_id = str(record.get("query_id") or "")
        if query_id:
            results[query_id] = record
    return results


def score(record: dict | None) -> float | None:
    if record is None:
        return None
    value = (record.get("eval") or {}).get("score")
    return float(value) if value is not None else None


def stop_reasons(records: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records.values():
        reason = str((record.get("search") or {}).get("stop_reason") or "?")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _health_line(name: str, records: dict[str, dict]) -> str:
    counts = stop_reasons(records)
    total = sum(counts.values()) or 1
    bad = sum(value for reason, value in counts.items()
              if reason.startswith("model_error") or reason == "fallback_error")
    top = ", ".join(f"{reason}={value}" for reason, value in
                    sorted(counts.items(), key=lambda item: -item[1])[:4])
    line = f"{name}: {top}"
    if bad / total > 0.05:
        line += (f"  ** WARNING: {bad}/{total} ({bad / total:.1%}) searches "
                 "ended in a model error; this run is effectively a coarse "
                 "baseline and not a valid method comparison. **")
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    baseline = load(Path(args.baseline))
    candidate = load(Path(args.candidate))
    common = sorted(set(baseline) & set(candidate))
    by_dataset: dict[str, list[tuple[float, float]]] = {}
    wins = losses = ties = 0
    for query_id in common:
        base = score(baseline[query_id])
        cand = score(candidate[query_id])
        if base is None or cand is None:
            continue
        dataset = str(candidate[query_id].get("dataset") or "unknown")
        by_dataset.setdefault(dataset, []).append((base, cand))
        if cand > base:
            wins += 1
        elif cand < base:
            losses += 1
        else:
            ties += 1

    print("| dataset | n | baseline | candidate | delta | wins | losses | ties |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    macro = []
    total_n = 0
    baseline_weighted = 0.0
    candidate_weighted = 0.0
    for dataset, pairs in sorted(by_dataset.items()):
        base = sum(item[0] for item in pairs) / len(pairs)
        cand = sum(item[1] for item in pairs) / len(pairs)
        w = sum(1 for a, b in pairs if b > a)
        l = sum(1 for a, b in pairs if b < a)
        t = len(pairs) - w - l
        macro.append(cand - base)
        total_n += len(pairs)
        baseline_weighted += sum(item[0] for item in pairs)
        candidate_weighted += sum(item[1] for item in pairs)
        print(f"| {dataset} | {len(pairs)} | {base:.4f} | {cand:.4f} | "
              f"{cand - base:+.4f} | {w} | {l} | {t} |")
    if macro:
        base_score = baseline_weighted / total_n
        cand_score = candidate_weighted / total_n
        print(f"| weighted score | {total_n} | {base_score:.4f} | "
              f"{cand_score:.4f} | {cand_score - base_score:+.4f} | "
              f"{wins} | {losses} | {ties} |")
        print(f"| macro | {total_n} | — | — | "
              f"{sum(macro) / len(macro):+.4f} | {wins} | {losses} | {ties} |")
    print(f"\npaired: wins={wins} losses={losses} ties={ties} "
          f"common={len(common)}")
    print("\nrun health (stop_reason):")
    print(_health_line("baseline", baseline))
    print(_health_line("candidate", candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
