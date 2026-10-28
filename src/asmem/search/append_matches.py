#!/usr/bin/env python3
"""Append memory records matching a regex as evidence.

This is a workspace helper for the bash agent; it is not a separate tool.
Usage (from the QA workspace)::

    python bin/append_matches.py 'caroline|lgbtq' --limit 10

It reads ``inputs/memory.jsonl`` and appends one evidence record per match to
``evidence.jsonl``.  Using a regex keeps the agent from hand-writing a fragile
``grep | jq`` pipeline; the resulting records have the same schema the verifier
and post-processor expect.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="case-insensitive regular expression")
    parser.add_argument("--memory", default="inputs/memory.jsonl")
    parser.add_argument("--out", default="evidence.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--score", type=float, default=0.8)
    parser.add_argument("--ids", action="store_true",
                        help="match record ids instead of text")
    args = parser.parse_args(argv)
    try:
        regex = re.compile(args.pattern, re.IGNORECASE)
    except re.error as exc:
        print(f"invalid regex: {exc}")
        return 2

    memory = Path(args.memory)
    if not memory.exists():
        print(f"memory not found: {memory}")
        return 2

    # The evidence file already contains the coarse shortlist.  Do not append
    # a record that is already there: the agent should rewrite or delete it,
    # not duplicate it.
    out = Path(args.out)
    existing_text: set[str] = set()
    existing_sources: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = str((item or {}).get("content") or "")
            if content:
                existing_text.add(" ".join(content.split()).lower())
            metadata = (item or {}).get("metadata") or {}
            sources = metadata.get("source") or []
            if isinstance(sources, list) and len(sources) == 1:
                existing_sources.add(str(sources[0]))

    rows: list[dict] = []
    skipped = 0
    skipped_ids: list[str] = []
    for line in memory.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        haystack = str(record.get("id") if args.ids else record.get("text") or "")
        if not regex.search(haystack):
            continue
        content = str(record.get("text") or "").strip()
        if not content:
            continue
        record_id = str(record.get("id") or "")
        key = " ".join(content.split()).lower()
        if key in existing_text or (record_id and record_id in existing_sources):
            skipped += 1
            if record_id and record_id not in skipped_ids:
                skipped_ids.append(record_id)
            continue
        rows.append({
            "content": content,
            "metadata": {
                "source": [record_id],
                "score": float(args.score),
            },
        })
        existing_text.add(key)
        if len(rows) >= max(1, args.limit):
            break

    if not rows:
        if skipped:
            ids = ", ".join(skipped_ids[:8])
            print(f"No new record matched {args.pattern!r}; {skipped} already in "
                  f"evidence.jsonl ({ids}). Read those records and rewrite the key "
                  "fact as a claim instead of searching again.")
        else:
            print(f"No memory record matched {args.pattern!r}.")
        return 1
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    suffix = f"; {skipped} already present" if skipped else ""
    print(f"Appended {len(rows)} record(s) to {out}{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
