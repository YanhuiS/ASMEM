"""Re-chunk an existing ``memory.jsonl`` with the current token chunk policy.

Add applies the policy to new input.  This script brings an older snapshot up
to date without changing its facts:

    python scripts/rechunk_memory.py \
        --input data/api_store/u_x/memory.jsonl \
        --output /tmp/memory.rechunked.jsonl

- rows and triples stay atomic;
- an already-split record (``meta.parts > 1``) is left untouched;
- the first part keeps the original id, later parts get ``<id>_p2``, ...
  so old references to the first chunk still resolve;
- ``meta.parent_id`` points at the original id;
- ``chunk_version`` and the embedding identity change together, so a
  re-chunked snapshot gets a fresh vector cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem.io import read_jsonl, write_jsonl  # noqa: E402
from asmem.schema import (  # noqa: E402
    DEFAULT_CHUNKING, chunk_units_tokens, chunk_version, tokens_for_kind,
)


def rechunk(records: list[dict], chunking: dict) -> tuple[list[dict], dict]:
    overlap = max(0, int(chunking.get("overlap_tokens", 40)))
    out: list[dict] = []
    split = kept = 0
    for record in records:
        meta = dict(record.get("meta") or {})
        kind = str(record.get("kind") or "record")
        text = str(record.get("text") or "")
        parts = int(meta.get("parts") or 1)
        limit = tokens_for_kind(kind, chunking)
        if parts > 1 or limit <= 0 or kind in {"row", "triple"}:
            meta.setdefault("chunk_version", chunk_version(chunking))
            out.append({**record, "meta": meta})
            kept += 1
            continue
        units = chunk_units_tokens(
            text, limit, overlap_tokens=overlap,
            min_tokens=int(chunking.get("min_tokens", 20)))
        if len(units) <= 1:
            meta.setdefault("chunk_version", chunk_version(chunking))
            out.append({**record, "meta": meta})
            kept += 1
            continue
        original_id = str(record.get("id") or f"r{len(out):06d}")
        for index, unit in enumerate(units, 1):
            record_id = original_id if index == 1 else f"{original_id}_p{index}"
            out.append({
                **record,
                "id": record_id,
                "text": unit["text"],
                "kind": "passage",
                "meta": {
                    **meta,
                    "parent_id": original_id,
                    "part": index,
                    "parts": len(units),
                    "parent_chars": len(text),
                    "parent_kind": kind,
                    "chunk_version": chunk_version(chunking),
                    "char_start": unit["char_start"],
                    "char_end": unit["char_end"],
                    "token_start": unit["token_start"],
                    "token_end": unit["token_end"],
                    "tokens": unit["tokens"],
                },
            })
        split += 1
    return out, {"input": len(records), "output": len(out),
                 "split_records": split, "kept_records": kept}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--message-tokens", type=int,
                        default=int(DEFAULT_CHUNKING["message_tokens"]))
    parser.add_argument("--passage-tokens", type=int,
                        default=int(DEFAULT_CHUNKING["passage_tokens"]))
    parser.add_argument("--record-tokens", type=int,
                        default=int(DEFAULT_CHUNKING["record_tokens"]))
    parser.add_argument("--overlap-tokens", type=int,
                        default=int(DEFAULT_CHUNKING["overlap_tokens"]))
    parser.add_argument("--min-tokens", type=int,
                        default=int(DEFAULT_CHUNKING["min_tokens"]))
    args = parser.parse_args(argv)
    chunking = {
        **DEFAULT_CHUNKING,
        "message_tokens": args.message_tokens,
        "passage_tokens": args.passage_tokens,
        "record_tokens": args.record_tokens,
        "overlap_tokens": args.overlap_tokens,
        "min_tokens": args.min_tokens,
    }
    source = Path(args.input)
    if not source.exists():
        print(f"input not found: {source}", file=sys.stderr)
        return 2
    records, stats = rechunk(read_jsonl(source), chunking)
    write_jsonl(Path(args.output), records)
    print(json.dumps({**stats, "chunk_version": chunk_version(chunking)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
