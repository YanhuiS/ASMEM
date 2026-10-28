#!/usr/bin/env python3
"""Deterministic coarse-recall CLI.

This is the first-stage retriever, exposed as a shell command so the postagent
does not have to construct BM25/dense/RRF by hand.  It is also the same code
path the runner uses for the initial candidate pool: ``run_coarse``.

Examples (from a QA workspace)::

    coarse --query "where did Alice move" --top 8
    coarse --query "Eli Manning sport" --mode dense --top 5
    coarse --query "..." --out inputs/coarse_run   # materialize candidates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from asmem.retrieval.coarse import run_coarse
from asmem.retrieval.parents import resolve_record


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    # The runner writes the ``retrieval`` block itself; older workspaces may
    # have written the full config.  Accept both.
    return value if isinstance(value.get("retrieval"), dict) else {"retrieval": value}


def _log_usage(info: dict[str, Any], path: Path) -> None:
    """Append provider-reported embedding usage for later cost accounting."""
    usage = info.get("embedding_usage") or {}
    if not isinstance(usage, dict) or not int(usage.get("total_tokens", 0) or 0):
        return
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(usage, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _in_range(record: dict[str, Any], start: str, end: str) -> bool:
    if not start and not end:
        return True
    value = str(record.get("time") or "")
    if not value:
        return False
    if start and value < start:
        return False
    if end and value[:len(end)] > end:
        return False
    return True


def _snippet(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if limit and len(text) > limit:
        return text[:limit].rstrip() + " ..."
    return text


def _render_hit(
    rank: int,
    record: dict[str, Any],
    *,
    text: str,
    why: str = "",
    chars: int = 600,
) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    parent = str(meta.get("parent_id") or record.get("id") or "")
    stamp = f"{record.get('time') or 'time?'}"
    role = str(record.get("role") or "-")
    head = f"#{rank} {record.get('id')} [{stamp}] ({role})"
    if why:
        head += f" why: {why}"
    if parent and parent != record.get("id"):
        head += f" parent: {parent}"
    return f"{head}\n  {_snippet(text, chars)}"


async def _search(
    memory: Path,
    query: str,
    *,
    mode: str,
    options: list[str],
    top: int,
    chars: int,
    start: str,
    end: str,
    resolve: str,
    config: dict[str, Any],
    cache_dir: Path,
    out_dir: Path | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    retrieval = config.setdefault("retrieval", {})
    if mode:
        retrieval["mode"] = mode
    result = await run_coarse(
        memory, query, options=options, config=config, cache_dir=cache_dir,
        out_dir=out_dir)
    _log_usage(result.info, Path(".embedding_usage.jsonl"))
    try:
        lines: list[str] = []
        emitted: list[dict[str, Any]] = []
        shown = 0
        for candidate in result.candidates:
            record = next(
                (item for item in result.records
                 if str(item.get("id")) == str(candidate.get("id"))), None)
            if record is None or not _in_range(record, start, end):
                continue
            shown += 1
            if shown > top:
                continue
            text = str(candidate.get("text") or "")
            if resolve in {"parent", "window"}:
                text, _ = resolve_record(
                    {**record, "meta": {**(record.get("meta") or {}),
                                        "resolve": resolve}},
                    result.records, route=str(result.info.get("route") or "dialogue"),
                    question=query, config=config)
            why = ",".join(str(item) for item in (candidate.get("why") or [])[:6])
            if candidate.get("neighbor_of"):
                why = f"neighbor_of={candidate['neighbor_of']} {why}".strip()
            lines.append(_render_hit(shown, record, text=text, why=why, chars=chars))
            emitted.append(candidate)
        if not lines:
            warning = "; ".join(result.info.get("warnings") or [])
            return ([f"no candidate for {query!r}"
                     + (f" ({warning})" if warning else "")], [])
        if len(result.candidates) > top:
            lines.append(f"({len(result.candidates) - top} more candidates; raise --top)")
        lines.append(
            "Use the ids above as metadata.source.  Neighbor hits are context, "
            "not direct matches.")
        return lines, emitted
    finally:
        if result.index is not None:
            await result.index.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coarse",
        description="Deterministic coarse retrieval (BM25/dense/exact/RRF).")
    parser.add_argument("--query", "-q", default="", help="natural-language query")
    parser.add_argument("--mode", choices=["hybrid", "bm25", "dense", "exact"],
                        default="")
    parser.add_argument("--top", "-k", type=int, default=8)
    parser.add_argument("--chars", type=int, default=600,
                        help="max characters of each record shown")
    parser.add_argument("--memory", default="inputs/memory.jsonl")
    parser.add_argument("--config", default="inputs/retrieval.json")
    parser.add_argument("--cache-dir", default="",
                        help="embedding cache (default: the runner's cache)")
    parser.add_argument("--date-from", default="", help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--date-to", default="", help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--resolve", choices=["chunk", "parent", "window"],
                        default="chunk")
    parser.add_argument("--json", action="store_true", help="emit candidates as JSONL")
    parser.add_argument("--options", default="", help="comma-separated MCQ options")
    parser.add_argument("--out", default="",
                        help="write candidates.jsonl + coarse.json to this directory")
    args = parser.parse_args(argv)

    memory = Path(args.memory)
    if not memory.exists():
        print(f"memory not found: {memory}", file=sys.stderr)
        return 2
    config = _load_config(Path(args.config))
    workspace_cache = str(
        (config.get("retrieval") or {}).get("_cache_dir") or "")
    cache_dir = Path(args.cache_dir or workspace_cache or ".index")
    options = [item.strip() for item in args.options.split(",") if item.strip()]
    top = max(1, args.top)

    if not args.query:
        print("pass --query", file=sys.stderr)
        return 2
    try:
        out_dir = Path(args.out) if args.out else None
        lines, emitted = asyncio.run(_search(
            memory, args.query, mode=args.mode, options=options, top=top,
            chars=args.chars, start=args.date_from, end=args.date_to,
            resolve=args.resolve, config=config,
            cache_dir=cache_dir, out_dir=out_dir))
    except Exception as exc:  # the postagent must see an error, not crash
        print(f"coarse search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if out_dir is not None:
        print(f"wrote {out_dir}/candidates.jsonl and {out_dir}/coarse.json")
    if args.json:
        for candidate in emitted:
            print(json.dumps(candidate, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return 0 if emitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
