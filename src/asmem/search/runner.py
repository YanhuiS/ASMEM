"""Search entry point and CLI.

The implementation is split by responsibility:

- :mod:`asmem.search.config`    defaults / YAML merge / model call policy
- :mod:`asmem.search.model`     model client wrapper + token accounting
- :mod:`asmem.search.workspace` per-request workspace + helper CLIs
- :mod:`asmem.search.pipeline`  coarse -> gate/seed -> agent -> postprocess -> verify

This module keeps the stable public names used by the HTTP API and the
benchmark runner, plus the ``python -m asmem.search`` CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ..io import read_jsonl
from ..log import Logger
from .config import CONFIG_FILE, DEFAULT_CONFIG, call_config, load_config
from .model import ModelRunner, _accepts_deadline
from .pipeline import _payload_key, run_question
from .workspace import _run_id

__all__ = [
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "ModelRunner",
    "call_config",
    "load_config",
    "run_question",
]


def _questions(args: argparse.Namespace) -> list[str]:
    if args.query:
        return [args.query]
    if args.query_file:
        items = read_jsonl(Path(args.query_file))
        out = []
        for item in items:
            value = item.get("question") or item.get("query") or item.get("text")
            if isinstance(value, str) and value.strip():
                out.append(value)
        return out[: args.max_qa] if args.max_qa else out
    return []


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m asmem.search",
                                     description="Search generalized memory with a code agent.")
    parser.add_argument("--memory", required=True, help="path to memory.jsonl")
    parser.add_argument("--query", default="", help="one question")
    parser.add_argument("--query-file", default="", help="JSONL with question/query/text")
    parser.add_argument("--config", default="", help="configs/asmem.yaml")
    parser.add_argument("--out", default="", help="run root")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-verify-rounds", type=int, default=-1)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config) if args.config else None)
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.max_verify_rounds >= 0:
        config["max_verify_rounds"] = args.max_verify_rounds
    memory = Path(args.memory)
    if not memory.exists():
        print(f"memory not found: {memory}", file=sys.stderr)
        raise SystemExit(2)
    questions = _questions(args)
    if not questions:
        print("no question: pass --query or --query-file", file=sys.stderr)
        raise SystemExit(2)
    run_root = Path(args.out) if args.out else None
    log = Logger.from_config(config, prefix="search")

    async def run_all() -> None:
        for question in questions:
            out = await run_question(memory_path=memory, question=question,
                                     config=config, run_root=run_root,
                                     log=log, dry_run=args.dry_run)
            if args.dry_run:
                continue
            summary = out["summary"]
            print(f"[{summary['sufficient']}] steps={summary['steps']} "
                  f"evidence={summary['evidence']} {question[:90]}")

    try:
        asyncio.run(run_all())
    finally:
        log.close()


if __name__ == "__main__":
    main()
