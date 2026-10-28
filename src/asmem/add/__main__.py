"""``python -m asmem.add`` -- normalize arbitrary data into ``memory.jsonl``.

Examples::

    python -m asmem.add --input data/Caroline_Melanie/sessions.jsonl \\
        --user caroline --session session_1 --out data/asmem

    python -m asmem.add --input data/api_store/u_x/requests.jsonl --legacy
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ..io import DATA_DIR, read_jsonl
from . import MEMORY_NAME, load_payload, normalize_input, write_memory

_SEQ = re.compile(r"_(\d+)$")


def slug(user_id: str) -> str:
    keep = [c if c.isalnum() or c in "._-" else "_" for c in user_id]
    return "".join(keep).strip("._-")[:80] or "user"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m asmem.add",
                                     description="Normalize data into the ASMEM memory format.")
    parser.add_argument("--input", required=True, help="JSON/JSONL file or a directory")
    parser.add_argument("--user", default="default", help="user id (directory name)")
    parser.add_argument("--session", default="session_1", help="session id for this batch")
    parser.add_argument("--session-index", type=int, default=1, help="numeric session index")
    parser.add_argument("--out", default="", help="output root (default data/asmem)")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="force one fixed chunk width (0 = per-kind policy)")
    parser.add_argument("--append", action="store_true", help="append instead of rewriting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.input)
    if not source.exists():
        print(f"input not found: {source}", file=sys.stderr)
        raise SystemExit(2)

    payload = load_payload(source)
    family, records = normalize_input(
        payload,
        session_id=args.session,
        session_index=args.session_index,
        max_chars=args.max_chars or None,
    )
    root = Path(args.out) if args.out else DATA_DIR / "asmem"
    target = root / slug(args.user) / MEMORY_NAME
    if args.append and target.exists():
        used = [str(item.get("id") or "") for item in read_jsonl(target)]
        sequence = 1 + max(
            (int(match.group(1)) for item in used if (match := _SEQ.search(item))),
            default=0,
        )
        for record in records:
            record["id"] = f"s{args.session_index}_{sequence:05d}"
            sequence += 1
    write_memory(target, records, append=args.append)
    print(f"add: {len(records)} record(s), family={family}, -> {target}")


if __name__ == "__main__":
    main()
