"""Per-user store administration: stats and prefix purge.

    python scripts/memory_admin.py --root data/api_store stats
    python scripts/memory_admin.py --root data/api_store \
        purge-prefix --user-prefix 'eval:run:' --confirm

Retention is a deployment/audit decision, so deletion is explicit and requires
``--confirm``.  ``purge-prefix`` matches the original ``user_id`` stored in
``meta.json``, not the slugged directory name.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _candidate_dirs(root: Path):
    """User directories in layout v2 (``users/*``) and legacy (``u_*``)."""
    seen = set()
    users = root / "users"
    if users.is_dir():
        for child in sorted(users.iterdir()):
            if child.is_dir():
                seen.add(child.resolve())
                yield child
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in {"users", "shared"}:
            if child.resolve() not in seen:
                yield child


def _users(root: Path):
    if not root.exists():
        return
    for child in _candidate_dirs(root):
        meta_path = child / "meta.json"
        user_id = child.name
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                user_id = str(meta.get("user_id") or user_id)
            except (json.JSONDecodeError, OSError):
                pass
        yield child, user_id


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").split("\n")
               if line.strip())


def stats(root: Path) -> dict:
    users = records = states = size = 0
    # Runs hardlink the user's memory/state.  Count each inode once so the
    # reported size is the real disk footprint, not the sum over every run
    # snapshot.  Symlinks (runs/*/bin -> shared/bin) are not counted.
    inodes: set[tuple[int, int]] = set()
    for directory, _ in _users(root) or ():
        users += 1
        records += _line_count(directory / "memory.jsonl")
        states += _line_count(directory / "state.jsonl")
        for path in directory.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key in inodes:
                continue
            inodes.add(key)
            size += stat.st_size
    return {"root": str(root), "users": users, "records": records,
            "state_entries": states, "bytes": size}


def purge_prefix(root: Path, prefix: str, *, confirm: bool) -> dict:
    if not prefix:
        raise ValueError("--user-prefix must not be empty")
    if not confirm:
        raise ValueError("purge-prefix requires --confirm")
    removed = []
    for directory, user_id in _users(root) or ():
        if user_id.startswith(prefix):
            shutil.rmtree(directory)
            removed.append(user_id)
    return {"root": str(root), "removed": len(removed), "users": removed[:50]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/api_store")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    purge = sub.add_parser("purge-prefix")
    purge.add_argument("--user-prefix", required=True)
    purge.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        if args.command == "stats":
            result = stats(root)
        else:
            result = purge_prefix(root, args.user_prefix, confirm=args.confirm)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
