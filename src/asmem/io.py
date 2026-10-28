"""Shared I/O: robust JSONL parsing, atomic writes, paths.

The only invariant: every step reads and writes JSONL files.  The parser is
deliberately forgiving because ``evidence.jsonl`` is written by a model.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    """Write through a temporary file in the same directory, then rename.

    The file is fsynced before the rename and the directory is fsynced after
    it, so a power loss leaves either the old file or the new one — never a
    zero-length file or a rename that is lost after the process reported
    success.  Directory fsync is best-effort (not available on every FS).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    """Parse JSONL, tolerating one-object-per-line, one-array-per-line,
    a whole pretty-printed JSON array, and truncated trailing objects."""
    records: list[dict[str, Any]] = []
    whole = text.strip()
    if not whole:
        return records

    # Fast path: one object per line (the normal case).
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            records.append(value)
        elif isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))

    if records:
        return records

    # Whole-file JSON (``jq -s`` without ``-c``).
    try:
        value = json.loads(whole)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

    # Truncated array: salvage complete objects by brace matching.
    decoder = json.JSONDecoder()
    index = 0
    while index < len(whole):
        start = whole.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(whole, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            records.append(value)
        index = end
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return parse_jsonl_text(path.read_text(encoding="utf-8", errors="replace"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in records)
    atomic_write_text(path, text)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    existing = read_jsonl(path)
    seen = {str(r.get("id")) for r in existing if r.get("id")}
    fresh = [r for r in records if not r.get("id") or str(r.get("id")) not in seen]
    if fresh:
        write_jsonl(path, existing + fresh)


def link_or_copy(source: Path, target: Path) -> str:
    """Hard-link when possible, otherwise copy.  Never fails the run.

    Returns how the file is shared — ``"same"``, ``"hardlink"``, ``"copy"`` or
    ``"failed"`` — so the caller can record whether a run snapshot really
    reuses the live file's inode (no extra disk) or had to copy it (different
    filesystem).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return "same"
    try:
        if target.exists() or target.is_symlink():
            target.unlink()
        os.link(source, target)
        return "hardlink"
    except OSError:
        try:
            target.write_bytes(source.read_bytes())
            return "copy"
        except OSError:
            return "failed"
