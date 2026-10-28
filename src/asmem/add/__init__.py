"""ASMEM add: receive data, normalize the format, write one JSONL memory.

No model calls.  No summaries.  No atom extraction.  Every input is mapped to
the generalized record schema in :mod:`asmem.schema`; the search agent reads
that single file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..io import append_jsonl, write_jsonl
from ..schema import normalize_payload

MEMORY_NAME = "memory.jsonl"


def load_payload(path: Path) -> Any:
    """Read a JSON/JSONL file, or every JSON/JSONL file in a directory."""
    if path.is_dir():
        items: list[Any] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in (".json", ".jsonl"):
                loaded = load_payload(child)
                items.extend(loaded if isinstance(loaded, list) else [loaded])
        return items
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        from ..io import parse_jsonl_text

        return parse_jsonl_text(text)
    return json.loads(text)


def normalize_input(
    payload: Any,
    *,
    session_id: str = "session_1",
    session_index: int = 1,
    start_seq: int = 1,
    max_chars: int | None = None,
    chunking: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    return normalize_payload(
        payload,
        session_id=session_id,
        session_index=session_index,
        start_seq=start_seq,
        max_chars=max_chars,
        chunking=chunking,
    )


def write_memory(path: Path, records: list[dict[str, Any]], *, append: bool = False) -> int:
    if append:
        append_jsonl(path, records)
        return len(records)
    write_jsonl(path, records)
    return len(records)
