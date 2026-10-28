"""Fast in-process replay of the processed add/search event stream.

Why not simply call ``UserStore.append`` for every add? That is O(n^2): it reads
and rewrites the whole ``memory.jsonl`` on each append. LongMemEval-m has ~475
adds and ~5k messages per user, which would rewrite gigabytes per user. Here we
keep the normalized records in memory, use the *same* ``normalize_messages``
function the store uses, and write the file only before a search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import write_jsonl
from ..schema import chunking_config, normalize_messages


class FastMemory:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self.sessions: dict[str, dict[str, int]] = {}
        self.next_index = 1
        self.applied: set[str] = set()
        self.dirty = False
        self.chunking = chunking_config(config)

    def add(self, event: dict[str, Any]) -> int:
        request_id = str(event.get("request_id") or "")
        if request_id and request_id in self.applied:
            return 0
        session_id = str(event.get("session_id") or "session_1")
        session = self.sessions.get(session_id)
        if session is None:
            session = {"index": self.next_index, "next": 1}
            self.sessions[session_id] = session
            self.next_index += 1
        _, records = normalize_messages(
            event.get("messages") or [],
            session_id=session_id,
            session_index=int(session["index"]),
            start_seq=int(session["next"]),
            chunking=self.chunking,
        )
        session["next"] += len(records)
        self.records.extend(records)
        if request_id:
            self.applied.add(request_id)
        self.dirty = bool(records) or self.dirty
        return len(records)

    def write(self, path: Path) -> None:
        write_jsonl(path, self.records)
        self.dirty = False

    def clear(self) -> None:
        self.records.clear()
        self.sessions.clear()
        self.next_index = 1
        self.applied.clear()
        self.dirty = False
