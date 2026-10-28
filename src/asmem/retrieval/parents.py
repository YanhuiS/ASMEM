"""Parent/chunk return policy.

Add always chunks long content.  At search time the final evidence unit is
chosen per record:

- an atomic record (row/triple/short message) is returned as a chunk;
- a chunk whose parent is small enough is expanded to the parent when the
  query is a summary/aggregation, when the chunk starts mid-sentence, or when
  the parent is cheap (<= ``parent_auto_chars``);
- a chunk whose parent is huge returns a local window (chunk +/- N siblings)
  instead of the whole document.

The agent may override the decision with ``metadata.resolve`` =
``"chunk" | "parent" | "window"``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

_PRONOUN_START = frozenset({
    "he", "she", "it", "they", "this", "that", "these", "those",
    "his", "her", "its", "their", "him", "them", "we", "our", "you",
})
_TERMINAL = ".!?。！？\"'”’)]}"


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("meta")
    return value if isinstance(value, dict) else {}


def parent_key(record: dict[str, Any]) -> str:
    return str(_meta(record).get("parent_id") or record.get("id") or "")


def group_by_parent(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[parent_key(record)].append(record)
    for items in groups.values():
        items.sort(key=lambda item: int(_meta(item).get("part") or 1))
    return groups


def parent_text(record: dict[str, Any], records: Sequence[dict[str, Any]]) -> str:
    parts = int(_meta(record).get("parts") or 1)
    if parts <= 1:
        return str(record.get("text") or "")
    key = parent_key(record)
    siblings = [item for item in records if parent_key(item) == key]
    siblings.sort(key=lambda item: int(_meta(item).get("part") or 1))
    text = "\n".join(str(item.get("text") or "") for item in siblings).strip()
    return text or str(record.get("text") or "")


def window_text(
    record: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    window: int = 1,
    max_chars: int = 4000,
) -> str:
    key = parent_key(record)
    siblings = [item for item in records if parent_key(item) == key]
    siblings.sort(key=lambda item: int(_meta(item).get("part") or 1))
    part = int(_meta(record).get("part") or 1)
    low, high = max(1, part - window), part + window
    pieces = [
        str(item.get("text") or "")
        for item in siblings
        if low <= int(_meta(item).get("part") or 1) <= high
    ]
    text = "\n".join(pieces).strip()
    if len(text) > max_chars:
        # Keep the anchor chunk and trim the edges rather than returning a
        # huge document.  The anchor is normally near the middle.
        anchor = str(record.get("text") or "")
        if len(anchor) >= max_chars:
            return anchor[:max_chars]
        return text[:max_chars]
    return text or str(record.get("text") or "")


def weak_boundary(text: str) -> bool:
    """Does this chunk look like it starts or ends mid-sentence?"""
    stripped = (text or "").strip()
    if not stripped:
        return True
    first = stripped[0]
    if first.islower():
        return True
    first_word = stripped.split(maxsplit=1)[0].strip(".,;:!?").lower()
    if first_word in _PRONOUN_START:
        return True
    if stripped[-1] not in _TERMINAL:
        return True
    return False


def choose_resolution(
    record: dict[str, Any],
    *,
    route: str,
    question: str,
    config: dict[str, Any] | None = None,
) -> str:
    config = config or {}
    meta = _meta(record)
    parts = int(meta.get("parts") or 1)
    if parts <= 1:
        return "chunk"
    kind = str(record.get("kind") or "record")
    if kind in {"row", "triple"}:
        return "chunk"
    parent_chars = int(meta.get("parent_chars") or 0)
    parent_max = int(config.get("parent_max_chars", 4000))
    auto_parent = int(config.get("parent_auto_chars", 1500))
    if str(config.get("parent_mode") or "auto").lower() == "chunk":
        return "chunk"
    if str(config.get("parent_mode") or "auto").lower() == "parent":
        return "parent" if parent_chars <= parent_max else "window"
    if route in {"summary", "aggregation"}:
        return "parent" if parent_chars <= parent_max else "window"
    if weak_boundary(str(record.get("text") or "")):
        return "parent" if parent_chars <= parent_max else "window"
    if parent_chars and parent_chars <= auto_parent:
        return "parent"
    return "chunk"


def resolve_record(
    record: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    route: str,
    question: str,
    config: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return ``(text, mode)`` for the final evidence unit."""
    config = config or {}
    override = str(_meta(record).get("resolve") or "").lower()
    mode = override if override in {"chunk", "parent", "window"} else choose_resolution(
        record, route=route, question=question, config=config)
    if mode == "parent":
        text = parent_text(record, records)
        max_chars = int(config.get("parent_max_chars", 4000))
        if len(text) > max_chars:
            text = text[:max_chars]
        return text or str(record.get("text") or ""), "parent"
    if mode == "window":
        return window_text(
            record, records,
            window=int(config.get("parent_window_chunks", 1)),
            max_chars=int(config.get("window_max_chars", 4000)),
        ), "window"
    return str(record.get("text") or ""), "chunk"
