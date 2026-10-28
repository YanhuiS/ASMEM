"""Evidence validation.  One record = one self-contained statement + sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..io import parse_jsonl_text, read_jsonl

EVIDENCE_NAME = "evidence.jsonl"


def _content(record: dict[str, Any]) -> str:
    # ``content`` is the evidence schema; ``memory`` is the v1 name; ``text``
    # is what a model writes when it copies a memory record.  Accept all
    # three so a formatting habit cannot throw the record away.
    for key in ("content", "memory", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sources(record: dict[str, Any]) -> list[str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    raw = metadata.get("source")
    if raw is None:
        raw = metadata.get("sources")
    if raw is None and record.get("id") and isinstance(record.get("text"), str):
        # A model sometimes copies a whole memory record:
        # {"id": "s1_00002", "text": "...", "source": "session_1"}.
        # ``source`` there is the logical session, not an evidence source id;
        # the record id is the citable anchor.
        raw = [record.get("id")]
    if raw is None:
        # Models often put the id at the top level: {"content":..., "source":"s1"}.
        raw = record.get("source")
    if raw is None:
        raw = record.get("sources")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item.strip()]


def normalize(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    content = _content(record)
    if not content:
        return None, "missing content"
    sources = _sources(record)
    if not sources:
        return None, "missing metadata.source"
    metadata = dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), dict) else {}
    metadata["source"] = sources
    try:
        metadata["score"] = float(metadata.get("score", 0.5))
    except (TypeError, ValueError):
        metadata["score"] = 0.5
    return {"content": content, "metadata": metadata}, ""


def validate_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read, normalize, deduplicate.  Returns (valid, rejected)."""
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(read_jsonl(path), 1):
        normalized, reason = normalize(record)
        if normalized is None:
            rejected.append({"line": index, "reason": reason})
            continue
        key = " ".join(normalized["content"].split()).lower()
        if key in seen:
            old = seen[key]
            merged = list(dict.fromkeys(
                old["metadata"]["source"] + normalized["metadata"]["source"]))
            old["metadata"]["source"] = merged
            old["metadata"]["score"] = max(old["metadata"]["score"],
                                           normalized["metadata"]["score"])
            continue
        seen[key] = normalized
        valid.append(normalized)
    return valid, rejected


def write_file(path: Path, records: list[dict[str, Any]]) -> None:
    from ..io import write_jsonl

    write_jsonl(path, records)


def read_file(path: Path) -> list[dict[str, Any]]:
    return [item for item in (normalize(record)[0] for record in read_jsonl(path)) if item]


def fingerprint(records: list[dict[str, Any]]) -> str:
    material = "\n".join(
        f"{' '.join(r.get('content', '').split()).lower()}|{','.join((r.get('metadata') or {}).get('source') or [])}"
        for r in records
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").split("\n")
               if line.strip())


def digest(path: Path, limit: int = 3) -> str:
    if not path.exists():
        return "[evidence.jsonl does not exist yet -- append a record to create it]"
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").split("\n")
             if line.strip()]
    if not lines:
        return "[evidence.jsonl is empty]"
    valid, rejected = validate_file(path)
    tail = "\n".join(f"  {line[:220]}" for line in lines[-limit:])
    state = f"{len(valid)} valid"
    if rejected:
        state += f", {len(rejected)} rejected (content or metadata.source missing)"
    return f"[evidence.jsonl: {state}; last {min(limit, len(lines))} line(s)]\n{tail}"


def parse_text(text: str) -> list[dict[str, Any]]:
    """Parse the evidence shapes a model actually replies with.

    Besides JSONL / JSON arrays, accept a wrapper object
    ``{"evidence": [ ... ]}`` and markdown-fenced JSON: small models often
    answer with one envelope instead of the schema the harness asks for.
    """
    records: list[dict[str, Any]] = []
    for record in parse_jsonl_text(text):
        nested = record.get("evidence")
        if isinstance(nested, list) and "content" not in record and "text" not in record:
            records.extend(item for item in nested if isinstance(item, dict))
        else:
            records.append(record)
    return records
