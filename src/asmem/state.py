"""Derived state index for triples and explicit UPDATE/DELETE records.

``memory.jsonl`` stays append-only: a later value never rewrites an earlier
row.  ``state.jsonl`` is the authoritative view of "which value is current",
derived deterministically from record order and the ``meta.mutation`` /
``meta.status`` fields written by :func:`asmem.schema.annotate_state`.

The derivation is the same whether it runs in Add (to persist the file) or in
the search index (when an older snapshot has no ``state.jsonl`` yet), so the
two views cannot drift.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl

STATE_NAME = "state.jsonl"


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("meta")
    return value if isinstance(value, dict) else {}


def _date(record: dict[str, Any], meta: dict[str, Any]) -> str:
    explicit = str(meta.get("valid_from") or "")
    if explicit:
        return explicit[:10]
    return str(record.get("time") or "")[:10]


def derive_state(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one state entry per state-bearing record, in record order."""
    entries: list[dict[str, Any]] = []
    active: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        meta = _meta(record)
        key = str(meta.get("state_key") or "")
        if not key:
            continue
        record_id = str(record.get("id") or "")
        value = str(meta.get("object") or record.get("text") or "")
        mutation = str(meta.get("mutation") or "none").lower()
        status = str(meta.get("status") or "active")
        valid_from = _date(record, meta)
        entry: dict[str, Any] = {
            "state_key": key,
            "value": value,
            "source": record_id,
            "valid_from": valid_from,
            "valid_to": str(meta.get("valid_to") or "")[:10],
            "status": status,
            "conflict": False,
            "supersedes": list(meta.get("supersedes") or []),
            "superseded_by": list(meta.get("superseded_by") or []),
        }
        if mutation in {"replace", "delete"}:
            for old in active[key]:
                if old["status"] != "active" or old["value"] == value:
                    continue
                old["status"] = "superseded" if mutation == "replace" else "deleted"
                old["valid_to"] = valid_from
                if record_id not in old["superseded_by"]:
                    old["superseded_by"].append(record_id)
                if old["source"] not in entry["supersedes"]:
                    entry["supersedes"].append(old["source"])
        if status == "active":
            others = [old for old in active[key]
                      if old["status"] == "active" and old["value"] != value]
            if others:
                entry["conflict"] = True
                for old in others:
                    old["conflict"] = True
        entries.append(entry)
        active[key].append(entry)
    return entries


def state_by_key(entries: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry.get("state_key") or "")].append(entry)
    return dict(grouped)


def read_state(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_state(path: Path, entries: list[dict[str, Any]]) -> None:
    write_jsonl(path, entries)
