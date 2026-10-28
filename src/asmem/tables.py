"""Optional read-only SQLite view of ``kind=row`` records.

Table/aggregation questions are much easier to answer with SQL than by reading
JSONL one line at a time.  The database is built in the QA workspace only when
the snapshot actually contains rows; it is never part of the memory store.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

_SQL_NAME = re.compile(r"[^A-Za-z0-9_]+")


def _column_name(value: Any, used: set[str]) -> str:
    name = _SQL_NAME.sub("_", str(value or "")).strip("_") or "col"
    if name[0].isdigit():
        name = f"c_{name}"
    base = name[:60]
    name = base
    suffix = 2
    while name.lower() in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name.lower())
    return name


def row_columns(records: Iterable[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        if str(record.get("kind") or "") != "row":
            continue
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        declared = meta.get("columns") or []
        data = meta.get("data")
        names = list(declared) if isinstance(declared, list) else []
        if isinstance(data, dict):
            names.extend(data.keys())
        for name in names:
            text = str(name)
            if text not in seen:
                seen.add(text)
                columns.append(text)
    return columns


def build_tables_db(records: Iterable[dict[str, Any]], path: Path) -> int:
    """Materialize rows into ``path``.  Returns the number of rows written."""
    records = list(records)
    rows = [record for record in records
            if str(record.get("kind") or "") == "row"]
    if not rows:
        return 0
    columns = row_columns(rows)
    sql_columns = []
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for column in columns:
        sql = _column_name(column, used)
        mapping[column] = sql
        sql_columns.append(f'"{sql}" TEXT')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            'CREATE TABLE rows (id TEXT, source TEXT, row_no INTEGER, data TEXT'
            + ("".join(f", {column}" for column in sql_columns)) + ")")
        for record in rows:
            meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
            data = meta.get("data")
            declared = meta.get("columns") if isinstance(meta.get("columns"), list) else []
            values: dict[str, Any] = {}
            if isinstance(data, dict):
                values.update(data)
            elif isinstance(data, list):
                for index, value in enumerate(data):
                    if index < len(declared):
                        values[str(declared[index])] = value
            payload = [
                str(record.get("id") or ""),
                str(record.get("source") or ""),
                int(meta.get("row") or 0),
                json.dumps(data, ensure_ascii=False) if data is not None else "{}",
            ]
            payload.extend(
                None if values.get(column) is None else str(values.get(column))
                for column in columns)
            placeholders = ", ".join("?" for _ in payload)
            connection.execute(
                f"INSERT INTO rows VALUES ({placeholders})", payload)
        connection.commit()
    finally:
        connection.close()
    return len(rows)
