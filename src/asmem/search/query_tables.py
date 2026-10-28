#!/usr/bin/env python3
"""Run a read-only SQL query against ``inputs/tables.db``.

    python bin/query_tables.py "SELECT name, MAX(closed) FROM rows"
    python bin/query_tables.py "SELECT * FROM rows WHERE name LIKE '%School%'" --json

The database is built by the runner from ``kind=row`` records.  Only SELECT /
WITH statements are allowed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="SELECT or WITH statement")
    parser.add_argument("--db", default="inputs/tables.db")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)
    path = Path(args.db)
    if not path.exists():
        print(f"table database not found: {path}", file=sys.stderr)
        return 2
    statement = args.query.strip()
    if not statement.lower().startswith(("select", "with")):
        print("only SELECT/WITH statements are allowed", file=sys.stderr)
        return 2
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(statement)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(max(1, args.limit))
    except sqlite3.Error as exc:
        print(f"SQL error: {exc}", file=sys.stderr)
        return 1
    finally:
        connection.close()
    if args.json:
        for row in rows:
            print(json.dumps(dict(zip(columns, row)), ensure_ascii=False))
    else:
        print("\t".join(columns))
        for row in rows:
            print("\t".join("" if value is None else str(value) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
