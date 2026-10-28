"""One generalized memory record and the normalizer that produces it.

Every input -- chat messages, documents, table rows, triples, agent traces --
becomes the same JSONL shape::

    {"id": "s1_00001", "text": "...", "kind": "message|passage|row|triple|record",
     "time": "2023-05-08T13:56:00Z", "role": "Caroline",
     "source": "session_1", "meta": {...}}

``text`` is what the search agent greps and reads.  ``meta`` keeps structured
fields so ``jq`` can still aggregate table rows.  Nothing is summarized and
nothing is rewritten: add is a pure format layer.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Sequence

from . import adapters

DEFAULT_MAX_CHARS = 1200
KINDS = ("message", "passage", "row", "triple", "record")

#: Chunk policy.  A message is a natural retrieval unit, so it gets a larger
#: threshold than a document passage; rows and triples are atomic.  The
#: overlap keeps a fact that straddles a boundary reachable from both chunks.
#: ``version`` participates in the embedding identity: changing the policy
#: invalidates the old vector cache instead of silently mixing chunkings.
DEFAULT_CHUNKING: dict[str, Any] = {
    "version": 3,
    # Token budgets.  Tokens are the same units BM25 sees: ASCII words, CJK
    # characters, and single punctuation marks.  A token budget is a much more
    # stable "length" than characters across English, Chinese, tables, and code.
    "message_tokens": 320,
    "passage_tokens": 320,
    "record_tokens": 320,
    "overlap_tokens": 40,
    "min_tokens": 20,
    # Legacy character thresholds (used only when a caller passes max_chars or
    # all token budgets are zero).
    "message_chars": 3000,
    "passage_chars": 1500,
    "record_chars": 2000,
    "overlap_chars": 120,
    "min_chars": 120,
}


def chunking_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read ``retrieval.chunking`` (or a top-level ``chunking``) with defaults."""
    config = config or {}
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    raw = retrieval.get("chunking") if isinstance(retrieval.get("chunking"), dict) else {}
    if not raw and isinstance(config.get("chunking"), dict):
        raw = config["chunking"]
    merged = dict(DEFAULT_CHUNKING)
    merged.update({key: value for key, value in (raw or {}).items() if value is not None})
    return merged


def chars_for_kind(kind: str, chunking: dict[str, Any] | None = None) -> int:
    """Per-kind chunk threshold; ``0`` means the unit is atomic."""
    cfg = chunking or DEFAULT_CHUNKING
    kind = _KIND_ALIASES.get(kind, kind)
    if kind in ("row", "triple"):
        return 0
    if kind == "message":
        return max(1, int(cfg.get("message_chars", 3000)))
    if kind == "passage":
        return max(1, int(cfg.get("passage_chars", 1500)))
    return max(1, int(cfg.get("record_chars", 2000)))


def tokens_for_kind(kind: str, chunking: dict[str, Any] | None = None) -> int:
    """Per-kind token budget; ``0`` means the unit is atomic."""
    cfg = chunking or DEFAULT_CHUNKING
    kind = _KIND_ALIASES.get(kind, kind)
    if kind in ("row", "triple"):
        return 0
    if kind == "message":
        return max(0, int(cfg.get("message_tokens", 320)))
    if kind == "passage":
        return max(0, int(cfg.get("passage_tokens", 320)))
    return max(0, int(cfg.get("record_tokens", 320)))


def chunk_version(chunking: dict[str, Any] | None = None) -> str:
    cfg = chunking or DEFAULT_CHUNKING
    return (
        f"v{int(cfg.get('version', 3))}:"
        f"{int(cfg.get('message_tokens', 320))}:"
        f"{int(cfg.get('passage_tokens', 320))}:"
        f"{int(cfg.get('record_tokens', 320))}:"
        f"{int(cfg.get('overlap_tokens', 40))}:"
        f"{int(cfg.get('min_tokens', 20))}"
    )


def make_record(
    record_id: str,
    text: str,
    *,
    kind: str = "record",
    time: str = "",
    role: str = "",
    source: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "text": text.strip(),
        "kind": kind if kind in KINDS else "record",
        "time": time or "",
        "role": role or "",
        "source": source or "",
        "meta": meta or {},
    }


_TRIPLE_LINE = re.compile(r"^(?:UPDATE:\s*)?(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$")


def annotate_state(record: dict[str, Any]) -> dict[str, Any]:
    """Attach deterministic triple/version fields when the shape is explicit.

    This is regex-only: ``UPDATE: s | r | o`` and plain ``s | r | o``.  It never
    guesses at natural-language updates; those stay ordinary records and the
    agent decides what to cite.
    """
    text = str(record.get("text") or "").strip()
    kind = str(record.get("kind") or "")
    if kind != "triple" and not (text.lower().startswith("update:") and "|" in text):
        return record
    update = text.lower().startswith("update:")
    body = text.split("(", 1)[0].strip()
    if update:
        body = body[len("update:"):].strip()
    match = _TRIPLE_LINE.match(body)
    if not match:
        return record
    subject, relation, obj = (part.strip() for part in match.groups())
    if not (subject and relation and obj):
        return record
    meta = dict(record.get("meta") or {})
    meta.setdefault("subject", subject)
    meta.setdefault("relation", relation)
    meta.setdefault("object", obj)
    meta.setdefault("state_key", f"{subject}|{relation}")
    meta.setdefault("status", "active")
    meta.setdefault("mutation", "replace" if update else "none")
    if not meta.get("valid_from") and record.get("time"):
        meta["valid_from"] = str(record.get("time"))[:10]
    record["meta"] = meta
    return record


def normalize_time(value: Any) -> str:
    """Unix ms / seconds / ISO / a few common strings -> ISO-8601 UTC."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e12:          # milliseconds
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OSError, OverflowError):
            return ""
    text = str(value).strip()
    if not text:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%b %d, %Y, %H:%M:%S",
                "%d %B, %Y", "%d %B %Y", "%B %d, %Y",
                "%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try:
            moment = datetime.strptime(text, fmt)
            return moment.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    try:
        return normalize_time(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except (ValueError, OSError):
        return ""


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    overlap_chars: int = 0,
) -> list[str]:
    """Split long content on paragraph, then sentence, then hard boundaries.

    ``overlap_chars`` prepends the tail of the previous chunk to the next one,
    so a fact that straddles a boundary is reachable from both retrieval units.
    The overlap is cut at a whitespace boundary when possible.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            sentence = sentence.strip()
            if not sentence:
                continue
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:].lstrip()
            if sentence:
                chunks.append(sentence)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) + len(chunk) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]}\n{chunk}"
        else:
            merged.append(chunk)

    overlap = max(0, int(overlap_chars))
    if overlap and len(merged) > 1:
        with_overlap = [merged[0]]
        for previous, current in zip(merged, merged[1:]):
            tail = previous[-overlap:]
            space = tail.find(" ")
            if 0 < space < len(tail) // 2:
                tail = tail[space + 1:]
            with_overlap.append(f"{tail.strip()}\n{current}" if tail.strip() else current)
        merged = with_overlap
    return merged


#: Chunk-length tokenizer.  It matches the indexing philosophy: an ASCII
#: word/number, one CJK character, or one punctuation mark.  A chunk boundary
#: therefore never falls inside an ASCII word; Chinese is counted per character.
_CHUNK_TOKEN = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _CHUNK_TOKEN.finditer(text or "")]


def token_count(text: str) -> int:
    return len(token_spans(text))


def _boundary_rank(gap: str) -> int:
    """0 = none, 1 = clause, 2 = sentence, 3 = paragraph."""
    if "\n\n" in gap:
        return 3
    if any(mark in gap for mark in ".!?。！？"):
        return 2
    if "\n" in gap or any(mark in gap for mark in ",;:，；："):
        return 1
    return 0


def _snap_end(
    text: str,
    spans: list[tuple[int, int]],
    start: int,
    end_raw: int,
    min_tokens: int,
) -> int:
    """Prefer a paragraph/sentence/clause boundary near the end of a window.

    The search is limited to the last 25% of the window so a boundary can
    neither shrink a chunk drastically nor create a gap with the next window.
    """
    if end_raw >= len(spans) or end_raw - start <= max(1, min_tokens):
        return end_raw
    floor = max(start + min_tokens, end_raw - max(1, (end_raw - start) // 4))
    best_rank = 0
    best = end_raw
    for position in range(end_raw, floor, -1):
        gap = text[spans[position - 1][1]:spans[position][0]]
        rank = _boundary_rank(gap)
        if rank > best_rank:
            best_rank, best = rank, position
            if rank == 3:
                break
    return best


def chunk_units_tokens(
    text: str,
    max_tokens: int,
    *,
    overlap_tokens: int = 0,
    min_tokens: int = 1,
) -> list[dict[str, Any]]:
    """Token-window chunking with boundary snapping and token overlap.

    Each unit carries its token/char offsets.  The window step is
    ``max_tokens - overlap_tokens``; after a boundary snap the next window
    starts ``overlap_tokens`` before the snapped end, so no source token is
    skipped and the boundary facts are visible from both sides.
    """
    text = (text or "").strip()
    spans = token_spans(text)
    if not spans:
        return []
    max_tokens = max(1, int(max_tokens))
    min_tokens = max(1, min(int(min_tokens), max_tokens))
    overlap = max(0, min(int(overlap_tokens), max_tokens // 2))
    step = max(1, max_tokens - overlap)
    units: list[dict[str, Any]] = []
    start = 0
    total = len(spans)
    while start < total:
        end_raw = min(total, start + max_tokens)
        end = _snap_end(text, spans, start, end_raw, min_tokens)
        if end <= start:
            end = end_raw
        char_start, char_end = spans[start][0], spans[end - 1][1]
        piece = text[char_start:char_end].strip()
        if piece:
            units.append({
                "text": piece,
                "token_start": start,
                "token_end": end,
                "tokens": end - start,
                "char_start": char_start,
                "char_end": char_end,
            })
        if end >= total:
            break
        # Keep the promised overlap while guaranteeing forward progress.
        next_start = max(start + 1, end - overlap)
        if next_start == start:
            next_start = start + step
        start = next_start
    return units


def chunk_text_tokens(
    text: str,
    max_tokens: int,
    *,
    overlap_tokens: int = 0,
    min_tokens: int = 1,
) -> list[str]:
    return [unit["text"] for unit in chunk_units_tokens(
        text, max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)]


def _json_units(text: str) -> list[tuple[str, str, dict[str, Any]]] | None:
    """Table / structured JSON -> one unit per row, else one record."""
    stripped = (text or "").strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    def rows(value: Any) -> list[tuple[str, str, dict[str, Any]]]:
        if isinstance(value, dict):
            for key in ("rows", "data", "records", "items"):
                if isinstance(value.get(key), list):
                    columns = value.get("columns") or value.get("headers")
                    return _rows_from(value[key], columns)
            return [("record", _render_dict(value), {"data": value})]
        if isinstance(value, list):
            return _rows_from(value, None)
        return [("record", str(value), {})]

    return rows(payload)


def _render_dict(value: dict[str, Any]) -> str:
    lines = []
    for key, item in value.items():
        rendered = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def _rows_from(items: Sequence[Any], columns: Sequence[Any] | None) -> list[tuple[str, str, dict[str, Any]]]:
    units: list[tuple[str, str, dict[str, Any]]] = []
    for index, row in enumerate(items):
        if isinstance(row, dict):
            text = "; ".join(
                f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
                for key, value in row.items()
            )
            units.append(("row", f"row {index + 1}: {text}", {"row": index + 1, "data": row}))
        elif isinstance(row, (list, tuple)) and columns and len(columns) == len(row):
            text = "; ".join(f"{col}: {val}" for col, val in zip(columns, row))
            units.append(("row", f"row {index + 1}: {text}",
                          {"row": index + 1, "columns": list(columns), "data": list(row)}))
        else:
            units.append(("row", f"row {index + 1}: {row}", {"row": index + 1}))
    return units


_KIND_ALIASES = {
    "dialogue": "message", "plain": "message", "narration": "message",
    "document": "passage", "facts": "record", "trajectory": "record",
    "qa-feedback": "record", "triples": "triple",
}


def content_units(
    text: str,
    *,
    kind: str = "record",
    max_chars: int | None = None,
    chunking: dict[str, Any] | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Turn one message/document into searchable units.

    Long text is always split (``chunk_text``) with a per-kind threshold and a
    small overlap; the caller attaches a shared ``parent_id`` and
    ``part/parts`` so retrieval can decide whether to return the chunk or the
    original parent at query time.  An explicit ``max_chars`` keeps the legacy
    fixed-width behaviour.
    """
    structured = _json_units(text)
    if structured:
        return structured
    original_kind = _KIND_ALIASES.get(kind, kind)
    if original_kind in ("triple", "triples"):
        return [("triple", text.strip(), {})]
    cfg = chunking or DEFAULT_CHUNKING
    original = original_kind if original_kind in KINDS else "record"
    if not max_chars:
        token_limit = tokens_for_kind(original_kind, cfg)
        if token_limit > 0:
            token_units = chunk_units_tokens(
                text, token_limit,
                overlap_tokens=int(cfg.get("overlap_tokens", 40)),
                min_tokens=int(cfg.get("min_tokens", 20)))
            if not token_units:
                return []
            if len(token_units) == 1:
                unit = token_units[0]
                return [(original, unit["text"], {
                    "char_start": unit["char_start"], "char_end": unit["char_end"],
                    "token_start": unit["token_start"], "token_end": unit["token_end"],
                    "tokens": unit["tokens"]})]
            return [("passage", unit["text"], {
                "part": index + 1, "parts": len(token_units),
                "char_start": unit["char_start"], "char_end": unit["char_end"],
                "token_start": unit["token_start"], "token_end": unit["token_end"],
                "tokens": unit["tokens"],
            }) for index, unit in enumerate(token_units)]

    # Legacy explicit-width path (tests, callers passing max_chars).
    limit = int(max_chars) if max_chars else chars_for_kind(original_kind, cfg)
    if limit <= 0:
        return [(original, text.strip(), {})]
    overlap = 0 if max_chars else max(0, int(cfg.get("overlap_chars", 0)))
    pieces = chunk_text(text, max_chars=limit, overlap_chars=overlap)
    if not pieces:
        return []
    if len(pieces) == 1:
        return [(original, pieces[0],
                 {"char_start": 0, "char_end": len(pieces[0]),
                  "token_start": 0, "token_end": token_count(pieces[0]),
                  "tokens": token_count(pieces[0])})]
    units: list[tuple[str, str, dict[str, Any]]] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        cursor = max(cursor, start + len(piece))
        units.append(("passage", piece, {
            "part": index + 1, "parts": len(pieces),
            "char_start": start, "char_end": start + len(piece),
            "token_start": 0, "token_end": token_count(piece),
            "tokens": token_count(piece),
        }))
    return units


def parent_metadata(
    *,
    parent_id: str,
    part: int,
    parts: int,
    parent_chars: int,
    parent_kind: str,
    chunking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fields that let the retrieval layer reassemble a split record."""
    return {
        "parent_id": parent_id,
        "part": int(part),
        "parts": int(parts),
        "parent_chars": int(parent_chars),
        "parent_kind": parent_kind,
        "chunk_version": chunk_version(chunking),
    }


def _content_text(content: Any) -> str:
    """Unordered content parts -> one searchable string (images keep a marker)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                ref = part.get("image_url") or part.get("url") or part.get("path")
                if isinstance(ref, dict):
                    ref = ref.get("url")
                if isinstance(ref, str) and ref.startswith("data:"):
                    # Inline base64 must never enter memory text: one image is
                    # up to 10 MiB and would poison BM25, evidence and prompts.
                    header, _, payload = ref.partition(",")
                    mime = (header[5:].split(";", 1)[0].strip()
                            if header.startswith("data:") else "")
                    size = (len(payload) * 3) // 4 if payload else 0
                    parts.append(f"[image: {mime or 'inline'}, ~{size} bytes]")
                else:
                    parts.append(f"[image: {ref}]" if ref else "[image]")
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {**message, "content": _content_text(message.get("content"))}
    return {"role": "user", "content": str(message)}


def normalize_messages(
    messages: Sequence[Any],
    *,
    session_id: str,
    session_index: int,
    start_seq: int = 1,
    max_chars: int | None = None,
    chunking: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize a batch of chat messages into generalized records.

    Returns ``(family, records)``.  ``adapters.adapt_messages`` recovers real
    speaker names / documents / triples from platform encodings; this function
    only splits and assigns stable ids.  ``max_chars`` forces one fixed width
    (legacy); otherwise the per-kind chunk policy applies.
    """
    cfg = dict(chunking or DEFAULT_CHUNKING)
    if max_chars:
        # A forced width is its own policy; the version must reflect that so a
        # later cache lookup cannot confuse it with the per-kind policy.
        cfg.update({"message_chars": int(max_chars), "passage_chars": int(max_chars),
                    "record_chars": int(max_chars), "overlap_chars": 0,
                    "message_tokens": 0, "passage_tokens": 0,
                    "record_tokens": 0, "overlap_tokens": 0})
    prepared = [_message_dict(message) for message in messages]
    batch = adapters.adapt_messages(prepared)
    family = batch.family
    records: list[dict[str, Any]] = []
    seq = start_seq
    for message_index, message in enumerate(batch.messages):
        original_kind = _KIND_ALIASES.get(message.kind, message.kind)
        units = content_units(
            message.content, kind=message.kind, max_chars=max_chars, chunking=cfg)
        # A split dialogue/document shares one parent (the original message);
        # every table row / triple is its own parent.
        shared_parent = len(units) > 1 and all(
            kind == "passage" for kind, _, _ in units)
        base_parent = f"p{session_index}_{message_index:05d}"
        parent_chars = len(message.content or "")
        for unit_index, (kind, text, extra) in enumerate(units):
            record_id = f"s{session_index}_{seq:05d}"
            seq += 1
            if kind in ("row", "triple"):
                parent_id = f"{base_parent}_u{unit_index:03d}"
                part, parts = 1, 1
            elif shared_parent:
                parent_id = base_parent
                part, parts = int(extra.get("part") or unit_index + 1), len(units)
            else:
                parent_id = base_parent
                part, parts = 1, 1
            record = make_record(
                record_id,
                text,
                kind=kind,
                time=normalize_time(message.timestamp),
                role=message.role,
                source=session_id,
                meta={
                    "family": family, "message": message_index,
                    "unit": unit_index,
                    **parent_metadata(
                        parent_id=parent_id, part=part, parts=parts,
                        parent_chars=parent_chars,
                        parent_kind=original_kind,
                        chunking=cfg,
                    ),
                    **extra,
                },
            )
            records.append(annotate_state(record))
    return family, records


def normalize_record(item: dict[str, Any], *, default_source: str = "",
                     record_id: str | None = None) -> dict[str, Any]:
    """Normalize one already-structured record (legacy JSONL, tables, etc.)."""
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    text = item.get("text") or item.get("content") or item.get("memory") or ""
    if not text and {"subject", "relation", "object"} <= set(item):
        text = adapters.render_triple(
            item, update=str(item.get("mutation", "")).lower() in ("replace", "update"))
    if not text:
        text = _render_dict(item)
    kind = item.get("kind") or item.get("type") or meta.get("type") or "record"
    if {"subject", "relation", "object"} <= set(item):
        kind = "triple"
    elif str(text).lower().startswith("update:") and "|" in str(text):
        kind = "triple"
    if kind in ("msg", "message"):
        kind = "message"
    elif kind in ("atom",):
        kind = "record"
    elif kind in ("summary",):
        kind = "record"
    extra = dict(item.get("meta") or {})
    if not (item.get("text") or item.get("content") or item.get("memory")):
        extra.setdefault("data", item)
    if meta:
        extra.setdefault("metadata", {k: v for k, v in meta.items() if k != "changelog"})
    source = item.get("source")
    if isinstance(source, list):
        source = source[0] if source else ""
    if not source:
        meta_source = meta.get("source")
        if isinstance(meta_source, list) and meta_source:
            source = meta_source[0]
        elif isinstance(meta_source, str):
            source = meta_source
    return annotate_state(make_record(
        record_id or str(item.get("id") or meta.get("id") or ""),
        str(text),
        kind=str(kind),
        time=normalize_time(item.get("time") or meta.get("time")),
        role=str(item.get("role") or meta.get("role") or ""),
        source=str(source or default_source),
        meta=extra,
    ))


def _expand_long_record(
    record: dict[str, Any],
    *,
    session_index: int,
    seq: int,
    max_chars: int | None = None,
    chunking: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Apply the same chunk policy to legacy/structured records.

    Short records keep their id.  Long ones become ``passage`` chunks that
    share a ``parent_id``; the retrieval layer can return the parent or the
    chunk.  Rows/triples are atomic and are never split.
    """
    cfg = dict(chunking or DEFAULT_CHUNKING)
    if max_chars:
        cfg.update({"message_chars": int(max_chars), "passage_chars": int(max_chars),
                    "record_chars": int(max_chars), "overlap_chars": 0,
                    "message_tokens": 0, "passage_tokens": 0,
                    "record_tokens": 0, "overlap_tokens": 0})
    text = str(record.get("text") or "")
    kind = str(record.get("kind") or "record")
    meta = dict(record.get("meta") or {})
    parent_id = str(record.get("id") or f"p{session_index}_{seq:05d}")
    units: list[dict[str, Any]] = []
    if max_chars or tokens_for_kind(kind, cfg) <= 0:
        limit = int(max_chars) if max_chars else chars_for_kind(kind, cfg)
        overlap = 0 if max_chars else max(0, int(cfg.get("overlap_chars", 0)))
        if limit <= 0 or len(text) <= limit:
            units = [{"text": text, "char_start": 0, "char_end": len(text),
                      "token_start": 0, "token_end": token_count(text),
                      "tokens": token_count(text)}]
        else:
            pieces = chunk_text(text, max_chars=limit, overlap_chars=overlap)
            cursor = 0
            for piece in pieces:
                start = text.find(piece, cursor)
                if start < 0:
                    start = cursor
                cursor = max(cursor, start + len(piece))
                units.append({"text": piece, "char_start": start,
                              "char_end": start + len(piece),
                              "token_start": 0, "token_end": token_count(piece),
                              "tokens": token_count(piece)})
    else:
        units = chunk_units_tokens(
            text, tokens_for_kind(kind, cfg),
            overlap_tokens=int(cfg.get("overlap_tokens", 40)),
            min_tokens=int(cfg.get("min_tokens", 20)))
    if not units:
        return [], seq
    if len(units) == 1:
        unit = units[0]
        meta.update(parent_metadata(
            parent_id=parent_id, part=1, parts=1, parent_chars=len(text),
            parent_kind=kind, chunking=cfg))
        meta.update({"char_start": unit["char_start"], "char_end": unit["char_end"],
                     "token_start": unit["token_start"], "token_end": unit["token_end"],
                     "tokens": unit["tokens"]})
        if unit["text"] != text:
            record = {**record, "text": unit["text"], "kind": "passage"}
        record["meta"] = meta
        return [annotate_state(record)], seq + 1
    out: list[dict[str, Any]] = []
    for part, unit in enumerate(units, 1):
        record_id = f"s{session_index}_{seq:05d}"
        seq += 1
        out.append(make_record(
            record_id, unit["text"], kind="passage",
            time=str(record.get("time") or ""),
            role=str(record.get("role") or ""),
            source=str(record.get("source") or ""),
            meta={
                "parent_id": parent_id, "part": part, "parts": len(units),
                "parent_chars": len(text), "parent_kind": kind,
                "chunk_version": chunk_version(cfg),
                "char_start": unit["char_start"], "char_end": unit["char_end"],
                "token_start": unit["token_start"], "token_end": unit["token_end"],
                "tokens": unit["tokens"],
                **{k: v for k, v in meta.items()
                   if k not in {"parent_id", "part", "parts", "parent_chars",
                                "parent_kind", "chunk_version", "char_start", "char_end",
                                "token_start", "token_end", "tokens"}},
            },
        ))
    return out, seq


def normalize_payload(
    payload: Any,
    *,
    session_id: str = "session_1",
    session_index: int = 1,
    start_seq: int = 1,
    max_chars: int | None = None,
    chunking: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize an arbitrary add payload: messages, a record, a list, text."""
    cfg = chunking or DEFAULT_CHUNKING
    if isinstance(payload, str):
        record = make_record(f"s{session_index}_{start_seq:05d}", payload,
                             kind="record", source=session_id)
        records, _ = _expand_long_record(record, session_index=session_index,
                                         seq=start_seq, max_chars=max_chars,
                                         chunking=cfg)
        return "plain", records
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return normalize_messages(payload["messages"], session_id=session_id,
                                  session_index=session_index,
                                  start_seq=start_seq, max_chars=max_chars,
                                  chunking=cfg)
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) and "role" in item and "content" in item
                           for item in payload):
            return normalize_messages(payload, session_id=session_id,
                                      session_index=session_index,
                                      start_seq=start_seq, max_chars=max_chars,
                                      chunking=cfg)
        records = []
        seq = start_seq
        for item in payload:
            if isinstance(item, dict):
                meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                record_id = (item.get("id") or meta.get("id")
                             or f"s{session_index}_{seq:05d}")
                record = normalize_record(item, default_source=session_id,
                                          record_id=str(record_id))
            else:
                record = make_record(f"s{session_index}_{seq:05d}", str(item),
                                     kind="record", source=session_id)
            expanded, seq = _expand_long_record(
                record, session_index=session_index, seq=seq,
                max_chars=max_chars, chunking=cfg)
            records.extend(expanded)
        return "records", records
    if isinstance(payload, dict):
        if "role" in payload and "content" in payload:
            return normalize_messages([payload], session_id=session_id,
                                      session_index=session_index,
                                      start_seq=start_seq, max_chars=max_chars,
                                      chunking=cfg)
        if any(isinstance(payload.get(key), list)
               for key in ("rows", "data", "records", "items")):
            units = _json_units(json.dumps(payload, ensure_ascii=False)) or []
            records = []
            seq = start_seq
            for unit_index, (kind, text, extra) in enumerate(units):
                record = make_record(
                    f"s{session_index}_{seq:05d}", text,
                    kind=kind, source=session_id, meta=extra)
                expanded, seq = _expand_long_record(
                    record, session_index=session_index, seq=seq,
                    max_chars=max_chars, chunking=cfg)
                records.extend(expanded)
            if records:
                return "table", records
        record = normalize_record(payload, default_source=session_id,
                                  record_id=f"s{session_index}_{start_seq:05d}")
        records, _ = _expand_long_record(record, session_index=session_index,
                                         seq=start_seq, max_chars=max_chars,
                                         chunking=cfg)
        return "record", records
    record = make_record(f"s{session_index}_{start_seq:05d}", str(payload),
                         kind="record", source=session_id)
    records, _ = _expand_long_record(record, session_index=session_index,
                                     seq=start_seq, max_chars=max_chars,
                                     chunking=cfg)
    return "plain", records
