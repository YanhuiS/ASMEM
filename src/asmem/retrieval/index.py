"""Per-user retrieval index: BM25 + optional dense vectors + exact/time/state.

The index is built from a memory snapshot.  Dense vectors are cached on disk by
:class:`asmem.retrieval.dense.EmbeddingCache`; adding records only embeds the
new texts.  If no encoder is configured or the endpoint fails, the index is
still usable through BM25/exact/time/state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from operator import mul
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..io import now_iso, read_jsonl
from ..schema import (
    chunk_text, chunk_text_tokens, chunk_version, chunking_config,
)
from ..state import derive_state
from .bm25 import BM25Index
from .dense import EmbeddingCache, build_embedder, cache_lock
from .tokenize import normalize_text, terms, tokenize

_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time_expr(text: str, *, now: datetime | None = None) -> dict[str, str]:
    """Parse a small set of deterministic date expressions.

    Returns ``{"start": iso, "end": iso, "kind": ...}`` or ``{}``.  Dates are
    UTC; an unparseable expression yields no filter (never a silent wrong one).
    """
    raw = (text or "").strip()
    if not raw:
        return {}
    lowered = raw.lower()
    padded = f" {lowered} "
    now = now or datetime.now(timezone.utc)
    kind = "range"
    if " as of " in padded:
        kind = "as_of"
    if " before " in padded:
        kind = "before"
    if " after " in padded:
        kind = "after"

    match = _DATE_RE.search(raw)
    if match:
        year, month, day = (int(item) for item in match.groups())
        try:
            start = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return {}
        if kind == "before":
            return {"start": "", "end": _iso(start - timedelta(seconds=1)), "kind": kind}
        if kind == "after":
            return {"start": _iso(start + timedelta(days=1)), "end": "", "kind": kind}
        return {"start": _iso(start), "end": _iso(start + timedelta(days=1) - timedelta(seconds=1)),
                "kind": kind}

    month_match = _MONTH_RE.search(raw)
    year_match = _YEAR_RE.search(raw)
    if month_match:
        month = _MONTHS[month_match.group(1).lower()]
        year = int(year_match.group(1)) if year_match else now.year
        try:
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = (datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
                   - timedelta(seconds=1))
        except ValueError:
            return {}
        return {"start": _iso(start), "end": _iso(end), "kind": "month"}
    if year_match:
        year = int(year_match.group(1))
        return {"start": _iso(datetime(year, 1, 1, tzinfo=timezone.utc)),
                "end": _iso(datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)),
                "kind": "year"}

    if "yesterday" in lowered:
        day = now - timedelta(days=1)
        return {"start": _iso(day.replace(hour=0, minute=0, second=0, microsecond=0)),
                "end": _iso(day.replace(hour=23, minute=59, second=59)), "kind": "day"}
    if "today" in lowered:
        return {"start": _iso(now.replace(hour=0, minute=0, second=0, microsecond=0)),
                "end": _iso(now.replace(hour=23, minute=59, second=59)), "kind": "day"}
    if "tomorrow" in lowered:
        day = now + timedelta(days=1)
        return {"start": _iso(day.replace(hour=0, minute=0, second=0, microsecond=0)),
                "end": _iso(day.replace(hour=23, minute=59, second=59)), "kind": "day"}
    if "last week" in lowered or "this week" in lowered or "next week" in lowered:
        monday = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        if "last week" in lowered:
            start, end = monday - timedelta(days=7), monday - timedelta(seconds=1)
        elif "next week" in lowered:
            start, end = monday + timedelta(days=7), monday + timedelta(days=14) - timedelta(seconds=1)
        else:
            start, end = monday, monday + timedelta(days=7) - timedelta(seconds=1)
        return {"start": _iso(start), "end": _iso(end), "kind": "week"}
    if ("last month" in lowered or "this month" in lowered
            or "next month" in lowered):
        year, month = now.year, now.month
        if "last month" in lowered:
            year, month = (year - 1, 12) if month == 1 else (year, month - 1)
        elif "next month" in lowered:
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = (datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
               - timedelta(seconds=1))
        return {"start": _iso(start), "end": _iso(end), "kind": "month"}
    if ("last year" in lowered or "this year" in lowered
            or "next year" in lowered):
        year = now.year + (-1 if "last year" in lowered else 1 if "next year" in lowered else 0)
        return {"start": _iso(datetime(year, 1, 1, tzinfo=timezone.utc)),
                "end": _iso(datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)),
                "kind": "year"}
    relative = re.search(
        r"\b(?:(\d+)\s+)?(day|week|month|year)s?\s+ago\b", lowered)
    if relative:
        count = int(relative.group(1) or 1)
        unit = relative.group(2)
        days = count * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        day = now - timedelta(days=days)
        if unit == "week":
            monday = (day - timedelta(days=day.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0)
            return {"start": _iso(monday),
                    "end": _iso(monday + timedelta(days=7) - timedelta(seconds=1)),
                    "kind": "relative"}
        if unit == "month":
            start = datetime(day.year, day.month, 1, tzinfo=timezone.utc)
            end = (datetime(day.year + (day.month == 12), (day.month % 12) + 1, 1,
                           tzinfo=timezone.utc) - timedelta(seconds=1))
            return {"start": _iso(start), "end": _iso(end), "kind": "relative"}
        if unit == "year":
            start = datetime(day.year, 1, 1, tzinfo=timezone.utc)
            end = datetime(day.year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
            return {"start": _iso(start), "end": _iso(end), "kind": "relative"}
        return {"start": _iso(day.replace(hour=0, minute=0, second=0, microsecond=0)),
                "end": _iso(day.replace(hour=23, minute=59, second=59)), "kind": "relative"}
    return {}


def _within(value: str, start: str, end: str) -> bool:
    if not value:
        return False
    if start and value < start:
        return False
    if end and value > end:
        return False
    return True


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("meta")
    return value if isinstance(value, dict) else {}


@dataclass
class RetrievalIndex:
    records: list[dict[str, Any]]
    bm25: BM25Index
    vectors: list[list[float]] | None = None
    #: One entry per record; a long record may be encoded as several windows.
    #: ``vectors`` keeps the first window for backward compatibility.
    vector_groups: list[list[list[float]]] | None = None
    #: Derived state entries (``state.jsonl`` view).
    state: list[dict[str, Any]] = field(default_factory=list)
    dim: int = 0
    embedder: Any = None
    #: Provider-reported embedding usage for this index's document/query
    #: encodings.  Accumulated per call, not by diffing the shared client, so
    #: concurrent searches cannot misattribute each other's tokens.
    embedding_usage_total: dict[str, int] = field(default_factory=dict)
    embedder_identity: str = ""
    dense_error: str = ""
    manifest_path: Path | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def dense_enabled(self) -> bool:
        return bool(self.vector_groups or self.vectors) and self.embedder is not None

    def embedding_usage(self) -> dict[str, int]:
        """Provider-reported embedding tokens/requests for this index."""
        return dict(self.embedding_usage_total)

    async def aclose(self) -> None:
        close = getattr(self.embedder, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass

    def exact_rank(self, phrases: Sequence[str], *, limit: int | None = None
                   ) -> list[tuple[int, float]]:
        if not phrases:
            return []
        scored: list[tuple[int, float]] = []
        for index, record in enumerate(self.records):
            text = normalize_text(str(record.get("text") or ""))
            score = 0.0
            for phrase in phrases:
                needle = normalize_text(phrase)
                if needle and needle in text:
                    score += 1.0 + math.log(1.0 + len(needle))
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit] if limit else scored

    def time_rank(self, time_expr: dict[str, str], *, limit: int | None = None
                  ) -> list[tuple[int, float]]:
        start = time_expr.get("start", "")
        end = time_expr.get("end", "")
        if not start and not end:
            return []
        scored: list[tuple[int, float]] = []
        for index, record in enumerate(self.records):
            meta = _meta(record)
            score = 0.0
            if _within(str(record.get("time") or ""), start, end):
                score = 1.0
            else:
                valid_from = str(meta.get("valid_from") or "")
                valid_to = str(meta.get("valid_to") or "")
                if valid_from and (not valid_to or valid_to >= start):
                    if not end or valid_from <= end:
                        score = 0.8
            if score:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit] if limit else scored

    def state_rank(self, query_terms: Iterable[str], *, limit: int | None = None
                   ) -> list[tuple[int, float]]:
        """Rank state-bearing records; current values beat superseded ones."""
        if not self.state:
            return []
        wanted = set(query_terms)
        by_id = {str(record.get("id") or ""): index
                 for index, record in enumerate(self.records)}
        scores: dict[int, float] = {}
        for entry in self.state:
            index = by_id.get(str(entry.get("source") or ""))
            if index is None:
                continue
            record = self.records[index]
            text = (f"{entry.get('state_key') or ''} "
                    f"{entry.get('value') or ''} {record.get('text') or ''}")
            overlap = len(wanted & terms(text))
            if not overlap:
                continue
            score = 1.0 + math.log(1.0 + overlap)
            if str(entry.get("status") or "active") != "active":
                score *= 0.3
            if entry.get("conflict"):
                score *= 0.9
            scores[index] = max(scores.get(index, 0.0), score)
        scored = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return scored[:limit] if limit else scored

    async def dense_rank(
        self,
        query: str,
        *,
        limit: int | None = None,
        options: Sequence[str] | None = None,
        option_weight: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Dense ranking with an optional MCQ option view.

        ``score = base + w * max(0, best_option - base)`` so a candidate answer
        can only raise a record's score, never lower it.
        """
        if not self.dense_enabled:
            return []
        views = [query] + [str(item) for item in (options or []) if str(item).strip()]
        usage: dict[str, int] = {}
        try:
            query_vectors = await self.embedder.embed(
                views, kind="query", usage_out=usage)
        except Exception as exc:  # noqa: BLE001 - dense is optional
            self.dense_error = f"{type(exc).__name__}: {exc}"
            return []
        for key, value in usage.items():
            self.embedding_usage_total[key] = (
                self.embedding_usage_total.get(key, 0) + value)
        base_vector = query_vectors[0]
        option_vectors = query_vectors[1:]
        weight = max(0.0, float(option_weight))
        groups = self.vector_groups or [[vector] for vector in (self.vectors or [])]
        scored: list[tuple[int, float]] = []
        for index, group in enumerate(groups):
            best = None
            for vector in group:
                if not vector:
                    continue
                base = sum(map(mul, base_vector, vector))
                score = base
                if option_vectors and weight:
                    option_best = max(
                        sum(map(mul, view, vector))
                        for view in option_vectors)
                    score = base + weight * max(0.0, option_best - base)
                best = score if best is None else max(best, score)
            if best is not None:
                scored.append((index, best))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit] if limit else scored


def _embed_text(record: dict[str, Any], config: dict[str, Any]) -> str:
    text = str(record.get("text") or "")
    if bool(config.get("include_role")) and record.get("role"):
        return f"{record.get('role')}: {text}"
    return text


async def build_index(
    memory_path: Path,
    config: dict[str, Any] | None = None,
    *,
    cache_dir: Path | None = None,
) -> RetrievalIndex:
    config = config or {}
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    bm25_cfg = retrieval.get("bm25") if isinstance(retrieval.get("bm25"), dict) else {}
    dense_cfg = retrieval.get("dense") if isinstance(retrieval.get("dense"), dict) else {}
    records = read_jsonl(memory_path)
    index = RetrievalIndex(
        records=records,
        bm25=BM25Index(
            [str(record.get("text") or "") for record in records],
            k1=float(bm25_cfg.get("k1", 1.2)),
            b=float(bm25_cfg.get("b", 0.75)),
        ),
        state=derive_state(records),
        stats={"records": len(records)},
    )
    if not records:
        return index

    embedder = build_embedder(dense_cfg)
    if embedder is None:
        index.dense_error = "dense disabled or no encoder configured"
        return index
    index.embedder = embedder
    index.embedding_usage_total = {}
    chunking = chunking_config(config)
    dense_max = int(dense_cfg.get("max_chars", 0) or 0)
    dense_overlap = max(0, int(dense_cfg.get("overlap_chars", 0) or 0))
    dense_tokens = int(dense_cfg.get("max_tokens", 320) or 0)
    dense_token_overlap = max(0, int(dense_cfg.get("overlap_tokens", 40) or 0))
    # The cache identity covers model, Add-time chunking, and the encoding
    # window.  Vectors computed over a different unit must not be reused.
    index.embedder_identity = (
        f"{getattr(embedder, 'identity', 'unknown')}"
        f"|chunk={chunk_version(chunking)}"
        f"|dense={dense_tokens}:{dense_token_overlap}:{dense_max}:{dense_overlap}")
    base_root = Path(cache_dir) if cache_dir else memory_path.parent / "index"
    identity_slug = hashlib.sha1(
        index.embedder_identity.encode("utf-8")).hexdigest()[:12]
    root = base_root / identity_slug

    # Encoding units: a record longer than ``dense.max_chars`` is encoded in
    # overlapping windows and scored by its best window.  This keeps the
    # embedding cache chunk-level and prevents a long record from being
    # truncated by the model's context limit.
    units: list[tuple[int, str]] = []
    for position, record in enumerate(records):
        text = _embed_text(record, dense_cfg)
        if dense_tokens > 0:
            pieces = chunk_text_tokens(
                text, dense_tokens, overlap_tokens=dense_token_overlap)
        elif dense_max > 0 and len(text) > dense_max:
            pieces = chunk_text(text, max_chars=dense_max, overlap_chars=dense_overlap)
        else:
            pieces = [text]
        units.extend((position, piece) for piece in pieces)
    texts = [text for _, text in units]

    def load_cache() -> tuple[
            list[list[float]], list[tuple[int, str]], dict[str, int], EmbeddingCache]:
        cache = EmbeddingCache(root, index.embedder_identity)
        vectors: list[list[float]] = []
        missing: list[tuple[int, str]] = []
        for position, text in enumerate(texts):
            vector = cache.get(text)
            if vector is None:
                vectors.append([])
                missing.append((position, text))
            else:
                vectors.append(vector)
        stats = {"cache_hits": cache.hits, "cache_misses": cache.misses,
                 "embedded": len(missing), "windows": len(texts)}
        return vectors, missing, stats, cache

    stats: dict[str, int] = {}
    try:
        async with cache_lock(root):
            vectors, missing, stats, cache = load_cache()
            try:
                if missing:
                    embedded = await embedder.embed(
                        [text for _, text in missing], kind="document",
                        usage_out=index.embedding_usage_total)
                    if len(embedded) != len(missing):
                        raise RuntimeError(
                            f"embedder returned {len(embedded)} rows for "
                            f"{len(missing)} texts")
                    for (position, text), vector in zip(missing, embedded):
                        vectors[position] = vector
                        cache.put(text, vector)
                    await asyncio.to_thread(cache.save)
            finally:
                cache.close()
    except Exception as exc:  # noqa: BLE001 - dense is optional
        index.dense_error = f"{type(exc).__name__}: {exc}"
        index.vectors = None
        index.vector_groups = None
        index.stats.update(stats)
        return index

    index.stats.update(stats)
    groups: list[list[list[float]]] = [[] for _ in records]
    for (position, _), vector in zip(units, vectors):
        if vector:
            groups[position].append(vector)
    if any(not group for group in groups):
        index.dense_error = "some embeddings are missing"
        index.vectors = None
        index.vector_groups = None
        return index
    index.vector_groups = groups
    index.vectors = [group[0] for group in groups]
    index.dim = len(index.vectors[0]) if index.vectors else 0
    manifest = {
        "built_at": now_iso(),
        "records": len(records),
        "dense": True,
        "identity": index.embedder_identity,
        "dim": index.dim,
        "chunking": chunking,
        "dense_windows": {"max_tokens": dense_tokens,
                          "overlap_tokens": dense_token_overlap,
                          "max_chars": dense_max, "overlap_chars": dense_overlap,
                          "units": len(units)},
        "embedding": stats,
        "bm25": {"k1": float(bm25_cfg.get("k1", 1.2)),
                 "b": float(bm25_cfg.get("b", 0.75))},
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        index.manifest_path = root / "manifest.json"
        fd, temp = tempfile.mkstemp(dir=str(root), prefix=".manifest.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, index.manifest_path)
    except OSError:
        pass
    return index
