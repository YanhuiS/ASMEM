"""Deterministic coarse recall: channels -> weighted RRF -> candidates.jsonl.

This module never calls a generative model.  Dense is optional; when disabled
or unavailable, BM25 + exact + time + state + session still produce a ranked,
diverse candidate pool for the bash agent.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .index import RetrievalIndex, build_index, parse_time_expr
from .parents import choose_resolution
from .router import route_query
from .tokenize import extract_phrases, jaccard, normalize_text, terms, tokenize

DEFAULT_CHANNELS = {
    "bm25": 0.5, "dense": 0.5, "exact": 0.3, "time": 0.2,
    "state": 0.2, "session": 0.2, "option": 0.2,
}
DEFAULT_COARSE = {
    "candidates": 80,
    "session_cap_ratio": 0.3,
    "channel_top_k": 3,
    "neighbor_window": 1,
    "neighbor_ratio": 0.2,
    "neighbor_max_chars": 600,
    "dedup_jaccard": 0.9,
    "rrf_k": 60,
    "option_view_weight": 0.3,
}


@dataclass
class CoarseResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    index: RetrievalIndex | None = None


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("meta")
    return value if isinstance(value, dict) else {}


def _source(record: dict[str, Any]) -> str:
    return str(record.get("source") or "")


def _rank_to_indices(ranking: Sequence[tuple[int, float]]) -> list[int]:
    return [index for index, _ in ranking]


def _session_ranking(
    rankings: dict[str, list[tuple[int, float]]],
    records: Sequence[dict[str, Any]],
    *,
    top_m: int = 3,
) -> list[tuple[int, float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for channel in ("bm25", "dense"):
        for index, score in rankings.get(channel, ()):
            scores[_source(records[index])].append(float(score))
    if not scores:
        return []
    source_score = {
        source: sum(sorted(values)[-top_m:]) / max(1, min(top_m, len(values)))
        for source, values in scores.items()
    }
    ordered = sorted(source_score, key=lambda source: (-source_score[source], source))
    rank_of = {source: rank for rank, source in enumerate(ordered, 1)}
    out = [
        (index, 1.0 / rank_of[_source(record)])
        for index, record in enumerate(records)
        if _source(record) in rank_of
    ]
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def _option_ranking(
    index: RetrievalIndex,
    options: Sequence[str],
) -> list[tuple[int, float]]:
    if not options:
        return []
    best: dict[int, float] = {}
    for option in options:
        for doc_index, score in index.bm25.search(tokenize(option)):
            best[doc_index] = max(best.get(doc_index, 0.0), score)
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return ranked


def _why(
    doc_index: int,
    record: dict[str, Any],
    *,
    query_terms: set[str],
    phrases: Sequence[str],
    channels: dict[str, float],
    options: Sequence[str],
) -> list[str]:
    why: list[str] = []
    matched = [term for term in sorted(query_terms)[:8]
               if term in terms(str(record.get("text") or ""))]
    for term in matched[:4]:
        why.append(f"bm25:{term}")
    text = normalize_text(str(record.get("text") or ""))
    for phrase in phrases:
        if normalize_text(phrase) in text:
            why.append(f"exact:{phrase}")
            if len(why) >= 6:
                break
    if "dense" in channels:
        why.append("dense")
    if "time" in channels:
        why.append("time")
    if "state" in channels:
        why.append("state")
    if "option" in channels:
        why.append("option")
    if "session" in channels:
        why.append("session")
    if not why and options:
        why.append("option")
    return why[:8]


def _neighbors(
    records: Sequence[dict[str, Any]],
    seeds: Sequence[dict[str, Any]],
    *,
    window: int,
    budget: int,
    already: set[int],
) -> list[dict[str, Any]]:
    if window <= 0 or budget <= 0:
        return []
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_source[_source(record)].append(index)
    for indices in by_source.values():
        indices.sort(key=lambda index: (
            str(records[index].get("time") or ""),
            int(_meta(records[index]).get("part") or 1),
            index,
        ))
    out: list[dict[str, Any]] = []
    for seed in seeds:
        if len(out) >= budget:
            break
        seed_index = int(seed["_index"])
        positions = by_source[_source(records[seed_index])]
        try:
            position = positions.index(seed_index)
        except ValueError:
            continue
        for distance in range(1, window + 1):
            for neighbor in (position - distance, position + distance):
                if len(out) >= budget:
                    break
                if neighbor < 0 or neighbor >= len(positions):
                    continue
                index = positions[neighbor]
                if index in already:
                    continue
                already.add(index)
                out.append({
                    "_index": index,
                    "neighbor_of": seed["id"],
                })
    return out


def _candidate(
    rank: int,
    index: int,
    records: Sequence[dict[str, Any]],
    *,
    rrf: float,
    channels: dict[str, float],
    why: list[str],
    neighbor_of: str | None,
    route: str,
    question: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    record = records[index]
    meta = _meta(record)
    return {
        "id": str(record.get("id") or ""),
        "rank": rank,
        "rrf": round(float(rrf), 6),
        "channels": {key: round(float(value), 6) for key, value in channels.items()},
        "why": why,
        "neighbor_of": neighbor_of,
        "text": str(record.get("text") or ""),
        "kind": str(record.get("kind") or "record"),
        "time": str(record.get("time") or ""),
        "role": str(record.get("role") or ""),
        "source": str(record.get("source") or ""),
        "meta": meta,
        "parent_id": str(meta.get("parent_id") or record.get("id") or ""),
        "part": int(meta.get("part") or 1),
        "parts": int(meta.get("parts") or 1),
        "parent_chars": int(meta.get("parent_chars") or len(str(record.get("text") or ""))),
        "return_hint": choose_resolution(
            record, route=route, question=question, config=config),
    }


async def run_coarse(
    memory_path: Path,
    question: str,
    *,
    options: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> CoarseResult:
    """Build the candidate pool and optionally materialize it as JSONL."""
    config = config or {}
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    coarse_cfg = dict(DEFAULT_COARSE)
    coarse_cfg.update(retrieval.get("coarse") or {})
    channels_cfg = dict(DEFAULT_CHANNELS)
    channels_cfg.update(retrieval.get("channels") or {})
    options = [str(item) for item in (options or []) if str(item).strip()]
    question = question or ""

    route = route_query(question, options)
    query_tokens = tokenize(question)
    phrases = extract_phrases(question)
    time_expr = parse_time_expr(question)
    index = await build_index(memory_path, config, cache_dir=cache_dir)
    records = index.records
    if not records:
        await index.aclose()
        info = {"query": question, "mode": str(retrieval.get("mode") or "hybrid"),
                "route": route.name, "records": 0,
                "candidates": 0, "warnings": ["empty memory"],
                "embedding_usage": index.embedding_usage()}
        return CoarseResult([], info, [], index)

    rankings: dict[str, list[tuple[int, float]]] = {}
    if query_tokens:
        rankings["bm25"] = index.bm25.search(query_tokens)
    dense = await index.dense_rank(
        question, options=options,
        option_weight=float(coarse_cfg.get("option_view_weight", 0.3)))
    if dense:
        rankings["dense"] = dense
    exact = index.exact_rank(phrases)
    if exact:
        rankings["exact"] = exact
    timed = index.time_rank(time_expr)
    if timed:
        rankings["time"] = timed
    state = index.state_rank(query_tokens)
    if state:
        rankings["state"] = state
    if options:
        option = _option_ranking(index, options)
        if option:
            rankings["option"] = option

    mode = str(retrieval.get("mode") or "hybrid").lower()
    allowed_channels: set[str] | None = None
    if mode == "bm25":
        allowed_channels = {"bm25", "session"}
    elif mode == "dense":
        allowed_channels = {"dense", "session"}
    if allowed_channels is not None:
        rankings = {name: ranking for name, ranking in rankings.items()
                    if name in allowed_channels}

    session = _session_ranking(rankings, records)
    if session:
        rankings["session"] = session

    # Drop deleted tombstones from the candidate pool.  History questions can
    # still reach them through full-file grep, but they should not win RRF.
    allowed = [
        index_ for index_, record in enumerate(records)
        if str(_meta(record).get("status") or "") != "deleted"
    ]
    allowed_set = set(allowed)
    for channel in list(rankings):
        rankings[channel] = [
            (index_, score) for index_, score in rankings[channel]
            if index_ in allowed_set
        ]

    k = int(coarse_cfg.get("rrf_k", 60))
    fused: dict[int, float] = defaultdict(float)
    doc_channels: dict[int, dict[str, float]] = defaultdict(dict)
    for channel, ranking in rankings.items():
        weight = float(channels_cfg.get(channel, 0.0))
        if weight <= 0:
            continue
        for rank, (index_, score) in enumerate(ranking, 1):
            fused[index_] += weight / (k + rank)
            doc_channels[index_][channel] = float(score)

    ordered = sorted(fused, key=lambda index_: (
        -fused[index_],
        -max(doc_channels[index_].values() or [0.0]),
        str(records[index_].get("time") or ""),
        index_,
    ))
    limit = max(1, int(coarse_cfg.get("candidates", 80)))
    session_cap = max(1, int(limit * float(coarse_cfg.get("session_cap_ratio", 0.3))))
    if route.name in {"summary", "aggregation"}:
        # These tasks need coverage across one long source, not diversity
        # across sources; do not let the per-source cap hide the rest.
        session_cap = limit

    selected: list[int] = []
    selected_set: set[int] = set()
    per_source: Counter[str] = Counter()

    def add(index_: int) -> bool:
        if index_ in selected_set or len(selected) >= limit:
            return False
        source = _source(records[index_])
        if per_source[source] >= session_cap:
            return False
        selected.append(index_)
        selected_set.add(index_)
        per_source[source] += 1
        return True

    # Every available channel gets a small guaranteed share; this prevents a
    # single channel's blind spot from removing all of its good hits.
    channel_top = int(coarse_cfg.get("channel_top_k", 3))
    for ranking in rankings.values():
        for index_, _ in ranking[:channel_top]:
            add(index_)

    dedup_threshold = float(coarse_cfg.get("dedup_jaccard", 0.9))
    for index_ in ordered:
        if len(selected) >= limit:
            break
        text = normalize_text(str(records[index_].get("text") or ""))
        if any(jaccard(terms(text), terms(str(records[other].get("text") or "")))
               >= dedup_threshold for other in selected):
            continue
        add(index_)

    # Relax the per-source cap if there are not enough candidates.
    if len(selected) < limit:
        for index_ in ordered:
            if index_ in selected_set:
                continue
            if len(selected) >= limit:
                break
            selected.append(index_)
            selected_set.add(index_)

    # Channel quotas decide *membership*; the final order is always the fused
    # score, otherwise a channel's top-3 would silently outrank a better hit.
    selected.sort(key=lambda index_: (
        -fused.get(index_, 0.0),
        -max(doc_channels.get(index_, {}).values() or [0.0]),
        str(records[index_].get("time") or ""),
        index_,
    ))
    seeds = [
        _candidate(
            rank, index_, records,
            rrf=fused.get(index_, 0.0),
            channels=doc_channels.get(index_, {}),
            why=_why(index_, records[index_],
                     query_terms=set(query_tokens), phrases=phrases,
                     channels=doc_channels.get(index_, {}), options=options),
            neighbor_of=None,
            route=route.name, question=question, config=config,
        )
        for rank, index_ in enumerate(selected, 1)
    ]
    neighbor_budget = int(limit * float(coarse_cfg.get("neighbor_ratio", 0.2)))
    neighbor_seeds = [{"id": item["id"], "_index": index_}
                      for item, index_ in zip(seeds, selected)]
    extra = _neighbors(
        records, neighbor_seeds,
        window=int(coarse_cfg.get("neighbor_window", 1)),
        budget=neighbor_budget,
        already=set(selected),
    )
    for offset, item in enumerate(extra, len(seeds) + 1):
        index_ = int(item["_index"])
        seeds.append(_candidate(
            offset, index_, records,
            rrf=0.0, channels={}, why=["neighbor"],
            neighbor_of=str(item["neighbor_of"]),
            route=route.name, question=question, config=config,
        ))

    warnings: list[str] = []
    if index.dense_error:
        warnings.append(f"dense: {index.dense_error}")
    if not rankings:
        warnings.append("no channel produced hits")
    info = {
        "query": question,
        "options": options,
        "mode": mode,
        "route": route.name,
        "route_reasons": route.reasons,
        "forbidden": list(route.forbidden),
        "tokens": query_tokens,
        "phrases": phrases,
        "time": time_expr,
        "channels": {
            name: {
                "weight": float(channels_cfg.get(name, 0.0)),
                "hits": len(ranking),
                "top": ranking[0][1] if ranking else 0.0,
            }
            for name, ranking in rankings.items()
        },
        "records": len(records),
        "state_entries": len(index.state),
        "seeds": len(selected),
        "neighbors": len(extra),
        "candidates": len(seeds),
        "dense": index.dense_enabled,
        "dense_dim": index.dim,
        "embedding": {
            key: index.stats.get(key)
            for key in ("cache_hits", "cache_misses", "embedded")
            if key in index.stats
        },
        # Provider-reported embedding usage for this search's document and
        # query encodings (cache hits report zero).
        "embedding_usage": index.embedding_usage(),
        "warnings": warnings,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
            for item in seeds:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        (out_dir / "coarse.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return CoarseResult(seeds, info, records, index)
