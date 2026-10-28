"""Evidence post-processing after the bash agent.

Responsibilities:

- reject fabricated source ids (except ``["query"]``);
- deduplicate and merge sources;
- choose the final unit: chunk / parent / local window (per record);
- preserve the agent's self-contained statement while adding parent context
  when the record was split at Add time;
- carry source observation time for downstream answer prompts;
- cap and sort evidence.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Sequence

from ..timeexpr import resolve_relative_dates
from .parents import group_by_parent, parent_key, resolve_record
from .tokenize import normalize_text, tokenize


def _meta(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata")
    return value if isinstance(value, dict) else {}


def _sources(item: dict[str, Any]) -> list[str]:
    raw = _meta(item).get("source")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if str(value).strip()]


def _content(item: dict[str, Any]) -> str:
    value = item.get("content")
    if not isinstance(value, str):
        value = item.get("memory")
    return value.strip() if isinstance(value, str) else ""


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _merge_context(agent_text: str, resolved: str, limit: int) -> str:
    agent_text = (agent_text or "").strip()
    resolved = (resolved or "").strip()
    if not agent_text:
        return _truncate(resolved, limit)
    if not resolved:
        return _truncate(agent_text, limit)
    # If the agent already copied the chunk verbatim, do not duplicate it.
    if normalize_text(agent_text) in normalize_text(resolved):
        return _truncate(resolved, limit)
    return _truncate(f"{agent_text}\n\n{resolved}", limit)


def _score(item: dict[str, Any]) -> float:
    try:
        return float(_meta(item).get("score", 0.5))
    except (TypeError, ValueError):
        return 0.5


def neighbor_records(
    anchor: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    window: int = 1,
) -> list[dict[str, Any]]:
    """Same-parent / same-source neighbours around ``anchor``.

    A split parent expands inside its own parts.  A standalone record expands
    along its session, ordered by ``(time, parent, part)`` — never by arrival
    order.  Table rows and triples stay atomic and are not expanded.
    """
    window = max(0, int(window))
    if window <= 0:
        return []
    meta = _meta(anchor)
    kind = str(anchor.get("kind") or "")
    out: list[dict[str, Any]] = []
    if int(meta.get("parts") or 1) > 1:
        key = parent_key(anchor)
        siblings = [record for record in records if parent_key(record) == key]
        siblings.sort(key=lambda record: int(_meta(record).get("part") or 1))
        anchor_id = str(anchor.get("id") or "")
        try:
            position = next(index for index, record in enumerate(siblings)
                            if str(record.get("id") or "") == anchor_id)
        except StopIteration:
            return []
        low, high = max(0, position - window), min(len(siblings), position + window + 1)
        out = siblings[low:high]
    elif kind not in {"row", "triple"}:
        source = str(anchor.get("source") or "")
        if not source:
            return []
        siblings = [record for record in records
                    if str(record.get("source") or "") == source]
        siblings.sort(key=lambda record: (
            str(record.get("time") or ""),
            parent_key(record),
            int(_meta(record).get("part") or 1),
            str(record.get("id") or ""),
        ))
        anchor_id = str(anchor.get("id") or "")
        try:
            position = next(index for index, record in enumerate(siblings)
                            if str(record.get("id") or "") == anchor_id)
        except StopIteration:
            return []
        low, high = max(0, position - window), min(len(siblings), position + window + 1)
        out = siblings[low:high]
    anchor_id = str(anchor.get("id") or "")
    return [record for record in out if str(record.get("id") or "") != anchor_id]


def expand_with_neighbors(
    item: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    window: int = 1,
    max_chars: int = 800,
) -> dict[str, Any]:
    """Attach the anchor's immediate neighbours to the evidence content.

    ``metadata.source`` keeps only the cited anchors; the appended records are
    listed in ``metadata.context_ids`` and are explicitly marked as context in
    the text.  This is the mandatory neighbourhood pass: an isolated sentence
    ("she moved there") is much less useful than the same sentence with the
    adjacent turns.
    """
    metadata = dict(_meta(item))
    sources = _sources(item)
    anchors = [source for source in sources if source != "query"]
    if not anchors or window <= 0:
        return item
    by_id = {str(record.get("id") or ""): record for record in records}
    anchor = next((by_id[source] for source in anchors if source in by_id), None)
    if anchor is None:
        return item
    neighbours = neighbor_records(anchor, records, window=window)
    if not neighbours:
        metadata.setdefault("anchor_id", str(anchor.get("id") or ""))
        metadata.setdefault("context_ids", [])
        return {**item, "metadata": metadata}
    budget = max(0, int(max_chars))
    chunks: list[str] = []
    context_ids: list[str] = []
    used = 0
    for record in neighbours:
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        allowance = budget - used
        if allowance <= 0:
            break
        if len(text) > allowance:
            text = text[:allowance].rstrip() + " ..."
        record_id = str(record.get("id") or "")
        chunks.append(f"[context {record_id}] {text}")
        context_ids.append(record_id)
        used += len(text) + 16
    if not chunks:
        return {**item, "metadata": metadata}
    content = str(item.get("content") or "").rstrip()
    merged = f"{content}\n\n" + "\n".join(chunks)
    metadata.update({
        "anchor_id": str(anchor.get("id") or ""),
        "context_ids": context_ids,
        "neighbor_window": window,
        "expanded": True,
    })
    return {"content": merged, "metadata": metadata}


def annotate_resolved_dates(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append deterministic relative-date resolutions to evidence content.

    The answer model only receives ``content`` (the API drops metadata), so a
    record that says "yesterday" must carry its resolved date in the text.
    The verifier and the answer then see the same absolute anchor.
    """
    for item in items:
        metadata = dict(_meta(item))
        observed = str(metadata.get("observed_at") or metadata.get("time") or "")
        content = _content(item)
        if not content or not observed or "[resolved:" in content:
            continue
        resolved = resolve_relative_dates(content, observed)
        if not resolved:
            continue
        rendered = "; ".join(f"{key} = {value}" for key, value in resolved.items())
        # Prepend, not append: per-item truncation keeps the head, and the
        # answer model must not lose the absolute anchor.
        item["content"] = f"[resolved: {rendered}]\n{content.lstrip()}"
        metadata["resolved_dates"] = resolved
        item["metadata"] = metadata
    return list(items)


def _neighbor_settings(config: dict[str, Any] | None) -> tuple[int, int]:
    config = config or {}
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    coarse = retrieval.get("coarse") if isinstance(retrieval.get("coarse"), dict) else {}
    window = int(config.get("evidence_neighbor_window",
                             coarse.get("neighbor_window", 1)))
    budget = int(config.get("neighbor_max_chars",
                            coarse.get("neighbor_max_chars", 600)))
    return max(0, window), max(0, budget)


def postprocess_evidence(
    evidence: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    question: str,
    route: str,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(evidence, report)`` ready for the API / bench."""
    config = config or {}
    max_chars = int(config.get("max_evidence_chars", 6000))
    max_items = int(config.get("max_evidence", 64))
    by_id = {str(record.get("id") or ""): record for record in records}
    by_parent = group_by_parent(records)
    merged: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    rejected: list[dict[str, Any]] = []
    report = {"in": len(evidence), "out": 0, "rejected": 0, "merged": 0,
              "resolutions": {}, "reject_reasons": {}}

    for item in evidence:
        content = _content(item)
        if not content:
            rejected.append({"reason": "missing content"})
            continue
        sources = _sources(item)
        if not sources:
            rejected.append({"reason": "missing metadata.source"})
            continue
        valid = [source for source in sources
                 if source == "query" or source in by_id or source in by_parent]
        invalid = [source for source in sources if source not in valid]
        if invalid:
            rejected.append({"reason": "unknown_source", "sources": invalid})
            continue
        metadata = dict(_meta(item))
        metadata["source"] = valid
        metadata["score"] = _score(item)
        # Everything reaching this function is agent-written (the seeded coarse
        # records are kept out of it by the runner).  Mark verbatim copies so
        # the final selector does not give them claim priority.
        metadata.setdefault("origin", "postagent")
        if "claim_type" not in metadata:
            copied = (len(valid) == 1 and valid[0] in by_id
                      and _is_copy(content, str(by_id[valid[0]].get("text") or "")))
            metadata["claim_type"] = "copy" if copied else "claim"
        resolved_text = content
        if len(valid) == 1 and valid[0] != "query":
            record = by_id.get(valid[0])
            if record is None:
                siblings = by_parent.get(valid[0]) or []
                record = siblings[0] if siblings else None
                if record is not None:
                    metadata["source"] = [str(record.get("id") or "")]
            if record is None:
                rejected.append({"reason": "unknown_source", "sources": valid})
                continue
            override = str(metadata.get("resolve") or "").lower()
            target = record
            if override in {"chunk", "parent", "window"}:
                target = {**record, "meta": {
                    **(record.get("meta") or {}), "resolve": override}}
            resolved, mode = resolve_record(
                target, records, route=route, question=question, config=config)
            metadata["resolution"] = mode
            metadata["parent_id"] = str(
                (record.get("meta") or {}).get("parent_id") or record.get("id") or "")
            metadata["chunk_part"] = int((record.get("meta") or {}).get("part") or 1)
            metadata["chunk_parts"] = int((record.get("meta") or {}).get("parts") or 1)
            if record.get("time"):
                metadata["observed_at"] = str(record["time"])
            if record.get("role"):
                metadata["speaker"] = str(record["role"])
            if mode == "chunk":
                resolved_text = content or resolved
            else:
                resolved_text = _merge_context(content, resolved, max_chars)
            report["resolutions"][mode] = report["resolutions"].get(mode, 0) + 1
        # Neighbourhood expansion is mandatory unless the record was already
        # resolved to its parent/window (which contains the neighbours).
        window, budget = _neighbor_settings(config)
        if metadata.get("resolution") not in {"parent", "window"}:
            expanded = expand_with_neighbors(
                {"content": resolved_text, "metadata": metadata}, records,
                window=window, max_chars=budget)
            resolved_text = str(expanded.get("content") or resolved_text)
            metadata = dict(expanded.get("metadata") or metadata)
            if metadata.get("context_ids"):
                report["expanded"] = report.get("expanded", 0) + 1
        key = normalize_text(resolved_text)
        if key in merged:
            old = merged[key]
            old_sources = list(dict.fromkeys(
                list(_meta(old).get("source") or []) + valid))
            _meta(old)["source"] = old_sources
            _meta(old)["score"] = max(_score(old), metadata["score"])
            report["merged"] += 1
            continue
        merged[key] = {
            "content": resolved_text,
            "metadata": metadata,
        }

    output = sorted(merged.values(), key=lambda item: -_score(item))
    output = output[:max_items]
    report["out"] = len(output)
    report["rejected"] = len(rejected)
    if rejected:
        reasons: dict[str, int] = {}
        for item in rejected:
            reason = str(item.get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        report["reject_reasons"] = reasons
    return output, report


def fallback_evidence(
    candidates: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    question: str,
    route: str,
    config: dict[str, Any] | None = None,
    limit: int = 8,
    verification: str = "coarse_fallback",
) -> list[dict[str, Any]]:
    """Use the top coarse candidates when the agent fails or writes nothing.

    The metadata carries the source observation time and speaker so the answer
    prompt can resolve relative dates and attribute first-person statements,
    exactly like agent-written evidence.
    """
    config = config or {}
    by_id = {str(record.get("id") or ""): record for record in records}
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    # Neighbour-only candidates are context, not anchors.  Expanding each
    # anchor below carries the same context without spending evidence slots.
    anchors = [item for item in candidates if not item.get("neighbor_of")]
    pool = anchors or list(candidates)
    window, neighbor_budget = _neighbor_settings(config)
    for candidate in pool:
        if len(out) >= limit:
            break
        record = by_id.get(str(candidate.get("id") or ""))
        if record is None:
            continue
        resolved, mode = resolve_record(
            record, records, route=route, question=question, config=config)
        key = normalize_text(resolved)
        if not key:
            continue
        if key in seen:
            target = out[seen[key]]
            sources = list(target["metadata"].get("source") or [])
            record_id = str(record.get("id") or "")
            if record_id and record_id not in sources:
                sources.append(record_id)
                target["metadata"]["source"] = sources
            continue
        rank = int(candidate.get("rank") or len(out) + 1)
        metadata: dict[str, Any] = {
            "source": [str(record.get("id") or "")],
            "score": 1.0 / (1.0 + rank),
            "verification": verification,
            "resolution": mode,
            "origin": "coarse",
            "claim_type": "raw",
            "parent_id": str((record.get("meta") or {}).get("parent_id")
                             or record.get("id") or ""),
        }
        if record.get("time"):
            metadata["observed_at"] = str(record["time"])
        if record.get("role"):
            metadata["speaker"] = str(record["role"])
        item: dict[str, Any] = {"content": resolved, "metadata": metadata}
        if mode not in {"parent", "window"}:
            item = expand_with_neighbors(
                item, records, window=window, max_chars=neighbor_budget)
        out.append(item)
        seen[key] = len(out) - 1
    return out


def _dedupe_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by normalized content, merging sources and scores."""
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for item in items:
        content = _content(item)
        if not content:
            continue
        key = normalize_text(content)
        if not key:
            continue
        if key in seen:
            target = out[seen[key]]
            target_meta = _meta(target)
            item_meta = _meta(item)
            sources = list(dict.fromkeys(
                [*_sources(target), *_sources(item)]))
            target_meta["source"] = sources
            target_meta["score"] = max(_score(target), _score(item))
            if len(content) > len(_content(target)):
                target["content"] = content
            continue
        copy = {"content": content, "metadata": dict(_meta(item))}
        copy["metadata"]["source"] = _sources(item)
        seen[key] = len(out)
        out.append(copy)
    return out


def fuse_evidence(
    agent_evidence: Sequence[dict[str, Any]],
    coarse_evidence: Sequence[dict[str, Any]],
    *,
    max_items: int = 64,
    coarse_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Fuse verified agent evidence with the coarse recall floor.

    The agent half is precision: self-contained, verified statements.  The
    coarse half is recall: the multi-channel RRF shortlist, which is broader
    than a vanilla dense top-k and already contains the answer record in most
    cases.  Each half gets a reserved share of the budget; unused slots are
    backfilled by the other half.  Output is interleaved so a context window
    truncated at an arbitrary point still contains both signals.
    """
    max_items = max(1, int(max_items))
    ratio = min(1.0, max(0.0, float(coarse_ratio)))
    agent = _dedupe_items(agent_evidence)
    coarse = _dedupe_items(coarse_evidence)

    # Merge coarse duplicates into the agent record instead of spending two
    # slots on the same content.
    agent_by_content = {normalize_text(_content(item)): item for item in agent}
    coarse_unique: list[dict[str, Any]] = []
    for item in coarse:
        target = agent_by_content.get(normalize_text(_content(item)))
        if target is None:
            coarse_unique.append(item)
            continue
        target_meta = _meta(target)
        target_meta["source"] = list(dict.fromkeys(
            [*_sources(target), *_sources(item)]))
        target_meta["score"] = max(_score(target), _score(item))
    coarse = coarse_unique

    coarse_slots = min(max_items, max(0, int(max_items * ratio + 0.5)))
    agent_slots = max_items - coarse_slots
    agent_n = min(len(agent), agent_slots)
    coarse_n = min(len(coarse), coarse_slots)

    remaining = max_items - agent_n - coarse_n
    if remaining > 0 and len(coarse) > coarse_n:
        take = min(remaining, len(coarse) - coarse_n)
        coarse_n += take
        remaining -= take
    if remaining > 0 and len(agent) > agent_n:
        agent_n += min(remaining, len(agent) - agent_n)

    agent_selected = agent[:agent_n]
    coarse_selected = coarse[:coarse_n]
    for item in agent_selected:
        item["metadata"]["origin"] = "postagent"
    for item in coarse_selected:
        item["metadata"]["origin"] = "coarse"

    out: list[dict[str, Any]] = []
    for index in range(max(len(agent_selected), len(coarse_selected))):
        if index < len(agent_selected):
            out.append(agent_selected[index])
        if index < len(coarse_selected):
            out.append(coarse_selected[index])
    return out[:max_items]


def _is_copy(content: str, record_text: str, *, min_chars: int = 200) -> bool:
    """True when ``content`` is a verbatim copy of one source record.

    Exact equality is checked before the length gate: LoCoMo dialogue turns
    are often shorter than 200 characters, so a verbatim record appended by
    ``append_matches.py`` used to be classified as a synthesized claim and
    then dropped for raw-only datasets.  Substring copies still need a
    minimum length so that a short claim that happens to occur inside a long
    document is not mistaken for its source.
    """
    body = normalize_text(content)
    source = normalize_text(record_text)
    if not body or not source:
        return False
    if body == source:
        return True
    return len(body) >= min_chars and body in source


def _smart_truncate(text: str, cap: int) -> str:
    """Cut at a sentence/paragraph boundary near ``cap`` instead of mid-word."""
    if cap <= 0 or len(text) <= cap:
        return text
    cut = text[:cap]
    floor = int(cap * 0.55)
    for marker in ("\n\n", "\n", ". ", "! ", "? "):
        index = cut.rfind(marker, floor)
        if index >= floor:
            return cut[: index + len(marker)].rstrip() + " ..."
    return cut.rstrip() + " ..."


def _diversity_key(item: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    """One unit for the per-parent cap.

    A short message split into a few chunks is one unit.  A long document split
    into dozens of passages is not: each passage carries different facts, so
    each is its own unit (otherwise a CL-bench document is capped at two
    chunks and the answer never sees the rest).
    """
    metadata = _meta(item)
    sources = _sources(item)
    parts = int(metadata.get("chunk_parts") or 0)
    if not parts and len(sources) == 1 and sources[0] in by_id:
        parts = int((by_id[sources[0]].get("meta") or {}).get("parts") or 0)
    if parts > 4 and len(sources) == 1:
        return sources[0]
    parent = str(metadata.get("parent_id") or "")
    if parent:
        return parent
    return sources[0] if sources else ""


def _fit(items: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]], *,
         budget: int, item_cap: int, parent_cap: int, max_items: int) -> list[dict[str, Any]]:
    """Dedupe, truncate, diversify, and fit ``items`` to the answer budget.

    The order of ``items`` is preserved.  No record is added: this only shapes
    the agent-curated evidence file, so deleting a seed is respected.
    """
    by_id = {str(record.get("id") or ""): record for record in records}
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    per_parent: dict[str, int] = {}
    used = 0
    for item in items:
        content = _content(item)
        if not content or len(out) >= max_items:
            continue
        content = _smart_truncate(content, item_cap)
        key = normalize_text(content)
        if not key:
            continue
        if key in seen:
            target = out[seen[key]]
            target_meta = _meta(target)
            target_meta["source"] = list(dict.fromkeys(
                [*_sources(target), *_sources(item)]))
            target_meta["score"] = max(_score(target), _score(item))
            continue
        metadata = _meta(item)
        is_claim = (metadata.get("origin") in {"agent", "postagent", "pipeline"}
                    and metadata.get("claim_type") == "claim")
        parent = _diversity_key(item, by_id)
        # Parent diversity protects raw seeds from one long conversation
        # filling the budget.  Claims are already capped and are the
        # computed/resolved value the pipeline exists to add; dropping one
        # because its parent also has two seeds silently loses that value.
        if parent and not is_claim and per_parent.get(parent, 0) >= parent_cap:
            continue
        length = len(content) + 80
        if used + length > budget and out:
            continue
        metadata = dict(_meta(item))
        if content != _content(item):
            metadata["truncated"] = True
        out.append({"content": content, "metadata": metadata})
        seen[key] = len(out) - 1
        if parent:
            per_parent[parent] = per_parent.get(parent, 0) + 1
        used += length
    return out


def seed_evidence(
    candidates: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    question: str,
    route: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The initial ``evidence.jsonl``: a char-budgeted coarse shortlist.

    This is the agent's starting point, not a separate recall channel.  The
    agent may keep, rewrite, or delete any of these records; whatever remains
    in the file is the final evidence.
    """
    config = config or {}
    limit = max(1, int(config.get("seed_evidence_limit", config.get("max_evidence", 64))))
    items = fallback_evidence(
        candidates, records, question=question, route=route, config=config,
        limit=limit, verification="coarse_seed")
    for item in items:
        metadata = _meta(item)
        metadata["origin"] = "coarse"
        metadata["claim_type"] = "raw"
        metadata["seeded"] = True
    return _fit(
        items, records,
        budget=int(config.get("final_evidence_chars", 15000)),
        item_cap=max(0, int(config.get("final_item_chars", 2500))),
        parent_cap=max(1, int(config.get("final_parent_cap", 2))),
        max_items=limit)


#: Intents where a synthesized claim adds something the raw record cannot:
#: a resolved contradiction, an ordered sequence, a computed total/duration,
#: or a two-sided comparison.  For the other intents, the ablation on the
#: v0.3 outputs showed the raw seeded records are better than paraphrased
#: claims (PersonaMem +0.115, LoCoMo category_2 +0.154, BEAM abstention
#: +0.289; claims still win for contradiction +0.221 and multi-session).
_KEEP_CLAIM_INTENTS = {
    "conflict", "yesno", "order", "aggregate", "duration", "compare",
    "multi_hop", "update",
}


#: Task families where the raw seeded records beat a small model's paraphrases
#: in the v0.3 ablation: conversational/personal memory (LoCoMo, PersonaMem)
#: and long-document rubric QA (CL-bench).  The caller declares the family;
#: unknown callers fall back to the intent policy.
_RAW_ONLY_DATASETS = {"locomo_refined", "personamem_32k", "clbench", "scriptmem"}


#: BEAM families where the claim ablation showed synthesis (both sides of a
#: contradiction, an ordered sequence, a comparison, or a computed interval)
#: beats the raw seeded records.  Every other BEAM family prefers raw records.
_BEAM_SYNTHESIS_TYPES = {
    "contradiction_resolution", "multi_session_reasoning",
    "event_ordering", "temporal_reasoning",
}


def claim_mode(dataset: str = "", intent: str = "", answer_type: str = "") -> str:
    """``"raw"`` (curate seeded records) or ``"synthesis"`` (write claims)."""
    family = str(dataset or "").strip()
    kind = str(answer_type or "").strip()
    if family == "beam_100k" and kind:
        return "synthesis" if kind in _BEAM_SYNTHESIS_TYPES else "raw"
    if family in _RAW_ONLY_DATASETS:
        return "raw"
    return "synthesis" if str(intent or "") in _KEEP_CLAIM_INTENTS else "raw"


def _is_seeded_item(item: dict[str, Any]) -> bool:
    metadata = _meta(item)
    return bool(
        metadata.get("seeded")
        or metadata.get("origin") == "coarse"
        or str(metadata.get("verification") or "").startswith("coarse")
    )


_ABSENCE_PREFIXES = (
    "no record establishes", "no record mentions", "no evidence",
    "the evidence does not", "not mentioned in",
)


def _is_absence_claim(item: dict[str, Any]) -> bool:
    text = _content(item).strip().lower()
    return text.startswith(_ABSENCE_PREFIXES)


def _absence_conflicts(item: dict[str, Any], raw: Sequence[dict[str, Any]],
                       question: str) -> bool:
    """An absence claim is wrong when a seeded/raw record covers the question.

    Small models sometimes append "No record establishes X" even though the
    shortlist contains a record about X.  That claim then pushes the answer
    model into a refusal.  Require at least two question content terms in one
    non-claim record before dropping the absence claim.
    """
    if not _is_absence_claim(item) or not question:
        return False
    keys = set(tokenize(question))
    if not keys:
        return False
    for record in raw:
        if len(keys & set(tokenize(_content(record)))) >= 2:
            return True
    return False


_DATE_LIKE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|(?:19|20)\d{2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
    re.IGNORECASE)
_NUMBER_LIKE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
#: Light verbs / fillers do not make a restatement into a new fact.  Without
#: this, "made clay pots" survives against a seed that says "made pots" just
#: because the tokenizer keeps "made" and "make" distinct.
_CLAIM_FILLER = frozenset({
    "made", "make", "took", "take", "got", "get", "gave", "give",
    "went", "go", "said", "say", "had", "have", "did", "do", "was",
    "were", "is", "are", "also", "just", "really", "very", "much",
    "many", "some", "any", "own", "thing", "things",
})


def _claim_adds_value(item: dict[str, Any], raw_by_source: dict[str, list[str]],
                      question: str) -> bool:
    """Whether a claim adds recall or a computed value over its cited seeds.

    The v0.3 ablation showed raw seeds beat small-model paraphrase on
    LoCoMo/PersonaMem/CL-bench.  This predicate keeps the claims that do add
    something -- a source the shortlist missed, a resolved date, a computed
    number, or a concrete detail absent from both the question and the cited
    record -- and drops the rest, which otherwise make a generic wording more
    salient than the specific seed.
    """
    sources = [source for source in _sources(item)
               if source and source != "query"]
    if not sources:
        return True
    if any(source not in raw_by_source for source in sources):
        return True
    text = _content(item)
    source_text = "\n".join(
        value for source in sources for value in raw_by_source[source])
    # A before/after relation across two records is a synthesis that no single
    # seed states, even when every content token is present in the seeds.
    if (len(sources) >= 2 and re.search(
            r"\b(before|after|earlier|later|prior|previous|next)\b",
            text, re.IGNORECASE)):
        return True
    claim_dates = {value.lower() for value in _DATE_LIKE.findall(text)}
    source_dates = {value.lower() for value in _DATE_LIKE.findall(source_text)}
    if claim_dates - source_dates:
        return True
    claim_numbers = set(_NUMBER_LIKE.findall(text))
    source_numbers = set(_NUMBER_LIKE.findall(source_text))
    if claim_numbers - source_numbers:
        return True
    question_tokens = set(tokenize(question))
    new_tokens = {
        token for token in ((set(tokenize(text)) - question_tokens)
                            - set(tokenize(source_text)))
        if token not in _CLAIM_FILLER
    }
    return bool(new_tokens)


def fit_evidence(
    items: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    intent: str = "",
    dataset: str = "",
    answer_type: str = "",
    question: str = "",
) -> list[dict[str, Any]]:
    """Final pass over the agent-curated file.

    Claims are interleaved with raw records (the answer prompt truncates),
    then everything is deduped and fit to the answer's character budget.
    Nothing is re-added: if the agent deleted a coarse record, it stays gone.

    Claims are kept only where synthesis is necessary.  For a
    preference/list/explain/temporal question the seeded records carry the
    user's own wording and are more reliable than a small model's paraphrase.
    For conversational/document families the ablation was strong enough to
    drop *paraphrases* entirely; BEAM keeps them for
    contradiction/order/aggregation.

    A claim that cites a record the coarse shortlist never retrieved is NOT a
    paraphrase: it is exactly the missing-hop evidence the postagent exists
    to produce.  Keeping it is what lets the verifier fail on the coarse
    seeds, the postagent fill the slot, and the answer see the new record
    (previously such a claim was dropped for ``_RAW_ONLY_DATASETS``, so every
    LoCoMo search ended coarse-only).
    """
    config = config or {}
    # The staged pipeline's claims are *answer-oriented extracts*, not generic
    # paraphrases: the model is told to quote the record and cite its id.  A
    # dataset-level raw-only default would silently discard that layer (the
    # exact failure that made every CL-bench/LoCoMo search end coarse-only),
    # so the pipeline can opt in explicitly.
    keep_claims = (bool(config.get("final_keep_claims", False))
                   or claim_mode(dataset, intent, answer_type) == "synthesis")
    claim_cap = max(0, int(config.get("final_claim_cap", 12)))
    claims: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for item in items:
        metadata = _meta(item)
        is_claim = (metadata.get("origin") in {"agent", "postagent"}
                    and metadata.get("claim_type") != "copy")
        if is_claim:
            claims.append(item)
        else:
            raw.append(item)

    # Sources already present as raw records.  A claim that only paraphrases
    # one of them adds no recall; a claim that introduces a new source does.
    raw_sources = {
        source
        for item in raw
        for source in _sources(item)
        if source and source != "query"
    }

    def _adds_new_source(item: dict[str, Any]) -> bool:
        sources = _sources(item)
        if any(source == "query" for source in sources):
            return True
        return any(
            source and source != "query" and source not in raw_sources
            for source in sources
        )

    if not keep_claims:
        claims = [item for item in claims
                  if ((_adds_new_source(item) or _is_absence_claim(item))
                      and not _absence_conflicts(item, raw, question))]
    else:
        # Even in synthesis mode, do not keep an absence claim contradicted by
        # a record the shortlist already contains.
        claims = [item for item in claims
                  if not _absence_conflicts(item, raw, question)]
    if claim_cap:
        claims = claims[:claim_cap]
    if claims and bool(config.get("final_drop_paraphrases", False)):
        # Keep a claim only when it adds recall or a computed value.  This is
        # the pipeline-specific version of the ablation rule above: a claim
        # that merely restates a seed ("made clay pots") can bury the seed's
        # specific detail ("a cup with a dog face") in the answer prompt.
        # Synthesis intents are exempt: a multi-hop chain or a conflict pair
        # is not a paraphrase even when the dataset family is "raw-only".
        if (claim_mode(dataset, intent, answer_type) == "raw"
                and intent.lower() not in _KEEP_CLAIM_INTENTS):
            raw_by_source: dict[str, list[str]] = {}
            for item in raw:
                for source in _sources(item):
                    if source and source != "query":
                        raw_by_source.setdefault(source, []).append(
                            _content(item))
            if raw_by_source:
                kept_claims: list[dict[str, Any]] = []
                for item in claims:
                    if _claim_adds_value(item, raw_by_source, question):
                        kept_claims.append(item)
                    else:
                        _meta(item)["dropped_reason"] = "paraphrase"
                claims = kept_claims

    # The agent's own raw records go near the front.  Seeded candidates are
    # ranked by RRF, but 20-30 of them fill the character budget; an appended
    # record at the end would be silently truncated before the answer sees
    # it.  Keep the single best seed first for context, then everything the
    # agent added, then the rest of the shortlist.
    seeded_raw = [item for item in raw if _is_seeded_item(item)]
    agent_raw = [item for item in raw if not _is_seeded_item(item)]
    agent_cap = max(0, int(config.get("final_agent_cap", 12)))
    if agent_cap:
        agent_raw = agent_raw[:agent_cap]
    raw = seeded_raw[:1] + agent_raw + seeded_raw[1:]

    claims.sort(key=lambda item: (-_score(item), len(_content(item))))
    # Start with the best coarse record, then alternate claim / raw.  A
    # claims-only prefix can bury the specific personal record the answer
    # needs (e.g. a user's actual Sunday routine behind generic advice);
    # alternating keeps both signals in the first few visible slots.
    ordered: list[dict[str, Any]] = []
    claim_index = raw_index = 0
    while claim_index < len(claims) or raw_index < len(raw):
        if raw_index < len(raw):
            ordered.append(raw[raw_index])
            raw_index += 1
        if claim_index < len(claims):
            ordered.append(claims[claim_index])
            claim_index += 1
    return _fit(
        ordered, records,
        budget=int(config.get("final_evidence_chars", 15000)),
        item_cap=max(0, int(config.get("final_item_chars", 2500))),
        parent_cap=max(1, int(config.get("final_parent_cap", 2))),
        max_items=max(1, int(config.get("max_evidence", 64))))


def select_final_evidence(
    agent_evidence: Sequence[dict[str, Any]],
    coarse_evidence: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Build the final evidence that the answer model will actually see.

    The answer prompt has a **character** budget (``answer.max_evidence_chars``,
    default 16k), while the search stage used to emit 64 item-count records.
    On BEAM/CL-bench only ~8-12 of those 64 were visible, and the agent's
    verbatim copies sat at the front.  This selector makes the search budget
    match the answer budget:

    - non-copy agent claims first (precision),
    - then the coarse shortlist by RRF rank (recall),
    - then agent records that are verbatim copies (raw fallback),
    - at most ``final_item_chars`` per record, ``final_parent_cap`` records per
      parent/source, and ``final_evidence_chars`` in total.

    A ratio of ``final_evidence_chars <= 0`` keeps the old 50/50 item fusion.
    """
    config = config or {}
    max_items = max(1, int(max_items or config.get("max_evidence", 64)))
    budget = int(config.get("final_evidence_chars", 15000))
    if budget <= 0:
        return fuse_evidence(
            agent_evidence, coarse_evidence, max_items=max_items,
            coarse_ratio=float(config.get("coarse_evidence_ratio", 0.5)))

    item_cap = max(0, int(config.get("final_item_chars", 2500)))
    parent_cap = max(1, int(config.get("final_parent_cap", 2)))
    claim_ratio = min(1.0, max(0.0, float(config.get("final_claim_ratio", 0.5))))
    claim_budget = int(budget * claim_ratio)
    by_id = {str(record.get("id") or ""): record for record in records}

    claims: list[dict[str, Any]] = []
    copies: list[dict[str, Any]] = []
    for item in agent_evidence:
        content = _content(item)
        if not content:
            continue
        sources = _sources(item)
        metadata = dict(_meta(item))
        metadata["origin"] = "postagent"
        is_copy = (len(sources) == 1 and sources[0] in by_id
                   and _is_copy(content, str(by_id[sources[0]].get("text") or "")))
        metadata["claim_type"] = "copy" if is_copy else "claim"
        normalized = {"content": content, "metadata": metadata}
        (copies if is_copy else claims).append(normalized)

    # Claims: higher score first; for equal scores prefer shorter (denser) text.
    claims.sort(key=lambda item: (-_score(item), len(_content(item))))
    copies.sort(key=lambda item: -_score(item))

    coarse: list[dict[str, Any]] = []
    for item in coarse_evidence:
        content = _content(item)
        if not content:
            continue
        metadata = dict(_meta(item))
        metadata["origin"] = "coarse"
        metadata["claim_type"] = "raw"
        coarse.append({"content": content, "metadata": metadata})
    coarse.sort(key=lambda item: -_score(item))

    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    per_parent: dict[str, int] = {}
    used = 0

    def parent_of(item: dict[str, Any]) -> str:
        """Diversity key for the per-parent cap.

        A short message split into a few chunks is one unit: cap its chunks.
        A long document split into dozens of passages is *not* one unit: its
        chunks carry different facts, so each chunk is its own unit.  Without
        this distinction, a CL-bench document (one parent, 38 parts) would be
        capped at two chunks and starve the answer of the document.
        """
        metadata = _meta(item)
        sources = _sources(item)
        parts = int(metadata.get("chunk_parts") or 0)
        if not parts and len(sources) == 1 and sources[0] in by_id:
            parts = int((by_id[sources[0]].get("meta") or {}).get("parts") or 0)
        if parts > 4 and len(sources) == 1:
            return sources[0]
        parent = str(metadata.get("parent_id") or "")
        if parent:
            return parent
        return sources[0] if sources else ""

    def add(item: dict[str, Any], *, cap: int, enforce_parent: bool = True) -> bool:
        nonlocal used
        if len(out) >= max_items:
            return False
        content = _smart_truncate(_content(item), item_cap)
        key = normalize_text(content)
        if not key:
            return False
        if key in seen:
            # Merge sources/scores into the earlier, higher-priority record.
            target = out[seen[key]]
            target_meta = _meta(target)
            target_meta["source"] = list(dict.fromkeys(
                [*_sources(target), *_sources(item)]))
            target_meta["score"] = max(_score(target), _score(item))
            return False
        parent = parent_of(item)
        if enforce_parent and parent and per_parent.get(parent, 0) >= parent_cap:
            return False
        length = len(content) + 80  # header overhead in the answer prompt
        if used + length > cap and out:
            return False
        metadata = dict(_meta(item))
        if content != _content(item):
            metadata["truncated"] = True
        out.append({"content": content, "metadata": metadata})
        seen[key] = len(out) - 1
        if parent:
            per_parent[parent] = per_parent.get(parent, 0) + 1
        used += length
        return True

    for item in claims:
        if used >= claim_budget:
            break
        add(item, cap=claim_budget)
    for item in coarse:
        if used >= budget:
            break
        add(item, cap=budget)
    for item in copies:
        if used >= budget:
            break
        add(item, cap=budget)
    # Backfill any claim that did not fit the reserved claim share.
    if used < budget:
        for item in claims:
            if used >= budget:
                break
            add(item, cap=budget)

    return out
