"""Answer-oriented evidence compiler (postagent v2).

The v0.7 postagent was an open-ended bash search loop: 92% of its searches
re-ran the same coarse retriever, 62% of queries ended with no agent-written
evidence, and the whole stage was accuracy-neutral against simply showing the
answer model the coarse shortlist.  The diagnosis (docs/04-postagent.md) is
that recall is not the bottleneck (gold is already in the coarse candidates
for ~90% of queries); the bottleneck is *selection and organisation*:

- list/count questions need every matching item, not the first one;
- a fact that appears only in a ``[context <id>]`` neighbour must be promoted
  to a first-class source or the answer prompt treats it as non-evidence;
- yes/no and "would ... likely" questions need the premise records plus one
  supported inference, otherwise the answer model refuses;
- dates need an explicit absolute anchor and arithmetic;
- only the remaining ~10% need a real search, and that search must use a
  distinctive anchor rather than a paraphrase of the question.

This module implements that flow in four deterministic stages around one
planner call: inventory -> plan -> gap-fill -> compile.  It never deletes
coarse seeds; selected records are promoted to the front and the shortlist
stays as a recall tail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from ..retrieval.parents import resolve_record
from ..retrieval.postprocess import (
    annotate_resolved_dates,
    expand_with_neighbors,
    seed_evidence as build_seed_evidence,
)
from .verifier import _object as parse_object

_CONTEXT = re.compile(r"\[context\s+([A-Za-z0-9_]+)\]")
_NORM = re.compile(r"[^a-z0-9]+")
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{1,2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|"
    r"nov|dec)[a-z]*)",
    re.IGNORECASE,
)

PLANNER_SYSTEM = """You compile evidence for a question-answering system. You see a candidate inventory of memory records. Decide whether the answer value is present, select the records that establish it, and (only if it is absent) name the missing slot and a concrete search anchor.

Return ONLY one JSON object, no prose:
{
  "answer_shape": "entity|date|number|list|yesno|inference|multi_hop|comparison|other",
  "answer_present": true,
  "selected": [{"id": "s1_00009", "role": "answer|bridge|premise|item", "value": "verbatim span"}],
  "items": [{"value": "verbatim item", "ids": ["s1_..."]}],
  "missing": {"slot": "", "anchor": "", "channel": "grep|time|state|dense", "date_from": "", "date_to": ""},
  "inference": {"text": "", "sources": []},
  "date_notes": ""
}

RULES
- ids MUST come from the inventory. Never invent an id.
- `value` must be copied verbatim from a record; do not paraphrase.
- list/count/aggregate: scan EVERY inventory record and return every distinct answer item in `items` (and its records in `selected`). Do not stop at the first item.
- date: select the record that states the event; put the absolute date (from [resolved]/observed_at) and the arithmetic in `date_notes`.
- yesno / inference ("would ... likely"): select the premise records and write ONE inference that follows directly from them. Do not add outside facts.
- multi_hop: select one record per hop with role=bridge, in chain order. If a hop is absent, set answer_present=false and name that hop in missing.slot.
- entity/number: select 1-3 short records that state the value.
- Ignore greetings, thanks, advice, plans and hypotheticals unless the question asks about them.
- If no record contains the answer, set answer_present=false and put exactly what is missing in missing.slot, with a distinctive anchor (proper noun, number, quoted phrase) for a gap search.
- Never write an absence claim. Never select the same record twice."""

PLANNER_USER = """QUESTION:
{question}

INTENT: {intent}
ANSWER TYPE: {answer_type}

CANDIDATE INVENTORY (id | time | speaker | rank | channels | text):
{inventory}

Return the JSON object now."""


@dataclass
class CompilerResult:
    evidence: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


#: Intent -> answer shape.  Shape decides the evidence budget and whether the
#: planner runs at all.  The v0.8 calibration showed the two levers are not
#: interchangeable: multi-hop/list gains come from a bigger recall tail, while
#: temporal and inference gains come from the planner's date/inference item.
SHAPE_BY_INTENT = {
    "multi_hop": "multi_hop", "aggregate": "multi_hop", "order": "multi_hop",
    "compare": "multi_hop", "summary": "multi_hop",
    "list": "list",
    "temporal": "date", "duration": "date",
    "yesno": "inference", "conflict": "inference", "update": "inference",
    "preference": "inference", "explain": "inference", "absence": "inference",
    "policy": "inference",
    "fact": "entity", "mcq": "entity", "self_contained": "entity",
}

DEFAULT_SHAPE_BUDGETS = {
    "multi_hop": 48000, "list": 48000, "number": 48000, "comparison": 48000,
    "date": 16000, "duration": 16000, "yesno": 16000, "inference": 16000,
    "entity": 24000, "other": 24000,
}

#: Shapes where one planner call pays off (date arithmetic, premise +
#: inference).  Recall shapes skip the planner and just widen the tail.
PLANNER_SHAPES = {"date", "duration", "yesno", "inference"}


_INFERENCE_CUE = re.compile(
    r"\b(?:would|could|likely|probably|might|seem|seems|expect)\b", re.IGNORECASE)
_DATE_CUE = re.compile(
    r"^\s*(?:when|what\s+(?:date|year|month|day)|how\s+long\s+ago|"
    r"how\s+many\s+(?:days|weeks|months|years))", re.IGNORECASE)


def route_shape(intent: str, answer_type: str = "",
                question: str = "") -> str:
    text = str(question or "")
    if _DATE_CUE.search(text):
        return "date"
    if _INFERENCE_CUE.search(text):
        return "inference"
    shape = SHAPE_BY_INTENT.get(str(intent or "").strip().lower())
    if shape:
        return shape
    # LoCoMo-style dataset labels as a fallback for callers that pass no intent.
    return {
        "category_1": "multi_hop", "category_2": "date",
        "category_3": "inference", "category_4": "entity",
    }.get(str(answer_type or "").strip().lower(), "other")


def _norm(text: str) -> str:
    return _NORM.sub(" ", (text or "").lower()).strip()


def _records_by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("id") or ""): record for record in records}


def build_inventory(
    candidates: Sequence[dict[str, Any]],
    seed_evidence: Sequence[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    question: str,
    preview_chars: int = 300,
    max_candidates: int = 80,
    promote_context: bool = True,
    max_chars: int = 30000,
) -> tuple[str, list[str], int]:
    """Render the planner inventory and list the promoted context ids."""
    # context id -> anchor rank, so promoted records sit next to their anchor.
    context_rank: dict[str, int] = {}
    if promote_context:
        for item in seed_evidence:
            metadata = item.get("metadata") or {}
            anchor = str(metadata.get("anchor_id") or "")
            rank = 0
            for index, candidate in enumerate(candidates):
                if str(candidate.get("id") or "") == anchor:
                    rank = index
                    break
            for cid in _CONTEXT.findall(str(item.get("content") or "")):
                context_rank.setdefault(cid, rank)
    promoted: list[str] = []
    lines: list[str] = []
    seen: set[str] = set()
    used = 0

    def add(record_id: str, rank: int, tag: str) -> None:
        nonlocal used
        if record_id in seen:
            return
        record = by_id.get(record_id)
        if record is None:
            return
        text = str(record.get("text") or "").replace("\n", " ")[:preview_chars]
        line = (f"- id={record_id} | {record.get('time') or ''} | "
                f"{record.get('role') or ''} | rank={rank} | {tag} | {text}")
        if used + len(line) + 1 > max_chars:
            return
        seen.add(record_id)
        lines.append(line)
        used += len(line) + 1

    for index, candidate in enumerate(list(candidates)[:max_candidates]):
        candidate_id = str(candidate.get("id") or "")
        record = by_id.get(candidate_id)
        if record is None:
            continue
        channels = candidate.get("channels") or {}
        top = ",".join(sorted(
            channels, key=lambda key: -float(channels.get(key) or 0))[:2])
        add(candidate_id, int(candidate.get("rank") or index + 1), top or "match")
        if promote_context:
            for cid in [key for key, value in context_rank.items()
                        if value == index]:
                if cid not in seen and cid in by_id:
                    promoted.append(cid)
                    add(cid, index + 1, "CONTEXT-ONLY")
    # context records whose anchor was not in the candidate prefix
    for cid in context_rank:
        if cid not in seen and cid in by_id:
            promoted.append(cid)
            add(cid, 0, "CONTEXT-ONLY")
    return "\n".join(lines), promoted, used


def _valid_ids(value: Any, by_id: dict[str, dict[str, Any]]) -> list[str]:
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value or []:
        record_id = str(item)
        if record_id in by_id and record_id not in out:
            out.append(record_id)
    return out


def _item_for_record(
    record: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    question: str,
    route: str,
    config: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    text, mode = resolve_record(
        record, records, route=route, question=question, config=config)
    item = {
        "content": text,
        "metadata": {
            "source": [str(record.get("id") or "")],
            "anchor_id": str(record.get("id") or ""),
            "score": 1.0,
            "origin": "compiler",
            "claim_type": "raw",
            "selection_role": role,
            "resolution": mode,
            "seeded": True,
            "parent_id": str((record.get("meta") or {}).get("parent_id")
                             or record.get("id") or ""),
            "observed_at": str(record.get("time") or ""),
            "speaker": str(record.get("role") or ""),
        },
    }
    window = int(config.get("evidence_neighbor_window", 1) or 0)
    if mode not in {"parent", "window"} and window:
        item = expand_with_neighbors(
            item, records, window=window,
            max_chars=int(config.get("neighbor_max_chars", 600) or 0))
    return item


def gapfill(
    seed_evidence: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Deterministic anchor search over the memory snapshot.

    Runs only when the planner says the answer is absent.  Every content token
    of the missing anchor (length >= 3) must appear in the record; date/number
    shapes additionally require a date/digit.  Records already visible in the
    seed tail are skipped, so this only adds genuinely new evidence.
    """
    missing = plan.get("missing") or {}
    anchor = str(missing.get("anchor") or "").strip().strip("\"'")
    if not anchor:
        return []
    shape = str(plan.get("answer_shape") or "")
    tokens = [token for token in _NORM.split(anchor.lower()) if len(token) >= 3][:4]
    if not tokens:
        return []
    visible: set[str] = set()
    for item in seed_evidence:
        metadata = item.get("metadata") or {}
        for record_id in metadata.get("source") or []:
            visible.add(str(record_id))
        for cid in _CONTEXT.findall(str(item.get("content") or "")):
            visible.add(cid)
    hits: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id") or "")
        if not record_id or record_id in visible:
            continue
        text = str(record.get("text") or "")
        lowered = text.lower()
        if not all(token in lowered for token in tokens):
            continue
        if shape in {"date", "duration"} and not (
                _DATE.search(text) or record.get("time")):
            continue
        if shape in {"number", "count"} and not re.search(r"\d", text):
            continue
        hits.append(record)
    hits.sort(key=lambda record: len(str(record.get("text") or "")))
    return hits[: max(0, limit)]


def compile_evidence(
    plan: dict[str, Any],
    seed_evidence: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    question: str,
    route: str,
    config: dict[str, Any],
    gap_records: Sequence[dict[str, Any]] = (),
    keep_tail: bool = True,
    keep_order: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn a plan + seed tail into the answer-visible evidence list."""
    window = int(config.get("evidence_neighbor_window", 1) or 0)
    max_select = max(1, int(config.get("compiler_max_select", 12)))
    max_items = max(1, int(config.get("compiler_max_items", 24)))
    stats: dict[str, Any] = {
        "shape": str(plan.get("answer_shape") or ""),
        "answer_present": bool(plan.get("answer_present")),
        "selected_ids": [],
        "gapfill_ids": [],
        "promoted_context": 0,
        "list_items": 0,
        "inference": False,
        "dropped_ids": [],
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_record(record: dict[str, Any], role: str) -> str:
        record_id = str(record.get("id") or "")
        if not record_id or record_id in seen or len(selected) >= max_select + 4:
            return ""
        seen.add(record_id)
        selected.append(_item_for_record(
            record, records, question=question, route=route,
            config=config, role=role))
        stats["selected_ids"].append(record_id)
        return record_id

    for record in gap_records:
        accepted = add_record(record, "gapfill")
        if accepted:
            stats["gapfill_ids"].append(accepted)
    for entry in plan.get("selected") or []:
        if not isinstance(entry, dict):
            continue
        record_id = str(entry.get("id") or "")
        record = by_id.get(record_id)
        if record is None:
            if record_id:
                stats["dropped_ids"].append(record_id)
            continue
        role = str(entry.get("role") or "answer")
        add_record(record, role)
    order = {"gapfill": -1, "answer": 0, "premise": 1, "bridge": 2,
             "promoted": 3, "item": 4}
    selected.sort(key=lambda item: order.get(
        str((item.get("metadata") or {}).get("selection_role") or ""), 9))
    selected = annotate_resolved_dates(selected)

    extras: list[dict[str, Any]] = []
    items = [item for item in plan.get("items") or []
             if isinstance(item, dict) and str(item.get("value") or "").strip()]
    shape = str(plan.get("answer_shape") or "")
    if items and (shape in {"list", "number", "comparison"} or len(items) >= 2):
        parts: list[str] = []
        sources: list[str] = []
        for item in items[:max_items]:
            ids = _valid_ids(item.get("ids"), by_id)
            if not ids:
                continue
            parts.append(f"{str(item.get('value')).strip()} ({', '.join(ids)})")
            for record_id in ids:
                if record_id not in sources:
                    sources.append(record_id)
        if parts:
            stats["list_items"] = len(parts)
            extras.append({
                "content": "[compiled list] " + "; ".join(parts),
                "metadata": {"source": sources, "origin": "compiler",
                             "claim_type": "list", "selection_role": "list",
                             "seeded": True, "score": 1.0},
            })
    inference = plan.get("inference") or {}
    inference_sources = _valid_ids(inference.get("sources"), by_id)
    if str(inference.get("text") or "").strip() and inference_sources:
        stats["inference"] = True
        extras.append({
            "content": "[inference] " + str(inference.get("text")).strip(),
            "metadata": {"source": inference_sources, "origin": "compiler",
                         "claim_type": "inference", "selection_role": "inference",
                         "seeded": True, "score": 1.0},
        })
    date_notes = str(plan.get("date_notes") or "").strip()
    if date_notes:
        date_sources: list[str] = []
        for item in selected:
            for sid in (item.get("metadata") or {}).get("source") or []:
                if sid not in date_sources:
                    date_sources.append(str(sid))
        if date_sources and (keep_order or not selected):
            extras.insert(0, {
                "content": f"[date] {date_notes}",
                "metadata": {"source": date_sources, "origin": "compiler",
                             "claim_type": "date", "selection_role": "date",
                             "seeded": True, "score": 1.0},
            })
        elif selected:
            selected[0]["content"] = (
                f"[date] {date_notes}\n" + str(selected[0].get("content") or ""))

    selected_anchors = set(seen)
    tail: list[dict[str, Any]] = []
    if keep_tail:
        for item in seed_evidence:
            metadata = item.get("metadata") or {}
            if str(metadata.get("origin") or "") in {"postagent", "agent"}:
                continue
            # With keep_order the tail stays intact: selected records that are
            # already seeds must remain visible in their RRF position. Only the
            # reorder path removes them (they are promoted to the front).
            if (not keep_order
                    and str(metadata.get("anchor_id") or "") in selected_anchors):
                continue
            tail.append(item)

    if keep_order:
        # Keep the coarse RRF order: promoting a wrong planner pick to the
        # front costs more than it gains (v0.8 calibration). Only records that
        # are not already in the tail -- context-promoted and gap-fill
        # records -- are added ahead of it, after the compiled extras.
        tail_ids: set[str] = set()
        for item in seed_evidence:
            metadata = item.get("metadata") or {}
            for rid in metadata.get("source") or []:
                tail_ids.add(str(rid))
            for cid in _CONTEXT.findall(str(item.get("content") or "")):
                tail_ids.add(cid)
        selected_new = [
            item for item in selected
            if str((item.get("metadata") or {}).get("anchor_id") or "")
            not in tail_ids]
        evidence = extras + selected_new + tail
        mandatory = len(extras) + len(selected_new)
    else:
        evidence = selected + extras + tail
        mandatory = len(selected) + len(extras)

    budget = int(config.get("compiler_final_chars",
                            config.get("final_evidence_chars", 15000)) or 15000)
    item_cap = int(config.get("final_item_chars", 2500) or 2500)
    mandatory = len(selected) + len(extras)
    out: list[dict[str, Any]] = []
    used = 0
    contents: set[str] = set()
    for index, item in enumerate(evidence):
        content = str(item.get("content") or "")
        if not content:
            continue
        key = _norm(content)
        if key and key in contents:
            continue
        if item_cap:
            content = content[:item_cap]
        if index >= mandatory and out and used + len(content) + 80 > budget:
            continue
        out.append({"content": content,
                    "metadata": dict(item.get("metadata") or {})})
        contents.add(key)
        used += len(content) + 80
    stats.update({"evidence": len(out), "chars": used,
                  "tail_items": max(0, len(out) - len(selected) - len(extras))})
    return out, stats


async def run_compiler(
    *,
    question: str,
    options: Sequence[str],
    intent: str,
    answer_type: str,
    route: str,
    candidates: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    seed_evidence: Sequence[dict[str, Any]],
    call_text: Callable[[list[dict[str, Any]]], Awaitable[str]],
    config: dict[str, Any],
    log: Any = None,
) -> CompilerResult:
    """Inventory -> planner -> deterministic gap-fill -> compiled evidence."""
    shape_routing = bool(config.get("compiler_shape_routing", False))
    shape = route_shape(intent, answer_type, question) if shape_routing else ""
    budgets = config.get("compiler_shape_budgets") or DEFAULT_SHAPE_BUDGETS
    try:
        budget = int(budgets.get(shape) or 0) if shape else 0
    except (TypeError, ValueError):
        budget = 0
    if not budget:
        budget = int(config.get("compiler_final_chars", 15000) or 15000)
    work_config = config
    if shape_routing:
        # Recall shapes need a wider tail, not more LLM work.  Build the seed
        # list at the shape budget so the tail is not silently capped by the
        # global final_evidence_chars.
        work_config = dict(config)
        work_config["final_evidence_chars"] = budget
        work_config["compiler_final_chars"] = budget
        seed_evidence = build_seed_evidence(
            candidates, records, question=question, route=route,
            config=work_config)
    if shape_routing and shape not in PLANNER_SHAPES:
        evidence = list(seed_evidence)
        stats = {
            "shape": shape, "routed_shape": shape, "budget": budget,
            "planner": False, "planner_parsed": False,
            "evidence": len(evidence),
            "chars": sum(len(str(item.get("content") or "")) + 80
                         for item in evidence),
            "selected_ids": [], "gapfill_ids": [],
        }
        return CompilerResult(evidence=evidence, plan={}, stats=stats)

    by_id = _records_by_id(records)
    inventory, promoted, inventory_chars = build_inventory(
        candidates, seed_evidence, by_id, question=question,
        preview_chars=int(work_config.get("compiler_preview_chars", 300)),
        max_candidates=int(work_config.get("compiler_max_candidates", 80)),
        promote_context=bool(work_config.get("compiler_promote_context", True)),
        max_chars=int(work_config.get("compiler_inventory_chars", 30000)))
    user = PLANNER_USER.format(
        question=question, intent=intent or "", answer_type=answer_type or "",
        inventory=inventory or "(no candidates)")
    if options:
        user += "\nOPTIONS: " + "; ".join(str(option) for option in options)
    plan: dict[str, Any] = {}
    try:
        text = await call_text([
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user},
        ])
        parsed = parse_object(text)
        if isinstance(parsed, dict):
            plan = parsed
    except Exception as exc:  # noqa: BLE001 - planner failure falls back to seeds
        if log is not None:
            log.warning(f"compiler planner failed: {type(exc).__name__}: {exc}")
        plan = {}
    gap_records: list[dict[str, Any]] = []
    if (not plan.get("answer_present")
            and bool(work_config.get("compiler_gapfill", True))):
        gap_records = gapfill(
            seed_evidence, records, plan,
            limit=int(work_config.get("compiler_gapfill_limit", 2)))
    keep_order = bool(work_config.get("compiler_keep_order", True))
    if shape_routing and shape in PLANNER_SHAPES:
        # A wrong planner pick must not push the date/premise tail down.
        keep_order = False
    evidence, stats = compile_evidence(
        plan, seed_evidence, records, by_id, question=question, route=route,
        config=work_config, gap_records=gap_records,
        keep_tail=bool(work_config.get("compiler_keep_tail", True)),
        keep_order=keep_order)
    stats.update({
        "shape": shape or str(plan.get("answer_shape") or ""),
        "routed_shape": shape,
        "budget": budget,
        "planner": True,
        "inventory_chars": inventory_chars,
        "promoted_context_ids": promoted[:20],
        "planner_parsed": bool(plan),
        "missing": plan.get("missing") or {},
        "gapfill_searched": not bool(plan.get("answer_present")),
    })
    return CompilerResult(evidence=evidence, plan=plan, stats=stats)


def run_fixed(
    *,
    question: str,
    intent: str,
    answer_type: str,
    route: str,
    candidates: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> CompilerResult:
    """Deterministic evidence assembly with no model calls.

    This is the stable postagent pipeline: shape routing chooses the evidence
    budget, the coarse shortlist is materialised with neighbourhood context,
    and the same RRF order is retained.  No planner, agent loop, or verifier
    can change the evidence set, so repeated runs differ only in answer output.
    """
    shape_routing = bool(config.get("compiler_shape_routing", False))
    shape = route_shape(intent, answer_type, question) if shape_routing else ""
    budgets = config.get("compiler_shape_budgets") or DEFAULT_SHAPE_BUDGETS
    try:
        budget = int(budgets.get(shape) or 0) if shape else 0
    except (TypeError, ValueError):
        budget = 0
    if not budget:
        budget = int(config.get("compiler_final_chars", 15000) or 15000)
    work_config = dict(config)
    work_config["final_evidence_chars"] = budget
    work_config["compiler_final_chars"] = budget
    seed = build_seed_evidence(
        candidates, records, question=question, route=route, config=work_config)
    evidence, stats = compile_evidence(
        {}, seed, records, _records_by_id(records), question=question,
        route=route, config=work_config, keep_tail=True, keep_order=True)
    stats.update({
        "mode": "fixed",
        "shape": shape or "other",
        "routed_shape": shape,
        "budget": budget,
        "planner": False,
        "planner_parsed": False,
        "selected_ids": [],
        "gapfill_ids": [],
        "promoted_context": 0,
    })
    return CompilerResult(evidence=evidence, plan={}, stats=stats)
