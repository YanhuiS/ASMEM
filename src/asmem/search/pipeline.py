"""The unified search pipeline: coarse -> gate/seed -> agent -> postprocess -> verify.

The pipeline is deliberately linear.  ``loop`` and ``compile`` differ only in
the agent stage; every other stage is shared.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .. import llm
from ..io import (PROJECT_ROOT, atomic_write_text, read_jsonl, write_jsonl)
from ..log import Logger
from ..tables import build_tables_db
from ..retrieval import (
    CoarseResult,
    annotate_resolved_dates,
    claim_mode,
    detect_intent,
    evaluate_gate,
    fit_evidence,
    gate_evidence_limit,
    gate_verify,
    postprocess_evidence,
    run_coarse,
    seed_evidence,
)
from ..retrieval.index import build_index
from ..retrieval.postprocess import fallback_evidence
from . import compiler, evidence, evidence_loop, prompts, verifier
from .fixed_pipeline import run_pipeline
from .model import (ModelRunner, _accepts_deadline, _read_usage_log,
                    _sum_usage, _token_block)
from .postagent import run_postagent
from .tools import build_tools
from .workspace import _brief, _run_id, _shell_prefix, prepare_workspace

def _attach_source_times(
    evidence: list[dict[str, Any]],
    memory_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy a source record's observation time into evidence metadata.

    The verifier needs it to resolve relative dates ("yesterday") and the
    answer prompt uses it as an anchor; it is observation time, never an
    event-time substitute.
    """
    by_id = {str(record.get("id") or ""): record for record in memory_records}
    by_parent: dict[str, dict[str, Any]] = {}
    for record in memory_records:
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        parent = str(meta.get("parent_id") or "")
        if parent and parent not in by_parent:
            by_parent[parent] = record
    for item in evidence:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            item["metadata"] = metadata
        sources = metadata.get("source") or []
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            record = by_id.get(str(source)) or by_parent.get(str(source))
            if record:
                if record.get("time"):
                    metadata["observed_at"] = str(record["time"])
                if record.get("role"):
                    metadata["speaker"] = str(record["role"])
                break
    return evidence


def _with_verification(items: list[dict[str, Any]],
                       label: str) -> list[dict[str, Any]]:
    """Clone coarse items with a stage-specific verification label."""
    return [{
        "content": str(item.get("content") or ""),
        "metadata": {**(item.get("metadata") or {}), "verification": label},
    } for item in items if str(item.get("content") or "").strip()]


def _compact_for_verifier(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep the verifier prompt bounded even if the agent over-appended."""
    limit = max(1, int(config.get("verifier_max_evidence", 24)))
    max_chars = max(200, int(config.get("verifier_max_chars", 3000)))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        content = str(item.get("content") or item.get("memory") or "").strip()
        if not content:
            continue
        key = " ".join(content.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + " ..."
        out.append({"content": content, "metadata": item.get("metadata") or {}})
        if len(out) >= limit:
            break
    return out



def _memory_fingerprint(memory_path: Path) -> str:
    """Cheap snapshot identity: a rewrite changes size or mtime.

    ``memory.jsonl`` is atomically replaced on every add, so ``(size, mtime)``
    is enough to tell whether a cached search result is stale.  Hashing the
    whole file on every request would make search O(N) before retrieval even
    starts.
    """
    try:
        stat = memory_path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return "0:0"


def _payload_key(memory_path: Path, question: str,
                 options: list[str] | None) -> str:
    material = json.dumps(
        {"question": question, "options": list(options or []),
         "memory": _memory_fingerprint(memory_path)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _retrieval_fingerprint(config: dict[str, Any]) -> str:
    """Stable hash of the deterministic coarse configuration.

    ``_cache_dir`` is per-run and the dense API key is not behavioural, so
    both are excluded; everything that changes recall (channels, candidate
    count, chunking, embedding model) is included.
    """
    retrieval = config.get("retrieval")
    clean = json.loads(json.dumps(retrieval if isinstance(retrieval, dict) else {}))
    clean.pop("_cache_dir", None)
    # ``hybrid`` / ``coarse`` / ``agent`` select which downstream stage uses
    # the candidates; they do not change run_coarse's scoring.  Only the
    # channel-restricting modes (bm25/dense) change the candidate set.
    if str(clean.get("mode") or "hybrid").lower() in {
            "hybrid", "coarse", "agent", ""}:
        clean["mode"] = "hybrid"
    dense = clean.get("dense")
    if isinstance(dense, dict):
        dense.pop("api_key", None)
    return hashlib.sha1(
        json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_coarse_source(
    source: Path,
    snapshot: Path,
    question: str,
    options: list[str],
    config: dict[str, Any],
    inputs: Path,
) -> CoarseResult | None:
    """Load a previous run's deterministic coarse output.

    Nothing after coarse is reused: the caller still runs the gate,
    pre-audit, postagent, verifier and answer from these candidates.  The
    function refuses a source whose retrieval config differs (unless
    ``reuse_coarse_strict`` is false) or whose candidate ids are not present
    in the current memory snapshot, so a stale cache can never silently
    corrupt a run.
    """
    source = Path(source)
    candidates_path = source / "candidates.jsonl"
    info_path = source / "coarse.json"
    if not candidates_path.exists() or not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(info, dict):
        return None
    cached_version = info.get("coarse_version")
    if cached_version is not None:
        try:
            if int(cached_version) != COARSE_CACHE_VERSION:
                return None
        except (TypeError, ValueError):
            return None
    cached_query = str(info.get("query") or "").strip()
    if cached_query and cached_query != question.strip():
        # A mis-keyed cache is worse than recomputing.
        return None
    if bool(config.get("reuse_coarse_strict", True)):
        cached_config = source / "retrieval.json"
        if cached_config.exists():
            try:
                cached = json.loads(cached_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict):
                if _retrieval_fingerprint({"retrieval": cached}) != \
                        _retrieval_fingerprint(config):
                    return None
    records = read_jsonl(snapshot)
    candidates = read_jsonl(candidates_path)
    if not candidates:
        return None
    by_id = {str(record.get("id") or "") for record in records}
    present = sum(1 for item in candidates
                  if str(item.get("id") or "") in by_id)
    if present < max(1, int(len(candidates) * 0.8)):
        # The snapshot changed (different user/add order); recompute instead.
        return None
    info = dict(info)
    info["query"] = question
    info["options"] = list(options or [])
    info["coarse_version"] = COARSE_CACHE_VERSION
    info.setdefault("records", len(records))
    info["reused"] = True
    info["reused_from"] = str(source)
    # Old embedding usage belongs to the source run, not this one.
    info["embedding_usage"] = {}
    info["embedding"] = {"cache_hits": 0, "cache_misses": 0, "embedded": 0}
    # Make the new workspace self-contained for debugging and for the agent's
    # helper commands (brief.md is regenerated from these candidates below).
    try:
        (inputs / "candidates.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n"
                    for item in candidates), encoding="utf-8")
        (inputs / "coarse.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return CoarseResult(candidates=candidates, info=info, records=records,
                        index=None)


def _bm25_only_config(config: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``config`` with dense retrieval disabled.

    Used when the embedding endpoint is slow/down: BM25 + exact + time + state
    are deterministic and fast, so they can still produce a coarse shortlist
    instead of an empty result and a hard timeout.
    """
    cfg = dict(config)
    retrieval = dict(cfg.get("retrieval") or {})
    retrieval["mode"] = "bm25"
    dense = dict(retrieval.get("dense") or {})
    dense.update({"mode": "off", "model": "", "base_url": ""})
    retrieval["dense"] = dense
    cfg["retrieval"] = retrieval
    return cfg


async def _coarse_or_empty(
    memory_path: Path,
    question: str,
    options: list[str],
    config: dict[str, Any],
    inputs: Path,
    *,
    cache_dir: Path | None = None,
) -> CoarseResult:
    try:
        return await run_coarse(
            memory_path, question, options=options, config=config,
            out_dir=inputs, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - coarse failure must not kill search
        # The usual failure here is the dense endpoint (timeout/connection).
        # Retry once with BM25-only before accepting an empty candidate pool.
        try:
            fallback = await run_coarse(
                memory_path, question, options=options,
                config=_bm25_only_config(config),
                out_dir=inputs, cache_dir=cache_dir)
            fallback.info.setdefault("warnings", []).append(
                f"bm25 fallback after {type(exc).__name__}: {exc}")
            return fallback
        except Exception as inner:  # noqa: BLE001
            records = read_jsonl(memory_path)
            return CoarseResult(
                candidates=[],
                info={"query": question, "route": "dialogue",
                      "records": len(records),
                      "warnings": [
                          f"coarse failed: {type(exc).__name__}: {exc}",
                          f"bm25 fallback failed: {type(inner).__name__}: {inner}",
                      ]},
                records=records,
                index=None,
            )


async def quick_coarse_fallback(
    *,
    memory_path: Path,
    question: str,
    config: dict[str, Any],
    options: list[str] | None = None,
    cache_dir: Path | None = None,
    limit: int = 12,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """A model-free, BM25-only shortlist for the API hard-timeout path.

    This is deliberately outside ``run_search``: it must work even when every
    search slot is occupied by a hung model call.  It never calls a generative
    or embedding model, and its own timeout is small.
    """
    cfg = _bm25_only_config(config)
    try:
        coro = run_coarse(
            memory_path, question, options=list(options or []),
            config=cfg, cache_dir=cache_dir)
        coarse = (await asyncio.wait_for(coro, timeout=timeout)
                  if timeout > 0 else await coro)
    except Exception:  # noqa: BLE001 - fallback failure is still a valid []
        return []
    route = str((coarse.info or {}).get("route") or "dialogue")
    items = fallback_evidence(
        coarse.candidates, coarse.records, question=question, route=route,
        config=config, limit=max(1, int(limit)),
        verification="bm25_fallback")
    return _attach_source_times(items, coarse.records)


#: One lock per deterministic run directory.  Two HTTP retries with the same
#: ``request_id`` serialize instead of racing on evidence.jsonl; the loser
#: then hits the finished summary and returns the cached result.
_RUN_LOCKS: dict[str, asyncio.Lock] = {}


#: Bump when coarse recall logic changes without a config change (router,
#: channel scoring, RRF).  Cached candidates from another version are refused.
COARSE_CACHE_VERSION = 1


def _run_lock(name: str) -> asyncio.Lock:
    lock = _RUN_LOCKS.get(name)
    if lock is None:
        lock = asyncio.Lock()
        _RUN_LOCKS[name] = lock
    return lock


def _cached_result(qa: Path, question: str) -> dict[str, Any] | None:
    """Return a finished run for the same request, if it is still intact."""
    summary_path = qa / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(summary, dict):
        return None
    stop_reason = str(summary.get("stop_reason") or "")
    if (summary.get("degraded")
            or stop_reason.startswith(("model_error", "fallback_error", "deadline"))):
        # A degraded run is a checkpoint, not a result: the retry resumes from
        # its evidence instead of freezing a low-quality answer in the cache.
        return None
    evidence_path = qa / evidence.EVIDENCE_NAME
    items = evidence.read_file(evidence_path) if evidence_path.exists() else []
    if int(summary.get("evidence") or 0) > 0 and not items:
        # A summary without its evidence file is a torn run: rerun rather than
        # return an answer with no sources.
        return None
    summary = dict(summary)
    summary["cached"] = True
    return {"question": question, "run_dir": str(qa), "evidence": items,
            "summary": summary, "cached": True}


def _run_label(qa: Path) -> str:
    """Human-readable run id for logs (flat v2 and nested v1 layouts)."""
    return qa.parent.name if qa.name == "qa" else qa.name



async def run_question(
    *,
    memory_path: Path,
    question: str,
    config: dict[str, Any],
    run_root: Path | None = None,
    model: ModelRunner | None = None,
    log: Logger | None = None,
    dry_run: bool = False,
    options: list[str] | None = None,
    cache_dir: Path | None = None,
    shared_dir: Path | None = None,
    dataset: str = "",
    answer_type: str = "",
    request_id: str = "",
    user_id: str = "",
    coarse_source: Path | None = None,
    fallback_path: Path | None = None,
) -> dict[str, Any]:
    """Entry point used by the HTTP API.

    ``request_id`` makes the call idempotent and resumable:

    * the run directory is derived from ``(user_id, request_id, query,
      options, memory snapshot)``;
    * a finished run returns its cached evidence immediately;
    * an interrupted run keeps the evidence.jsonl written by the previous
      attempt and continues from it (the coarse seed is not re-applied);
    * concurrent duplicates wait on one lock instead of racing.

    Without ``request_id`` the behaviour is the old one: a fresh timestamped
    run every call.
    """
    normalized = [str(item) for item in (options or []) if str(item).strip()]
    if not request_id:
        return await run_search(
            memory_path=memory_path, question=question, config=config,
            run_root=run_root, model=model, log=log, dry_run=dry_run,
            options=normalized, cache_dir=cache_dir, shared_dir=shared_dir,
            dataset=dataset, answer_type=answer_type,
            coarse_source=coarse_source, fallback_path=fallback_path)
    request_key = f"{user_id}\0{request_id}"
    payload_key = _payload_key(memory_path, question, normalized)
    name = _run_id(question, request_key, payload_key)
    lock = _run_lock(name)
    try:
        async with lock:
            return await run_search(
                memory_path=memory_path, question=question, config=config,
                run_root=run_root, model=model, log=log, dry_run=dry_run,
                options=normalized, cache_dir=cache_dir, shared_dir=shared_dir,
                dataset=dataset, answer_type=answer_type, request_key=request_key,
                payload_key=payload_key, coarse_source=coarse_source,
                fallback_path=fallback_path)
    finally:
        # Keep the lock table bounded on a long-lived API process.
        if len(_RUN_LOCKS) > 1024:
            for key, item in list(_RUN_LOCKS.items()):
                if key != name and not item.locked():
                    _RUN_LOCKS.pop(key, None)


async def run_search(
    *,
    memory_path: Path,
    question: str,
    config: dict[str, Any],
    run_root: Path | None = None,
    model: ModelRunner | None = None,
    log: Logger | None = None,
    dry_run: bool = False,
    options: list[str] | None = None,
    cache_dir: Path | None = None,
    shared_dir: Path | None = None,
    dataset: str = "",
    answer_type: str = "",
    request_key: str = "",
    payload_key: str = "",
    coarse_source: Path | None = None,
    fallback_path: Path | None = None,
) -> dict[str, Any]:
    log = log or Logger.from_config(config, prefix="search")
    options = [str(item) for item in (options or []) if str(item).strip()]
    run_root = run_root or (PROJECT_ROOT / "data" / "asmem_runs")
    started = time.perf_counter()
    qa, inputs, resumed = prepare_workspace(
        memory_path, question, run_root, options, config, cache_dir,
        request_key=request_key, payload_key=payload_key,
        shared_dir=shared_dir)

    if dry_run:
        text = prompts.system_prompt() + "\n\n" + prompts.user_prompt(question, options)
        log.info(text)
        return {"question": question, "run_dir": str(qa), "prompt": text,
                "evidence": [], "summary": {"dry_run": True}}

    if request_key:
        cached = _cached_result(qa, question)
        if cached is not None:
            log.info(f"cached search result for request (run {_run_label(qa)})")
            return cached
        if resumed:
            log.info(f"resuming interrupted search (run {_run_label(qa)})")

    owned = model is None
    model = model or ModelRunner(config)
    usage_tracker = getattr(model, "usage", None)
    usage_before = (
        usage_tracker.snapshot() if callable(getattr(usage_tracker, "snapshot", None))
        else {})
    # Whole-request wall-clock budget.  It covers coarse recall, the agent
    # loop, retries and verifier calls, so a hanging endpoint can never keep
    # an HTTP search open indefinitely.
    budget = float(config.get("search_timeout", 0) or 0)
    deadline = started + budget if budget > 0 else None
    # Split the model budget by role.  ``postagent_calls`` counts only model
    # calls made inside the postagent loop; verifier calls are counted
    # separately so a run can be audited without guessing.
    agent_stats = {"postagent_calls": 0, "postagent_invocations": 0,
                   "verifier_calls": 0, "planner_calls": 0}
    postagent_mode = str(config.get("postagent_mode") or "loop").strip().lower()
    fixed_pipeline = postagent_mode == "fixed"
    # The staged pipeline is also deterministic in its orchestration: a fixed
    # number of model turns, no open-ended "keep searching" loop.  It must not
    # be short-circuited by the coarse gate, because the gate's six-record
    # shortlist is exactly the baseline it is meant to improve on.
    pipeline_mode = postagent_mode in {"pipeline", "staged"}
    # Permissive gate + evidence loop: only questions whose coarse evidence
    # clearly lacks the answer slot pay for the second retrieval loop.
    gate_loop_mode = postagent_mode in {"gate_loop", "evidence_loop"}

    supports_call_deadline = _accepts_deadline(getattr(model, "call", None))
    supports_text_deadline = _accepts_deadline(getattr(model, "text", None))

    async def postagent_call(messages: list[dict[str, Any]],
                             schemas: list[dict[str, Any]] | None = None) -> ModelReply:
        agent_stats["postagent_calls"] += 1
        if supports_call_deadline:
            return await model.call(messages, schemas, deadline=deadline)
        return await model.call(messages, schemas)

    async def verify_call(messages: list[dict[str, Any]]) -> str:
        agent_stats["verifier_calls"] += 1
        if supports_text_deadline:
            return await model.text(messages, deadline=deadline)
        return await model.text(messages)

    async def planner_call(messages: list[dict[str, Any]]) -> str:
        # The compiler planner is a search-stage model call, but not a
        # postagent tool-loop call; keep it separately attributable.
        agent_stats["planner_calls"] += 1
        agent_stats["postagent_calls"] += 1
        reply = (await model.call(messages, None, deadline=deadline)
                 if supports_call_deadline
                 else await model.call(messages, None))
        return str(getattr(reply, "text", "") or "")

    def verify_options() -> dict[str, Any]:
        return {
            "strict": bool(config.get("verifier_strict", True)),
            "require_claims": bool(config.get("verifier_require_claims", True)),
            "grounding_threshold": float(
                config.get("verifier_grounding_threshold", 0.35)),
            "min_key_coverage": float(
                config.get("verifier_min_key_coverage", 0.6)),
            "max_context_tokens": int(
                config.get("verifier_max_context_tokens", 24000) or 0),
        }
    # Coarse must read the same snapshot the agent sees.  If memory.jsonl was
    # atomically replaced after prepare_workspace, the hardlink still points to
    # the old inode; reading the original path would mix two versions.
    snapshot = inputs / "memory.jsonl"
    coarse: CoarseResult | None = None
    coarse_reused = False
    if coarse_source is not None:
        coarse = _load_coarse_source(
            coarse_source, snapshot, question, options, config, inputs)
        coarse_reused = coarse is not None
        if coarse_reused:
            log.info(f"reused coarse candidates from {coarse_source}")
        else:
            log.info(f"coarse reuse skipped (missing/stale/incompatible): "
                     f"{coarse_source}")
    if coarse is None:
        coarse_remaining = (None if deadline is None
                            else max(1.0, deadline - time.monotonic()))
        if coarse_remaining is None:
            coarse = await _coarse_or_empty(
                snapshot, question, options, config, inputs, cache_dir=cache_dir)
        else:
            try:
                coarse = await asyncio.wait_for(
                    _coarse_or_empty(
                        snapshot, question, options, config, inputs,
                        cache_dir=cache_dir),
                    timeout=coarse_remaining)
            except asyncio.TimeoutError:
                # Coarse recall (usually the embedding call) exceeded the
                # request budget.  Do not return an empty candidate pool:
                # BM25 is deterministic and fast, so one short retry still
                # gives the agent and the fallback path a shortlist.
                try:
                    coarse = await asyncio.wait_for(
                        _coarse_or_empty(
                            snapshot, question, options,
                            _bm25_only_config(config), inputs,
                            cache_dir=cache_dir),
                        timeout=max(5.0, min(30.0, coarse_remaining)))
                    coarse.info.setdefault("warnings", []).append(
                        "dense coarse stage timed out; bm25 fallback")
                except Exception:  # noqa: BLE001
                    coarse = CoarseResult(
                        candidates=[],
                        info={"query": question, "route": "dialogue",
                              "records": len(read_jsonl(snapshot)),
                              "warnings": ["coarse deadline exceeded"]},
                        records=read_jsonl(snapshot))
    coarse.info.setdefault("coarse_version", COARSE_CACHE_VERSION)
    route = str((coarse.info or {}).get("route") or "dialogue")
    candidates = coarse.candidates
    records = coarse.records
    if any(str(record.get("kind") or "") == "row" for record in records):
        try:
            written = build_tables_db(records, inputs / "tables.db")
            if written:
                coarse.info["tables_db"] = written
        except (sqlite3.Error, OSError, ValueError):
            pass
    intent = detect_intent(question, route, options)
    # brief.md is the legacy loop's first-step crutch.  The staged pipeline
    # injects the same coarse records directly into its first prompt, so it
    # neither writes nor reads a brief.
    brief = _brief(candidates, question, intent) if candidates and not pipeline_mode else ""
    if brief:
        try:
            (inputs / "brief.md").write_text(brief, encoding="utf-8")
        except OSError:
            pass
    mode = str((config.get("retrieval") or {}).get("mode") or "hybrid").lower()
    if mode == "agent":
        # Ablation: no coarse shortlist; the agent greps the full snapshot.
        candidates = []
        coarse_reused = False
        coarse.info["mode"] = "agent"
        (inputs / "candidates.jsonl").write_text("", encoding="utf-8")

    # Materialize a model-free shortlist BEFORE the first model call and, when
    # the API supplied a path, write it to disk.  If the request then hits the
    # hard timeout while the gate/loop is hanging, the API can still return
    # these records instead of an empty ``data: []``.
    fallback_items: list[dict[str, Any]] = []
    if candidates and route != "self_contained":
        try:
            fallback_limit = int(
                config.get("fallback_evidence_limit",
                           gate_evidence_limit(config))
                or gate_evidence_limit(config))
            fallback_items = _attach_source_times(
                fallback_evidence(
                    candidates, records, question=question, route=route,
                    config=config, limit=max(1, fallback_limit),
                    verification="coarse_fallback"),
                records)
        except Exception as exc:  # noqa: BLE001 - fallback must not kill search
            fallback_items = []
            log.warning(
                f"fallback evidence failed: {type(exc).__name__}: {exc}")
    if fallback_path is not None:
        try:
            evidence.write_file(Path(fallback_path), fallback_items)
        except Exception:  # noqa: BLE001 - fallback must not kill search
            pass

    local = dict(config)
    local["question"] = question
    local["options"] = options
    local["intent"] = intent
    local["brief"] = brief
    local["claim_mode"] = claim_mode(dataset, intent, answer_type)
    local["dataset"] = dataset
    local["answer_type"] = answer_type
    local["route"] = route
    local["deadline"] = deadline

    result = None
    verdict = verifier.Verdict(sufficient=False)
    postagent_verdict = verdict
    raw_evidence: list[dict[str, Any]] = []
    coarse_seed: list[dict[str, Any]] = []
    agent_count = 0
    coarse_seed_count = 0
    degraded = False
    stop_reason = ""
    resolution: dict[str, Any] = {}
    gate_info: dict[str, Any] | None = None
    gate_skipped = False
    compile_info: dict[str, Any] | None = None
    try:
        if route == "self_contained":
            # The answer inputs are in the query; do not spend the agent budget
            # retrieving unrelated memory (the memory may even be empty).
            raw_evidence = [{
                "content": question[: int(config.get("max_evidence_chars", 6000))],
                "metadata": {"source": ["query"], "score": 0.6,
                             "verification": "self_contained"},
            }]
            verdict = verifier.Verdict(sufficient=True, missing=[],
                                       reasoning="self-contained query")
            stop_reason = "self_contained"
        elif not records:
            return {
                "question": question, "run_dir": str(qa), "evidence": [],
                "summary": {"records": 0, "coarse": coarse.info, "route": route,
                            "steps": 0, "sufficient": False, "evidence": 0,
                            "degraded": False, "seconds": 0.0,
                            "tokens": _token_block(
                                (coarse.info or {}).get("embedding_usage") or {},
                                {})},
            }
        else:
            # Coarse-only mode: return the shortlist as evidence, no agent and
            # no verifier.  This is the latency/availability baseline.
            if mode == "coarse":
                raw_evidence = _with_verification(fallback_items, "coarse_mode")
                verdict = verifier.Verdict(
                    sufficient=bool(raw_evidence), missing=[],
                    reasoning="coarse-only mode")
                stop_reason = "coarse_mode"
                gate_skipped = True
                gate_info = {"sufficient": bool(raw_evidence),
                             "reasons": ["mode:coarse"],
                             "coverage": 0.0, "channels": [],
                             "candidate_id": "", "verified": False}

            # Coarse-sufficiency gate: a simple query whose top candidates
            # already cover it can skip the agent loop.  If a verifier
            # is enabled, it audits the coarse evidence first; an insufficient
            # verdict falls through to the agent, so the gate can only save
            # work, not hide a bad answer.
            gate = evaluate_gate(candidates, question, route, config)
            if mode != "coarse":
                gate_info = gate.as_dict()
            if (gate.sufficient and mode != "coarse" and not resumed
                    and not fixed_pipeline and not pipeline_mode
                    and not gate_loop_mode):
                coarse_evidence = _with_verification(
                    fallback_items, "coarse_sufficient")
                if gate_verify(config) and coarse_evidence:
                    coarse_evidence = _attach_source_times(coarse_evidence, records)
                    verdict = await verifier.verify(
                        question, _compact_for_verifier(coarse_evidence, config),
                        verify_call, memory_records=records,
                        forbidden=(coarse.info or {}).get("forbidden"),
                        intent=intent, **verify_options())
                    gate.verified = True
                    gate_info = gate.as_dict()
                    if verdict.sufficient:
                        raw_evidence = coarse_evidence
                        stop_reason = "coarse_sufficient"
                        gate_skipped = True
                    else:
                        gate.sufficient = False
                        gate.reasons.append("verifier_insufficient")
                        gate_info = gate.as_dict()
                elif coarse_evidence:
                    raw_evidence = coarse_evidence
                    verdict = verifier.Verdict(
                        sufficient=True, missing=[], reasoning="coarse gate accepted")
                    stop_reason = "coarse_sufficient"
                    gate_skipped = True
                else:
                    gate.sufficient = False
                    gate.reasons.append("empty_coarse_evidence")
                    gate_info = gate.as_dict()

            # Permissive gate: the deterministic gate above only accepts
            # high-coverage dialogue queries.  Everything else gets one cheap
            # "is the slot actually present?" check before we pay for the
            # second retrieval loop.  This gate is intentionally biased to
            # accept: a false "answerable" costs one answer, a false
            # "unanswerable" costs several model calls on an already-correct
            # question.
            gate_missing: list[str] = []
            force_intents = {
                str(item).lower()
                for item in (config.get("loop_force_intents") or [])}
            accept_types = {
                str(item).lower()
                for item in (config.get("loop_accept_answer_types") or [])}
            force_loop = str(intent or "").lower() in force_intents
            accept_coarse = str(answer_type or "").lower() in accept_types
            if (gate_loop_mode and mode != "coarse" and not gate_skipped
                    and not resumed and route != "self_contained"):
                coarse_evidence = _with_verification(
                    fallback_items, "coarse_gate")
                if coarse_evidence:
                    coarse_evidence = _attach_source_times(
                        coarse_evidence, records)
                    gate_info = dict(gate_info or {})
                    if accept_coarse:
                        # Long-document/rubric answers need coverage, not a
                        # second chunk search; extra hits dilute the prompt.
                        raw_evidence = coarse_evidence
                        verdict = verifier.Verdict(
                            sufficient=True, missing=[],
                            reasoning="rubric coarse evidence accepted")
                        stop_reason = "coarse_rubric_accepted"
                        gate_skipped = True
                        gate_info["rubric_accepted"] = True
                    else:
                        check_consistency = (
                            bool(config.get("gate_check_consistency", False))
                            and str(answer_type).lower() not in {"rubric"}
                            and str(intent).lower() in {
                                "fact", "temporal", "duration", "yesno",
                                "update", "conflict", "compare", "order",
                                "multi_hop", "aggregate"})
                        gate_verdict = await verifier.gate_verify(
                            question,
                            _compact_for_verifier(coarse_evidence, config),
                            verify_call,
                            max_context_tokens=int(
                                config.get(
                                    "verifier_max_context_tokens", 24000) or 0),
                            check_consistency=check_consistency)
                        gate_info["permissive_gate"] = gate_verdict.sufficient
                        gate_info["permissive_missing"] = list(
                            gate_verdict.missing)
                        gate_info["forced_loop"] = force_loop
                        if gate_verdict.sufficient and not force_loop:
                            raw_evidence = coarse_evidence
                            verdict = gate_verdict
                            stop_reason = "coarse_gate_accepted"
                            gate_skipped = True
                        else:
                            hints = list(gate_verdict.missing)
                            slot = str(
                                (gate_verdict.checks or {}).get(
                                    "gate_slot") or "").strip()
                            if slot:
                                hints.append(f"the exact {slot} value")
                            gate_missing = list(dict.fromkeys(hints))
                else:
                    gate_missing = ["no coarse candidates"]

            if gate_skipped:
                evidence.write_file(qa / evidence.EVIDENCE_NAME, raw_evidence)
            else:
                # The coarse shortlist is not a parallel recall channel: it is
                # the agent's INITIAL evidence.jsonl.  The agent keeps, rewrites,
                # or deletes these records and appends claims; the file it
                # leaves behind is the final evidence.
                leftover = (evidence.read_file(qa / evidence.EVIDENCE_NAME)
                            if resumed else [])
                if leftover:
                    # A previous attempt was interrupted mid-run.  Continue
                    # from the evidence it already curated instead of
                    # resetting the file to the original shortlist.
                    coarse_seed = leftover
                else:
                    coarse_seed = seed_evidence(
                        candidates, records, question=question, route=route,
                        config=config)
                    evidence.write_file(qa / evidence.EVIDENCE_NAME, coarse_seed)
                tool_names = tuple(config.get("tools") or ("bash",))
                if pipeline_mode:
                    # The staged pipeline has exactly three tools.  `write`
                    # appends, `edit` replaces text inside one record, and
                    # neither can delete a coarse seed or rewrite the file.
                    tools = build_tools(
                        cwd=qa,
                        shell_command_prefix=_shell_prefix(),
                        bash_timeout=float(config.get("bash_timeout", 60)),
                        names=("bash", "write", "edit"),
                        write_replace=False,
                        edit_append=False,
                        text_edit=True,
                        write_metadata={
                            "origin": "agent", "claim_type": "claim"},
                        write_max_chars=int(config.get(
                            "pipeline_claim_chars_long",
                            config.get("pipeline_claim_chars", 3000))),
                    )
                else:
                    tools = build_tools(
                        cwd=qa,
                        shell_command_prefix=_shell_prefix(),
                        bash_timeout=float(config.get("bash_timeout", 60)),
                        names=tool_names,
                    )
                memory_records = records

                audit_holder: dict[str, Any] = {}

                async def _audit(items: list[dict[str, Any]]) -> list[str]:
                    if not items:
                        return ["no evidence records were written"]
                    items = _attach_source_times(items, memory_records)
                    audit_verdict = await verifier.verify(
                        question, _compact_for_verifier(items, config), verify_call,
                        memory_records=memory_records,
                        forbidden=(coarse.info or {}).get("forbidden"),
                        intent=intent, **verify_options())
                    audit_holder["verdict"] = audit_verdict
                    if audit_verdict.sufficient:
                        return []
                    return audit_verdict.missing or [
                        "the evidence does not answer the question"]

                rounds = int(config.get("max_verify_rounds", 0)) + 1
                hints: list[str] = []
                # Audit the coarse shortlist BEFORE spending agent steps.  The
                # verifier names the missing slot, so the postagent starts
                # from that slot instead of re-discovering it after step 4;
                # and when the shortlist is already sufficient we skip the
                # agent entirely (the coarse half does the easy queries, the
                # agent only the hard ones).
                initial_verdict: verifier.Verdict | None = None
                initial_fingerprint = ""
                coverage = float((gate_info or {}).get("coverage") or 0.0)
                min_coverage = float(config.get("pre_audit_min_coverage", 0.25) or 0.0)
                audit_worthwhile = route == "dialogue" or coverage >= min_coverage
                if (bool(config.get("pre_audit", True))
                        and not gate_loop_mode
                        and postagent_mode not in {"compile", "fixed", "pipeline", "staged"}
                        and audit_worthwhile
                        and int(config.get("audit_after_step", 4)) > 0
                        and coarse_seed and mode != "coarse"
                        and route != "self_contained"):
                    try:
                        initial_missing = await _audit(coarse_seed)
                    except Exception as exc:  # noqa: BLE001 - auditing is best-effort
                        log.warning(f"pre-audit failed: {type(exc).__name__}: {exc}")
                        initial_missing = []
                    initial_verdict = audit_holder.get("verdict")
                    initial_fingerprint = evidence.fingerprint(coarse_seed)
                    if (initial_verdict is not None and initial_verdict.sufficient
                            and not initial_missing):
                        raw_evidence = coarse_seed
                        verdict = initial_verdict
                        stop_reason = "coarse_sufficient"
                        gate_skipped = True
                    elif initial_verdict is not None and initial_verdict.parsed:
                        hints = list(initial_missing)
                if gate_skipped:
                    rounds = 0
                if not gate_skipped and gate_loop_mode:
                    # Reuse the coarse index when it exists; a cached coarse
                    # result does not carry one, so build it once here (the
                    # document embeddings are already in the cache).
                    loop_index = coarse.index
                    owned_index = False
                    if loop_index is None:
                        loop_index = await build_index(
                            snapshot, config, cache_dir=cache_dir)
                        owned_index = True
                    try:
                        loop_result = await evidence_loop.run_evidence_loop(
                            workspace=qa, index=loop_index,
                            call_model=postagent_call,
                            config={
                                **config,
                                "question": question,
                                "options": list(options or []),
                                "intent": intent,
                                "deadline": deadline,
                            },
                            missing=gate_missing,
                            fallback_path=fallback_path)
                    finally:
                        if owned_index and loop_index is not None:
                            await loop_index.aclose()
                    result = loop_result
                    evidence.write_file(
                        qa / evidence.EVIDENCE_NAME, loop_result.evidence)
                    raw_evidence = _attach_source_times(
                        loop_result.evidence, records)
                    verdict = verifier.Verdict(
                        sufficient=bool(loop_result.evidence),
                        missing=list(loop_result.missing),
                        reasoning=f"evidence_loop:{loop_result.stop_reason}")
                    stop_reason = loop_result.stop_reason
                    compile_info = dict(loop_result.stats)
                    rounds = 0
                elif (not gate_skipped and postagent_mode == "compile"
                        and not resumed):
                    compiled = await compiler.run_compiler(
                        question=question, options=list(options or []),
                        intent=intent, answer_type=answer_type, route=route,
                        candidates=candidates, records=records,
                        seed_evidence=coarse_seed,
                        call_text=planner_call, config=config, log=log)
                    evidence.write_file(
                        qa / evidence.EVIDENCE_NAME, compiled.evidence)
                    raw_evidence = _attach_source_times(
                        compiled.evidence, records)
                    compile_info = dict(compiled.stats)
                    compile_info["plan"] = {
                        "shape": compiled.plan.get("answer_shape"),
                        "answer_present": compiled.plan.get("answer_present"),
                        "missing": compiled.plan.get("missing"),
                        "selected": [
                            str(item.get("id"))
                            for item in (compiled.plan.get("selected") or [])
                            if isinstance(item, dict)],
                        "items": len(compiled.plan.get("items") or []),
                    }
                    stop_reason = ("compiled_gapfill"
                                   if compile_info.get("gapfill_ids")
                                   else "compiled")
                    rounds = 0
                    log.info(
                        f"compiler: shape={compile_info.get('shape')} "
                        f"present={compile_info.get('answer_present')} "
                        f"selected={len(compile_info.get('selected_ids') or [])} "
                        f"gapfill={len(compile_info.get('gapfill_ids') or [])} "
                        f"evidence={compile_info.get('evidence')}")
                if not gate_skipped and fixed_pipeline and not resumed:
                    fixed = compiler.run_fixed(
                        question=question, intent=intent,
                        answer_type=answer_type, route=route,
                        candidates=candidates, records=records, config=config)
                    evidence.write_file(qa / evidence.EVIDENCE_NAME, fixed.evidence)
                    raw_evidence = _attach_source_times(fixed.evidence, records)
                    compile_info = dict(fixed.stats)
                    compile_info["plan"] = {
                        "shape": None, "answer_present": None, "missing": None,
                        "selected": [], "items": 0,
                    }
                    verdict = verifier.Verdict(
                        sufficient=bool((gate_info or {}).get("sufficient")),
                        missing=[], reasoning="fixed deterministic pipeline")
                    stop_reason = "fixed_pipeline"
                    rounds = 0
                    log.info(
                        f"fixed postagent: shape={compile_info.get('shape')} "
                        f"budget={compile_info.get('budget')} "
                        f"evidence={compile_info.get('evidence')} "
                        f"chars={compile_info.get('chars')}")
                if not gate_skipped and pipeline_mode and not resumed:
                    fixed = await run_pipeline(
                        workspace=qa, tools=tools, call_model=postagent_call,
                        schemas=model.schemas, config=local, log=log)
                    evidence.write_file(
                        qa / evidence.EVIDENCE_NAME, fixed.evidence)
                    raw_evidence = _attach_source_times(fixed.evidence, records)
                    compile_info = dict(fixed.stats)
                    compile_info["plan"] = {
                        "shape": None, "answer_present": None, "missing": None,
                        "selected": [], "items": 0,
                    }
                    verdict = verifier.Verdict(
                        sufficient=bool(fixed.evidence),
                        missing=[], reasoning="staged pipeline")
                    stop_reason = "pipeline"
                    result = fixed
                    rounds = 0
                    log.info(
                        f"staged pipeline: stages={len(compile_info.get('stages') or [])} "
                        f"writes={compile_info.get('writes')} "
                        f"edits={compile_info.get('edits')} "
                        f"bash={compile_info.get('bash')} "
                        f"evidence={len(fixed.evidence)}")
                for round_index in range(rounds):
                    agent_stats["postagent_invocations"] += 1
                    result = await run_postagent(
                        workspace=qa, tools=tools, call_model=postagent_call,
                        schemas=model.schemas, config=local, hints=hints or None,
                        audit=_audit if int(config.get("audit_after_step", 4)) > 0 else None)
                    current_file = evidence.read_file(qa / evidence.EVIDENCE_NAME)
                    raw_evidence = _attach_source_times(current_file, records)
                    if (result is not None and result.audit_at_finish
                            and result.audit_sufficient
                            and evidence.fingerprint(current_file) == result.audit_fingerprint):
                        # The postagent's last action was the audit itself; the
                        # file cannot have changed since.  Reuse the audit's
                        # structured checks rather than fabricating a bare
                        # verdict.
                        verdict = audit_holder.get("verdict") or verifier.Verdict(
                            sufficient=True, missing=[],
                            reasoning="mid-loop audit")
                    elif (initial_verdict is not None
                            and evidence.fingerprint(current_file) == initial_fingerprint):
                        # The agent changed nothing; the pre-audit verdict is
                        # still the verdict on this exact file.
                        verdict = initial_verdict
                    else:
                        verdict = await verifier.verify(
                            question, _compact_for_verifier(raw_evidence, config),
                            verify_call, memory_records=records,
                            forbidden=(coarse.info or {}).get("forbidden"),
                            intent=intent, **verify_options())
                    log.info(f"round {round_index + 1}: steps={result.steps} "
                             f"stop={result.stop_reason} evidence={len(raw_evidence)} "
                             f"sufficient={verdict.sufficient}")
                    if verdict.sufficient or round_index + 1 >= rounds:
                        break
                    hints = verdict.missing
                if result is not None:
                    stop_reason = result.stop_reason

        # Final evidence = exactly the file the agent curated.  Seeded coarse
        # records are kept as-is (deleting one in the file is respected); only
        # agent-written records go through evidence validation/resolution.
        max_items = max(1, int(config.get("max_evidence", 64)))
        valid, rejected = evidence.validate_file(qa / evidence.EVIDENCE_NAME)
        source_items = valid or raw_evidence
        seeded = [
            item for item in source_items
            if (item.get("metadata") or {}).get("seeded")
            or str((item.get("metadata") or {}).get("verification") or "").startswith("coarse")
        ]
        seeded_ids = {id(item) for item in seeded}
        agent_only = [item for item in source_items if id(item) not in seeded_ids]
        processed_agent, resolution = postprocess_evidence(
            agent_only, records, question=question, route=route, config=config)
        final_config = config
        if pipeline_mode:
            # Pipeline claims are answer-oriented extracts with explicit
            # source ids.  Keep them even for the raw-only dataset families;
            # the generic claim_mode policy would otherwise discard the layer
            # this pipeline exists to add.
            final_config = dict(config)
            final_config["final_keep_claims"] = True
            # Keep answer-oriented claims, but drop the ones that merely
            # restate a seeded record.  The pipeline's claim layer exists to
            # add resolved dates/counts/chains, not to paraphrase a seed into
            # a more generic (and more salient) sentence.
            final_config["final_drop_paraphrases"] = True
        processed = fit_evidence(
            annotate_resolved_dates(seeded + processed_agent), records,
            config=final_config, intent=intent, dataset=dataset,
            answer_type=answer_type, question=question)
        if route == "self_contained" and not processed:
            # The answer inputs are in the query itself.  Post-processing
            # rejects source=["query"] as an unknown memory id, so keep the
            # self-contained record verbatim instead of returning [].
            processed = annotate_resolved_dates(raw_evidence)
        agent_count = sum(1 for item in processed
                          if (item.get("metadata") or {}).get("origin")
                          in {"agent", "postagent"})
        coarse_seed_count = sum(1 for item in processed
                                if (item.get("metadata") or {}).get("origin") == "coarse")

        if not processed and candidates and route != "self_contained":
            processed = annotate_resolved_dates(fallback_evidence(
                candidates, records, question=question, route=route, config=config,
                limit=max_items))
            degraded = True
        elif (not gate_skipped and mode != "coarse"
              and not agent_count and coarse_seed_count):
            # The agent ran but produced no usable evidence; the result is
            # coarse-only.  Gate/coarse-mode paths are intentionally coarse.
            degraded = True
        # Verify the EXACT evidence the answer model will see.  Post-processing
        # resolves parents, adds neighbourhood context and truncates to the
        # answer budget; a verdict on the pre-processed file can therefore
        # disagree with what the answer actually gets.  Precision-first:
        # sufficient means this final list passed the strict audit.
        postagent_verdict = verdict
        if (mode != "coarse" and route != "self_contained"
                and not fixed_pipeline and not pipeline_mode
                and not gate_loop_mode):
            final_records = _compact_for_verifier(processed, config) if processed else []
            verdict = await verifier.verify(
                question, final_records, verify_call,
                memory_records=records,
                forbidden=(coarse.info or {}).get("forbidden"),
                intent=intent, **verify_options())
            if verdict.sufficient != postagent_verdict.sufficient:
                log.info(f"final verifier: postagent={postagent_verdict.sufficient} "
                         f"final={verdict.sufficient} evidence={len(processed)}")
        evidence.write_file(qa / evidence.EVIDENCE_NAME, processed)
        raw_evidence = processed
    except Exception as exc:  # noqa: BLE001 - fallback beats a 5xx
        log.error(f"search failed, using coarse fallback: {type(exc).__name__}: {exc}")
        # Keep anything the agent already wrote: one bad model reply must not
        # throw away verified evidence.  Only fall back to the shortlist when
        # the evidence file is empty.
        existing = evidence.read_file(qa / evidence.EVIDENCE_NAME)
        if existing:
            raw_evidence = _attach_source_times(existing, records)
        elif records and candidates:
            raw_evidence = fallback_evidence(
                candidates, records, question=question, route=route, config=config,
                limit=max(1, int(config.get("max_evidence", 64))))
        else:
            raw_evidence = []
        degraded = True
        stop_reason = "fallback_error"
        raw_evidence = annotate_resolved_dates(raw_evidence)
        evidence.write_file(qa / evidence.EVIDENCE_NAME, raw_evidence)
    finally:
        if coarse.index is not None:
            await coarse.index.aclose()
        if owned:
            await model.aclose()

    usage_after = (
        usage_tracker.snapshot() if callable(getattr(usage_tracker, "snapshot", None))
        else {})
    llm_usage = llm.usage_delta(usage_before, usage_after)
    embedding_usage = _sum_usage(
        (coarse.info or {}).get("embedding_usage"),
        _read_usage_log(qa / ".embedding_usage.jsonl"),
    )
    token_block = _token_block(embedding_usage, llm_usage)

    if result is not None and result.history and bool(config.get("save_transcript", True)):
        try:
            with (qa / "agent_history.jsonl").open("w", encoding="utf-8") as handle:
                for message in result.history:
                    handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        except OSError:
            pass
    if result is not None and bool(config.get("save_steps", True)):
        try:
            write_jsonl(qa / "steps.jsonl", list(result.step_log))
        except OSError:
            pass

    summary = {
        "question": question,
        "run_dir": str(qa),
        "steps": result.steps if result else 0,
        "stop_reason": stop_reason,
        "sufficient": verdict.sufficient,
        "missing": verdict.missing,
        "intent": intent,
        "answer_type": answer_type,
        "claim_mode": claim_mode(dataset, intent, answer_type),
        "audited": bool(result.audited) if result else False,
        "audit_missing": list(result.audit_missing) if result else [],
        "final_chars": sum(len(str(item.get("content") or "")) + 80
                           for item in raw_evidence),
        "claims": sum(1 for item in raw_evidence
                      if (item.get("metadata") or {}).get("claim_type") == "claim"),
        "copies": sum(1 for item in raw_evidence
                      if (item.get("metadata") or {}).get("claim_type") == "copy"),
        "evidence": len(raw_evidence),
        "evidence_mix": {
            "postagent": agent_count,
            "coarse_seed": coarse_seed_count,
            "final": len(raw_evidence),
        },
        "rejected": int(resolution.get("rejected") or 0),
        "resolution": resolution,
        "coarse": coarse.info,
        "coarse_reused": coarse_reused,
        "coarse_source": str(coarse_source) if coarse_reused and coarse_source else "",
        "route": route,
        "gate": gate_info,
        "gate_skipped": gate_skipped,
        "compile": compile_info,
        "degraded": degraded,
        "cached": False,
        "resumed": resumed,
        "search_timeout": round(budget, 1) if deadline is not None else 0,
        "timed_out": bool(
            deadline is not None and time.monotonic() >= deadline),
        "model_calls": model.calls,
        "postagent_calls": agent_stats["postagent_calls"],
        "postagent_invocations": agent_stats["postagent_invocations"],
        "planner_calls": agent_stats["planner_calls"],
        "verifier_calls": agent_stats["verifier_calls"],
        "verifier": {
            "sufficient": verdict.sufficient,
            "parsed": verdict.parsed,
            "checks": dict(verdict.checks),
            "reasoning": verdict.reasoning,
            # Keep the raw verdict for audits; truncate so a verbose model
            # cannot bloat every run summary.
            "raw": (verdict.raw or "")[:2000],
        },
        "verifier_postagent": {
            "sufficient": postagent_verdict.sufficient,
            "parsed": postagent_verdict.parsed,
            "checks": dict(postagent_verdict.checks),
        },
        "tokens": token_block,
        "seconds": round(time.perf_counter() - started, 3),
    }
    # Atomic: a kill during this write must leave either the previous (absent)
    # summary or a complete one, never a half-written cache entry.
    atomic_write_text(
        qa / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2))
    return {"question": question, "run_dir": str(qa), "evidence": raw_evidence,
            "summary": summary}


