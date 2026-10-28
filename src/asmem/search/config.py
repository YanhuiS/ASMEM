"""Search configuration: defaults, merge, and model call policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..io import PROJECT_ROOT

CONFIG_FILE = PROJECT_ROOT / "configs" / "asmem.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "generator": {
        "base_url": "http://127.0.0.1:30000/v1",
        "api_key": "sglang",
        "model": "local",
        "temperature": 0.2,
        "max_tokens": 4096,
        "enable_thinking": False,
    },
    "native_tools": False,
    "tools": ["read", "write", "edit", "bash"],
    "max_steps": 6,
    "max_verify_rounds": 0,
    "no_progress_patience": 2,
    "observation_max_chars": 6000,
    "write_nudge_after": 2,
    "max_empty_finish_refusals": 1,
    "max_parse_retries": 1,
    "repeat_limit": 1,
    "context_max_chars": 50000,
    "context_keep_recent": 6,
    # Token-level preflight for the agent transcript; 20k input + the model's
    # 8k output stays inside a 32k endpoint context.
    "agent_max_context_tokens": 20000,
    "bash_timeout": 60,
    # Whole-request budget.  The runner stops the loop at this deadline; the
    # bench also wraps the call in a hard cap so a non-cooperative endpoint
    # cannot outlive it.
    "search_timeout": 300,
    "search_hard_timeout": 360,
    # Queue and fallback policy.  The hard timeout covers semaphore wait; a
    # request that cannot get a slot within ``search_queue_timeout`` returns
    # the model-free coarse fallback instead of queueing for the whole budget.
    "search_queue_timeout": 60,
    "fallback_enabled": True,
    "fallback_timeout": 20,
    "fallback_concurrency": 8,
    "fallback_evidence_limit": 12,
    "fallback_keep_files": False,
    # A process-wide endpoint breaker (0 disables).  Unlike the per-run
    # breaker, this stops every new search from retrying a dead endpoint.
    "llm_global_breaker_failures": 0,
    "llm_global_breaker_cooldown": 30,
    # Preflight cap for every model prompt and context-overflow recovery.
    # 0 disables the preflight; the API config enables 24000.
    "model_input_token_cap": 0,
    "model_output_reserve_tokens": 4096,
    "context_overflow_retries": 1,
    # Output caps by role: gate/verifier replies are short JSON, the postagent
    # needs room for tool calls/claims.
    "verifier_max_output_tokens": 2048,
    "postagent_max_output_tokens": 4096,
    # Refuse a reused coarse source when the deterministic retrieval config
    # fingerprint differs (candidate-id validation is always applied).
    "reuse_coarse_strict": True,
    "concurrency": 4,
    # Consecutive model failures (transport/5xx, not truncation) that open the
    # per-run circuit breaker.  A dead endpoint then fails fast instead of
    # multiplying its timeout by every remaining step.
    "llm_breaker_failures": 2,
    # Total evidence budget.  Half is reserved for the agent and half for the
    # coarse shortlist; unused slots are backfilled by the other half.
    "max_evidence": 64,
    "coarse_evidence_ratio": 0.5,
    "max_evidence_chars": 6000,
    # Precision-first verifier.  It sees the same evidence the answer model
    # sees, requires grounded claims, and checks key-term coverage.
    "verifier_max_evidence": 24,
    "verifier_max_chars": 3000,
    "verifier_max_context_tokens": 24000,
    "verifier_strict": True,
    "verifier_require_claims": True,
    "verifier_grounding_threshold": 0.35,
    "verifier_min_key_coverage": 0.6,
    "max_append_per_call": 8,
    # A single bash command is capped before execution.  The truncation
    # failures in v0.2 were 8k-token jq filters built from repeated `test(...)`
    # clauses; the limit turns that into one cheap error observation.
    "max_command_chars": 800,
    "max_model_errors": 2,
    # Mid-loop audit: after this step (and before the last), the verifier's
    # `missing` list is injected as a hint, so the remaining steps are targeted.
    "audit_after_step": 3,
    # Audit the coarse shortlist before step 1: sufficient skips the agent,
    # otherwise the missing slots become the first postagent hints.
    "pre_audit": True,
    # Only audit the shortlist when it has a plausible chance of being
    # sufficient.  A multi-hop/document query with near-zero key coverage
    # cannot pass; spending a verifier call on it just slows the run down.
    "pre_audit_min_coverage": 0.25,
    # Capture a search hit the model did not append (append_matches dedupes).
    "auto_capture_hits": True,
    "auto_capture_limit": 2,
    "auto_capture_total": 6,
    # postagent v2: ``compile`` runs one answer-oriented planner call and a
    # deterministic evidence compiler instead of the bash search loop.
    # ``loop`` keeps the v0.7 behaviour for A/B runs.
    "postagent_mode": "loop",
    "compiler_max_candidates": 80,
    "compiler_preview_chars": 300,
    "compiler_inventory_chars": 30000,
    "compiler_max_select": 12,
    "compiler_max_items": 24,
    "compiler_keep_tail": True,
    "compiler_keep_order": True,
    "compiler_promote_context": True,
    "compiler_final_chars": 15000,
    "compiler_gapfill": True,
    "compiler_gapfill_limit": 2,
    # Shape-aware budgets: recall shapes (multi-hop/list) get a wider tail;
    # temporal/inference run the planner at a tighter budget.
    "compiler_shape_routing": True,
    "compiler_shape_budgets": {
        "multi_hop": 48000, "list": 48000, "number": 48000,
        "comparison": 48000, "date": 48000, "duration": 48000,
        "yesno": 48000, "inference": 48000, "entity": 48000, "other": 48000,
    },
    # Final evidence is selected against the ANSWER model's character budget.
    # Item-count fusion (64 records) was invisible past ~12 records.
    "final_evidence_chars": 15000,
    "final_item_chars": 2500,
    "final_parent_cap": 2,
    "final_claim_ratio": 0.5,
    "final_claim_cap": 12,
    "final_agent_cap": 12,
    # Pipeline only: drop a claim that only restates a seeded record and adds
    # no new source/date/number/detail.  The raw seed is already evidence.
    "final_drop_paraphrases": False,
    # If a literal grep in the gap stage matches nothing, deterministically
    # rewrite the same anchors as a `coarse --query` (no model call).
    "auto_coarse_fallback": True,
    "auto_coarse_total": 3,
    "auto_coarse_limit": 3,
    # Gate + evidence loop policy.  High-risk intents always get the loop even
    # when the gate accepts (a present-but-wrong date/ordering is the common
    # failure).  Long-document rubric answers skip the loop: extra chunks
    # dilute coverage more than they help recall.
    "loop_force_intents": [],
    "loop_accept_answer_types": [],
    # Second, stricter look at an accepted gate answer for single-value/
    # ordering intents.  Catches a quote about a different event; rubric/
    # long-document answers skip it.
    "gate_check_consistency": False,
    # Neighbour expansion is mandatory: every cited anchor carries its
    # immediate same-source neighbours as marked context.
    "evidence_neighbor_window": 1,
    "neighbor_max_chars": 600,
    "save_transcript": True,
    "save_steps": True,
    "parent_max_chars": 4000,
    "parent_auto_chars": 1500,
    "parent_window_chunks": 1,
    "window_max_chars": 4000,
    "retrieval": {
        "mode": "hybrid",
        "coarse": {
            "candidates": 80,
            "session_cap_ratio": 0.3,
            "channel_top_k": 3,
            "neighbor_window": 1,
            "neighbor_ratio": 0.2,
            "neighbor_max_chars": 600,
            "dedup_jaccard": 0.9,
            "rrf_k": 60,
            "option_view_weight": 0.3,
        },
        "chunking": {
            "version": 3,
            "message_tokens": 320,
            "passage_tokens": 320,
            "record_tokens": 320,
            "overlap_tokens": 40,
            "min_tokens": 20,
            # Legacy char thresholds, used only by explicit max_chars callers.
            "message_chars": 3000,
            "passage_chars": 1500,
            "record_chars": 2000,
            "overlap_chars": 120,
            "min_chars": 120,
        },
        "channels": {
            "bm25": 0.5, "dense": 0.5, "exact": 0.3, "time": 0.2,
            "state": 0.2, "session": 0.2, "option": 0.2,
        },
        "gate": {
            "enabled": True,
            "verify": True,
            "min_coverage": 0.6,
            "min_channels": 1,
            "max_terms": 12,
            "min_text_chars": 40,
            "evidence_limit": 6,
            "allow_routes": ["dialogue"],
        },
        "bm25": {"k1": 1.2, "b": 0.75},
        "dense": {
            "mode": "auto", "model": "", "base_url": "", "api_key": "",
            "batch_size": 16, "concurrency": 4, "max_retries": 3,
            "dim": 128, "dimensions": 0, "query_prefix": "",
            "document_prefix": "", "include_role": False,
            # Long records are encoded in overlapping token windows (max-pooled).
            "max_tokens": 320, "overlap_tokens": 40,
            # Legacy char window, used when max_tokens is 0.
            "max_chars": 0, "overlap_chars": 0,
        },
    },
    "log": {"level": "info"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    target = Path(path).expanduser() if path else CONFIG_FILE
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            _deep_merge(config, loaded)
    return config


def call_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config.get("generator") or {})
    for key in ("timeout", "connect_timeout", "max_retries", "backoff_base",
                "rate_limit_backoff", "max_backoff", "backoff_jitter",
                "concurrency", "max_connections", "max_keepalive_connections",
                "trust_env", "retry_truncated"):
        if key in config:
            merged[key] = config[key]
    merged.setdefault("max_retries", 3)
    merged.setdefault("timeout", 300)
    return merged

