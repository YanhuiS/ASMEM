"""YAML configuration for the benchmark experiment.

The file is intentionally one flat YAML:

- top-level ``generator`` / ``max_steps`` / ``native_tools`` ... are exactly the
  keys :mod:`asmem.search.runner` already understands, so the same file drives
  the search agent;
- ``experiment`` controls the replay/parallelism;
- ``answer`` and ``eval`` can override the generator (e.g. use a stronger judge).

Precedence for the answer/eval model::

    top-level generator  <  answer.generator / eval.generator
    top-level generator  <  answer/eval scalar keys (temperature, max_tokens, ...)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from ..io import PROJECT_ROOT
from ..search import runner
from . import prompts

DEFAULT_DATASETS = [
    "longmemeval_oracle", "longmemeval_s", "longmemeval_m", "locomo_refined",
    "beam_100k", "beam_500k", "beam_1m", "beam_10m",
    "personamem_32k", "personamem_128k",
    "clbench", "scriptmem",
]

EXPERIMENT_DEFAULTS: dict[str, Any] = {
    "data_dir": "data/processed",
    "datasets": "all",
    "out_dir": "data/bench_runs",
    "index_cache_dir": "",     # empty -> {out_dir}/_index_cache
    "run_name": "",
    "max_users": 0,
    "limit_queries_per_user": 0,
    "user_concurrency": 16,
    "search_concurrency": 16,
    "answer_concurrency": 8,
    "eval_concurrency": 8,
    "pending_per_user": 8,
    "resume": True,
    "overwrite": False,
    "save_memory": False,
    "skip_search_if_no_memory": True,
    "skip_add": False,
    "skip_search": False,
    "skip_answer": False,
    "skip_eval": False,
    "force_answer": False,      # reuse search evidence but re-run answer/eval
    "force_eval": False,        # reuse search+answer but re-run eval only
    "force_search": False,      # ignore cached evidence and re-run search
    "reuse_coarse_from": "",    # run name/path: reuse its deterministic coarse stage
    "mock": False,
    # Liveness: log in-flight searches/answers and any QA slower than this.
    "heartbeat_seconds": 60,
    "qa_warn_seconds": 300,
    "log_level": "info",
}

ANSWER_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_evidence_chars": 16000,
    # Per-call and per-stage guards.  A 4o-mini call normally returns in a few
    # seconds; 180s is already pathological.  Retries are generous so a
    # transient timeout still produces an answer, while total_timeout bounds
    # the whole stage (retries included) so one QA cannot hang the run.
    "timeout": 180,
    "connect_timeout": 10,
    "max_retries": 5,
    "backoff_base": 1.5,
    "rate_limit_backoff": 5,
    "max_backoff": 30,
    "total_timeout": 300,
    "max_context_tokens": 24000,
    "system": prompts.ANSWER_SYSTEM,
    "system_by_dataset": {
        "scriptmem": prompts.SCRIPTMEM_SYSTEM,
        "personamem_32k": prompts.PERSONALIZED_SYSTEM,
        "personamem_128k": prompts.PERSONALIZED_SYSTEM,
    },
    "user": prompts.ANSWER_USER,
    "styles": {},
    # dataset/answer_type -> extra instruction, applied on top of styles.
    # Used for the abstention and instruction-following families without
    # changing every answer in that dataset.
    "type_styles": {},
    # Datasets where the verifier-gated strong answerer consistently loses to
    # the normal answerer.  This is a routing guard, not a model setting.
    "no_strong_datasets": [],
    "generator": {},
}

EVAL_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "mode": "auto",                 # auto | mcq | f1 | judge | rubric | none
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_rubrics": 100,
    "max_answer_chars": 12000,
    "timeout": 180,
    "connect_timeout": 10,
    "max_retries": 5,
    "backoff_base": 1.5,
    "rate_limit_backoff": 5,
    "max_backoff": 30,
    "total_timeout": 300,
    "max_context_tokens": 24000,
    # Keep each rubric response small enough to finish inside max_tokens.
    "rubric_batch": 20,
    "judge": True,
    "rubric": True,
    "locomo_official": True,
    "locomo_official_user": prompts.LOCOMO_REFINED_JUDGE_USER,
    "primary_by_dataset": {},
    "styles": dict(prompts.JUDGE_STYLES),
    "system": prompts.JUDGE_SYSTEM,
    "user": prompts.JUDGE_USER,
    "rubric_system": prompts.RUBRIC_SYSTEM,
    "rubric_user": prompts.RUBRIC_USER,
    "generator": {},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def project_path(value: str | Path | None, default: str = "") -> Path:
    text = str(value or default)
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_datasets(value: Any) -> list[str]:
    if value is None or str(value).strip().lower() == "all":
        return list(DEFAULT_DATASETS)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(DEFAULT_DATASETS)


def _text_model_config(
    base_generator: dict[str, Any],
    section: dict[str, Any],
    *,
    concurrency: int,
) -> dict[str, Any]:
    """Apply ``answer``/``eval`` overrides on top of the top-level generator."""
    config = copy.deepcopy(base_generator)
    config.update(copy.deepcopy(section.get("generator") or {}))
    # A few scalar keys are common enough to allow directly in the section.
    for key in ("model", "base_url", "api_key", "temperature", "max_tokens",
                "enable_thinking", "timeout", "connect_timeout", "max_retries",
                "backoff_base", "rate_limit_backoff", "max_backoff", "backoff_jitter",
                "retry_truncated"):
        if key in section and section[key] is not None:
            config[key] = section[key]
    config["concurrency"] = max(1, int(concurrency))
    config.setdefault("max_connections", max(32, config["concurrency"] * 2))
    config.setdefault("max_keepalive_connections", config["concurrency"])
    return config


def load_bench_config(path: str | Path) -> dict[str, Any]:
    """Load and resolve one experiment YAML."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")

    experiment = deep_merge(EXPERIMENT_DEFAULTS, raw.get("experiment") or {})
    experiment["datasets"] = normalize_datasets(experiment.get("datasets"))
    experiment["user_concurrency"] = max(1, int(experiment["user_concurrency"]))
    experiment["search_concurrency"] = max(1, int(experiment["search_concurrency"]))
    experiment["answer_concurrency"] = max(1, int(experiment["answer_concurrency"]))
    experiment["eval_concurrency"] = max(1, int(experiment["eval_concurrency"]))
    experiment["pending_per_user"] = max(1, int(experiment["pending_per_user"]))
    experiment["max_users"] = max(0, int(experiment["max_users"]))
    experiment["limit_queries_per_user"] = max(0, int(experiment["limit_queries_per_user"]))

    answer = deep_merge(ANSWER_DEFAULTS, raw.get("answer") or {})
    ev = deep_merge(EVAL_DEFAULTS, raw.get("eval") or {})

    # The search runner reads the same file. Its defaults + top-level keys are
    # exactly what we want, then we override the client pool size.
    search = runner.load_config(config_path)
    search["concurrency"] = experiment["search_concurrency"]
    top_log = raw.get("log") if isinstance(raw.get("log"), dict) else {}
    search["log"] = {
        "level": top_log.get("level") or "warn",
        "output": top_log.get("output") or "",
    }

    base_generator = dict(raw.get("generator") or {})
    answer_generator = _text_model_config(
        base_generator, answer, concurrency=experiment["answer_concurrency"])
    eval_generator = _text_model_config(
        base_generator, ev, concurrency=experiment["eval_concurrency"])

    return {
        "path": str(config_path),
        "raw": raw,
        "experiment": experiment,
        "search": search,
        "answer": answer,
        "eval": ev,
        "answer_generator": answer_generator,
        "eval_generator": eval_generator,
    }


def redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolved config snapshot safe to write next to the run (no API keys)."""
    snapshot = copy.deepcopy(config.get("raw") or {})
    snapshot["experiment"] = copy.deepcopy(config.get("experiment") or {})
    snapshot["answer"] = copy.deepcopy(config.get("answer") or {})
    snapshot["eval"] = copy.deepcopy(config.get("eval") or {})
    snapshot["answer_generator"] = copy.deepcopy(config.get("answer_generator") or {})
    snapshot["eval_generator"] = copy.deepcopy(config.get("eval_generator") or {})
    # Keep the search settings that affect the agent, without duplicating the
    # generator (already top-level) or the internal log section.
    snapshot["search"] = {
        key: copy.deepcopy(value)
        for key, value in (config.get("search") or {}).items()
        if key not in {"generator", "log", "answer", "eval", "experiment"}
    }
    if "api_key" in snapshot:
        snapshot["api_key"] = "***"
    for key in ("generator", "answer_generator", "eval_generator"):
        generator = snapshot.get(key)
        if isinstance(generator, dict) and "api_key" in generator:
            generator["api_key"] = "***"
    for section in ("answer", "eval"):
        value = snapshot.get(section)
        if isinstance(value, dict):
            generator = value.get("generator")
            if isinstance(generator, dict) and "api_key" in generator:
                generator["api_key"] = "***"
    return snapshot


def config_json(config: dict[str, Any]) -> str:
    return json.dumps(redacted_config(config), ensure_ascii=False, indent=2)
