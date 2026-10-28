"""Aggregate QA results into ``summary.json`` / ``summary.md``."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

METRIC_NAMES = (
    "score", "token_f1", "exact_match", "mcq_exact",
    "judge_correct", "rubric_score",
)


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values if value is not None]
    return round(sum(items) / len(items), 4) if items else None


def _metric_values(results: list[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for result in results:
        evaluation = result.get("eval") or {}
        if name == "score":
            value = evaluation.get("score")
        else:
            value = (evaluation.get("metrics") or {}).get(name)
        if value is not None:
            values.append(value)
    return values


def _add_usage(target: dict[str, Any], source: dict[str, Any] | None) -> None:
    for key, value in (source or {}).items():
        if isinstance(value, dict):
            bucket = target.setdefault(key, {})
            if isinstance(bucket, dict):
                _add_usage(bucket, value)
        elif isinstance(value, (int, float)):
            target[key] = int(target.get(key, 0)) + int(value)


def _token_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    per_query: list[int] = []
    for result in results:
        tokens = result.get("tokens") or {}
        if not isinstance(tokens, dict):
            continue
        _add_usage(totals, tokens)
        value = tokens.get("total_tokens")
        if isinstance(value, (int, float)):
            per_query.append(int(value))
    if not totals:
        return {}
    if per_query:
        totals["avg_total_tokens"] = round(sum(per_query) / len(per_query), 2)
    return totals


def summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(result.get("status") or "unknown") for result in results)
    search = [result.get("search") or {} for result in results]
    sufficient = [1.0 if item.get("sufficient") else 0.0
                  for item in search if item.get("sufficient") is not None]
    steps = [float(item["steps"]) for item in search
             if isinstance(item.get("steps"), (int, float))]
    search_errors = sum(1 for item in search if item.get("error"))
    postagent_calls = [float(item["postagent_calls"]) for item in search
                       if isinstance(item.get("postagent_calls"), (int, float))]
    verifier_calls = [float(item["verifier_calls"]) for item in search
                      if isinstance(item.get("verifier_calls"), (int, float))]
    verifier_sufficient = [
        1.0 if (item.get("verifier") or {}).get("sufficient") else 0.0
        for item in search if (item.get("verifier") or {}).get("sufficient") is not None
    ]
    verifier_unparsed = sum(
        1 for item in search if (item.get("verifier") or {}).get("parsed") is False)
    eval_errors = sum(
        1 for result in results
        if (result.get("eval") or {}).get("errors")
        and (result.get("eval") or {}).get("mode") != "search_failed"
    )
    summary: dict[str, Any] = {
        "n": len(results),
        "status": dict(statuses),
        "search_errors": search_errors,
        "eval_errors": eval_errors,
        "score": _mean(_metric_values(results, "score")),
        "token_f1": _mean(_metric_values(results, "token_f1")),
        "exact_match": _mean(_metric_values(results, "exact_match")),
        "mcq_exact": _mean(_metric_values(results, "mcq_exact")),
        "judge_correct": _mean(_metric_values(results, "judge_correct")),
        "rubric_score": _mean(_metric_values(results, "rubric_score")),
        "sufficient_rate": _mean(sufficient),
        "avg_steps": _mean(steps),
        "avg_postagent_calls": _mean(postagent_calls),
        "postagent_calls_total": int(sum(postagent_calls)),
        "avg_verifier_calls": _mean(verifier_calls),
        "verifier_calls_total": int(sum(verifier_calls)),
        "verifier_sufficient_rate": _mean(verifier_sufficient),
        "verifier_unparsed": verifier_unparsed,
        "avg_search_seconds": _mean(
            (result.get("timings") or {}).get("search_s") for result in results),
        "avg_answer_seconds": _mean(
            (result.get("timings") or {}).get("answer_s") for result in results),
        "avg_eval_seconds": _mean(
            (result.get("timings") or {}).get("eval_s") for result in results),
    }
    embedding = Counter()
    for item in search:
        counts = (item.get("coarse") or {}).get("embedding") or {}
        for key in ("cache_hits", "cache_misses", "embedded"):
            value = counts.get(key)
            if value is not None:
                embedding[key] += int(value)
    if embedding:
        summary["embedding"] = dict(embedding)
    tokens = _token_totals(results)
    if tokens:
        summary["tokens"] = tokens
    return {key: value for key, value in summary.items()
            if value is not None and value != {}}


def summarize(results: list[dict[str, Any]], *, config: dict[str, Any] | None = None,
              elapsed_seconds: float | None = None) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        dataset = str(result.get("dataset") or "unknown")
        answer_type = str(result.get("answer_type") or "unknown")
        by_dataset[dataset].append(result)
        by_type[f"{dataset}/{answer_type}"].append(result)

    by_dataset_summary = {
        name: summarize_group(items) for name, items in sorted(by_dataset.items())
    }
    dataset_scores = [item["score"] for item in by_dataset_summary.values()
                      if item.get("score") is not None]
    summary = {
        "overall": summarize_group(results),
        "macro_score": _mean(dataset_scores),
        "by_dataset": by_dataset_summary,
        "by_answer_type": {
            name: summarize_group(items) for name, items in sorted(by_type.items())
        },
        "elapsed_seconds": round(elapsed_seconds, 1) if elapsed_seconds else None,
    }
    if config is not None:
        summary["config_path"] = config.get("path")
        summary["experiment"] = config.get("experiment")
        summary["answer"] = {
            key: config.get("answer", {}).get(key)
            for key in ("enabled", "max_evidence_chars", "max_tokens")
        }
        summary["answer"]["model"] = (config.get("answer_generator") or {}).get("model")
        summary["eval"] = {
            key: config.get("eval", {}).get(key)
            for key in ("enabled", "mode", "judge", "rubric", "primary_by_dataset")
        }
        summary["eval"]["model"] = (config.get("eval_generator") or {}).get("model")
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Benchmark run summary", ""]
    overall = summary.get("overall") or {}
    lines.append(f"- QA records: **{overall.get('n', 0)}**")
    if summary.get("macro_score") is not None:
        lines.append(f"- macro score (per-dataset mean): **{_fmt(summary['macro_score'])}**")
    for key in ("score", "token_f1", "mcq_exact", "judge_correct", "rubric_score",
                "sufficient_rate", "avg_steps", "avg_search_seconds",
                "avg_answer_seconds", "avg_eval_seconds", "search_errors",
                "eval_errors"):
        if key in overall:
            lines.append(f"- {key}: {_fmt(overall[key])}")
    embedding = overall.get("embedding") or {}
    if embedding:
        lines.append(
            f"- embedding cache: hits {embedding.get('cache_hits', 0)}, "
            f"misses {embedding.get('cache_misses', 0)}, "
            f"embedded {embedding.get('embedded', 0)}")
    tokens = overall.get("tokens") or {}
    if tokens:
        lines.append(
            f"- tokens: total **{tokens.get('total_tokens', 0)}** "
            f"(embedding {((tokens.get('embedding') or {}).get('total_tokens', 0))}, "
            f"postagent {((tokens.get('postagent') or {}).get('total_tokens', 0))}, "
            f"verifier {((tokens.get('verifier') or {}).get('total_tokens', 0))}, "
            f"answer {((tokens.get('answer') or {}).get('total_tokens', 0))}, "
            f"eval {((tokens.get('eval') or {}).get('total_tokens', 0))})")
    if summary.get("elapsed_seconds"):
        lines.append(f"- elapsed: {summary['elapsed_seconds']} s")
    if overall.get("status"):
        lines.append("- status: " + ", ".join(
            f"{key}={value}" for key, value in sorted(overall["status"].items())))
    lines.append("")

    lines.append("## By dataset")
    lines.append("")
    lines.append("| dataset | n | score | f1 | mcq | judge | rubric | sufficient | steps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in (summary.get("by_dataset") or {}).items():
        lines.append(
            f"| {name} | {item.get('n', 0)} | {_fmt(item.get('score'))} | "
            f"{_fmt(item.get('token_f1'))} | {_fmt(item.get('mcq_exact'))} | "
            f"{_fmt(item.get('judge_correct'))} | {_fmt(item.get('rubric_score'))} | "
            f"{_fmt(item.get('sufficient_rate'))} | {_fmt(item.get('avg_steps'))} |"
        )
    lines.append("")

    lines.append("## Token usage by dataset")
    lines.append("")
    lines.append("| dataset | embedding | postagent | verifier | answer | eval | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, item in (summary.get("by_dataset") or {}).items():
        usage = item.get("tokens") or {}
        def _t(key: str) -> int:
            return int(((usage.get(key) or {}).get("total_tokens", 0)) or 0)
        lines.append(
            f"| {name} | {_t('embedding')} | {_t('postagent')} | {_t('verifier')} | "
            f"{_t('answer')} | {_t('eval')} | {usage.get('total_tokens', 0)} |")
    lines.append("")

    lines.append("## By dataset / answer type")
    lines.append("")
    lines.append("| dataset/type | n | score | f1 | mcq | judge | rubric |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, item in (summary.get("by_answer_type") or {}).items():
        lines.append(
            f"| {name} | {item.get('n', 0)} | {_fmt(item.get('score'))} | "
            f"{_fmt(item.get('token_f1'))} | {_fmt(item.get('mcq_exact'))} | "
            f"{_fmt(item.get('judge_correct'))} | {_fmt(item.get('rubric_score'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(render_markdown(summary), encoding="utf-8")
