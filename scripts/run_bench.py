#!/usr/bin/env python3
"""Run the processed benchmark datasets end to end.

    python scripts/run_bench.py --config configs/bench.yaml
    python scripts/run_bench.py --config configs/bench.yaml \
        --datasets beam_100k,clbench --max-users 2 --limit-queries 1 --mock

The pipeline is: replay ``add``/``search`` events in order -> answer from the
returned evidence -> evaluate against the gold answer/rubrics. See
``docs/07-benchmark.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem.bench.config import (  # noqa: E402
    config_json,
    load_bench_config,
    normalize_datasets,
    project_path,
)
from asmem.bench.pipeline import Experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay benchmark add/search events, answer, and evaluate.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "bench.yaml"))
    parser.add_argument("--datasets", default="",
                        help="comma list; overrides experiment.datasets")
    parser.add_argument("--max-users", type=int, default=-1,
                        help="per dataset; overrides experiment.max_users")
    parser.add_argument("--limit-queries", type=int, default=-1,
                        help="per user; overrides experiment.limit_queries_per_user")
    parser.add_argument("--run-name", default="",
                        help="overrides experiment.run_name")
    parser.add_argument("--postagent-mode", default="",
                        choices=["", "loop", "compile", "fixed", "pipeline",
                                 "gate_loop", "evidence_loop"],
                        help="overrides postagent_mode (search stage)")
    parser.add_argument("--retrieval-mode", default="",
                        choices=["", "hybrid", "coarse", "agent"],
                        help="overrides retrieval.mode")
    parser.add_argument("--out-dir", default="",
                        help="overrides experiment.out_dir")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing run dir before starting")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore predictions.jsonl and start fresh")
    parser.add_argument("--skip-add", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-answer", action="store_true",
                        help="reuse cached search evidence but re-run answer/eval")
    parser.add_argument("--force-eval", action="store_true",
                        help="reuse cached search+answer but re-run eval only")
    parser.add_argument("--force-search", action="store_true",
                        help="ignore cached evidence and re-run search")
    parser.add_argument("--reuse-coarse-from", default="",
                        help="run name or path: reuse its deterministic coarse "
                             "candidates, then re-run postagent/verifier/answer")
    parser.add_argument("--no-reuse-coarse", action="store_true",
                        help="ignore experiment.reuse_coarse_from and compute coarse")
    parser.add_argument("--search-only", action="store_true",
                        help="same as --skip-answer --skip-eval")
    parser.add_argument("--coarse-only", action="store_true",
                        help="run only deterministic coarse recall, save the "
                             "candidate pool, and skip postagent/answer/eval; "
                             "use the run name later as --reuse-coarse-from")
    parser.add_argument("--mock", action="store_true",
                        help="no model calls; deterministic mock search/answer/eval")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the resolved plan without running")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved config and exit")
    parser.add_argument("--verbose", action="store_true")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    exp = config["experiment"]
    if args.datasets:
        exp["datasets"] = normalize_datasets(args.datasets)
    if args.max_users >= 0:
        exp["max_users"] = args.max_users
    if args.limit_queries >= 0:
        exp["limit_queries_per_user"] = args.limit_queries
    if args.run_name:
        exp["run_name"] = args.run_name
    if args.postagent_mode:
        config["search"]["postagent_mode"] = args.postagent_mode
    if args.retrieval_mode:
        config["search"].setdefault("retrieval", {})["mode"] = args.retrieval_mode
    if args.out_dir:
        exp["out_dir"] = args.out_dir
    if args.overwrite:
        exp["overwrite"] = True
    if args.no_resume:
        exp["resume"] = False
    if args.skip_add:
        exp["skip_add"] = True
    if args.skip_search:
        exp["skip_search"] = True
    if args.skip_answer:
        exp["skip_answer"] = True
    if args.skip_eval:
        exp["skip_eval"] = True
    if args.force_answer:
        exp["force_answer"] = True
    if args.force_eval:
        exp["force_eval"] = True
    if args.force_search:
        exp["force_search"] = True
    if args.no_reuse_coarse:
        exp["reuse_coarse_from"] = ""
    if args.reuse_coarse_from:
        exp["reuse_coarse_from"] = args.reuse_coarse_from
    if args.search_only:
        exp["skip_answer"] = True
        exp["skip_eval"] = True
    if args.coarse_only:
        # Deterministic coarse cache: one run computes and saves every
        # query's candidate pool; later loop runs reuse it.
        exp["skip_answer"] = True
        exp["skip_eval"] = True
        exp["reuse_coarse_from"] = ""
        config.setdefault("search", {}).setdefault(
            "retrieval", {})["mode"] = "coarse"
        config["search"]["postagent_mode"] = "loop"
    if args.mock:
        exp["mock"] = True
    if args.verbose:
        config["search"].setdefault("log", {})
        config["search"]["log"]["level"] = "info"


def plan(config: dict) -> dict:
    data_dir = project_path(config["experiment"]["data_dir"])
    datasets = []
    for name in config["experiment"]["datasets"]:
        path = data_dir / f"{name}.jsonl"
        item = {"dataset": name, "path": str(path), "exists": path.exists()}
        if path.exists():
            item["bytes"] = path.stat().st_size
            with open(path, encoding="utf-8") as handle:
                item["users"] = sum(1 for _ in handle)
        datasets.append(item)
    return {"config": json.loads(config_json(config)),
            "run_dir": str(project_path(config["experiment"]["out_dir"])
                           / (config["experiment"].get("run_name") or "<timestamp>")),
            "datasets": datasets}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        config = load_bench_config(args.config)
    except FileNotFoundError:
        print(f"config not found: {args.config}", file=sys.stderr)
        raise SystemExit(2)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    apply_overrides(config, args)

    if args.print_config:
        print(config_json(config))
        return

    if args.dry_run:
        print(json.dumps(plan(config), ensure_ascii=False, indent=2))
        return

    experiment = Experiment(config)
    try:
        summary = asyncio.run(experiment.run())
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\ninterrupted; completed results are in predictions.jsonl", file=sys.stderr)
        raise SystemExit(130)

    overall = summary.get("overall") or {}
    print(json.dumps({
        "run_dir": str(experiment.run_dir),
        "n": overall.get("n"),
        "score": overall.get("score"),
        "token_f1": overall.get("token_f1"),
        "mcq_exact": overall.get("mcq_exact"),
        "judge_correct": overall.get("judge_correct"),
        "rubric_score": overall.get("rubric_score"),
        "status": overall.get("status"),
        "summary": str(experiment.run_dir / "summary.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
