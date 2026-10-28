"""Async benchmark pipeline: replay -> search -> answer -> eval.

Parallelism has three layers:

- ``user_concurrency`` users are replayed at once; each user's events stay in
  order (streaming searches must see the memory state at that point);
- ``search_concurrency`` caps concurrent search-agent runs;
- ``answer_concurrency`` / ``eval_concurrency`` cap the answer and judge calls.

Within a user, answer/eval runs as a background task, so the next add/search can
start while the previous QA is being answered. ``pending_per_user`` bounds that
backlog.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from .. import llm
from ..api.store import slug
from ..io import now_iso
from ..log import Logger
from ..search import runner
from .answer import Answerer, TextModel
from .config import config_json, load_bench_config, project_path
from .judge import Judge, looks_like_refusal
from .replay import FastMemory
from .report import summarize, write_report


_LIST_QUESTION = re.compile(
    r"^(?:what|which|who|list|name)\b.*\b(?:items|books|pets|artists|bands|"
    r"places|things|activities|hobbies|movies|films|games|countries|cities|"
    r"songs|events|projects|awards|sports|foods|dishes|people|friends|"
    r"jobs|trips|purchases|gifts)\b|^how\s+many\b",
    re.IGNORECASE)


def is_list_question(question: str, answer_type: str = "",
                     intent: str = "") -> bool:
    """True for enumeration/count questions.

    These are the queries where the v0.8/0.9 calibration still lost most to
    incomplete enumeration (103 of 261 failed-set errors); routing only this
    shape to a stronger answerer is cheap and does not disturb single-fact
    questions.
    """
    if _LIST_QUESTION.search(question or ""):
        return True
    return str(intent or "").lower() in {"list", "aggregate"}


_STRONG_INTENTS = {
    "multi_hop", "temporal", "duration", "yesno", "conflict", "update",
    "aggregate", "order", "compare", "list", "explain", "preference",
}


def uses_strong_answerer(question: str, answer_type: str = "",
                         intent: str = "",
                         sufficient: bool | None = None) -> bool:
    """Shapes where a stronger answerer pays for itself.

    The route is based on the **intent router** (semantic, dataset-agnostic),
    not on dataset-specific answer labels:

    - reasoning-heavy intents (multi-hop, temporal, list, yes/no, conflict,
      comparison, aggregation, explanation) use the strong answerer;
    - fact/entity/unknown intents use it only when the verifier says the
      evidence is insufficient (a verification-gated escalation);
    - explicit enumeration/count questions are always strong.

    The v1.1 calibration on LoCoMo-Refined showed this split recovers the
    budget/compiler gains without the verbosity penalty the strong model
    incurs on already-sufficient single-fact questions.
    """
    name = str(intent or "").strip().lower()
    if name in _STRONG_INTENTS:
        return True
    if is_list_question(question):
        return True
    return sufficient is False


class ModelView:
    """Per-search view over a shared ``ModelRunner`` (local call counter)."""

    def __init__(self, shared: runner.ModelRunner) -> None:
        self.shared = shared
        self.calls = 0
        # Per-search tracker: concurrent searches share the client but not the
        # token counters, so per-query stats stay attributable.
        self.usage = llm.UsageTracker()

    @property
    def schemas(self) -> Any:
        return self.shared.schemas


    async def call(self, messages: list[dict[str, Any]],
                   schemas: list[dict[str, Any]] | None = None,
                   usage: Any = None, deadline: float | None = None) -> Any:
        self.calls += 1
        return await self.shared.call(messages, schemas, usage=self.usage,
                                      deadline=deadline)

    async def text(self, messages: list[dict[str, Any]],
                   usage: Any = None, deadline: float | None = None) -> str:
        self.calls += 1
        return await self.shared.text(messages, usage=self.usage,
                                      deadline=deadline)

    async def aclose(self) -> None:
        # The shared model owns the client.
        return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def mock_search(records: list[dict[str, Any]], question: str,
                top_k: int = 3) -> dict[str, Any]:
    """Deterministic stand-in for the code agent (``experiment.mock: true``)."""
    if not records:
        return {
            "evidence": [],
            "search": {"steps": 0, "sufficient": False, "missing": [],
                       "evidence": 0, "mock": True},
        }
    query_words = _words(question)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        overlap = len(query_words & _words(str(record.get("text") or "")))
        scored.append((overlap, -index, record))
    scored.sort(reverse=True)
    picked = [record for overlap, _, record in scored[:top_k] if overlap > 0]
    if not picked:
        picked = [record for _, _, record in scored[:top_k]]
    evidence = [
        {
            "content": str(record.get("text") or ""),
            "metadata": {"source": [str(record.get("id") or "")], "score": 1.0},
        }
        for record in picked
    ]
    return {
        "evidence": evidence,
        "search": {"steps": 0, "sufficient": bool(evidence), "missing": [],
                   "evidence": len(evidence), "mock": True},
    }


_FAILED_STATUSES = {
    "search_failed", "answer_failed", "eval_failed", "internal_error",
}


class QAProgress:
    """Live QA-level progress: completed count, running accuracy, ETA.

    One unit is one ``search`` event (a QA), not one user. The bar is seeded
    with already-complete cached results so a resume starts at the real point.
    """

    def __init__(self, total: int | None, seeded: list[dict[str, Any]] | None = None):
        self.results: dict[str, dict[str, Any]] = {}
        self.scored = 0
        self.correct = 0.0
        self.failed = 0
        for result in seeded or []:
            self._account(result, 1)
            qid = str(result.get("query_id") or "")
            if qid:
                self.results[qid] = result
        self.total = total
        if total is not None:
            self.total = max(total, len(self.results))
        self.bar: Any = None
        try:
            from tqdm import tqdm
            self.bar = tqdm(
                total=self.total,
                initial=len(self.results),
                desc="QA",
                unit="qa",
                dynamic_ncols=True,
            )
            self._refresh()
        except ImportError:  # pragma: no cover
            self.bar = None

    def _account(self, result: dict[str, Any], sign: int) -> None:
        evaluation = result.get("eval") or {}
        score = evaluation.get("score")
        if score is not None:
            self.scored += sign
            self.correct += sign * float(score)
        if str(result.get("status") or "") in _FAILED_STATUSES:
            self.failed += sign

    def add(self, result: dict[str, Any]) -> None:
        qid = str(result.get("query_id") or "")
        old = self.results.get(qid)
        if old is not None:
            self._account(old, -1)
        self.results[qid] = result
        self._account(result, 1)
        if self.bar is not None:
            self.bar.n = len(self.results)
            if self.total is not None and self.bar.n > self.bar.total:
                self.bar.total = self.bar.n
            self._refresh()

    def _refresh(self) -> None:
        if self.bar is None:
            return
        finished = len(self.results)
        unscored = finished - self.scored
        if self.scored:
            accuracy = self.correct / self.scored
            postfix = (f"acc={accuracy:.4f} ({self.correct:.1f}/{self.scored}) "
                       f"fail={self.failed} unscored={unscored}")
        else:
            postfix = f"acc=- scored=0 fail={self.failed} unscored={unscored}"
        self.bar.set_postfix_str(postfix)

    def set_dataset(self, name: str) -> None:
        if self.bar is not None:
            self.bar.set_description(f"{name} QA")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class Experiment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.exp = config["experiment"]
        self.search_cfg = config["search"]
        self.eval_required = (
            bool(config["eval"].get("enabled"))
            and not self.exp.get("skip_eval")
            and str(config["eval"].get("mode") or "auto").lower() != "none"
        )
        self.data_dir = project_path(self.exp["data_dir"])
        self.out_dir = project_path(self.exp["out_dir"])
        cache_root = str(self.exp.get("index_cache_dir") or "").strip()
        self.index_cache_dir = (
            project_path(cache_root) if cache_root
            else self.out_dir / "_index_cache")
        run_name = str(self.exp.get("run_name") or "").strip()
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        self.exp["run_name"] = self.run_name
        self.run_dir = self.out_dir / self.run_name
        self.memory_dir = self.run_dir / "memory"
        self.search_dir = self.run_dir / "search"
        self.log_dir = self.run_dir / "logs"
        self.predictions_path = self.run_dir / "predictions.jsonl"
        self.search_cache_path = self.run_dir / "search_cache.jsonl"

        self.cache: dict[str, dict[str, Any]] = {}
        self.search_partials: dict[str, dict[str, Any]] = {}
        self.write_lock: asyncio.Lock | None = None
        self.writer: Any = None
        self.search_writer: Any = None
        self.search_sem: asyncio.Semaphore | None = None
        self.answer_sem: asyncio.Semaphore | None = None
        self.eval_sem: asyncio.Semaphore | None = None

        self.search_model: runner.ModelRunner | None = None
        self.answer_model: TextModel | None = None
        self.eval_model: TextModel | None = None
        self.answerer: Answerer | None = None
        self.judge: Judge | None = None
        self.search_log: Logger | None = None
        self.log: Logger | None = None
        self.progress: QAProgress | None = None
        # Optional previous run whose deterministic coarse stage is reused.
        self.reuse_coarse_from = str(self.exp.get("reuse_coarse_from") or "").strip()
        self.coarse_index: dict[str, Path] = {}
        # Liveness counters for the heartbeat.  Without them a run that is
        # slowly chewing through long documents looks identical to a hung one.
        self.search_inflight = 0
        self.answer_inflight = 0
        self.search_started: dict[str, float] = {}
        self.answer_started: dict[str, float] = {}
        self.heartbeat_task: asyncio.Task | None = None

    # -- setup -------------------------------------------------------------
    def _prepare(self) -> None:
        if self.run_dir.exists():
            if self.exp.get("overwrite"):
                shutil.rmtree(self.run_dir)
            elif not self.exp.get("resume"):
                raise FileExistsError(
                    f"{self.run_dir} exists; set experiment.overwrite=true, "
                    "experiment.resume=true, or choose another run_name")
        for path in (self.run_dir, self.memory_dir, self.search_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

        (self.run_dir / "config.used.json").write_text(
            config_json(self.config), encoding="utf-8")
        (self.run_dir / "config.used.yaml").write_text(
            yaml.safe_dump(json.loads(config_json(self.config)),
                           allow_unicode=True, sort_keys=False),
            encoding="utf-8")

        if self.exp.get("resume") and self.search_cache_path.exists():
            for line in self.search_cache_path.read_text(
                    encoding="utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query_id = str(result.get("query_id") or "")
                if query_id:
                    self.search_partials[query_id] = result
        if self.exp.get("resume") and self.predictions_path.exists():
            for line in self.predictions_path.read_text(
                    encoding="utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query_id = str(result.get("query_id") or "")
                if query_id:
                    # Final predictions always win over a cached search-only
                    # snapshot for the same query_id.
                    self.cache[query_id] = result
        mode = "a" if self.exp.get("resume") and self.predictions_path.exists() else "w"
        search_mode = "a" if self.exp.get("resume") and self.search_cache_path.exists() else "w"
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = self.predictions_path.open(mode, encoding="utf-8")
        self.search_writer = self.search_cache_path.open(search_mode, encoding="utf-8")

        log_config = self.config["search"].get("log") or {}
        self.search_log = Logger.from_config(
            {"log": {"level": log_config.get("level", "warn"),
                     "output": str(self.log_dir)}},
            prefix="search",
        )
        self.log = Logger.from_config(
            {"log": {"level": str(self.exp.get("log_level") or "info"),
                     "output": str(self.log_dir)}},
            prefix="bench",
        )

    def _setup_models(self) -> None:
        mock = bool(self.exp.get("mock"))
        self.search_cfg = self.config["search"]
        self.search_cfg.setdefault("log", {})
        self.search_cfg["log"]["output"] = str(self.log_dir)
        self.search_sem = asyncio.Semaphore(self.exp["search_concurrency"])
        self.answer_sem = asyncio.Semaphore(self.exp["answer_concurrency"])
        self.eval_sem = asyncio.Semaphore(self.exp["eval_concurrency"])
        self.write_lock = asyncio.Lock()

        if not mock and not self.exp.get("skip_search"):
            self.search_model = runner.ModelRunner(self.search_cfg)
        if (self.config["answer"].get("enabled") and not mock
                and not self.exp.get("skip_answer")):
            self.answer_model = TextModel(
                self.config["answer_generator"], role="answer")
        if (self.config["eval"].get("enabled") and not mock
                and not self.exp.get("skip_eval")):
            self.eval_model = TextModel(
                self.config["eval_generator"], role="eval")
        self.answerer = Answerer(self.config["answer"], self.answer_model, mock=mock)
        self.judge = Judge(self.config["eval"], self.eval_model, mock=mock)
        # Optional stronger answerer used only when the small model refuses
        # despite non-empty evidence: a refusal is cheap to detect and the
        # v0.8 calibration recovered ~27% of them.
        self.escalation_model = None
        self.escalation_answerer = None
        escalation = (self.config["answer"].get("escalation_generator")
                      or self.config.get("answer_escalation_generator"))
        if (escalation and self.config["answer"].get("escalate_on_refusal", True)
                and not mock and not self.exp.get("skip_answer")):
            self.escalation_model = TextModel(escalation, role="answer")
            self.escalation_answerer = Answerer(
                self.config["answer"], self.escalation_model)
        # Enumeration/count and multi-hop/temporal/inference questions get a
        # stronger answerer: v1.1 calibration showed +37 net correct on
        # category_1/2/3 failed queries on the same 48k evidence.
        self.list_model = None
        self.list_answerer = None
        strong_generator = (self.config["answer"].get("strong_generator")
                            or self.config["answer"].get("list_generator"))
        if (strong_generator and not mock
                and not self.exp.get("skip_answer")):
            self.list_model = TextModel(strong_generator, role="answer")
            self.list_answerer = Answerer(
                self.config["answer"], self.list_model)

    async def _close_models(self) -> None:
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
        for model in (self.answer_model, self.eval_model,
                      self.escalation_model, self.list_model):
            if model is not None:
                try:
                    await model.aclose()
                except Exception:  # noqa: BLE001
                    pass
        if self.search_model is not None:
            try:
                await self.search_model.aclose()
            except Exception:  # noqa: BLE001
                pass
        for logger in (self.search_log, self.log):
            if logger is not None:
                logger.close()
        if self.progress is not None:
            self.progress.close()
        for writer in (self.writer, self.search_writer):
            if writer is not None:
                writer.close()

    # -- progress ----------------------------------------------------------
    def _estimate_qa_total(self) -> int | None:
        """Estimate the number of QA/search events for the global progress bar.

        The processed ``manifest.json`` gives exact users/search counts, so a
        full run is exact. With ``--max-users`` / ``--limit-queries`` the total
        is an estimate (clamped by the per-dataset average); the bar/ETA adjust
        as the actual count is reached.
        """
        if not self.data_dir.exists():
            return None
        manifest: dict[str, Any] = {}
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        entries = manifest.get("datasets") or {}
        max_users = int(self.exp.get("max_users") or 0)
        limit = int(self.exp.get("limit_queries_per_user") or 0)
        total = 0.0
        unknown = False

        for name in self.exp["datasets"]:
            path = self.data_dir / f"{name}.jsonl"
            if not path.exists():
                continue
            entry = entries.get(name)
            if entry:
                users = int(entry.get("users") or 0)
                searches = int(entry.get("search_events") or 0)
            elif path.stat().st_size <= 50_000_000:
                users = searches = 0
                try:
                    with open(path, encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            users += 1
                            searches += int((record.get("stats") or {}).get("search_events") or 0)
                except (OSError, json.JSONDecodeError):
                    unknown = True
                    continue
            else:
                # Avoid parsing a multi-GB file just to size the progress bar.
                unknown = True
                continue

            if users <= 0:
                continue
            if max_users:
                # The number of users is small in a debug run; read their stats
                # for an exact total even when a per-user query limit is set.
                exact = 0
                seen = 0
                try:
                    with open(path, encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            count = int((record.get("stats") or {}).get("search_events") or 0)
                            exact += min(count, limit) if limit else count
                            seen += 1
                            if seen >= max_users:
                                break
                except (OSError, json.JSONDecodeError):
                    exact = 0
                if exact:
                    total += exact
                    continue
            selected_users = min(users, max_users) if max_users else users
            average = searches / users
            per_user = min(limit, average) if limit else average
            total += selected_users * per_user

        if unknown or total <= 0:
            return None
        return max(1, int(round(total)))

    def _init_progress(self) -> None:
        total = self._estimate_qa_total()
        force = bool(self.exp.get("force_answer")
                     or self.exp.get("force_eval")
                     or self.exp.get("force_search"))
        seeded = []
        selected = set(self.exp["datasets"])
        if not force:
            seeded = [
                result for result in self.cache.values()
                if str(result.get("dataset") or "") in selected
                and self._is_complete_for_config(result)
            ]
        self.progress = QAProgress(total, seeded)

    # -- coarse reuse ------------------------------------------------------
    @staticmethod
    def _source_inputs(source: Path, run_dir: str) -> Path | None:
        """Resolve a source run's ``inputs`` dir from a recorded ``run_dir``.

        Support both layouts: v2 flat ``<run>/inputs`` and v1
        ``<run>/qa/inputs``.  The recorded path may come from another host or
        an older run-directory name, so only the part after ``/search/`` is
        used and re-rooted at the local source run.
        """
        candidates: list[Path] = []
        raw = Path(run_dir)
        if raw.exists():
            candidates.extend([raw / "inputs", raw / "qa" / "inputs", raw])
        marker = "/search/"
        if marker in run_dir:
            tail = run_dir.split(marker, 1)[1]
            base = source / "search" / tail
            candidates.extend([base / "inputs", base / "qa" / "inputs", base])
        try:
            local_tail = raw.relative_to("/").as_posix()
        except ValueError:
            local_tail = run_dir.lstrip("/")
        candidates.append(source / local_tail)
        for candidate in candidates:
            if ((candidate / "candidates.jsonl").exists()
                    and (candidate / "coarse.json").exists()):
                return candidate
        return None

    def _load_coarse_index(self) -> None:
        """Index deterministic coarse outputs for reuse.

        By default the *current* run's own previous attempt is the source:
        coarse is deterministic, so an interrupted run (or a restart with the
        same run_name) should not recompute candidates.  Setting
        ``reuse_coarse_from`` points at another run instead.
        """
        explicit = bool(self.reuse_coarse_from)
        raw_text = self.reuse_coarse_from or str(self.run_dir)
        raw = Path(raw_text).expanduser()
        if not raw.is_absolute():
            local = self.out_dir / raw_text
            raw = local if local.exists() else project_path(raw_text)
        source = raw.resolve()
        if not source.exists():
            self.log.warn(f"coarse reuse source {source} does not exist")
            return
        mapping: dict[str, Path] = {}
        for name in ("predictions.jsonl", "search_cache.jsonl"):
            path = source / name
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query_id = str(record.get("query_id") or "")
                run_dir = str(((record.get("search") or {}).get("run_dir")
                               or record.get("run_dir") or ""))
                if not query_id or not run_dir:
                    continue
                inputs = self._source_inputs(source, run_dir)
                if inputs is not None:
                    mapping[query_id] = inputs
        self.coarse_index = mapping
        origin = self.reuse_coarse_from or "current run"
        if mapping:
            self.log.info(
                f"coarse reuse: {len(mapping)} queries from {origin} "
                f"(postagent/verifier/answer will re-run)")
        elif explicit:
            self.log.warn(
                f"reuse_coarse_from={source}: no candidates.jsonl/coarse.json found")

    def _start_heartbeat(self) -> None:
        seconds = float(self.exp.get("heartbeat_seconds", 60) or 0)
        if seconds <= 0 or self.log is None:
            return
        self.heartbeat_task = asyncio.create_task(self._heartbeat(seconds))

    async def _heartbeat(self, seconds: float) -> None:
        warn_after = float(self.exp.get("qa_warn_seconds", 300) or 0)
        while True:
            await asyncio.sleep(seconds)
            done = len(self.progress.results) if self.progress else 0
            total = self.progress.total if self.progress else None
            now = time.monotonic()
            slow: list[str] = []
            for label, started in (("search", self.search_started),
                                   ("answer", self.answer_started)):
                if not warn_after:
                    continue
                for query_id, begin in started.items():
                    elapsed = now - begin
                    if elapsed >= warn_after:
                        slow.append(f"{label}:{query_id}={elapsed:.0f}s")
            slow.sort(key=lambda item: -float(item.rsplit("=", 1)[-1].rstrip("s")))
            suffix = ("; slow=" + ", ".join(slow[:3])) if slow else ""
            self.log.info(
                f"heartbeat: done={done}/{total} "
                f"searches={self.search_inflight} answers={self.answer_inflight}"
                f"{suffix}")

    # -- main --------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            self._prepare()
            self._load_coarse_index()
            self._setup_models()
            self._init_progress()
            self._start_heartbeat()
            for dataset in self.exp["datasets"]:
                path = self.data_dir / f"{dataset}.jsonl"
                if not path.exists():
                    self.log.warn(f"{dataset}: {path} does not exist; skipped")
                    continue
                try:
                    await self._run_dataset(dataset, path)
                except Exception as exc:  # noqa: BLE001 - one dataset must not kill the run
                    self.log.error(
                        f"{dataset}: dataset failed: {type(exc).__name__}: {exc}")
        finally:
            await self._close_models()
        results = list(self.cache.values())
        summary = summarize(results, config=self.config,
                            elapsed_seconds=time.perf_counter() - started)
        write_report(self.run_dir / "summary.json", summary)
        return summary

    async def _run_dataset(self, dataset: str, path: Path) -> None:
        max_users = int(self.exp.get("max_users") or 0)
        total = 0
        with open(path, encoding="utf-8") as handle:
            for _ in handle:
                total += 1
                if max_users and total >= max_users:
                    break
        if max_users:
            total = min(total, max_users)
        self.log.info(f"{dataset}: {total} users from {path}")
        if self.progress is not None:
            self.progress.set_dataset(dataset)

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=max(1, self.exp["user_concurrency"] * 4))

        async def producer() -> None:
            sent = 0
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if max_users and sent >= max_users:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    await queue.put(record)
                    sent += 1

        async def worker() -> None:
            while True:
                record = await queue.get()
                if record is None:
                    queue.task_done()
                    return
                try:
                    await self._process_user(dataset, record)
                except Exception as exc:  # noqa: BLE001
                    self.log.error(
                        f"{dataset}/{record.get('user_id')}: "
                        f"{type(exc).__name__}: {exc}")
                finally:
                    queue.task_done()

        tasks = [asyncio.create_task(worker())
                 for _ in range(self.exp["user_concurrency"])]
        producer_error: Exception | None = None
        try:
            await producer()
        except Exception as exc:  # noqa: BLE001
            producer_error = exc
        finally:
            # Always release the workers, even if reading the file failed.
            for _ in range(self.exp["user_concurrency"]):
                await queue.put(None)
        await asyncio.gather(*tasks)
        if producer_error is not None:
            raise producer_error
        self.log.info(f"{dataset}: done")

    # -- one user ----------------------------------------------------------
    async def _process_user(self, dataset: str, record: dict[str, Any]) -> None:
        user_id = str(record.get("user_id") or "unknown")
        events = record.get("events") or []
        search_events = [event for event in events if event.get("type") == "search"]
        limit = int(self.exp.get("limit_queries_per_user") or 0)
        if limit:
            search_events = search_events[:limit]
        if not search_events:
            return
        force_answer = bool(self.exp.get("force_answer"))
        force_eval = bool(self.exp.get("force_eval"))
        force_search = bool(self.exp.get("force_search"))
        if (not force_answer and not force_eval and not force_search
                and all(self._is_complete_for_config(
                    self.cache.get(str(event.get("query_id") or "")))
                        for event in search_events)):
            if not self.exp.get("save_memory"):
                shutil.rmtree(self.memory_dir / slug(user_id), ignore_errors=True)
            return
        stop_step = int(search_events[-1].get("step") or 0)

        # If every search already has reusable evidence, we never need the
        # memory at all (e.g. --force-answer, or a resume where only answer/eval
        # failed). This avoids replaying millions of LongMemEval-m adds.
        all_reusable = (not force_search) and all(
            (self._is_complete_for_config(cached)
             or (cached and self._can_reuse_search(
                 cached, force=force_answer or force_eval)))
            or (partial and self._can_reuse_search(
                partial, force=force_answer or force_eval))
            for cached, partial in (
                (self.cache.get(str(event.get("query_id") or "")),
                 self.search_partials.get(str(event.get("query_id") or "")))
                for event in search_events
            )
        )
        skip_adds = (bool(self.exp.get("skip_add"))
                     or (all_reusable and not self.exp.get("save_memory")))

        user_dir = self.memory_dir / slug(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        memory_path = user_dir / "memory.jsonl"
        memory = FastMemory(self.search_cfg)
        pending: list[asyncio.Task] = []
        add_batch: list[dict[str, Any]] = []

        def flush_adds() -> None:
            if add_batch:
                for add_event in add_batch:
                    memory.add(add_event)
                add_batch.clear()

        async def handle_search(event: dict[str, Any]) -> None:
            query_id = str(event.get("query_id") or "")
            cached = self.cache.get(query_id)
            if (self._is_complete_for_config(cached)
                    and not (force_answer or force_eval or force_search)):
                return
            stage = cached or self.search_partials.get(query_id)
            reuse_search = (
                not force_search
                and stage is not None
                and self._can_reuse_search(
                    stage, force=force_answer or force_eval)
            )
            reuse_predicted: str | None = None
            if reuse_search:
                # Skip the already-completed search. Reuse the answer too
                # unless the caller explicitly asked to re-answer or there is
                # no usable answer yet.
                search_info = stage.get("search") or {}
                evidence = stage.get("evidence") or []
                if not force_answer and str(stage.get("predicted") or "").strip():
                    reuse_predicted = str(stage["predicted"])
            else:
                # LongMemEval adds hundreds of sessions before its single
                # search; batch them in one worker thread instead of one
                # to_thread call per add.
                if add_batch:
                    await asyncio.to_thread(flush_adds)
                search_info, evidence = await self._do_search(
                    dataset, user_id, event, memory, memory_path)
                # Persist the expensive search result before answer/eval: a
                # crash here still leaves a resumable evidence snapshot.
                stage_result = self._make_result(
                    dataset, user_id, event, evidence, search_info,
                    predicted="", answer_error=None, evaluation={},
                    timings={"search_s": search_info.get("seconds")},
                    status="search_only")
                stage_result["stage"] = "search"
                await self._write_search_partial(stage_result)
            task = asyncio.create_task(
                self._answer_eval(
                    dataset, user_id, event, evidence, search_info,
                    reuse_predicted=reuse_predicted))
            pending.append(task)
            if len(pending) >= int(self.exp["pending_per_user"]):
                await pending.pop(0)

        add_events = [
            event for event in events
            if event.get("type") == "add"
            and int(event.get("step") or 0) <= stop_step
            and not skip_adds
        ]
        search_steps = [int(event.get("step") or 0) for event in search_events]
        first_search = min(search_steps)
        last_search = max(search_steps)
        # If no add falls *between* two searches, the memory is static for every
        # query (LongMemEval, LoCoMo, CL-bench, ScriptMem; trailing adds after
        # the last search do not matter). Run those searches concurrently
        # instead of serialising them per user.
        static_memory = not any(
            first_search < int(event.get("step") or 0) < last_search
            for event in add_events
        )

        try:
            if static_memory:
                add_batch.extend(add_events)
                if add_batch:
                    await asyncio.to_thread(flush_adds)
                if memory.records or not self.exp.get("skip_search_if_no_memory"):
                    await asyncio.to_thread(memory.write, memory_path)
                await asyncio.gather(*(handle_search(event) for event in search_events))
            else:
                for event in events:
                    if int(event.get("step") or 0) > stop_step:
                        break
                    if event.get("type") == "add":
                        if not skip_adds:
                            add_batch.append(event)
                        continue
                    if event.get("type") != "search":
                        continue
                    await handle_search(event)
                if add_batch:
                    await asyncio.to_thread(flush_adds)
            if self.exp.get("save_memory") and memory.dirty:
                await asyncio.to_thread(memory.write, memory_path)
        finally:
            # Even if an add/search raised, let the already-scheduled answer/
            # eval tasks finish before the user workspace is cleaned up.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if not self.exp.get("save_memory"):
                try:
                    memory_path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    user_dir.rmdir()
                except OSError:
                    pass

    async def _do_search(self, dataset: str, user_id: str,
                         event: dict[str, Any], memory: FastMemory,
                         memory_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        query = str(event.get("query") or "")
        options = [str(item) for item in (event.get("options") or []) if str(item).strip()]
        query_id = str(event.get("query_id") or "")
        if self.exp.get("skip_search"):
            return {"skipped": True, "steps": 0, "sufficient": None,
                    "evidence": 0}, []
        if (self.exp.get("skip_search_if_no_memory")
                and not memory.records):
            return {"skipped": "no_memory", "steps": 0, "sufficient": None,
                    "evidence": 0}, []

        started = time.perf_counter()
        if self.exp.get("mock"):
            out = mock_search(memory.records, query, top_k=3)
            search_info = dict(out["search"])
            search_info["seconds"] = round(time.perf_counter() - started, 3)
            return search_info, self._attach_record_times(
                out["evidence"], memory.records)

        if self.search_model is None:
            return {"error": "search model is not configured", "steps": 0,
                    "sufficient": False, "evidence": 0, "seconds": 0.0}, []
        if memory.dirty or not memory_path.exists():
            await asyncio.to_thread(memory.write, memory_path)
        search_timeout = float(self.search_cfg.get("search_timeout", 0) or 0)
        hard_timeout = float(self.search_cfg.get("search_hard_timeout", 0) or 0)
        if hard_timeout <= 0 and search_timeout > 0:
            hard_timeout = search_timeout + 60
        started = time.perf_counter()
        try:
            # Acquire the concurrency slot BEFORE starting the wall clock.
            # The old order let a task wait many minutes in the semaphore
            # queue and then begin with an already-expired search deadline,
            # which looked like a 40-minute "slow search" and could degrade
            # the result to a coarse fallback.
            async with self.search_sem:
                started = time.perf_counter()
                self.search_inflight += 1
                self.search_started[str(query_id)] = started
                view = ModelView(self.search_model)
                try:
                    call = runner.run_question(
                        memory_path=memory_path,
                        question=query,
                        options=options,
                        config=self.search_cfg,
                        run_root=self.search_dir / slug(user_id) / slug(query_id),
                        model=view,
                        log=(self.search_log.bind(slug(user_id))
                             if self.search_log is not None else None),
                        cache_dir=self.index_cache_dir / dataset / slug(user_id),
                        dataset=dataset,
                        answer_type=str(event.get("answer_type") or ""),
                        coarse_source=self.coarse_index.get(str(query_id)),
                    )
                    out = (await asyncio.wait_for(call, timeout=hard_timeout)
                           if hard_timeout > 0 else await call)
                finally:
                    self.search_inflight -= 1
                    self.search_started.pop(str(query_id), None)
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "steps": 0,
                "sufficient": False,
                "evidence": 0,
                "seconds": round(time.perf_counter() - started, 3),
            }, []
        summary = out.get("summary") or {}
        evidence = out.get("evidence") or []
        evidence = self._attach_record_times(evidence, memory.records)
        search_info = {
            "steps": summary.get("steps"),
            "stop_reason": summary.get("stop_reason"),
            "sufficient": summary.get("sufficient"),
            "missing": summary.get("missing") or [],
            "evidence": len(evidence),
            "evidence_mix": summary.get("evidence_mix"),
            "intent": summary.get("intent"),
            "audited": summary.get("audited"),
            "audit_missing": summary.get("audit_missing"),
            "final_chars": summary.get("final_chars"),
            "claims": summary.get("claims"),
            "copies": summary.get("copies"),
            "model_calls": view.calls,
            "tokens": summary.get("tokens"),
            "postagent_calls": summary.get("postagent_calls"),
            "postagent_invocations": summary.get("postagent_invocations"),
            "planner_calls": summary.get("planner_calls"),
            "verifier_calls": summary.get("verifier_calls"),
            "compile": summary.get("compile"),
            "verifier": summary.get("verifier"),
            "verifier_postagent": summary.get("verifier_postagent"),
            "run_dir": out.get("run_dir"),
            "route": summary.get("route"),
            "gate": summary.get("gate"),
            "gate_skipped": summary.get("gate_skipped"),
            "degraded": summary.get("degraded"),
            "resolution": summary.get("resolution"),
            "coarse": summary.get("coarse"),
            "coarse_reused": summary.get("coarse_reused"),
            "coarse_source": summary.get("coarse_source"),
            "seconds": round(time.perf_counter() - started, 3),
        }
        return search_info, evidence

    @staticmethod
    def _is_complete(cached: dict[str, Any] | None,
                     *, require_eval: bool = False) -> bool:
        """Final result is complete only if its search did not fail.

        Older runs could mark a search-error result as ``ok``; treating it as
        complete would skip exactly the work that needs to be redone. If eval
        is required, an ``ok`` without a score is also incomplete (e.g. a run
        made with ``--skip-eval``).
        """
        if not cached or cached.get("status") != "ok":
            return False
        if (cached.get("search") or {}).get("error"):
            return False
        if require_eval and (cached.get("eval") or {}).get("score") is None:
            return False
        return True

    def _is_complete_for_config(self, cached: dict[str, Any] | None) -> bool:
        gold = (cached or {}).get("gold") or {}
        has_gold = bool(gold.get("answer") or gold.get("answer_letters")
                        or gold.get("rubrics"))
        return self._is_complete(
            cached, require_eval=self.eval_required and has_gold)

    @staticmethod
    def _attach_record_times(evidence: list[dict[str, Any]],
                             records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the source record's observation time to evidence metadata.

        The search agent is told to write self-contained evidence, but relative
        dates ("yesterday", "last Saturday") are common in LoCoMo. Carrying the
        memory record's timestamp into the answer prompt makes those resolvable
        without changing the evidence content.
        """
        if not evidence or not records:
            return evidence
        by_id = {
            str(record.get("id")): record
            for record in records if record.get("id")
        }
        for item in evidence:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                item["metadata"] = metadata
            sources = metadata.get("source") or []
            if isinstance(sources, str):
                sources = [sources]
            for source in sources:
                record = by_id.get(str(source))
                if record and record.get("time"):
                    metadata["time"] = record["time"]
                    break
        return evidence

    @staticmethod
    def _can_reuse_search(cached: dict[str, Any], *, force: bool = False) -> bool:
        if cached.get("evidence") is None:
            return False
        if (cached.get("search") or {}).get("error"):
            return False
        if force:
            return True
        return cached.get("status") in {
            "search_only", "answer_only", "answer_failed", "eval_failed",
        }

    @staticmethod
    def _make_result(dataset: str, user_id: str, event: dict[str, Any],
                     evidence: list[dict[str, Any]], search_info: dict[str, Any],
                     *, predicted: str, answer_error: str | None,
                     evaluation: dict[str, Any], timings: dict[str, Any],
                     status: str, tokens: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "user_id": user_id,
            "query_id": str(event.get("query_id") or ""),
            "query": str(event.get("query") or ""),
            "answer_type": event.get("answer_type"),
            "status": status,
            "gold": {
                "answer": event.get("answer"),
                "answer_type": event.get("answer_type"),
                "answer_letters": event.get("answer_letters") or [],
                "rubrics": event.get("rubrics") or [],
                "options": event.get("options"),
                "answer_meta": event.get("answer_meta") or {},
            },
            "evidence": evidence,
            "predicted": predicted,
            "answer_error": answer_error,
            "search": search_info,
            "eval": evaluation,
            "timings": timings,
            "tokens": tokens or search_info.get("tokens") or {},
            "created_at": now_iso(),
        }

    async def _answer_eval(self, dataset: str, user_id: str,
                           event: dict[str, Any], evidence: list[dict[str, Any]],
                           search_info: dict[str, Any],
                           reuse_predicted: str | None = None) -> None:
        # A failed search is an infrastructure failure, not an answer. Do not
        # ask the answer/judge models (they would judge a refusal from empty
        # evidence); record it as a zero-credit failure instead.
        if search_info.get("error"):
            evaluation = {
                "mode": "search_failed",
                "primary": "search_failed",
                "score": 0.0,
                "metrics": {"search_failed": 1.0},
                "errors": [str(search_info.get("error"))],
                "judge_raw": "",
                "rubric_raw": "",
            }
            result = self._make_result(
                dataset, user_id, event, evidence, search_info,
                predicted="", answer_error=None, evaluation=evaluation,
                timings={"search_s": search_info.get("seconds")},
                status="search_failed")
            await self._write_result(result)
            return

        query = str(event.get("query") or "")
        gold = {
            "answer": event.get("answer"),
            "answer_type": event.get("answer_type"),
            "answer_letters": event.get("answer_letters") or [],
            "rubrics": event.get("rubrics") or [],
            "options": event.get("options"),
            "answer_meta": event.get("answer_meta") or {},
        }
        timings = {"search_s": search_info.get("seconds")}
        predicted = ""
        answer_error: str | None = None
        evaluation: dict[str, Any] = {}
        usage_tracker = llm.UsageTracker()
        usage_token = llm.CURRENT_USAGE.set(usage_tracker)
        live_id = str(event.get("query_id") or "")
        self.answer_inflight += 1
        self.answer_started[live_id] = time.monotonic()

        try:
            if reuse_predicted is not None:
                # The answer stage is already done; go straight to eval.
                predicted = reuse_predicted
            elif self.config["answer"].get("enabled") and not self.exp.get("skip_answer"):
                answer_budget = float(
                    self.config["answer"].get("total_timeout", 0) or 0)
                answerer = self.answerer
                no_strong = {
                    str(item) for item in
                    (self.config["answer"].get("no_strong_datasets") or [])
                }
                if (self.list_answerer is not None
                        and dataset not in no_strong
                        and uses_strong_answerer(
                            query, str(gold.get("answer_type") or ""),
                            str(search_info.get("intent") or ""),
                            search_info.get("sufficient"))):
                    answerer = self.list_answerer
                    timings["strong_answerer"] = True
                async with self.answer_sem:
                    started = time.perf_counter()
                    call = answerer.answer(
                        dataset=dataset, question=query, evidence=evidence, gold=gold)
                    try:
                        result = await asyncio.wait_for(
                            call, timeout=answer_budget + 30 if answer_budget > 0 else None)
                    except asyncio.TimeoutError:
                        result = {
                            "answer": "",
                            "error": (f"answer stage exceeded {answer_budget:.0f}s "
                                      "(including retries)"),
                        }
                predicted = str(result.get("answer") or "")
                answer_error = result.get("error")
                timings["answer_s"] = round(time.perf_counter() - started, 3)
                # Refusal verification: the small model sometimes says "not in
                # the evidence" while a record states the answer. Re-answer the
                # same evidence with the stronger model; keep it only when it
                # actually produces a non-refusal.
                if (self.escalation_answerer is not None and not answer_error
                        and predicted.strip() and evidence
                        and looks_like_refusal(predicted)):
                    esc_started = time.perf_counter()
                    esc_call = self.escalation_answerer.answer(
                        dataset=dataset, question=query, evidence=evidence,
                        gold=gold)
                    try:
                        esc_result = await asyncio.wait_for(
                            esc_call,
                            timeout=answer_budget + 30 if answer_budget > 0 else None)
                    except asyncio.TimeoutError:
                        esc_result = {"answer": "", "error": "escalation timeout"}
                    escalated = str(esc_result.get("answer") or "")
                    if (escalated.strip() and not esc_result.get("error")
                            and not looks_like_refusal(escalated)):
                        predicted = escalated
                        timings["answer_escalated"] = True
                    timings["answer_escalation_s"] = round(
                        time.perf_counter() - esc_started, 3)
                if (predicted.strip() and not answer_error
                        and self.config["eval"].get("enabled")
                        and not self.exp.get("skip_eval")):
                    # Persist the answer stage before eval: a crash during eval
                    # can resume with only eval to redo.
                    answer_stage = self._make_result(
                        dataset, user_id, event, evidence, search_info,
                        predicted=predicted, answer_error=None, evaluation={},
                        timings=dict(timings), status="answer_only")
                    answer_stage["stage"] = "answer"
                    await self._write_search_partial(answer_stage)

            if self.config["eval"].get("enabled") and not self.exp.get("skip_eval"):
                if answer_error and not predicted.strip():
                    # The answer stage already failed (timeout / context / model
                    # error).  Do not spend a judge call on an empty answer;
                    # record a zero so the result is complete and resumable.
                    evaluation = {
                        "mode": "answer_failed", "primary": "answer_failed",
                        "score": 0.0, "metrics": {"answer_failed": 1.0},
                        "errors": [str(answer_error)],
                        "judge_raw": "", "rubric_raw": "",
                    }
                else:
                    eval_budget = float(
                        self.config["eval"].get("total_timeout", 0) or 0)
                    async with self.eval_sem:
                        started = time.perf_counter()
                        call = self.judge.evaluate(
                            dataset=dataset, question=query,
                            predicted=predicted, gold=gold)
                        try:
                            evaluation = await asyncio.wait_for(
                                call, timeout=eval_budget + 30 if eval_budget > 0 else None)
                        except asyncio.TimeoutError:
                            evaluation = {
                                "mode": "auto", "primary": None, "score": None,
                                "metrics": {},
                                "errors": [f"eval stage exceeded {eval_budget:.0f}s "
                                           "(including retries)"],
                                "judge_raw": "", "rubric_raw": "",
                            }
                    timings["eval_s"] = round(time.perf_counter() - started, 3)
        except Exception as exc:  # noqa: BLE001 - never lose the QA record
            evaluation.setdefault("errors", []).append(
                f"pipeline: {type(exc).__name__}: {exc}")
        finally:
            self.answer_inflight -= 1
            self.answer_started.pop(live_id, None)
            llm.CURRENT_USAGE.reset(usage_token)

        usage = usage_tracker.as_dict()
        roles = usage.get("by_role") or {}
        answer_usage = roles.get("answer") or {}
        eval_usage = roles.get("eval") or {}
        tokens = dict(search_info.get("tokens") or {})
        tokens["answer"] = answer_usage
        tokens["eval"] = eval_usage
        tokens["total_tokens"] = (
            int(tokens.get("total_tokens", 0) or 0)
            + int(answer_usage.get("total_tokens", 0) or 0)
            + int(eval_usage.get("total_tokens", 0) or 0))

        has_gold = bool(gold.get("answer") or gold.get("answer_letters")
                        or gold.get("rubrics"))
        eval_mode = str(self.config["eval"].get("mode") or "auto").lower()
        eval_enabled = bool(self.config["eval"].get("enabled")) and not self.exp.get("skip_eval")
        eval_expected = eval_enabled and eval_mode != "none" and has_gold
        answer_done = bool(predicted.strip()) and not answer_error
        eval_done = eval_expected and evaluation.get("score") is not None
        if answer_error:
            status = "answer_failed"
        elif not answer_done:
            # No usable answer yet (answer skipped or produced empty output).
            status = "search_only"
        elif eval_done:
            status = "ok"
        elif eval_expected:
            status = "eval_failed"
        else:
            # Eval was intentionally skipped or has no gold; the answer stage
            # is complete, so a later resume can start at eval.
            status = "answer_only"

        result = self._make_result(
            dataset, user_id, event, evidence, search_info,
            predicted=predicted, answer_error=answer_error,
            evaluation=evaluation, timings=timings, status=status,
            tokens=tokens)
        await self._write_result(result)

    async def _write_search_partial(self, result: dict[str, Any]) -> None:
        line = json.dumps(result, ensure_ascii=False) + "\n"
        async with self.write_lock:
            self.search_writer.write(line)
            self.search_writer.flush()
            self.search_partials[str(result.get("query_id"))] = result

    async def _write_result(self, result: dict[str, Any]) -> None:
        line = json.dumps(result, ensure_ascii=False) + "\n"
        async with self.write_lock:
            self.writer.write(line)
            self.writer.flush()
            self.cache[str(result.get("query_id"))] = result
            if self.progress is not None:
                self.progress.add(result)


def make_experiment(config_path: str | Path) -> Experiment:
    return Experiment(load_bench_config(config_path))
