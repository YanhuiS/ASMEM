"""Evaluation: deterministic metrics plus optional LLM graders.

Two kinds of judge calls are supported, both one call per QA:

- **answer judge** for free-form answers (LongMemEval, PersonaMem);
- **rubric judge** for rubric-based datasets (BEAM, CL-bench), which grades
  every rubric in one JSON response instead of one call per rubric.
"""

from __future__ import annotations

import json
import re
import string
import time
from typing import Any

from .. import llm
from . import prompts

_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)

#: Answers that refuse because no evidence was found. A factual gold answer
#: cannot be satisfied by one of these, regardless of what a weak judge says.
_REFUSAL_RE = re.compile(
    r"""^\s*(?:
        the\s+(?:provided\s+)?(?:evidence|memory|context)\s+
            (?:does\s+not|doesn't|did\s+not|didn't|doesnt)\s+
            (?:provide|contain|mention|include|state|specify)
        |(?:there\s+is|there's)\s+no\s+(?:evidence|information)
        |(?:i\s+)?(?:do\s+not|don't|dont)\s+know
        |i\s+(?:cannot|can't|cant|am\s+unable\s+to)\s+(?:answer|determine|find)
        |(?:not\s+enough|insufficient)\s+information
        |no\s+(?:relevant\s+)?information\s+(?:is|was)\s+(?:provided|given|available)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

#: Gold answers that themselves say the question is unanswerable. A refusal
#: from the answerer can be correct for these.
_GOLD_UNANSWERABLE_RE = re.compile(
    r"\b(?:no\s+information|not\s+mentioned|no\s+evidence|not\s+provided|"
    r"cannot\s+be\s+determined|unable\s+to\s+determine|not\s+specified|"
    r"there\s+is\s+no\s+(?:information|evidence))\b",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    value = (text or "").strip()
    return bool(value) and len(value) <= 1200 and bool(_REFUSAL_RE.search(value))


def gold_is_unanswerable(answer: str, answer_meta: dict[str, Any] | None = None) -> bool:
    meta = answer_meta if isinstance(answer_meta, dict) else {}
    if bool(meta.get("unanswerable")):
        return True
    return bool(_GOLD_UNANSWERABLE_RE.search(answer or ""))


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = (text or "").lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(predicted: str, gold: str) -> float:
    pred = normalize_text(predicted).split()
    target = normalize_text(gold).split()
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    common: dict[str, int] = {}
    for token in target:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in pred:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize_text(predicted) == normalize_text(gold) else 0.0


def parse_letters(text: str) -> list[str]:
    """Extract option letters from an MCQ answer, without matching prose words."""
    value = (text or "").strip()
    patterns = [
        r"\(([A-F](?:\s*[,;/]\s*[A-F])*)\)",
        r"^\s*([A-F](?:\s*[,;/]\s*[A-F])*)\s*$",
        r"(?:answer|option)s?\s*(?:is|:)\s*\(?([A-F](?:\s*[,;/]\s*[A-F])*)\)?",
        r"(?:^|\n)\s*([A-F])[.)]\s",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            return [item.upper() for item in re.findall(r"[A-F]", match.group(1).upper())]
    return []


def mcq_exact(
    predicted: str,
    answer_letters: list[str],
    *,
    options: list[str] | None = None,
    ordering: bool = False,
) -> tuple[float, list[str]]:
    letters = parse_letters(predicted)
    gold = [str(item).upper() for item in answer_letters or []]
    if not letters and options:
        # The model may have written the option text without its letter.
        lowered = normalize_text(predicted)
        for index, option in enumerate(options):
            letter = chr(ord("A") + index)
            option_text = normalize_text(re.sub(r"^\s*[A-F][.)]\s*", "", str(option)))
            if lowered and option_text and (lowered in option_text or option_text in lowered):
                letters = [letter]
                break
    if not letters or not gold:
        return 0.0, letters
    if ordering:
        correct = letters == gold
    elif len(gold) > 1:
        correct = set(letters) == set(gold)
    else:
        correct = len(letters) == 1 and letters[0] == gold[0]
    return (1.0 if correct else 0.0), letters


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def strip_json_comments(text: str) -> str:
    """Remove ``//`` / ``/* */`` comments and trailing commas from JSON.

    Small judges sometimes annotate a results array (``true,  // 1``).  Those
    comments are valid in JavaScript but not JSON, and used to turn a
    perfectly usable rubric verdict into ``eval_failed``.  The scanner keeps
    string contents untouched.
    """
    out: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text):
            following = text[index + 1]
            if following == "/":
                index += 2
                while index < len(text) and text[index] not in "\r\n":
                    index += 1
                continue
            if following == "*":
                index += 2
                while index + 1 < len(text) and not (
                        text[index] == "*" and text[index + 1] == "/"):
                    index += 1
                index += 2
                continue
        out.append(char)
        index += 1
    cleaned = "".join(out)
    return re.sub(r",\s*([}\]])", r"\1", cleaned)


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    sanitized = strip_json_comments(stripped)
    candidates = [
        stripped, sanitized,
        *[match.group(0) for match in _OBJECT_RE.finditer(text or "")],
        *[match.group(0) for match in _OBJECT_RE.finditer(sanitized)],
    ]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass", "1"}
    return bool(value)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class Judge:
    def __init__(self, config: dict[str, Any], model: Any | None,
                 *, mock: bool = False) -> None:
        self.config = config
        self.model = model
        self.mock = mock

    async def evaluate(
        self,
        *,
        dataset: str,
        question: str,
        predicted: str,
        gold: dict[str, Any],
    ) -> dict[str, Any]:
        answer = str(gold.get("answer") or "")
        letters = list(gold.get("answer_letters") or [])
        rubrics = [str(item) for item in (gold.get("rubrics") or []) if str(item).strip()]
        options = gold.get("options") or []
        answer_meta = gold.get("answer_meta")
        if not isinstance(answer_meta, dict):
            answer_meta = {}
        ordering = str(answer_meta.get("qa_type") or "").lower() in {"ordering", "order"}

        metrics: dict[str, Any] = {}
        errors: list[str] = []
        judge_raw = ""
        rubric_raw = ""

        if answer and not letters:
            # For MCQ, comparing the full option text as a free-form answer is
            # misleading; the primary metric is mcq_exact.
            # Some datasets (LoCoMo-Refined) ship several acceptable complete
            # answers; take the best lexical match against any of them.
            acceptable = [
                str(item).strip()
                for item in (answer_meta.get("acceptable_answers") or [])
                if str(item).strip()
            ]
            candidates = [answer] + [item for item in acceptable if item != answer]
            metrics["token_f1"] = max(token_f1(predicted, item) for item in candidates)
            metrics["exact_match"] = max(exact_match(predicted, item) for item in candidates)
        if letters:
            score, parsed = mcq_exact(predicted, letters, options=options, ordering=ordering)
            metrics["mcq_exact"] = score
            metrics["mcq_pred_letters"] = parsed
            metrics["mcq_gold_letters"] = letters

        mode = str(self.config.get("mode") or "auto").lower()
        total_timeout = float(self.config.get("total_timeout", 0) or 0)
        deadline = time.monotonic() + total_timeout if total_timeout > 0 else None
        if mode == "auto":
            if letters:
                mode = "mcq"
            elif rubrics and self.config.get("rubric", True):
                mode = "rubric"
            elif answer and self.config.get("judge", True):
                mode = "judge"
            else:
                mode = "f1"
        if mode == "none":
            return {
                "mode": "none", "primary": None, "score": None, "metrics": {},
                "errors": [], "judge_raw": "", "rubric_raw": "",
            }

        if mode == "judge" and answer:
            if not predicted.strip():
                metrics["judge_correct"] = 0.0
            elif (looks_like_refusal(predicted)
                  and not gold_is_unanswerable(answer, answer_meta)
                  and metrics.get("token_f1", 0.0) < 0.1):
                # A weak judge can accept "the evidence does not say" as
                # CORRECT. For an answerable gold, a refusal is wrong.
                metrics["judge_correct"] = 0.0
                metrics["judge_refusal_guard"] = 1.0
            elif self.mock:
                metrics["judge_correct"] = 1.0
            else:
                try:
                    if (dataset == "locomo_refined"
                            and self.config.get("locomo_official", True)):
                        correct, judge_raw = await self._judge_locomo_official(
                            question, answer, predicted, answer_meta,
                            deadline=deadline)
                    else:
                        correct, judge_raw = await self._judge_answer(
                            dataset, question, answer, predicted, answer_meta,
                            deadline=deadline)
                    metrics["judge_correct"] = 1.0 if correct else 0.0
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"judge: {type(exc).__name__}: {exc}")
                    metrics["judge_correct"] = None
        elif mode == "rubric" and rubrics:
            limit = max(1, int(self.config.get("max_rubrics", 100)))
            used = rubrics[:limit]
            if not predicted.strip():
                metrics["rubric_score"] = 0.0
                metrics["rubric_passed"] = 0
                metrics["rubric_total"] = len(used)
            elif self.mock:
                metrics["rubric_score"] = 1.0
                metrics["rubric_passed"] = len(used)
                metrics["rubric_total"] = len(used)
            else:
                try:
                    results, rubric_raw = await self._judge_rubrics(
                        question, predicted, used, deadline=deadline)
                    passed = sum(1 for item in results[:len(used)] if item)
                    metrics["rubric_score"] = passed / len(used)
                    metrics["rubric_passed"] = passed
                    metrics["rubric_total"] = len(used)
                    metrics["rubric_results"] = results[:len(used)]
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"rubric: {type(exc).__name__}: {exc}")
                    metrics["rubric_score"] = None
                    metrics["rubric_total"] = len(used)

        primary = self._primary(dataset, metrics, mode)
        score = metrics.get(primary)
        if score is None:
            fallback = metrics.get("token_f1")
            score = fallback
        return {
            "mode": mode,
            "primary": primary,
            "score": score,
            "metrics": metrics,
            "errors": errors,
            "judge_raw": judge_raw[:4000],
            "rubric_raw": rubric_raw[:4000],
        }

    def _primary(self, dataset: str, metrics: dict[str, Any], mode: str) -> str:
        override = (self.config.get("primary_by_dataset") or {}).get(dataset)
        if override:
            return str(override)
        if mode == "mcq" or "mcq_exact" in metrics:
            return "mcq_exact"
        if mode == "rubric" or "rubric_score" in metrics:
            return "rubric_score"
        if mode == "judge" or "judge_correct" in metrics:
            return "judge_correct"
        return "token_f1"

    # -- model calls -------------------------------------------------------
    async def _call(self, messages: list[dict[str, Any]], *,
                    deadline: float | None = None) -> str:
        """One bounded judge call with a context-window preflight.

        A judge prompt is normally small, but rubric batches and long answers
        can grow; shrink to ``max_context_tokens`` and retry once with half
        the budget if the endpoint still reports an overflow.
        """
        if self.model is None:
            raise RuntimeError("eval model is not configured")
        budget = int(self.config.get("max_context_tokens", 0) or 0)
        reserve = int(getattr(self.model, "config", {}).get("max_tokens", 2048))
        fitted = (llm.fit_messages(messages, budget, reserve_tokens=reserve)
                  if budget > 0 else messages)
        try:
            return await self.model.text(fitted, deadline=deadline)
        except llm.ContextOverflow:
            if budget <= 0:
                raise
            smaller = llm.fit_messages(
                fitted, max(4000, budget // 2), reserve_tokens=reserve)
            return await self.model.text(smaller, deadline=deadline)

    async def _judge_locomo_official(
        self, question: str, gold: str, predicted: str,
        answer_meta: dict[str, Any] | None = None,
        *, deadline: float | None = None,
    ) -> tuple[bool, str]:
        """Official LoCoMo-Refined judge: one ``label`` call per gold candidate."""
        if self.model is None:
            raise RuntimeError("eval model is not configured")
        meta = answer_meta if isinstance(answer_meta, dict) else {}
        candidates = [gold]
        candidates.extend(
            str(item).strip()
            for item in (meta.get("acceptable_answers") or [])
            if str(item).strip()
        )
        candidates = [item for item in dict.fromkeys(candidates) if item]
        if not candidates:
            candidates = [""]

        generated = predicted[: int(self.config.get("max_answer_chars", 12000))]
        raws: list[str] = []
        errors: list[str] = []
        for candidate in candidates:
            user = str(self.config.get("locomo_official_user")
                       or prompts.LOCOMO_REFINED_JUDGE_USER)
            for key, value in {
                "{question}": question,
                "{gold_answer}": candidate,
                "{generated_answer}": generated,
                "{gold}": candidate,
                "{predicted}": generated,
            }.items():
                user = user.replace(key, value)
            try:
                raw = await self._call(
                    [{"role": "user", "content": user}], deadline=deadline)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            raws.append(f"[gold={candidate}] {raw}")
            value = parse_json_object(raw) or {}
            label = str(value.get("label") or "").strip().upper()
            if label == "CORRECT":
                return True, "\n".join(raws)
            if label == "WRONG":
                continue
            if "correct" in value:
                if _bool(value.get("correct")):
                    return True, "\n".join(raws)
                continue
            if re.search(r"\bCORRECT\b", raw, re.IGNORECASE) and not re.search(
                    r"\bWRONG\b", raw, re.IGNORECASE):
                return True, "\n".join(raws)
        if errors and len(errors) == len(candidates):
            raise RuntimeError("LoCoMo judge failed: " + "; ".join(errors))
        return False, "\n".join(raws)

    async def _judge_answer(self, dataset: str, question: str, gold: str,
                            predicted: str,
                            answer_meta: dict[str, Any] | None = None,
                            *, deadline: float | None = None) -> tuple[bool, str]:
        if self.model is None:
            raise RuntimeError("eval model is not configured")
        user = str(self.config.get("user") or prompts.JUDGE_USER)
        for key, value in {
            "{question}": question,
            "{gold}": gold,
            "{predicted}": predicted[: int(self.config.get("max_answer_chars", 12000))],
        }.items():
            user = user.replace(key, value)

        styles = self.config.get("styles") or {}
        style = styles.get(dataset) if isinstance(styles, dict) else None
        if style:
            user += f"\n\nGrading note:\n{style}"
        context_lines = []
        meta = answer_meta if isinstance(answer_meta, dict) else {}
        for key in ("preference", "topic_preference", "updated", "prev_pref",
                    "who", "sensitive_info", "conversation_scenario"):
            value = meta.get(key)
            if value not in (None, "", False, "False", "false"):
                context_lines.append(f"{key}: {value}")
        if context_lines:
            user += "\n\nAdditional context:\n" + "\n".join(context_lines)
        acceptable = [
            str(item).strip()
            for item in (meta.get("acceptable_answers") or [])
            if str(item).strip() and str(item).strip() != gold
        ]
        if acceptable:
            user += "\n\nOther acceptable gold answers (any one is correct):\n" + \
                "\n".join(f"- {item}" for item in acceptable)
        messages = [
            {"role": "system", "content": str(self.config.get("system") or prompts.JUDGE_SYSTEM)},
            {"role": "user", "content": user},
        ]
        raw = await self._call(messages, deadline=deadline)
        value = parse_json_object(raw) or {}
        if "correct" not in value:
            match = re.search(r'"correct"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if match:
                return match.group(1).lower() == "true", raw
            # Last-resort text fallback: "Correct: yes" / "Incorrect".
            if re.search(r"\b(incorrect|not correct|wrong)\b", raw, re.IGNORECASE):
                return False, raw
            if re.search(r"\b(correct|yes)\b", raw, re.IGNORECASE):
                return True, raw
            raise ValueError(f"judge did not return 'correct': {raw[:300]}")
        return _bool(value.get("correct")), raw

    async def _judge_rubrics(self, question: str, answer: str,
                             rubrics: list[str],
                             *, deadline: float | None = None) -> tuple[list[bool], str]:
        """Grade rubrics in bounded batches.

        A 100-rubric response easily exceeds the judge's ``max_tokens`` and is
        then truncated mid-JSON (observed: ``eval_failed`` with a partial
        ``results`` array).  Batching keeps each response parseable; the
        comment/trailing-comma sanitizer handles ``true, // 1`` annotations.
        """
        if self.model is None:
            raise RuntimeError("eval model is not configured")
        if not rubrics:
            return [], ""
        batch_size = max(1, int(self.config.get("rubric_batch", 20)))
        results: list[bool] = []
        raws: list[str] = []
        for start in range(0, len(rubrics), batch_size):
            batch = rubrics[start:start + batch_size]
            parsed, raw = await self._judge_rubric_batch(
                question, answer, batch, deadline=deadline)
            results.extend(parsed)
            raws.append(raw)
        return results, "\n".join(raws)

    async def _judge_rubric_batch(self, question: str, answer: str,
                                  rubrics: list[str],
                                  *, deadline: float | None = None) -> tuple[list[bool], str]:
        rubric_text = "\n".join(
            f"{index}. {item}" for index, item in enumerate(rubrics, 1))
        user = str(self.config.get("rubric_user") or prompts.RUBRIC_USER)
        for key, value in {
            "{question}": question,
            "{answer}": answer[: int(self.config.get("max_answer_chars", 12000))],
            "{rubrics}": rubric_text,
        }.items():
            user = user.replace(key, value)
        messages = [
            {"role": "system", "content": str(
                self.config.get("rubric_system") or prompts.RUBRIC_SYSTEM)},
            {"role": "user", "content": user},
        ]
        last_raw = ""
        for attempt in range(2):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("eval deadline exceeded")
            last_raw = await self._call(messages, deadline=deadline)
            value = parse_json_object(last_raw) or {}
            parsed = _bool_list(value.get("results"))
            if parsed is None:
                parsed = _bool_list_from_text(last_raw)
            if parsed is not None:
                if len(parsed) < len(rubrics):
                    parsed.extend([False] * (len(rubrics) - len(parsed)))
                return parsed[:len(rubrics)], last_raw
            # One strict retry: the model annotated the array with comments or
            # wrapped it in prose.  Ask for exactly the JSON shape.
            messages = [
                *messages,
                {"role": "assistant", "content": last_raw or "{}"},
                {"role": "user", "content":
                 f"Return ONLY one JSON object with exactly {len(rubrics)} booleans: "
                 '{"results":[true,false,...]}. No comments, no markdown, no explanation.'},
            ]
        raise ValueError(
            f"rubric judge did not return 'results': {last_raw[:300]}")


def _bool_list(value: Any) -> list[bool] | None:
    if not isinstance(value, list):
        return None
    return [_bool(item) for item in value]


def _bool_list_from_text(text: str) -> list[bool] | None:
    """Salvage results from fenced/prose responses or one-flag-per-line."""
    cleaned = strip_json_comments(text or "")
    stripped = cleaned.strip()
    if stripped.startswith("["):
        try:
            value = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            value = None
        if isinstance(value, list):
            return [_bool(item) for item in value]
    match = re.search(r'"results"\s*:\s*\[([^\]]*)\]', cleaned, re.DOTALL)
    if match:
        items = re.findall(r"\b(true|false|yes|no|pass|fail)\b",
                           match.group(1), re.IGNORECASE)
        if items:
            return [item.lower() in {"true", "yes", "pass"} for item in items]
    lines = re.findall(
        r"(?m)^\s*(?:rubric\s*)?\d+\s*[.):-]?\s*(pass|fail|true|false)\b",
        cleaned, re.IGNORECASE)
    if lines:
        return [item.lower() in {"pass", "true"} for item in lines]
    return None
