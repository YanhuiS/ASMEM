"""Answer step: turn search evidence into a final answer."""

from __future__ import annotations

import re
import time
from typing import Any

from .. import llm
from . import prompts
from .judge import parse_letters

#: Small models often echo the prompt's trailing "Answer:" label.  It is not
#: part of the answer and makes the judge's inclusion check noisier.
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:final\s+answer|answer|response)\s*[:\-]\s*", re.IGNORECASE)


def clean_answer(text: str) -> str:
    value = str(text or "").strip()
    stripped = _ANSWER_PREFIX.sub("", value).strip()
    return stripped or value


def _letters(text: str) -> bool:
    return bool(parse_letters(text or ""))


class TextModel:
    """Small shared wrapper around the OpenAI async client (text completions)."""

    def __init__(self, config: dict[str, Any], *, role: str = "text") -> None:
        self.config = llm.merge_call_config(config)
        self.client = llm.make_client(config)
        self.calls = 0
        self.role = role
        self.usage = llm.UsageTracker()

    async def text(self, messages: list[dict[str, Any]],
                   *, deadline: float | None = None) -> str:
        self.calls += 1
        return await llm.chat_completion(
            self.client, self.config, messages,
            usage=llm.CURRENT_USAGE.get() or self.usage, role=self.role,
            deadline=deadline)

    async def aclose(self) -> None:
        await self.client.close()


def format_evidence(evidence: list[dict[str, Any]], max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for index, record in enumerate(evidence, 1):
        content = str(record.get("content") or "").strip()
        if not content:
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        sources = metadata.get("source") or []
        source_text = ", ".join(str(item) for item in sources) if sources else "unknown"
        time_text = str(metadata.get("time") or metadata.get("observed_at") or "").strip()
        speaker = str(metadata.get("speaker") or "").strip()
        header_parts = [source_text]
        if speaker:
            header_parts.append(f"speaker={speaker}")
        if time_text:
            header_parts.append(f"observed_at={time_text}")
        header = "(" + "; ".join(header_parts) + ")"
        line = f"[{index}] {header} {content}"
        if used + len(line) + 1 > max_chars and lines:
            lines.append(f"... ({len(evidence) - index + 1} more evidence records omitted)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "(no evidence records)"


def format_options(options: Any) -> str:
    if not options:
        return ""
    items = [str(item).strip() for item in options if str(item).strip()]
    if not items:
        return ""
    return "\n\nOptions:\n" + "\n".join(items)


class Answerer:
    def __init__(self, config: dict[str, Any], model: TextModel | None,
                 *, mock: bool = False) -> None:
        self.config = config
        self.model = model
        self.mock = mock

    async def answer(
        self,
        *,
        dataset: str,
        question: str,
        evidence: list[dict[str, Any]],
        gold: dict[str, Any],
    ) -> dict[str, Any]:
        if self.mock:
            text = gold.get("answer")
            if not text and gold.get("answer_letters"):
                text = "(" + ", ".join(gold["answer_letters"]) + ")"
            return {"answer": text or "MOCK ANSWER", "error": None, "mock": True}

        style = prompts.answer_style(dataset, self.config.get("styles"))
        type_styles = self.config.get("type_styles") or {}
        answer_type = str(gold.get("answer_type") or "").strip()
        if isinstance(type_styles, dict):
            candidates = [f"{dataset}/{answer_type}"]
            # BEAM variants share answer contracts.  Configs historically used
            # beam_100k/* keys; without this fallback beam_500k silently lost
            # the abstention and instruction-following guards.
            if dataset.startswith("beam_"):
                candidates.append(f"beam_100k/{answer_type}")
            candidates.append(answer_type)
            for key in candidates:
                extra = type_styles.get(key)
                if isinstance(extra, str) and extra.strip():
                    style = (style + "\n\n" + extra).strip()
                    break
        systems = self.config.get("system_by_dataset") or {}
        system = (systems.get(dataset) if isinstance(systems, dict) else None)
        system = system or self.config.get("system") or prompts.ANSWER_SYSTEM
        user_template = str(self.config.get("user") or prompts.ANSWER_USER)

        # Two independent guards:
        # * ``total_timeout`` bounds the whole answer stage (including retries),
        #   so one slow endpoint call cannot hold a QA open indefinitely.
        # * ``max_context_tokens`` preflight-shrinks the prompt; if the endpoint
        #   still reports a context overflow, halve the evidence and retry.
        total_timeout = float(self.config.get("total_timeout", 0) or 0)
        deadline = time.monotonic() + total_timeout if total_timeout > 0 else None
        input_budget = int(self.config.get("max_context_tokens", 0) or 0)
        reserve = int(getattr(self.model, "config", {}).get("max_tokens", 4096)) if self.model else 4096
        evidence_limit = int(self.config.get("max_evidence_chars", 16000))

        if self.model is None:
            return {"answer": "", "error": "answer model is not configured"}
        last_error: str | None = None
        for attempt in range(3):
            evidence_text = format_evidence(evidence, evidence_limit)
            user = user_template
            for key, value in {
                "{question}": question,
                "{evidence}": evidence_text,
                "{options}": format_options(gold.get("options")),
                "{style}": style,
                "{dataset}": dataset,
            }.items():
                user = user.replace(key, value)
            messages = [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": user},
            ]
            if input_budget > 0:
                messages = llm.fit_messages(
                    messages, input_budget, reserve_tokens=reserve)
            try:
                text = await self.model.text(messages, deadline=deadline)
                # MCQ robustness: a small model occasionally returns "" or "()".
                # One strict retry is cheap and cannot change a valid answer.
                if gold.get("options") and not _letters(text):
                    retry_messages = [
                        *messages,
                        {"role": "assistant", "content": text or "()"},
                        {"role": "user", "content":
                         "Return ONLY the option letter(s) in parentheses, e.g. (B) or "
                         "(A, C). Do not explain."},
                    ]
                    if input_budget > 0:
                        retry_messages = llm.fit_messages(
                            retry_messages, input_budget, reserve_tokens=reserve)
                    retry = await self.model.text(retry_messages, deadline=deadline)
                    if _letters(retry):
                        text = retry
                return {"answer": clean_answer(text), "error": None}
            except llm.TruncatedCompletion as exc:
                # A long rubric answer that hit max_tokens is still usable;
                # retrying the same prompt would truncate again and cost a lot
                # of wall clock.  Keep the partial content.
                partial = clean_answer(getattr(exc, "content", "") or "")
                if partial:
                    return {"answer": partial, "error": None, "truncated": True}
                return {"answer": "", "error": "truncated completion"}
            except llm.ContextOverflow as exc:
                last_error = f"context overflow: {exc}"
                if evidence_limit <= 1000:
                    break
                evidence_limit = max(1000, evidence_limit // 2)
                continue
            except Exception as exc:  # noqa: BLE001 - a failed answer must not kill the run
                return {"answer": "", "error": f"{type(exc).__name__}: {exc}"}
        return {"answer": "", "error": last_error or "context window exceeded"}
