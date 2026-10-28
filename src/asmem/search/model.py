"""Model wrapper, call policy, and token accounting."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from .. import llm
from .config import call_config
from .postagent import ModelReply
from . import toolcfg

class ModelRunner:
    """Wraps the LLM client and normalizes native/text tool-calling.

    A small per-run circuit breaker keeps a dead endpoint from dominating the
    request: after ``llm_breaker_failures`` consecutive transport/model
    failures (truncation is a content problem, not an outage) every further
    call fails immediately with ``ModelUnavailable``.  The postagent then keeps
    whatever evidence it already wrote and the runner falls back to coarse
    candidates, so the HTTP request still completes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.full_config = config
        self.config = call_config(config)
        self.native = bool(config.get("native_tools"))
        self.schemas = toolcfg.tool_schemas() if self.native else None
        self.client = llm.make_client(self.config)
        self.calls = 0
        self.usage = llm.UsageTracker()
        self.fail_streak = 0
        self.breaker_threshold = max(1, int(config.get("llm_breaker_failures", 2)))
        self.last_error = ""
        # Process-wide endpoint breaker (opt-in).  A per-run breaker cannot
        # stop a dead endpoint from being retried by every new search.
        self.global_threshold = max(
            0, int(config.get("llm_global_breaker_failures", 0) or 0))
        self.global_cooldown = max(
            0.0, float(config.get("llm_global_breaker_cooldown", 30) or 30))
        self.breaker_key = (
            f"{self.config.get('base_url', '')}|{self.config.get('model', '')}")
        # Token-level preflight and context-overflow recovery.  Long prompts
        # are both slower and more likely to hit the endpoint's context limit;
        # cap them before the request and shrink further on a 400.
        self.input_cap = max(
            0, int(config.get("model_input_token_cap", 0) or 0))
        self.output_reserve = max(
            256, int(config.get("model_output_reserve_tokens", 4096) or 4096))
        self.overflow_retries = max(
            0, int(config.get("context_overflow_retries", 1) or 0))
        # Output budgets are per role.  A gate/verifier reply is a small JSON
        # object; letting it generate 8k tokens only adds latency and makes a
        # timeout more likely.
        base_max = max(64, int(self.config.get("max_tokens", 4096) or 4096))
        self.call_config = dict(self.config)
        self.call_config["max_tokens"] = max(64, min(
            base_max,
            int(config.get("postagent_max_output_tokens", base_max)
                or base_max)))
        self.verifier_config = dict(self.config)
        self.verifier_config["max_tokens"] = max(64, min(
            base_max,
            int(config.get("verifier_max_output_tokens", base_max)
                or base_max)))

    def _breaker_message(self) -> str:
        return (f"model circuit open after {self.fail_streak} consecutive "
                f"failure(s): {self.last_error}")

    def _check_breaker(self) -> None:
        if self.fail_streak >= self.breaker_threshold:
            raise llm.ModelUnavailable(self._breaker_message())
        if self.global_threshold > 0:
            allowed, message = llm.GLOBAL_BREAKER.allow(
                self.breaker_key, self.global_threshold, self.global_cooldown)
            if not allowed:
                raise llm.ModelUnavailable(message)

    def _record_failure(self, exc: Exception) -> None:
        # Truncation means the endpoint answered; the content was just too
        # long.  Context overflow is a prompt-size problem, not an outage.
        # Neither must trip the transport breaker.
        if isinstance(exc, (llm.TruncatedCompletion, llm.ContextOverflow)):
            return
        self.fail_streak += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.global_threshold > 0:
            llm.GLOBAL_BREAKER.failure(
                self.breaker_key, self.global_threshold)

    def _record_success(self) -> None:
        self.fail_streak = 0
        if self.global_threshold > 0:
            llm.GLOBAL_BREAKER.success(self.breaker_key)

    def _fit(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.input_cap <= 0:
            return messages
        return llm.fit_messages(
            messages, self.input_cap, reserve_tokens=self.output_reserve)

    def _shrink(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """A smaller prompt after a context-overflow 400, or ``None``."""
        base = self.input_cap or int(
            self.full_config.get("agent_max_context_tokens", 32000) or 32000)
        reserve = min(self.output_reserve, max(256, base // 4))
        for ratio in (0.5, 0.25):
            candidate = llm.fit_messages(
                messages, max(2048, int(base * ratio)),
                reserve_tokens=reserve)
            if candidate != messages:
                return candidate
        # The prompt fits the configured cap but the endpoint still rejected
        # it (its real window is smaller).  Force one hard cut so the caller
        # gets a smaller prompt instead of retrying the same one.
        out = [dict(message) for message in messages]
        positions = [index for index, message in enumerate(out)
                     if message.get("role") != "system"]
        if not positions:
            return None
        longest = max(
            positions, key=lambda index: len(str(out[index].get("content") or "")))
        content = str(out[longest].get("content") or "")
        if len(content) <= 512:
            return None
        out[longest]["content"] = (
            content[: len(content) // 2].rstrip()
            + "\n\n[... shrunk to fit the model context ...]")
        return out

    async def call(self, messages: list[dict[str, Any]],
                   schemas: list[dict[str, Any]] | None = None,
                   usage: llm.UsageTracker | None = None,
                   deadline: float | None = None) -> ModelReply:
        self._check_breaker()
        self.calls += 1
        tracker = usage or self.usage

        async def _invoke(current: list[dict[str, Any]]) -> ModelReply:
            if self.native:
                message = await llm.chat_completion_message(
                    self.client, self.call_config, current,
                    tools=schemas or self.schemas,
                    usage=tracker, role="postagent", deadline=deadline)
                text = message.content or ""
                calls: list[ToolCall] = []
                assistant: dict[str, Any] = {"role": "assistant", "content": text}
                raw_calls = getattr(message, "tool_calls", None) or []
                if raw_calls:
                    assistant["tool_calls"] = []
                    for item in raw_calls:
                        try:
                            args = json.loads(item.function.arguments or "{}")
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        calls.append(ToolCall(item.function.name, args, id=item.id))
                        assistant["tool_calls"].append({
                            "id": item.id, "type": "function",
                            "function": {"name": item.function.name,
                                         "arguments": item.function.arguments or "{}"},
                        })
                return ModelReply(text=text, calls=calls,
                                  assistant_entry=assistant, raw=message)
            text = await llm.chat_completion(
                self.client, self.call_config, current,
                usage=tracker, role="postagent", deadline=deadline)
            calls = toolcfg.parse_tool_calls(text)
            return ModelReply(text=text, calls=calls,
                              assistant_entry={"role": "assistant", "content": text})

        current = self._fit(messages)
        attempts = 0
        while True:
            try:
                reply = await _invoke(current)
                self._record_success()
                return reply
            except llm.ContextOverflow as exc:
                shrunk = self._shrink(current)
                if shrunk is None or attempts >= self.overflow_retries:
                    self._record_failure(exc)
                    raise
                current = shrunk
                attempts += 1
            except Exception as exc:
                self._record_failure(exc)
                raise

    async def text(self, messages: list[dict[str, Any]],
                   usage: llm.UsageTracker | None = None,
                   deadline: float | None = None) -> str:
        self._check_breaker()
        self.calls += 1
        tracker = usage or self.usage

        async def _invoke(current: list[dict[str, Any]]) -> str:
            return await llm.chat_completion(
                self.client, self.verifier_config, current,
                usage=tracker, role="verifier", deadline=deadline)

        current = self._fit(messages)
        attempts = 0
        while True:
            try:
                text = await _invoke(current)
                self._record_success()
                return text
            except llm.ContextOverflow as exc:
                shrunk = self._shrink(current)
                if shrunk is None or attempts >= self.overflow_retries:
                    self._record_failure(exc)
                    raise
                current = shrunk
                attempts += 1
            except Exception as exc:
                self._record_failure(exc)
                raise

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:  # noqa: BLE001 - shutdown must not mask the result
            pass


def _accepts_deadline(fn: Any) -> bool:
    """Whether a model wrapper's ``call``/``text`` accepts ``deadline=``.

    ``ModelRunner`` does, but lightweight wrappers (bench ``ModelView``, test
    stubs, third-party integrations) may only take ``messages``.  Passing an
    unsupported keyword would turn a working search into ``TypeError``; the
    deadline is a best-effort cap, so it is simply omitted for those.
    """
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "deadline" in params or any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in params.values())



def _sum_usage(*values: dict[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        for key, item in (value or {}).items():
            if isinstance(item, (int, float)):
                out[key] = out.get(key, 0) + int(item)
    return out


def _read_usage_log(path: Path) -> dict[str, int]:
    """Sum embedding usage written by ``coarse`` subprocesses."""
    if not path.exists():
        return {}
    total: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            total = _sum_usage(total, value)
    return total


def _token_block(embedding: dict[str, Any], llm_usage: dict[str, Any]) -> dict[str, Any]:
    roles = llm_usage.get("by_role") or {}
    postagent = roles.get("postagent") or {}
    verifier = roles.get("verifier") or {}
    embedding_total = int(embedding.get("total_tokens", 0) or 0)
    llm_total = int(llm_usage.get("total_tokens", 0) or 0)
    return {
        "embedding": dict(embedding),
        "postagent": dict(postagent),
        "verifier": dict(verifier),
        "llm": dict(llm_usage),
        "total_tokens": embedding_total + llm_total,
    }


