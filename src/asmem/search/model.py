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
        self.config = call_config(config)
        self.native = bool(config.get("native_tools"))
        self.schemas = toolcfg.tool_schemas() if self.native else None
        self.client = llm.make_client(self.config)
        self.calls = 0
        self.usage = llm.UsageTracker()
        self.fail_streak = 0
        self.breaker_threshold = max(1, int(config.get("llm_breaker_failures", 2)))
        self.last_error = ""

    def _breaker_message(self) -> str:
        return (f"model circuit open after {self.fail_streak} consecutive "
                f"failure(s): {self.last_error}")

    def _check_breaker(self) -> None:
        if self.fail_streak >= self.breaker_threshold:
            raise llm.ModelUnavailable(self._breaker_message())

    def _record_failure(self, exc: Exception) -> None:
        # Truncation means the endpoint answered; the content was just too
        # long.  It must not trip the transport breaker.
        if isinstance(exc, llm.TruncatedCompletion):
            return
        self.fail_streak += 1
        self.last_error = f"{type(exc).__name__}: {exc}"

    async def call(self, messages: list[dict[str, Any]],
                   schemas: list[dict[str, Any]] | None = None,
                   usage: llm.UsageTracker | None = None,
                   deadline: float | None = None) -> ModelReply:
        self._check_breaker()
        self.calls += 1
        tracker = usage or self.usage
        try:
            if self.native:
                message = await llm.chat_completion_message(
                    self.client, self.config, messages, tools=schemas or self.schemas,
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
                reply = ModelReply(text=text, calls=calls,
                                   assistant_entry=assistant, raw=message)
            else:
                text = await llm.chat_completion(
                    self.client, self.config, messages,
                    usage=tracker, role="postagent", deadline=deadline)
                calls = toolcfg.parse_tool_calls(text)
                reply = ModelReply(text=text, calls=calls,
                                   assistant_entry={"role": "assistant", "content": text})
        except Exception as exc:
            self._record_failure(exc)
            raise
        self.fail_streak = 0
        return reply

    async def text(self, messages: list[dict[str, Any]],
                   usage: llm.UsageTracker | None = None,
                   deadline: float | None = None) -> str:
        self._check_breaker()
        self.calls += 1
        try:
            text = await llm.chat_completion(
                self.client, self.config, messages,
                usage=usage or self.usage, role="verifier", deadline=deadline)
        except Exception as exc:
            self._record_failure(exc)
            raise
        self.fail_streak = 0
        return text

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


