"""LLM 调用：重试、退避、截断检测。

add（抽取原子记忆）、answer（依证据作答）、eval（judge）、search（verifier 与 agent）
四处都要调 chat completion，且都要处理同一批坑：

- **429 / 503** —— 按 ``Retry-After`` 或 ``rate_limit_backoff`` 退避；其它错误普通指数退避；
  统一叠抖动，避免并发请求同时重试再次打爆限流。
- **空回复** —— 小模型关掉思考后偶发，补 ``reasoning`` 兜底，仍空则当瞬时错误重试。
- **被 max_tokens 截断**（``finish_reason == "length"``）—— 内容必然是不完整的 JSON；
  思考长度会波动，重试常能成功。所以单独成类 ``TruncatedCompletion``，让调用方能区分
  "截断了"和"真挂了"。
- **qwen3 关思考** —— 只有模型名含 ``qwen3`` 才发 ``chat_template_kwargs``；
  其它端点会因未知参数直接 400。
- **连接建立超时单独设短** —— 远程端点在高并发下会丢一部分**新建**连接；``connect_timeout``
  比读超时短得多，好让失败快速发生、把重试预算花在重试而不是等内核超时（见 ``make_client``）。

放在顶层而不是藏进 add/：这些都是**端点行为**，与"抽取记忆"这个业务无关。
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import random
import re
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import openai

logger = logging.getLogger(__name__)

#: Task-local usage collector.  The answer and judge models are shared across
#: concurrent QA tasks; a context variable gives each task its own exact token
#: attribution without serializing the model calls.
CURRENT_USAGE: ContextVar["UsageTracker | None"] = ContextVar(
    "asmem_current_usage", default=None)

# 谈判用的默认参数。调用方用 ``merge_call_config`` 把自己的模型配置叠上去。
DEFAULT_CALL_CONFIG: dict[str, Any] = {
    "base_url": "http://127.0.0.1:30000/v1",
    "api_key": "sglang",
    "model": "local",
    "temperature": 0.2,
    "max_tokens": 4096,
    "timeout": 300,
    "connect_timeout": 10.0,
    "max_retries": 3,
    "backoff_base": 2.0,
    "rate_limit_backoff": 10.0,
    "max_backoff": 60.0,
    "backoff_jitter": 0.2,
    "enable_thinking": False,
    # Retrying a response that hit max_tokens is usually wasted work: the
    # same long prompt tends to truncate again.  Search can opt out; the
    # answer stage instead keeps the partial content.
    "retry_truncated": True,
}

# ``chat_completion`` 真正会读的键。配置里的其它键（业务参数）原样忽略。
CALL_KEYS = (
    "base_url", "api_key", "model", "temperature", "max_tokens", "enable_thinking",
    "timeout", "connect_timeout", "max_retries", "backoff_base", "rate_limit_backoff",
    "max_backoff", "backoff_jitter", "retry_truncated",
)


class UsageTracker:
    """Accumulate token usage across chat-completion calls.

    Roles keep the search budget auditable: ``postagent`` and ``verifier`` are
    separate, so a run can report exactly how much the agent loop cost versus
    the audit.  The tracker is additive and cheap; callers snapshot it before a
    stage and subtract to get a per-query delta.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.by_role: dict[str, dict[str, int]] = {}

    @staticmethod
    def read(usage: Any) -> tuple[int, int, int]:
        if usage is None:
            return 0, 0, 0
        prompt = int(getattr(usage, "prompt_tokens", None)
                     or getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", None)
                         or getattr(usage, "output_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", None) or (prompt + completion))
        return prompt, completion, total

    def add(self, usage: Any, role: str = "") -> None:
        prompt, completion, total = self.read(usage)
        if prompt == completion == total == 0:
            return
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        key = role or "default"
        bucket = self.by_role.setdefault(
            key, {"calls": 0, "prompt_tokens": 0,
                  "completion_tokens": 0, "total_tokens": 0})
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt
        bucket["completion_tokens"] += completion
        bucket["total_tokens"] += total

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "by_role": {role: dict(values) for role, values in self.by_role.items()},
        }

    def snapshot(self) -> dict[str, Any]:
        return self.as_dict()


def usage_delta(before: dict[str, Any] | None,
                after: dict[str, Any] | None) -> dict[str, Any]:
    """Subtract two ``UsageTracker.as_dict()`` snapshots."""
    before = before or {}
    after = after or {}
    out: dict[str, Any] = {
        "calls": int(after.get("calls", 0)) - int(before.get("calls", 0)),
        "prompt_tokens": int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)),
        "completion_tokens": int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0)),
        "total_tokens": int(after.get("total_tokens", 0)) - int(before.get("total_tokens", 0)),
        "by_role": {},
    }
    roles = set((before.get("by_role") or {})) | set((after.get("by_role") or {}))
    for role in roles:
        old = (before.get("by_role") or {}).get(role) or {}
        new = (after.get("by_role") or {}).get(role) or {}
        out["by_role"][role] = {
            key: int(new.get(key, 0)) - int(old.get(key, 0))
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
    return out


class TruncatedCompletion(RuntimeError):
    """模型输出因达到 ``max_tokens`` 而被截断（``finish_reason == 'length'``）。

    ``content`` 是**被截断的那段正文** —— 调用方可以按对象边界抢救出前面完整的记录
    （见 ``io.salvage_json_records``），而不是把整轮产出当作没发生。
    """

    def __init__(self, message: str, content: str = "") -> None:
        super().__init__(message)
        self.content = content


class ModelUnavailable(RuntimeError):
    """模型端点在一次请求（含重试）内不可用，或已没有剩余时间预算。

    调用方应当把它当成**降级信号**而不是程序错误：postagent 会保留已经写好的证据，
    runner 会退回粗筛候选，API 仍然返回 200。
    """


class ContextOverflow(ModelUnavailable):
    """请求超过了端点的上下文窗口（通常是 400 context_length_exceeded）。

    这类错误**重试同一个 prompt 永远不会成功**，所以不按普通瞬时错误重试；
    调用方应当缩小证据/历史后重试，或降级返回。
    """


class EndpointBreaker:
    """Process-wide circuit breaker for one ``(base_url, model)`` endpoint.

    The per-run breaker in :class:`asmem.search.model.ModelRunner` protects one
    search from a dead endpoint, but every new HTTP request constructs a new
    runner, so an outage still causes every request to pay the same timeout.
    This shared breaker opens after ``threshold`` failures and lets all new
    searches fail fast into the deterministic coarse fallback until the
    cooldown expires.
    """

    def __init__(self) -> None:
        self._state: dict[str, dict[str, float]] = {}

    def allow(self, key: str, threshold: int, cooldown: float) -> tuple[bool, str]:
        if threshold <= 0:
            return True, ""
        state = self._state.get(key)
        if not state:
            return True, ""
        opened = float(state.get("opened_at") or 0.0)
        if not opened:
            return True, ""
        elapsed = time.monotonic() - opened
        if elapsed >= cooldown:
            # Cooldown elapsed: allow one probe.  Keep the failure count at
            # the threshold so a failed probe re-opens immediately; a success
            # resets it in ``success``.
            state["opened_at"] = 0.0
            return True, ""
        return False, (
            f"global model circuit open for {cooldown - elapsed:.0f}s "
            f"(endpoint {key})")

    def failure(self, key: str, threshold: int) -> None:
        if threshold <= 0:
            return
        state = self._state.setdefault(
            key, {"failures": 0.0, "opened_at": 0.0})
        state["failures"] = float(state.get("failures") or 0.0) + 1.0
        if state["failures"] >= threshold:
            state["opened_at"] = time.monotonic()

    def success(self, key: str) -> None:
        state = self._state.get(key)
        if state is not None:
            state["failures"] = 0.0
            state["opened_at"] = 0.0

    def reset(self) -> None:
        self._state.clear()


#: Shared by every ModelRunner in this process.  Opt-in via
#: ``llm_global_breaker_failures > 0``.
GLOBAL_BREAKER = EndpointBreaker()


def merge_call_config(config: dict[str, Any]) -> dict[str, Any]:
    """把调用方的配置叠到默认值上，只保留 ``CALL_KEYS``。

    为什么只挑键：配置文件通常把业务参数和模型参数混在一层（``search.yaml`` 里既有
    ``max_steps`` 也有 ``model``）。这里做一次收窄，让 ``chat_completion`` 拿到的永远
    是干净的调用参数 —— 否则少一个键就是运行时 ``KeyError``。
    """
    merged = dict(DEFAULT_CALL_CONFIG)
    for key in CALL_KEYS:
        if key in config and config[key] is not None:
            merged[key] = config[key]
    return merged


def estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate (no tokenizer dependency).

    English averages ~3.6 chars/token; CJK averages closer to 1.1.  Rounding
    up is deliberate: the budget must never be underestimated, otherwise a
    "fits" prompt still overflows at the endpoint.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    other_chars = len(text) - ascii_chars
    return int(ascii_chars / 3.6 + other_chars / 1.1) + 1


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + "\n\n[... truncated to fit the context window ...]"


def fit_messages(messages: list[dict[str, Any]], max_input_tokens: int,
                 *, reserve_tokens: int = 0) -> list[dict[str, Any]]:
    """Shrink a prompt so ``input + reserve`` fits ``max_input_tokens``.

    System messages are kept intact (they carry the output contract); the
    longest non-system messages are truncated from the end.  The answer and
    judge prompts put the evidence last, so this drops the least relevant
    records first.  This is a preflight guard: it prevents a 400
    ``context_length_exceeded`` instead of retrying one.
    """
    budget = max(512, int(max_input_tokens) - int(reserve_tokens))
    out = [dict(message) for message in messages]
    total = sum(estimate_tokens(str(item.get("content") or "")) for item in out)
    if total <= budget:
        return out
    order = sorted(
        (index for index, item in enumerate(out) if item.get("role") != "system"),
        key=lambda index: len(str(out[index].get("content") or "")),
        reverse=True,
    )
    for index in order:
        others = sum(
            estimate_tokens(str(item.get("content") or ""))
            for position, item in enumerate(out) if position != index
        )
        allowed = max(64, budget - others)
        content = str(out[index].get("content") or "")
        if estimate_tokens(content) > allowed:
            out[index]["content"] = _truncate_to_tokens(content, allowed)
        total = sum(estimate_tokens(str(item.get("content") or "")) for item in out)
        if total <= budget:
            break
    return out


_CONTEXT_HINTS = (
    "context_length_exceeded", "context length", "maximum context",
    "too many tokens", "reduce the length", "prompt is too long",
    "exceeds the maximum", "input is too long",
)


def is_context_overflow(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(hint in text for hint in _CONTEXT_HINTS)


def make_client(config: dict[str, Any], *, sdk_retries: int = 0) -> openai.AsyncOpenAI:
    """按配置建一个异步客户端。

    ``timeout`` 是**读超时**（一次生成可能很慢，要留足）；``connect_timeout`` 单独设、
    且**明显更短**。原因：远程端点在高并发下会丢掉一部分**新建**连接（SYN 被丢，实测
    16 并发时约一半失败，而串行复用连接几乎不失败）。若不单独限时，一次连接失败要等内核
    SYN 重传超时（~60s）才轮到重试 —— 既慢，又把重试预算耗在"等超时"上。短 connect
    超时让失败**快速**发生，退避后重试才有意义。

    **连接池按并发配**（``max_connections`` / ``max_keepalive_connections``）：默认池只留
    少量 keep-alive 连接，32 路并发时每批请求都要**新建**连接 —— 而那正是远程端点会丢的
    东西（同上）。把 keep-alive 上限设成并发量，池里的连接就能被下一批直接复用，
    新建连接数降到接近 0（``llm.max_connections`` / ``llm.max_keepalive_connections`` 可覆盖，
    ``max_keepalive_connections`` 缺省取 ``concurrency``）。

    连接池这一段是**加速件**，不是依赖：拿不到 HTTPX 时退回 SDK 自建客户端（超时照旧），
    功能一模一样，只是池子不能用并发量去调。**绝不能因为一个可选优化让整条链路起不来。**
    OpenAI 3.x 使用 ``httpx2``；旧版 SDK 使用 ``httpx``，所以这里按顺序兼容两者。
    ``trust_env=false`` 可关闭环境代理（本地端点/无网络测试常用），默认仍读取代理环境。
    """
    merged = merge_call_config(config)
    read_timeout = float(merged.get("timeout", 300))
    connect_timeout = float(merged.get("connect_timeout", 10.0))

    httpx = None
    for module_name in ("httpx2", "httpx"):
        try:
            httpx = importlib.import_module(module_name)
            break
        except ImportError:
            continue
    if httpx is None:                       # pragma: no cover - 取决于运行环境
        return openai.AsyncOpenAI(
            base_url=merged["base_url"],
            api_key=merged.get("api_key") or "not-needed",
            timeout=openai.Timeout(
                connect=connect_timeout, read=read_timeout,
                write=read_timeout, pool=read_timeout,
            ),
            max_retries=max(0, int(sdk_retries)),
        )

    concurrency = max(1, int(config.get("concurrency", 8) or 8))
    max_connections = int(config.get("max_connections", max(32, concurrency * 2)) or 0)
    keepalive = int(config.get("max_keepalive_connections", concurrency) or 0)
    # 超时**建在 http_client 上**：传了 http_client 时，SDK 直接用它（不再另建一个默认客户端），
    # 所以这里的 Timeout 就是生效的那一份 —— 不要再给 AsyncOpenAI 传 timeout，免得两处各说各话。
    timeout = httpx.Timeout(
        connect=connect_timeout, read=read_timeout, write=read_timeout, pool=read_timeout,
    )
    explicit_trust_env = config.get("trust_env")
    if explicit_trust_env is None:
        # Environment proxies must not intercept loopback endpoints (local
        # SGLang/vLLM) and make otherwise offline tests depend on SOCKS extras.
        hostname = (urlsplit(str(merged.get("base_url") or "")).hostname or "").lower()
        trust_env = hostname not in {"127.0.0.1", "::1", "localhost"}
    else:
        trust_env = bool(explicit_trust_env)
    return openai.AsyncOpenAI(
        base_url=merged["base_url"],
        api_key=merged.get("api_key") or "not-needed",
        # Chat defaults to 0 because chat_completion owns retries.  The
        # embedding backend passes its configured retry count explicitly.
        max_retries=max(0, int(sdk_retries)),
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections or None,
                max_keepalive_connections=keepalive or None,
            ),
            timeout=timeout,
            trust_env=trust_env,
        ),
    )


# ---------------------------------------------------------------------------
# 退避
# ---------------------------------------------------------------------------

def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after_seconds(exc: Exception) -> float | None:
    """解析 ``Retry-After`` 响应头（秒数或 HTTP-date 两种形式）。"""
    response = getattr(exc, "response", None)
    raw = response.headers.get("Retry-After") if response is not None else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(raw)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def backoff_seconds(exc: Exception, attempt: int, config: dict[str, Any]) -> float:
    """本次失败后该等多久。429/503 用更长的 ``rate_limit_backoff``，其余用 ``backoff_base``。"""
    base = float(config.get("backoff_base", 2.0))
    cap = float(config.get("max_backoff", 120.0))
    jitter = float(config.get("backoff_jitter", 0.5))
    status = _status_code(exc)

    if status == 429:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = min(retry_after, cap)
        else:
            delay = min(float(config.get("rate_limit_backoff", 10.0)) * 2 ** (attempt - 1), cap)
    elif status == 503:
        delay = min(float(config.get("rate_limit_backoff", 10.0)) * 2 ** (attempt - 1), cap)
    else:
        delay = min(base * 2 ** (attempt - 1), cap)
    return delay + random.uniform(0, jitter * delay)


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------

async def _complete_choice(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    require_text: bool = True,
    usage: UsageTracker | None = None,
    role: str = "",
    deadline: float | None = None,
) -> Any:
    """调一次 chat completion，返回 ``choices[0]``。失败按 ``max_retries`` 重试。

    ``tools`` 非空时走 **OpenAI 原生函数调用**（``tools=`` 参数）。这一模式下结果是
    ``message.tool_calls``，``content`` 可以是空的 —— 所以此时传 ``require_text=False``，
    把"空正文"从错误降级为合法状态（"既没有正文也没有工具调用"仍然算失败、要重试）。

    配置里若指定了与 ``client`` 不同的 ``base_url``，**显式切换** —— OpenAI SDK 的默认
    client 会忽略请求级 base_url，导致请求被静默发到旧地址（这个坑踩过）。

    ``deadline`` 是 ``time.monotonic()`` 时间戳（可选）。它把"单次调用 + 重试 + 退避"
    整体限制在一个墙钟预算内：每次尝试的读超时取 ``min(timeout, 剩余时间)``，剩余时间
    不足时立即抛 ``ModelUnavailable``，而不是让一次 300 秒的读超时把整个请求拖死。
    """
    config = merge_call_config(config)
    base_url = config.get("base_url")
    if base_url and str(client.base_url).rstrip("/") != str(base_url).rstrip("/"):
        client = client.with_options(base_url=base_url)

    kwargs: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": config.get("temperature", 0.2),
        "max_tokens": config.get("max_tokens", 4096),
        "stream": False,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        # agent loop 一步只执行一个调用，所以要求端点一次只给一个（否则并行调用会被丢弃）。
        # 这是可选参数：不支持的端点 400，下面的循环会把它去掉再试。
        kwargs["parallel_tool_calls"] = False
    # 只有 qwen3 端点支持关闭思考；其它端点会因未知参数 400。
    if "qwen3" in str(config.get("model", "")).lower() and config.get("enable_thinking") is False:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    non_retryable = {400, 401, 403, 404, 422}
    read_timeout = float(config.get("timeout", 300))
    max_attempts = max(1, int(config.get("max_retries", 3)))
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        remaining: float | None = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelUnavailable(
                    "model call skipped: request deadline exceeded"
                    + (f" (last error: {describe_error(last_error)})" if last_error else ""))
        call_kwargs = dict(kwargs)
        if remaining is not None:
            # The SDK applies this as the per-request read timeout.
            call_kwargs["timeout"] = max(0.05, min(read_timeout, remaining))
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**call_kwargs),
                # Defense in depth: even an endpoint that streams headers and
                # then stalls cannot outlive the request budget.
                timeout=None if remaining is None else remaining + 5.0,
            )
            # Count before validating: a truncated or empty response was still
            # billed by the provider.
            if usage is not None:
                usage.add(getattr(response, "usage", None), role)
            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None)
            content = message.content or ""
            if require_text:
                if not content.strip():
                    # 关思考后偶发空回复：补 reasoning 兜底，仍空则当瞬时错误重试
                    content = (getattr(message, "reasoning", None) or "").strip()
                if not content:
                    raise RuntimeError("empty completion content")
                if getattr(choice, "finish_reason", None) == "length":
                    raise TruncatedCompletion(
                        f"completion truncated by max_tokens (tail: {content[-80:]!r})",
                        content=content,
                    )
            else:
                # 原生工具调用：有 tool_calls 就成功；都没有才算失败。
                if not tool_calls and not content.strip():
                    raise RuntimeError("empty completion content (no tool_calls)")
                if not tool_calls and getattr(choice, "finish_reason", None) == "length":
                    raise TruncatedCompletion(
                        f"completion truncated by max_tokens (tail: {content[-80:]!r})",
                        content=content,
                    )
            return choice
        except openai.APIStatusError as exc:
            last_error = exc
            # Endpoint-side context overflow: retrying the same prompt cannot
            # help.  Raise a distinct error so the caller shrinks the prompt.
            if exc.status_code == 400 and is_context_overflow(exc):
                raise ContextOverflow(describe_error(exc)) from exc
            # 端点不认识 ``parallel_tool_calls``（可选参数）就 400 —— 去掉它再试一次，
            # 而不是把整次调用判死。工具本身仍能正常工作，只是可能一次给多个调用。
            if (exc.status_code == 400 and "parallel_tool_calls" in kwargs
                    and "parallel_tool_calls" in str(exc)):
                kwargs.pop("parallel_tool_calls", None)
                continue
            if exc.status_code in non_retryable:
                raise
        except Exception as exc:  # noqa: BLE001 - 连接/超时/解析等瞬时错误
            if (isinstance(exc, TruncatedCompletion)
                    and not config.get("retry_truncated", True)):
                # The caller can use the partial content; retrying the same
                # max_tokens produces the same truncation and is very slow.
                raise
            last_error = exc
        if attempt < max_attempts:
            delay = backoff_seconds(last_error, attempt, config)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
            logger.warning(
                "llm retry role=%s model=%s attempt=%d/%d in=%.1fs error=%s",
                role or "default", config.get("model", "?"), attempt,
                max_attempts, delay, describe_error(last_error))
            await asyncio.sleep(delay)

    if isinstance(last_error, TruncatedCompletion):
        raise TruncatedCompletion(f"chat completion failed after retries: {last_error}")
    raise ModelUnavailable(
        f"chat completion failed after {max_attempts} attempt(s): "
        f"{describe_error(last_error)}")


def describe_error(exc: Exception | None) -> str:
    """把异常写成带**底层原因**的一行。

    OpenAI SDK 的连接失败只把顶层异常写成 ``APIConnectionError: Connection error.``
    —— 真正有用的信息（``ConnectError`` / ``ConnectTimeout``）在 ``__cause__`` 里，不打出来
    就完全没法判断"是超时、被重置、还是连不上"。
    """
    if exc is None:
        return "unknown error"
    text = f"{type(exc).__name__}: {exc}"
    cause: BaseException | None = exc
    seen = set()
    while getattr(cause, "__cause__", None) is not None and id(cause) not in seen:
        seen.add(id(cause))
        cause = cause.__cause__
    if cause is not None and cause is not exc:
        text += f" (cause: {type(cause).__name__}: {cause!r})"
    return text


async def chat_completion(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    usage: UsageTracker | None = None,
    role: str = "",
    deadline: float | None = None,
) -> str:
    """调一次 chat completion 并返回正文（文本协议：正文就是全部信息）。"""
    choice = await _complete_choice(
        client, config, messages, require_text=True,
        usage=usage or CURRENT_USAGE.get(), role=role, deadline=deadline)
    message = choice.message
    content = message.content or ""
    if not content.strip():
        # 关思考后偶发空回复：补 reasoning 兜底（返回值不再校验，故这里再兜一次）
        content = (getattr(message, "reasoning", None) or "").strip()
    return content


async def chat_completion_message(
    client: openai.AsyncOpenAI,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    usage: UsageTracker | None = None,
    role: str = "",
    deadline: float | None = None,
) -> Any:
    """调一次并返回 assistant **message 对象**（含原生 ``tool_calls``）。

    给"用原生函数调用而不是文本协议"的模型（如 gpt-4o-mini）用：``tools`` 非空时走
    ``tools=``，``message.tool_calls`` 是要执行的动作，``message.content`` 是纯文本
    （在 agent 协议里就是"做完了"信号）。调用方负责把两者折回自己的协议。
    """
    choice = await _complete_choice(
        client, config, messages, tools=tools or None, require_text=not tools,
        usage=usage or CURRENT_USAGE.get(), role=role, deadline=deadline,
    )
    return choice.message
