"""The postagent's file tools: ``read`` / ``write`` / ``edit`` / ``bash``.

- ``read``: UTF-8 read, truncated to 2000 lines / 50KB, with offset/limit.
- ``write``: append by default (``mode="replace"`` rewrites the whole file).
  This is the everyday evidence writer.
- ``edit``: exact ``oldText`` -> ``newText`` replacement, or an append alias;
  used for surgical corrections.
- ``bash``: stdout+stderr, tail-truncated, timeout kills the process group;
  supports ``shell_command_prefix``.

Writes are confined to the workspace.  ``inputs/`` (the memory snapshot) and
``memory.jsonl`` / ``state.jsonl`` are read-only at the tool layer; bash is not
a security boundary and must be sandboxed in production.

This module is pure CPU / stdlib: no model calls, no network.  Input errors
raise ``ToolError`` and become one failed observation, never a crashed loop.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import signal
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

# 输出截断上限（与 reference/tools.py 一致）
DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
UTF8_BOM = "﻿"


class ToolError(ValueError):
    """工具入参非法 / 前置条件不满足。会被渲染成失败的观测返回给模型。"""


@dataclass
class ToolResult:
    """一次工具调用的结果。

    ``text``     —— 给模型看的观测（会被写进对话历史）
    ``details``  —— 结构化明细（落 ``steps.jsonl``、供日志与排错，不进 prompt）
    ``ok``       —— 是否成功。失败只用于记账与"无进展检测"，不中断循环。
    """

    text: str
    details: dict[str, Any] = field(default_factory=dict)
    ok: bool = True


@dataclass
class ToolDefinition:
    """一个工具：给模型看的元信息 + JSON Schema + 异步执行器。

    ``description`` / ``guidelines`` / ``input_schema`` 由 ``tool.json`` 提供
    （统一管理、统一注入上下文）；执行器在本文件里按名字注册。
    """

    name: str
    description: str
    guidelines: tuple[str, ...]
    input_schema: Mapping[str, Any]
    executor: Callable[[Mapping[str, Any]], Awaitable[ToolResult]]
    # 一次调用是否**可能**改变工作目录的内容。无进展检测用它判断"这一步有没有可能推进"。
    mutating: bool = False

    async def run(self, args: Mapping[str, Any]) -> ToolResult:
        return await self.executor(args)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def atomic_write_text(path: Path, content: str) -> None:
    """**原子**写文本：先写同目录临时文件 → fsync → ``os.replace``。

    为什么必须这样：agent 会在多轮之间用 ``write`` 重写 ``evidence.jsonl``。若直接
    ``write_text``，模型输出被 ``max_tokens`` 截断时就会**原地留下一个残缺文件** ——
    而截断发生在写到一半时，文件长度甚至可能被截短，之前写好的证据同时丢失。
    原子替换保证任何时刻磁盘上的文件要么是旧版本、要么是完整的新版本。

    临时文件必须放**同一目录**（跨设备 ``os.replace`` 不是原子的）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        # 任何失败都不要留下临时文件（它会被 agent 的 `ls`/`find` 看到，造成困惑）
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class Truncation:
    """截断结果 + 元信息（``to_details`` 进 JSONL，``note`` 进观测文本）。"""

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False

    def to_details(self) -> dict[str, Any]:
        return {
            "truncated": self.truncated,
            "truncated_by": self.truncated_by,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "output_lines": self.output_lines,
            "output_bytes": self.output_bytes,
        }


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def truncate_head(content: str, *, max_lines: int, max_bytes: int) -> Truncation:
    """保留开头（``read`` 用）。"""
    lines = _split_lines_for_counting(content)
    total_lines, total_bytes = len(lines), len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return Truncation(content, False, None, total_lines, total_bytes, total_lines, total_bytes)

    first_line_bytes = len(lines[0].encode()) if lines else 0
    if first_line_bytes > max_bytes:
        return Truncation("", True, "bytes", total_lines, total_bytes, 0, 0,
                          first_line_exceeds_limit=True)

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode()) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes
    output = "\n".join(output_lines)
    return Truncation(output, True, truncated_by, total_lines, total_bytes,
                      len(output_lines), len(output.encode()))


def truncate_tail(content: str, *, max_lines: int, max_bytes: int) -> Truncation:
    """保留结尾（``bash`` 用：报错和最终结果都在尾部）。"""
    lines = _split_lines_for_counting(content)
    total_lines, total_bytes = len(lines), len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return Truncation(content, False, None, total_lines, total_bytes, total_lines, total_bytes)

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode()) + (1 if output_lines else 0)
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                clipped = line.encode()[-max_bytes:].decode(errors="ignore")
                output_lines.insert(0, clipped)
                output_bytes = len(clipped.encode())
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes
    output = "\n".join(output_lines)
    return Truncation(output, True, truncated_by, total_lines, total_bytes,
                      len(output_lines), len(output.encode()), last_line_partial=last_line_partial)


# ---------------------------------------------------------------------------
# 入参解析（一律抛 ToolError，让模型知道错在哪）
# ---------------------------------------------------------------------------

def _str_arg(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _optional_int_arg(arguments: Mapping[str, Any], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, Any], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{name} must be a number")
    return float(value)


def _path_arg(arguments: Mapping[str, Any], name: str, *, cwd: Path) -> Path:
    value = _str_arg(arguments, name).strip()
    if not value:
        raise ToolError(f"{name} must not be empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _ensure_inside(path: Path, root: Path) -> None:
    """路径约束：``path`` 解析后必须落在 ``root`` 内。

    用 ``resolve()``（follow symlink）而不是字面比较 —— 否则 ``qa_0/inputs`` 这个指向
    只读输入目录的软链就能被用来写到外面去。
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ToolError(
            f"refusing to write outside the workspace: {path} "
            f"(resolves to {resolved}, workspace is {root.resolve()})"
        ) from None


def _reject_readonly(path: Path, root: Path) -> None:
    """The memory snapshot and derived state are read-only at the tool layer."""
    resolved = path.resolve()
    inputs = (root / "inputs").resolve()
    if resolved == inputs or inputs in resolved.parents:
        raise ToolError(f"refusing to write read-only inputs: {path}")
    if resolved.name in {"memory.jsonl", "state.jsonl"}:
        raise ToolError(f"refusing to write the memory snapshot: {path}")


# ---------------------------------------------------------------------------
# 每个路径一把写锁（同进程内串行化，避免并发写同一文件交错）
# ---------------------------------------------------------------------------

_file_locks: dict[Path, asyncio.Lock] = {}


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._key = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _file_locks.setdefault(self._key, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, *_exc: object) -> None:
        if self._lock is not None:
            self._lock.release()


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def create_read_tool(
    *,
    cwd: Path,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ToolDefinition:
    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        raw_path = _str_arg(arguments, "path")
        path = _path_arg(arguments, "path", cwd=cwd)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")
        if offset is not None and offset < 0:
            raise ToolError("offset must be at least 0")
        if limit is not None and limit < 1:
            raise ToolError("limit must be at least 1")
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")

        text = path.read_bytes().decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        all_lines = text.split("\n")
        start_line = 0 if offset is None or offset == 0 else offset - 1
        if start_line >= len(all_lines):
            raise ToolError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        user_limited: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected, max_lines=max_lines, max_bytes=max_bytes)
        start_display = start_line + 1
        details: dict[str, Any] = {
            "path": str(path),
            "lines": len(all_lines),
            "bytes": len(text.encode()),
            "truncation": truncation.to_details(),
        }

        if truncation.first_line_exceeds_limit:
            output = (
                f"[Line {start_display} is "
                f"{format_size(len(all_lines[start_line].encode()))}, exceeds the "
                f"{format_size(max_bytes)} limit. Read a slice of it with bash instead, e.g. "
                f"`sed -n '{start_display}p' {raw_path} | head -c {max_bytes}`]"
            )
        else:
            output = truncation.content
            end_display = start_display + max(truncation.output_lines, 1) - 1
            next_offset = end_display + 1
            if truncation.truncated:
                why = "lines" if truncation.truncated_by == "lines" else format_size(max_bytes)
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"(limit: {why}). Use offset={next_offset} to continue.]"
                )
            elif user_limited is not None and start_line + user_limited < len(all_lines):
                remaining = len(all_lines) - (start_line + user_limited)
                output += (
                    f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                )
        return ToolResult(text=output or "(empty file)", details=details)

    return ToolDefinition(
        name="read",
        description="",              # 由 tool.json 提供
        guidelines=(),
        input_schema={},
        executor=execute,
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------

def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _count_occurrences(content: str, text: str) -> int:
    count, start = 0, 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _edits_arg(arguments: Mapping[str, Any]) -> list[dict[str, str]]:
    value = arguments.get("edits")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            value = parsed
    if not isinstance(value, list) or not value:
        raise ToolError(
            "edits must be a non-empty array of {oldText, newText} objects"
        )
    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolError(f"edits[{index}] must be an object")
        old_text, new_text = item.get("oldText"), item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolError(f"edits[{index}].oldText and edits[{index}].newText must be strings")
        edits.append({"oldText": old_text, "newText": new_text})
    return edits


def apply_edits(normalized: str, edits: list[dict[str, str]], path: str) -> tuple[str, str]:
    """把若干条精确替换应用到 ``normalized``，返回 ``(原文, 新文)``。

    **全部校验通过才返回** —— 任一条 ``oldText`` 找不到或出现多次，就抛错，文件不动。
    """
    normalized_edits = [
        {"oldText": _normalize_to_lf(e["oldText"]), "newText": _normalize_to_lf(e["newText"])}
        for e in edits
    ]
    total = len(normalized_edits)
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolError(f"edits[{index}].oldText must not be empty in {path}")

    spans: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        occurrences = _count_occurrences(normalized, edit["oldText"])
        label = "the text" if total == 1 else f"edits[{index}]"
        if occurrences == 0:
            raise ToolError(
                f"Could not find {label} in {path}. oldText must match exactly, "
                f"including all whitespace and newlines."
            )
        if occurrences > 1:
            raise ToolError(
                f"Found {occurrences} occurrences of {label} in {path}. Each oldText must be "
                f"unique -- add more surrounding context to disambiguate."
            )
        start = normalized.index(edit["oldText"])
        spans.append((start, start + len(edit["oldText"]), edit["newText"]))

    previous_end = -1
    for start, end, _ in sorted(spans):
        if start < previous_end:
            raise ToolError("Edits must not overlap")
        previous_end = end

    new_content = normalized
    for start, end, new_text in sorted(spans, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized:
        raise ToolError(
            f"No changes made to {path}: the replacement produced identical content."
        )
    return normalized, new_content


def _record_text(record: Any) -> str:
    """一条记录的正文字段（新版 ``content``，兼容旧版 ``memory``）。"""
    if not isinstance(record, dict):
        return ""
    value = record.get("content")
    if not isinstance(value, str):
        value = record.get("memory")
    return value.strip() if isinstance(value, str) else ""


def _record_sources(record: Any) -> set[str]:
    """一条记录 cite 的 source id 集合。"""
    if not isinstance(record, dict):
        return set()
    raw = (record.get("metadata") or {}).get("source") or []
    return {s for s in raw if isinstance(s, str) and s}


def near_duplicate_of(appended: str, previous: str) -> tuple[int, str] | None:
    """追加内容里是否含有"**同一事实、且没带新 source**"的记录。

    返回 ``(已有行号, 已有正文片段)``，没有则 ``None``。

    实测背景：agent 会反复追加同一个事实的换说法（31% 的 QA 结束时存在近重复对），
    既不加信息，又让 evidence 膨胀、稀释下游 answer 的注意力。判据复用
    ``similarity.is_same_fact``（带主语/专有名词护栏，不会把 "pet named pup" 与
    "pet named Luna" 并掉）。

    **带了新 source 的重复一律放行** —— 那是"给同一条事实补来源"或"把两条并成一条"，
    是正当操作（``is_same_fact`` 判的是正文，来源是新信息）。
    """
    existing: list[tuple[int, Any]] = []
    for lineno, line in enumerate(previous.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict):
            existing.append((lineno, record))
    if not existing:
        return None

    for line in appended.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        text = _record_text(record)
        if not text:
            continue
        new_sources = _record_sources(record)
        for lineno, old in existing:
            old_text = _record_text(old)
            if not old_text or " ".join(text.split()).lower() != " ".join(old_text.split()).lower():
                continue
            if new_sources - _record_sources(old):
                continue        # 带了新来源 → 补来源/合并，放行
            return lineno, old_text
    return None


def create_edit_tool(*, cwd: Path, confine: bool = True,
                     default_write_file: str = "evidence.jsonl",
                     reject_near_duplicates: bool = True,
                     allow_append: bool = True) -> ToolDefinition:
    """``edit`` —— **唯一的写文件工具**（覆盖写已删除）。

    两种模式，靠参数二选一：

    - **replace**：``edits=[{oldText, newText}, ...]``，精确替换。用于修正已有记录。
    - **append**：``append="<新记录>"``，把文本追加到文件末尾（**文件不存在则创建**）。

    ``default_write_file`` 是缺 ``path`` 时的默认写目标：search 是 ``evidence.jsonl``，
    add 侧是 ``atommem.jsonl``（它唯一的产物）。

    **为什么要有 append 模式**：这套流程是"每看完一个 session 就记一条"，也就是**逐条追加**。
    早先让模型用替换模式自己造 ``oldText``（拿已有的最后一行当锚点），实测 8B 经常复不出那
    一行精确文本 —— 要么近似、要么留空，``edit`` 失败后退回 ``write`` 整份覆盖，把前面几个
    session 的证据全冲掉。**追加是这套流程的一等操作，就该是一等参数**，不该让模型用替换去模拟。
    同时 append 模式**文件不存在时直接创建**，修掉了"第一次追加时 evidence.jsonl 还不存在、
    edit 直接报 File not found"这个 bug —— 反复出现的 edit 失败里，第一个 session 那次必然踩到。

    追加的防护：``append`` 文本为空 → 拒绝（空写只可能是截断）；追加内容与文件末尾已有的
    最后一条**逐字重复** → 拒绝（模型有时会把刚刚写过的又追加一遍）；追加内容与**任意一条**
    已有记录是**同一事实、且没带新 source** → 也拒绝（``reject_near_duplicates``，换措辞的
    重复追加只涨体积不涨信息）。
    """
    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        # 缺 `path` 时默认 ``default_write_file``：这是本流程**唯一**要写的文件，而模型实测会
        # 忘记填 path（无论 append 还是 replace），旧行为是报 "path must be a string, got
        # NoneType" —— 然后它把整条一模一样的调用重复 10 次，步数全烧光、0 条证据。schema 里
        # 本就写着"path 省略时默认 evidence.jsonl"，所以两种模式都替它补上（不只是 append）。
        if arguments.get("path") is None and (
            arguments.get("append") is not None or arguments.get("edits") is not None
        ):
            arguments = {**arguments, "path": default_write_file}
        path = _path_arg(arguments, "path", cwd=cwd)
        if confine:
            _ensure_inside(path, cwd)
            _reject_readonly(path, cwd)
        if path.is_dir():
            raise ToolError(f"Could not edit file: {path}. Path is a directory.")

        # ---- append 模式（文件不存在则创建）----
        if "append" in arguments and arguments.get("append") is not None:
            if not allow_append:
                raise ToolError(
                    "this edit tool is replacement-only. Use {"
                    '"oldText": "...", "newText": "..."} to change evidence, '
                    "or the write tool to append a new record.")
            appended = _str_arg(arguments, "append")
            if not appended.strip():
                raise ToolError(
                    "append must not be empty -- an empty append is almost always a truncated "
                    "or malformed record. Write the record text inside `append`."
                )
            existed = path.exists()
            if existed:
                previous = path.read_text(encoding="utf-8", errors="replace")
            else:
                previous = ""
            # 逐字重复防护：新内容与文件里已有的最后一条完全相同 → 拒绝。
            # 模型偶尔会把刚写过的记录又追加一遍（它看不到自己写的内容时尤其如此）。
            stripped = appended.strip()
            if existed and stripped:
                existing_lines = [line.strip() for line in previous.splitlines() if line.strip()]
                if existing_lines and existing_lines[-1] == stripped:
                    raise ToolError(
                        f"That record is already the LAST line in {path.name} -- the file did "
                        f"not change. Append a DIFFERENT record, or reply with plain text if "
                        f"you are done."
                    )
            # 近重复防护：与已有记录"同一事实、且没带新 source" → 拒绝。换措辞的重复追加
            # 不加信息，还稀释下游 answer 的注意力（实测 31% 的 QA 出现近重复对）。
            if existed and stripped and reject_near_duplicates:
                duplicate = near_duplicate_of(appended, previous)
                if duplicate is not None:
                    lineno, old_text = duplicate
                    snippet = old_text[:140] + ("…" if len(old_text) > 140 else "")
                    raise ToolError(
                        f"You already recorded this fact at line {lineno} of {path.name}:\n"
                        f"  {snippet}\n"
                        f"Re-appending it in different words does not add evidence. If you found "
                        f"additional provenance, EDIT that line to add the new source id; "
                        f"otherwise search for a DIFFERENT fact."
                    )
            new_content = previous + appended + ("" if appended.endswith("\n") else "\n")
            async with _FileLock(path):
                atomic_write_text(path, new_content)
            added_lines = appended.count("\n") + (0 if appended.endswith("\n") else 1)
            return ToolResult(
                text=(
                    f"Appended {len(appended)} characters ({added_lines} line(s)) to "
                    f"{path.name}"
                    + (f" ({len(previous.splitlines())} -> {len(new_content.splitlines())} lines)."
                       if existed else f" (created it).")
                ),
                details={
                    "path": str(path), "mode": "append",
                    "appended_chars": len(appended), "created": not existed,
                    "previous_bytes": len(previous.encode()),
                    "bytes": len(new_content.encode()),
                },
            )

        # ---- replace 模式（精确替换）----
        edits = _edits_arg(arguments)
        if not path.exists():
            raise ToolError(
                f"Could not edit file: {path}. File not found. "
                f"(To create it with your first record, use append instead: "
                f'{{"tool":"edit","args":{{"path":"{path.name}","append":"<record>"}}}})'
            )

        async with _FileLock(path):
            raw = path.read_text(encoding="utf-8")
            bom, content = (UTF8_BOM, raw[1:]) if raw.startswith(UTF8_BOM) else ("", raw)
            ending = "\r\n" if content.find("\r\n") != -1 and (
                content.find("\n") == -1 or content.find("\r\n") < content.find("\n")
            ) else "\n"
            normalized = _normalize_to_lf(content)
            base_content, new_content = apply_edits(normalized, edits, str(path))
            restored = new_content.replace("\n", "\r\n") if ending == "\r\n" else new_content
            atomic_write_text(path, bom + restored)

        old_lines, new_lines = base_content.splitlines(), new_content.splitlines()
        diff = "\n".join(difflib.ndiff(old_lines, new_lines))
        return ToolResult(
            text=f"Successfully applied {len(edits)} replacement(s) to {path}.\n\n{diff}",
            details={"path": str(path), "mode": "replace", "edits": len(edits), "diff": diff},
        )

    return ToolDefinition(
        name="edit", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


def create_text_edit_tool(*, cwd: Path, confine: bool = True,
                          default_write_file: str = "evidence.jsonl"
                          ) -> ToolDefinition:
    """Replace text inside one evidence record's ``content``.

    Editing raw JSONL with ``oldText`` that spans a whole serialized object is
    brittle for a small model.  This tool takes the *content* substring, finds
    the single record that contains it (optionally restricted by ``id``), and
    replaces it in place.  The record count never changes, so an edit can
    correct a date or a value but cannot delete a coarse seed.
    """
    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        if arguments.get("path") is None:
            arguments = {**arguments, "path": default_write_file}
        path = _path_arg(arguments, "path", cwd=cwd)
        if confine:
            _ensure_inside(path, cwd)
            _reject_readonly(path, cwd)
        if arguments.get("append") is not None:
            raise ToolError(
                "this edit tool is replacement-only. Use {"
                '"oldText": "...", "newText": "..."} to change evidence, '
                "or the write tool to append a new record.")
        old_text = _str_arg(arguments, "oldText")
        new_text = _str_arg(arguments, "newText")
        if not old_text.strip():
            raise ToolError("oldText must not be empty")
        if not new_text.strip():
            raise ToolError(
                "newText must not be empty; this tool replaces text and does "
                "not delete records")
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        wanted = arguments.get("id", arguments.get("source"))
        wanted_ids = ({str(wanted)} if isinstance(wanted, str)
                      else {str(item) for item in wanted} if isinstance(wanted, list)
                      else set())

        async with _FileLock(path):
            raw = path.read_text(encoding="utf-8", errors="replace")
            records: list[dict[str, Any]] = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    raise ToolError(
                        "evidence.jsonl contains a non-JSON line; fix it before "
                        "using edit") from None
                if not isinstance(value, dict):
                    raise ToolError("evidence.jsonl lines must be JSON objects")
                records.append(value)

            matches: list[int] = []
            for index, record in enumerate(records):
                content = _record_text(record)
                if old_text not in content:
                    continue
                if wanted_ids:
                    meta = record.get("metadata")
                    sources = set()
                    if isinstance(meta, dict):
                        sources.update(str(item) for item in (meta.get("source") or []))
                    if record.get("id"):
                        sources.add(str(record["id"]))
                    if not (sources & wanted_ids):
                        continue
                matches.append(index)
            if not matches:
                raise ToolError(
                    f"oldText not found in any evidence record: {old_text[:120]!r}")
            if len(matches) > 1:
                raise ToolError(
                    f"oldText occurs in {len(matches)} records; add `id` to "
                    "disambiguate or include more surrounding text")
            record = records[matches[0]]
            key = next((name for name in ("content", "memory", "text")
                        if isinstance(record.get(name), str)), "")
            before = str(record.get(key) or "")
            after = before.replace(old_text, new_text, 1)
            record[key] = after
            records[matches[0]] = record
            final = "".join(json.dumps(item, ensure_ascii=False) + "\n"
                            for item in records)
            atomic_write_text(path, final)

        return ToolResult(
            text=(f"Replaced {len(old_text)} characters in record "
                  f"{matches[0] + 1} of {len(records)}.\n"
                  f"before: {before[:200]}\nafter:  {after[:200]}"),
            details={"path": str(path), "mode": "replace",
                     "record": matches[0] + 1, "records": len(records)},
        )

    return ToolDefinition(
        name="edit", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def _normalize_write_content(content: str) -> str:
    """Turn the common model encodings into one JSON object per line.

    Models often send a JSON object as a string, or double-escape newlines
    (``\\n``) between records.  Writing that verbatim produces an invalid JSONL
    file, and the evidence validator then silently drops the record.  When the
    content is JSON we canonicalize it; when it is plain text we leave it alone.
    """
    text = (content or "").strip()
    if not text:
        return ""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in value)
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n")
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            lines.append(line)
            continue
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def _jsonl_ok(content: str) -> bool:
    for line in (content or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(value, dict):
            return False
    return True


def _validate_evidence_content(content: str, path: Path) -> None:
    """Evidence must be JSONL, not prose; a bad append is worse than an error."""
    if path.name != "evidence.jsonl":
        return
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            raise ToolError(
                "evidence.jsonl must contain one JSON object per line. "
                "Send a claim with content + source (the harness wraps it), or "
                "use `python bin/append_matches.py 'term' --limit 5` for a raw record."
            ) from None
        if not isinstance(value, dict) or not any(
                isinstance(value.get(key), str)
                for key in ("content", "text", "memory")):
            raise ToolError(
                "each evidence record needs a non-empty content/text/memory field"
            )
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        sources = (metadata.get("source") or metadata.get("sources")
                   or value.get("source") or value.get("sources"))
        if sources is None and value.get("id") and (
                value.get("text") or value.get("memory")):
            sources = [value["id"]]
        if isinstance(sources, str):
            sources = [sources]
        if not sources:
            raise ToolError(
                "evidence record needs metadata.source (the ids you read), "
                "or pass a top-level `source` argument to write"
            )


def create_write_tool(*, cwd: Path, confine: bool = True,
                      default_write_file: str = "evidence.jsonl",
                      allow_replace: bool = True,
                      record_metadata: Mapping[str, Any] | None = None,
                      max_content_chars: int = 0) -> ToolDefinition:
    """Append evidence by default; ``mode="replace"`` rewrites a small file.

    Append is the normal path: the agent keeps the coarse seeds and adds one
    record/claim at a time.  Replace is explicit because an accidental
    overwrite would delete the evidence the answer is supposed to see.

    ``allow_replace=False`` makes append the only legal mode.  The fixed
    pipeline uses this: its ``write`` tool cannot destroy the coarse seeds.
    ``record_metadata`` is merged into the record the harness wraps around a
    plain-text claim, so a pipeline write is attributable to its stage.
    ``max_content_chars`` rejects an oversized claim instead of silently
    flooding the answer prompt.
    """
    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        if arguments.get("path") is None:
            arguments = {**arguments, "path": default_write_file}
        path = _path_arg(arguments, "path", cwd=cwd)
        if confine:
            _ensure_inside(path, cwd)
            _reject_readonly(path, cwd)
        if path.is_dir():
            raise ToolError(f"Could not write file: {path}. Path is a directory.")

        content = arguments.get("content")
        if not isinstance(content, str):
            content = arguments.get("text")
        if not isinstance(content, str) or not content.strip():
            raise ToolError("content must be a non-empty string")
        content = _normalize_write_content(content)
        if not content:
            raise ToolError("content must contain at least one record")
        if max_content_chars and len(content) > max_content_chars:
            raise ToolError(
                f"content is {len(content)} characters; the limit is "
                f"{max_content_chars}. Write one compact record at a time.")

        source = arguments.get("source", arguments.get("sources"))
        if isinstance(source, str):
            source = [source]
        if isinstance(source, list):
            source = [str(item) for item in source if str(item).strip()]
        else:
            source = []
        if path.name == "evidence.jsonl" and source and not _jsonl_ok(content):
            try:
                score = float(arguments.get("score", 0.8))
            except (TypeError, ValueError):
                score = 0.8
            metadata: dict[str, Any] = {"source": source, "score": score}
            if record_metadata:
                metadata.update(dict(record_metadata))
            content = json.dumps({
                "content": content.strip(),
                "metadata": metadata,
            }, ensure_ascii=False)
        _validate_evidence_content(content, path)
        mode = str(arguments.get("mode") or "append").strip().lower()
        if not allow_replace and mode == "replace":
            raise ToolError(
                "this write tool is append-only. Put the new record in `content`; "
                "do not replace the file.")
        if mode not in {"append", "replace"}:
            raise ToolError("mode must be 'append' or 'replace'")

        async with _FileLock(path):
            if mode == "replace":
                final = content if content.endswith("\n") else content + "\n"
                atomic_write_text(path, final)
                return ToolResult(
                    text=(f"Replaced {path.name} with {len(content)} characters "
                          f"({len(final.splitlines())} line(s))."),
                    details={"path": str(path), "mode": "replace",
                             "bytes": len(final.encode()), "lines": len(final.splitlines())},
                )

            existed = path.exists()
            previous = (path.read_text(encoding="utf-8", errors="replace")
                        if existed else "")
            stripped = content.strip()
            if existed:
                lines = [line.strip() for line in previous.splitlines() if line.strip()]
                if lines and lines[-1] == stripped:
                    raise ToolError(
                        f"That record is already the LAST line in {path.name}. "
                        "Append a different record, or finish.")
            final = previous + content + ("" if content.endswith("\n") else "\n")
            atomic_write_text(path, final)
            return ToolResult(
                text=(f"Appended {len(content)} characters to {path.name} "
                      f"({0 if not existed else len(previous.splitlines())} -> "
                      f"{len(final.splitlines())} lines)."),
                details={"path": str(path), "mode": "append",
                         "appended_chars": len(content), "created": not existed,
                         "bytes": len(final.encode()), "lines": len(final.splitlines())},
            )

    return ToolDefinition(
        name="write", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

def _shell_executable() -> str | None:
    """本机 bash 的绝对路径；找不到返回 ``None``。

    这是名为 ``bash`` 的工具，命令就该由 bash 解释 —— 尤其是引号规则（``'.content="x"'``
    这类在 POSIX shell 里合法、在 Windows ``cmd.exe`` 里会被吃掉引号）。只有当本机确实
    没有 bash 时才退回 ``create_subprocess_shell``（POSIX 上是 ``/bin/sh``，Windows 上是
    ``cmd.exe``）—— 那里命令的语法就不保证了，但至少不是直接崩。

    路径必须是**绝对路径**：POSIX 上 ``execvp`` 会查 PATH，Windows 的 ``CreateProcess``
    不会（它要求带扩展名），裸 ``bash`` 直接 ``WinError 2``。
    """
    import shutil

    return shutil.which("bash")


def _fix_grep_alternation(command: str) -> str:
    """给"用了 ``|`` 却没加 ``-E``"的 grep 补上 ``-E``。

    **这是 harness 该替模型兜住的一类错**，不是 prompt 能解决的：``grep "study|learn"``
    在基本正则里 ``|`` 是**字面竖线**，于是这条命令永远匹配不到任何东西，返回空 ——
    模型得到的观测是"没找到"，它据此判定"库里没有"，然后要么换词、要么直接放弃。
    实测一次全量里 **95 次带 ``|`` 的 grep 有 63 次没加 ``-E``**，直接造成了若干条
    零证据 QA（如 qa2 "What fields would Caroline pursue" 连查 12 次全是这种空 grep）。

    判定要保守：只处理**确实在正则里用了 ``|`` 分隔**的情形。所以取第一条 ``grep``
    之后引号里的那个 pattern 看有没有 ``|`` —— 不看整条命令（命令里其它地方的 ``|``
    是管道，不能当 alternation）。已经带 ``-E`` / ``-e`` 的不动。
    """
    import re

    def fix_one(match: re.Match[str]) -> str:
        head, quote, pattern = match.group(1), match.group(2), match.group(3)
        if "|" not in pattern or "\\|" in pattern:
            # No alternation, or basic-regex alternation (\|) that already works.
            return match.group(0)
        # head 是 "grep -in " 这种（到 pattern 的开引号为止）；已有 -E/-e 就不动
        if re.search(r"(?:^|\s)-[A-Za-z]*[Ee]", head):
            return match.group(0)
        # 在 grep 与其选项之后插入 -E：把 "grep -in " -> "grep -Ein "
        new_head = re.sub(r"\bgrep\b(\s+-[A-Za-z]+)?", lambda m: "grep" +
                          (m.group(1).rstrip() + "E" if m.group(1) else " -E"), head, count=1)
        if new_head == head:  # 兜底：没匹配到就粗暴加 -E
            new_head = head.replace("grep", "grep -E", 1)
        return new_head + quote + pattern + quote

    # 匹配 grep ... "<pattern>" 或 grep ... '<pattern>'（只看第一个引号串）
    return re.sub(r'(\bgrep\b[^"\']*?)(["\'])([^"\']*)\2', fix_one, command, count=1)


def _prioritize_corpus_hits(text: str) -> str:
    """Generalized memory has one file, so hit order needs no adjustment."""
    return text


def create_bash_tool(
    *,
    cwd: Path,
    shell_command_prefix: str | None = None,
    default_timeout: float | None = None,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ToolDefinition:
    prefix = shell_command_prefix.strip() if shell_command_prefix else None
    shell_executable = _shell_executable()

    async def execute(arguments: Mapping[str, Any]) -> ToolResult:
        command = _str_arg(arguments, "command")
        fixed = _fix_grep_alternation(command)
        note = ""
        if fixed != command:
            note = (
                f"\n\n[note: your grep used `|` as alternation but lacked `-E`, so it would "
                f"have matched a literal `|` and found nothing. Ran it as: "
                f"{fixed.splitlines()[0][:160]}]"
            )
            command = fixed
        timeout = _optional_float_arg(arguments, "timeout")
        if timeout is not None and timeout <= 0:
            raise ToolError("timeout must be greater than 0")
        effective_timeout = timeout if timeout is not None else default_timeout
        start = monotonic()
        # 用 ``bash -c`` 执行，而不是 create_subprocess_shell(executable=...)。
        #
        # 两个理由：① 前缀里是 POSIX 语法（``export PATH="a:b"``），只有 bash 会解释；
        # ② Windows 上 ``create_subprocess_shell`` 会把 executable 拼进 ``cmd.exe /c``
        # 的命令行 —— 实测 ``C:\Program Files\...\bash.EXE`` 被拆成 ``/c/Program: Files\...``
        # 直接 "No such file or directory"。走 bash -c 没有这一层：脚本是**一个参数**，
        # 引号规则由我们自己定。
        if shell_executable:
            argv = [shell_executable, "-c", f"{prefix}\n{command}" if prefix else command]
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
        else:
            process = await asyncio.create_subprocess_shell(
                f"{prefix}\n{command}" if prefix else command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
        communicate = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            output_bytes, _ = await asyncio.wait_for(communicate, timeout=effective_timeout)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_process_tree(process)
            try:
                output_bytes, _ = await communicate
            except asyncio.CancelledError:
                output_bytes = b""
        except asyncio.CancelledError:
            _kill_process_tree(process)
            communicate.cancel()
            raise

        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output, max_lines=max_lines, max_bytes=max_bytes)
        # 空输出**保持为空**，不在这里补一个 "(no output)" 占位串。
        #
        # 原因（实测踩的坑）：占位串会让 `agent.render_observation` 里"命令成功但没有任何
        # 输出"的检测**永远不成立**，于是那条专门解释"jq 选不到东西"的提示成了死代码。
        # 实测后果是一条 QA 因此空转 6 步：模型查一个不存在的 id → 得到 "(no output)" →
        # 以为是环境问题 → 换个写法再查 → 又被重复检测拒掉 → 循环到预算耗尽。
        # 观测文本由渲染层统一决定（见 render_observation），工具只负责如实返回空串。
        output_text = truncation.content
        # 语料 grep 的命中按记忆层重排（atommem -> msgmem -> summem）。只在**没有截断**时做：
        # 截断过的话尾部的 "[Showing lines X-Y ...]" 是按原始行号写的，重排会让它对不上。
        if not truncation.truncated:
            output_text = _prioritize_corpus_hits(output_text)
        full_output_path: str | None = None
        if truncation.truncated:
            full_output_path = _write_temp_output(output)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text += (
                    f"\n\n[Showing the last {format_size(truncation.output_bytes)} of line "
                    f"{end_line}. Full output: {full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {full_output_path}]"
                )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = (
                f"Command timed out after {effective_timeout:g} seconds"
                if effective_timeout else "Command timed out"
            )
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = f"{output_text}\n\n{status}" if output_text else status
        output_text += note

        return ToolResult(
            text=output_text,
            details={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_details(),
                "full_output_path": full_output_path,
                "output_bytes": len(output.encode()),
                "output_lines": truncation.total_lines,
            },
            ok=not timed_out and exit_code in (0, None),
        )

    return ToolDefinition(
        name="bash", description="", guidelines=(), input_schema={},
        executor=execute, mutating=True,
    )


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """杀掉整个进程组 —— 否则管道/复合命令的子进程会继续跑（参考实现同此处理）。"""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return


def _write_temp_output(output: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="asmem-bash-", suffix=".log", delete=False
    ) as handle:
        handle.write(output)
        return handle.name


# ---------------------------------------------------------------------------
# 工具集
# ---------------------------------------------------------------------------

def build_tools(
    *,
    cwd: Path,
    shell_command_prefix: str | None = None,
    bash_timeout: float | None = None,
    confine: bool = True,
    default_write_file: str = "evidence.jsonl",
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    reject_near_duplicates: bool = True,
    names: tuple[str, ...] = ("read", "write", "edit", "bash"),
    write_replace: bool = True,
    edit_append: bool = True,
    text_edit: bool = False,
    write_metadata: Mapping[str, Any] | None = None,
    write_max_chars: int = 0,
) -> dict[str, ToolDefinition]:
    """Build the default file/search tool set.

    ``read`` / ``write`` / ``edit`` cover the file operations the postagent
    needs; ``bash`` remains the escape hatch for grep/sed/jq and the workspace
    search CLI.  A caller can restrict ``names`` for tests or ablations.
    """
    available: dict[str, ToolDefinition] = {
        "read": create_read_tool(cwd=cwd, max_lines=max_lines,
                                 max_bytes=max_bytes),
        "write": create_write_tool(
            cwd=cwd, confine=confine, default_write_file=default_write_file,
            allow_replace=write_replace, record_metadata=write_metadata,
            max_content_chars=write_max_chars),
        "edit": (create_text_edit_tool(
            cwd=cwd, confine=confine, default_write_file=default_write_file)
            if text_edit else create_edit_tool(
                cwd=cwd, confine=confine, default_write_file=default_write_file,
                reject_near_duplicates=reject_near_duplicates,
                allow_append=edit_append)),
        "bash": create_bash_tool(
            cwd=cwd,
            shell_command_prefix=shell_command_prefix,
            default_timeout=bash_timeout,
            max_lines=max_lines,
            max_bytes=max_bytes,
        ),
    }
    return {name: available[name] for name in names if name in available}
