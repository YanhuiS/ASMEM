"""Tool-call parsing and schemas.  Small on purpose."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*|\s*```\s*$")


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def key(self) -> str:
        return f"{self.tool}:{json.dumps(self.args, sort_keys=True, ensure_ascii=False)}"

    def describe(self) -> str:
        text = json.dumps(self.args, ensure_ascii=False)
        return f"{self.tool}({text[:160]})"


def _objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object in the text, including inside <tool_call> blocks."""
    found: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for block in _TOOL_CALL_BLOCK.findall(text) or [text]:
        candidate = _FENCE.sub("", block.strip()).strip()
        index = 0
        while index < len(candidate):
            start = candidate.find("{", index)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(candidate, start)
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(value, dict):
                found.append(value)
            index = end
    return found


def _to_call(obj: dict[str, Any]) -> ToolCall | None:
    name = obj.get("tool") or obj.get("name")
    if not isinstance(name, str):
        # OpenAI function-call shape inside text.
        function = obj.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            name = function["name"]
            obj = function
    if not isinstance(name, str):
        return None
    args = obj.get("args", obj.get("arguments", obj.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name, args)


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for obj in _objects(text or ""):
        call = _to_call(obj)
        if call is not None:
            calls.append(call)
    return calls


def looks_like_call(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("{") or "<tool_call>" in stripped


def tool_schemas() -> list[dict[str, Any]]:
    """Default agent tools: read / write / edit / bash."""
    return [
        {"type": "function", "function": {
            "name": "read",
            "description": "Read a workspace file, optionally an offset/limit slice.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            }, "required": ["path"]},
        }},
        {"type": "function", "function": {
            "name": "write",
            "description": (
                "Append text to a workspace file (default evidence.jsonl). "
                "For a claim, pass content plus source ids and the harness wraps "
                "it into an evidence record. Use mode='replace' only to rewrite "
                "the whole file."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "array", "items": {"type": "string"}},
                "score": {"type": "number"},
                "mode": {"type": "string", "enum": ["append", "replace"]},
            }, "required": ["content"]},
        }},
        {"type": "function", "function": {
            "name": "edit",
            "description": (
                "Replace exact text in a workspace file. Each oldText must be "
                "unique; all edits are validated before the file is written."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "edits": {"type": "array", "items": {"type": "object", "properties": {
                    "oldText": {"type": "string"},
                    "newText": {"type": "string"},
                }, "required": ["oldText", "newText"]}},
            }, "required": ["edits"]},
        }},
        {"type": "function", "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace (grep, jq, sed, date, "
                "python). Use coarse for deterministic retrieval."
            ),
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
            }, "required": ["command"]},
        }},
    ]
