"""Pure-CPU self-test: normalization, idempotent add, tool parsing, agent loop.

    python scripts/test_asmem.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem.add import normalize_input  # noqa: E402
from asmem.api.store import RequestConflictError, UserStore  # noqa: E402
from asmem.search import evidence, toolcfg  # noqa: E402
from asmem.search.postagent import ModelReply, run_postagent  # noqa: E402
from asmem.search.tools import build_tools  # noqa: E402


def test_dialogue() -> None:
    messages = [
        {"role": "user", "content": "Alice: I moved to Paris last week.", "timestamp": 1683540000000},
        {"role": "user", "content": "Bob: Nice! Is it warm there?", "timestamp": 1683540060000},
    ]
    family, records = normalize_input(messages, session_id="session_1", session_index=1)
    assert family == "dialogue", family
    assert records[0]["role"] == "Alice"
    assert records[0]["text"] == "I moved to Paris last week."
    assert records[0]["time"].startswith("2023-05-08")
    assert records[0]["id"] == "s1_00001"


def test_table() -> None:
    payload = {"columns": ["name", "closed"], "rows": [["A School", "1960"], ["B School", "2006"]]}
    family, records = normalize_input(payload, session_id="session_2", session_index=2)
    assert family == "table"
    assert len(records) == 2
    assert "name: B School" in records[1]["text"]
    assert records[1]["meta"]["data"] == ["B School", "2006"]


def test_triple() -> None:
    payload = {"subject": "Microsoft", "relation": "CEO", "object": "Steve Jobs"}
    _, records = normalize_input(payload, session_id="session_3", session_index=3)
    assert records[0]["text"] == "Microsoft | CEO | Steve Jobs"


def test_tool_parse() -> None:
    text = '{"tool":"bash","args":{"command":"grep -in x inputs/memory.jsonl"}}'
    calls = toolcfg.parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].tool == "bash"
    native = '{"name":"bash","arguments":"{\\"command\\":\\"grep x memory.jsonl\\"}"}'
    calls = toolcfg.parse_tool_calls(native)
    assert calls[0].tool == "bash" and calls[0].args["command"].startswith("grep")
    assert toolcfg.parse_tool_calls("I am done.") == []


async def test_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = UserStore("alice", Path(tmp))
        messages = [{"role": "user", "content": "Alice: hello"}]
        assert await store.append("r1", "s1", messages) == 1
        assert await store.append("r1", "s1", messages) == 0  # idempotent
        assert len(store.records()) == 1
        try:
            await store.append("r1", "s1", [{"role": "user", "content": "different"}])
        except RequestConflictError:
            pass
        else:
            raise AssertionError("same request_id with a different payload must conflict")
        assert (not store.state_path.exists()
                or not store.state_path.read_text(encoding="utf-8").strip())

        update = [{"role": "user",
                   "content": "UPDATE: quarterback | sport | American football"}]
        await store.append("r2", "s2", update)
        replacement = [{"role": "user",
                        "content": "UPDATE: quarterback | sport | Muay Thai"}]
        await store.append("r3", "s3", replacement)
        lines = [json.loads(line) for line in
                 store.state_path.read_text(encoding="utf-8").splitlines() if line]
        assert lines[0]["state_key"] == "quarterback|sport"
        assert lines[0]["status"] == "superseded"
        assert lines[0]["superseded_by"] == [lines[1]["source"]]
        assert lines[1]["status"] == "active"
        assert lines[1]["supersedes"] == [lines[0]["source"]]


async def test_agent_implicit_evidence() -> None:
    """A small model may reply with an evidence record instead of a bash call."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "inputs").mkdir()
        (workspace / "inputs" / "memory.jsonl").write_text(
            json.dumps({"id": "s1_00001", "text": "Alice lives in Paris.",
                        "kind": "message", "time": "", "role": "Alice",
                        "source": "s1", "meta": {}}) + "\n", encoding="utf-8")
        tools = build_tools(cwd=workspace, shell_command_prefix="export LC_ALL=C")

        def reply(text: str) -> ModelReply:
            return ModelReply(text=text, calls=toolcfg.parse_tool_calls(text),
                              assistant_entry={"role": "assistant", "content": text})

        script = [
            reply(json.dumps({"content": "Alice lives in Paris.",
                              "metadata": {"source": ["s1_00001"], "score": 0.9}})),
            reply("done"),
        ]

        async def call_model(messages, schemas):
            return script.pop(0)

        result = await run_postagent(
            workspace=workspace, tools=tools, call_model=call_model,
            schemas=None, config={"question": "Where does Alice live?",
                                  "max_steps": 4})
        assert result.stop_reason == "sufficient", result.stop_reason
        valid, rejected = evidence.validate_file(workspace / "evidence.jsonl")
        assert len(valid) == 1 and not rejected


async def test_agent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "inputs").mkdir()
        (workspace / "inputs" / "memory.jsonl").write_text(
            json.dumps({"id": "s1_00001", "text": "Alice lives in Paris.",
                        "kind": "message", "time": "", "role": "Alice",
                        "source": "s1", "meta": {}}) + "\n", encoding="utf-8")
        tools = build_tools(cwd=workspace, shell_command_prefix="export LC_ALL=C",
                            names=("bash",))

        def reply(text: str) -> ModelReply:
            return ModelReply(text=text, calls=toolcfg.parse_tool_calls(text),
                              assistant_entry={"role": "assistant", "content": text})

        record = json.dumps({"content": "Alice lives in Paris.",
                             "metadata": {"source": ["s1_00001"], "score": 0.9}})
        append = (
            "cat >> evidence.jsonl <<'EOF'\n" + record + "\nEOF\n"
        )
        script = [
            reply('{"tool":"bash","args":{"command":"grep -in Paris inputs/memory.jsonl"}}'),
            reply(json.dumps({"tool": "bash", "args": {"command": append}})),
            reply("done"),
        ]

        async def call_model(messages, schemas):
            return script.pop(0)

        result = await run_postagent(
            workspace=workspace, tools=tools, call_model=call_model,
            schemas=None, config={"question": "Where does Alice live?",
                                  "max_steps": 6,
                                  "no_progress_patience": 3})
        assert result.stop_reason == "sufficient", result.stop_reason
        valid, rejected = evidence.validate_file(workspace / "evidence.jsonl")
        assert len(valid) == 1 and not rejected
        assert valid[0]["content"] == "Alice lives in Paris."


def main() -> None:
    test_dialogue()
    test_table()
    test_triple()
    test_tool_parse()
    asyncio.run(test_store())
    asyncio.run(test_agent_implicit_evidence())
    asyncio.run(test_agent())
    print("test_asmem: OK")


if __name__ == "__main__":
    main()
