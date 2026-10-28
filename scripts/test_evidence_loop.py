#!/usr/bin/env python3
"""CPU-only tests for the gate + evidence loop.

    python scripts/test_evidence_loop.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem.retrieval.index import build_index  # noqa: E402
from asmem.search import evidence  # noqa: E402
from asmem.search.postagent import ModelReply  # noqa: E402
from asmem.search.evidence_loop import (  # noqa: E402
    _parse_action, run_evidence_loop, search_index)
from asmem.search.verifier import (  # noqa: E402
    _answer_supported, _gate_shape_ok)


def test_action_parse() -> None:
    assert _parse_action('{"action":"search","query":"x"}')[0] == "search"
    assert _parse_action('{"tool":"search","query":"x"}')[0] == "search"
    assert _parse_action('{"action":"take_note","indices":[1]}')[0] == "note"
    assert _parse_action('{"action":"finish"}')[0] == "finish"
    assert _parse_action("no json here")[0] == ""


def test_gate_checks() -> None:
    records = [{
        "content": "I'm still thinking about counseling and mental health work.",
        "metadata": {"source": ["s1"], "speaker": "Caroline"},
    }]
    assert not _answer_supported(
        "Yes, Caroline would pursue writing as a career option.",
        records[0]["content"],
        "Would Caroline pursue writing as a career option?")
    assert _answer_supported(
        "last year", "I painted that sunrise last year.",
        "When did Melanie paint a sunrise?")
    assert not _gate_shape_ok(
        "How long has Caroline had her current group of friends?",
        "since I was 17 or so", records,
        answer="about three years")


def test_search_and_loop() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        memory = root / "memory.jsonl"
        rows = [
            {"id": "s1", "text": "Alice moved to Paris in May 2023.",
             "time": "2023-05-01", "role": "Alice", "source": "s1"},
            {"id": "s2", "text": "Bob visited Rome in June 2023.",
             "time": "2023-06-01", "role": "Bob", "source": "s2"},
            {"id": "s3", "text": "Alice likes painting sunsets.",
             "time": "2023-07-01", "role": "Alice", "source": "s3"},
        ]
        memory.write_text("".join(json.dumps(row) + "\n" for row in rows),
                          encoding="utf-8")
        evidence.write_file(root / "evidence.jsonl", [{
            "content": "Alice moved to Paris in May 2023.",
            "metadata": {"source": ["s1"], "origin": "coarse",
                         "seeded": True}}])
        config = {"retrieval": {"dense": {"mode": "off"}}}

        async def run() -> None:
            index = await build_index(memory, config)
            hits = await search_index(index, "Where did Alice move?", top_k=2)
            assert hits and hits[0]["id"] == "s1", hits
            hits = await search_index(index, "Alice", exclude={"s1"}, top_k=2)
            assert all(hit["id"] != "s1" for hit in hits), hits

            replies = [
                ModelReply(text=json.dumps({
                    "action": "search", "query": "Bob Rome"}),
                    calls=[], assistant_entry={}),
                ModelReply(text=json.dumps({
                    "action": "take_note", "indices": [1]}),
                    calls=[], assistant_entry={}),
                ModelReply(text=json.dumps({"action": "finish"}),
                    calls=[], assistant_entry={}),
            ]
            seen: list[str] = []

            async def call_model(messages, schemas=None):  # noqa: ANN001
                seen.append(messages[-1]["content"])
                return replies[len(seen) - 1]

            result = await run_evidence_loop(
                workspace=root, index=index, call_model=call_model,
                config={
                    "question": "Where did Bob visit?",
                    "options": [],
                    "loop_max_steps": 3,
                    "loop_top_k": 3,
                    "loop_auto_note": 0,
                    "loop_note_chars": 500,
                    "loop_evidence_chars": 3000,
                    "loop_item_chars": 200,
                    "agent_max_context_tokens": 8000,
                    "pipeline_reserve_tokens": 1024,
                },
                missing=["where Bob visited"])
            await index.aclose()

            assert result.stop_reason == "evidence_loop_finish", result.stop_reason
            assert result.steps == 3, result.steps
            assert result.stats["notes"] == 1, result.stats
            assert any("Bob visited Rome" in item["content"]
                       for item in result.evidence), result.evidence
            assert "MISSING SLOTS" in seen[0], seen[0][:300]

        asyncio.run(run())


def main() -> int:
    test_action_parse()
    test_gate_checks()
    test_search_and_loop()
    print("evidence loop tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
