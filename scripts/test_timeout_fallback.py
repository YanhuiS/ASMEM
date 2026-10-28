"""Pure-CPU tests for the timeout fallback and context-overflow recovery.

    python scripts/test_timeout_fallback.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem import llm  # noqa: E402
from asmem.api.store import UserStore  # noqa: E402
from asmem.io import write_jsonl  # noqa: E402
from asmem.search import runner  # noqa: E402
from asmem.search.model import ModelRunner  # noqa: E402
from asmem.search.pipeline import (  # noqa: E402
    _bm25_only_config, quick_coarse_fallback, run_search,
)


def _memory(path: Path) -> None:
    write_jsonl(path, [{
        "id": "s1_00001",
        "text": "Alice lives in Paris and works as a teacher.",
        "kind": "message",
        "time": "2024-01-01T00:00:00Z",
        "role": "Alice",
        "source": "s1",
        "meta": {},
    }])


def test_bm25_config_isolated() -> None:
    config = {"retrieval": {"mode": "hybrid", "dense": {"mode": "on",
                                                       "model": "m"}}}
    fallback = _bm25_only_config(config)
    assert fallback["retrieval"]["mode"] == "bm25"
    assert fallback["retrieval"]["dense"]["mode"] == "off"
    # The caller's config is untouched.
    assert config["retrieval"]["mode"] == "hybrid"
    assert config["retrieval"]["dense"]["mode"] == "on"


def test_quick_coarse_fallback() -> None:
    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = root / "memory.jsonl"
            _memory(memory)
            config = {
                "retrieval": {
                    "mode": "hybrid",
                    "dense": {"mode": "off"},
                    "coarse": {"candidates": 8},
                },
                "fallback_evidence_limit": 4,
            }
            items = await quick_coarse_fallback(
                memory_path=memory, question="Where does Alice live?",
                config=config, limit=4, timeout=5.0)
            assert items, items
            assert any("Alice" in item["content"] for item in items)
            assert items[0]["metadata"]["verification"] == "bm25_fallback"
    asyncio.run(main())


class _DeadModel:
    def __init__(self) -> None:
        self.calls = 0
        self.usage = llm.UsageTracker()
        self.schemas = None

    async def call(self, messages, schemas=None, usage=None, deadline=None):
        raise llm.ModelUnavailable("dead endpoint")

    async def text(self, messages, usage=None, deadline=None):
        raise llm.ModelUnavailable("dead endpoint")


def test_run_search_writes_fallback_early() -> None:
    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = root / "memory.jsonl"
            _memory(memory)
            fallback = root / "fallback.jsonl"
            config = runner.load_config(None)
            config.update({
                "postagent_mode": "gate_loop",
                "search_timeout": 5,
                "pre_audit": False,
                "fallback_evidence_limit": 4,
            })
            config["retrieval"] = {
                "mode": "bm25",
                "dense": {"mode": "off"},
                "coarse": {"candidates": 8},
                "gate": {"evidence_limit": 4},
            }
            result = await run_search(
                memory_path=memory, question="Where does Alice live?",
                config=config, run_root=root / "runs",
                model=_DeadModel(), options=[], fallback_path=fallback)
            assert result["evidence"], result["summary"]
            assert fallback.exists() and fallback.stat().st_size > 0
            lines = [json.loads(line) for line in fallback.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            assert lines and any("Alice" in line.get("content", "")
                                 for line in lines)
    asyncio.run(main())


def test_model_runner_shrinks_on_context_overflow() -> None:
    async def main() -> None:
        original = llm.chat_completion
        calls: list[list[dict]] = []

        async def fake(client, config, messages, **kwargs):
            calls.append(messages)
            if len(calls) == 1:
                raise llm.ContextOverflow("context_length_exceeded")
            return "ok"

        llm.chat_completion = fake  # type: ignore[assignment]
        try:
            model = ModelRunner({
                "generator": {
                    "base_url": "http://127.0.0.1:1/v1",
                    "model": "fake",
                    "api_key": "x",
                    "max_retries": 1,
                },
                "model_input_token_cap": 1000,
                "model_output_reserve_tokens": 256,
                "context_overflow_retries": 1,
                "trust_env": False,
            })
            long = "evidence " * 400
            text = await model.text([
                {"role": "system", "content": "system"},
                {"role": "user", "content": long},
            ])
            assert text == "ok"
            assert len(calls) == 2, len(calls)
            assert len(str(calls[1][-1]["content"])) < len(
                str(calls[0][-1]["content"]))
        finally:
            llm.chat_completion = original  # type: ignore[assignment]
    asyncio.run(main())


def test_global_breaker() -> None:
    breaker = llm.EndpointBreaker()
    assert breaker.allow("k", 2, 30)[0] is True
    breaker.failure("k", 2)
    breaker.failure("k", 2)
    allowed, message = breaker.allow("k", 2, 30)
    assert allowed is False and "circuit" in message
    # Cooldown 0 always allows the probe.
    assert breaker.allow("k", 2, 0)[0] is True
    breaker.success("k")
    assert breaker.allow("k", 2, 30)[0] is True


def test_api_hard_timeout_returns_fallback() -> None:
    from fastapi.testclient import TestClient

    import asmem.api.server as server_mod
    from asmem.api.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = UserStore("alice", root)
        # Write the memory directly: creating an asyncio lock in one loop and
        # using it in TestClient's loop is not supported.
        write_jsonl(store.memory_path, [{
            "id": "s1_00001", "text": "Alice lives in Paris.",
            "kind": "message", "time": "", "role": "Alice",
            "source": "s1", "meta": {},
        }])
        original = server_mod.runner.run_question

        async def slow(**kwargs):
            # Emulate run_search's early fallback write, then hang.
            path = kwargs.get("fallback_path")
            if path is not None:
                write_jsonl(Path(path), [{
                    "content": "Alice lives in Paris (fallback file).",
                    "metadata": {"source": ["s1_00001"], "score": 0.9},
                }])
            await asyncio.sleep(10)
            return {"evidence": [], "summary": {}}

        server_mod.runner.run_question = slow  # type: ignore[assignment]
        try:
            app = create_app({
                "store_dir": str(root),
                "search_timeout": 30,
                "search_hard_timeout": 0.5,
                "search_queue_timeout": 2,
                "fallback_enabled": True,
                "fallback_timeout": 5,
                "fallback_concurrency": 2,
                "fallback_evidence_limit": 4,
            })
            with TestClient(app) as client:
                response = client.post("/search", json={
                    "query": "Where does Alice live?",
                    "options": [],
                    "user_id": "alice",
                    "top_k": 10,
                })
                assert response.status_code == 200
                data = response.json()["data"]
                assert data, response.text
                assert any("fallback file" in item["content"] for item in data)
        finally:
            server_mod.runner.run_question = original  # type: ignore[assignment]


def main() -> None:
    test_bm25_config_isolated()
    test_quick_coarse_fallback()
    test_run_search_writes_fallback_early()
    test_model_runner_shrinks_on_context_overflow()
    test_global_breaker()
    test_api_hard_timeout_returns_fallback()
    print("test_timeout_fallback: OK")


if __name__ == "__main__":
    main()
