"""Pure-CPU resilience tests: crash recovery, resume, cache, breaker, deadline.

    python scripts/test_resilience.py

No network and no model calls: every failure path uses stubs.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem import llm  # noqa: E402
from asmem.api.store import (META_BACKUP_NAME, UserStore,  # noqa: E402
                             _payload_hash, ensure_layout, recover_all)
from asmem.io import atomic_write_text  # noqa: E402
from asmem.retrieval.dense import OpenAIEmbedder  # noqa: E402
from asmem.search import runner  # noqa: E402
from asmem.search.postagent import run_postagent  # noqa: E402

MESSAGES = [{"role": "user", "content": "Alice: I live in Paris."}]


def _record() -> dict:
    return {"id": "s1_00001", "text": "Alice lives in Paris.", "kind": "message",
            "time": "", "role": "Alice", "source": "s1", "meta": {}}


def _write_memory(path: Path, records: list[dict]) -> None:
    atomic_write_text(path, "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records))


async def test_recover_pending() -> None:
    """A crash after the WAL entry but before the memory append is settled."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = UserStore("alice", root)
        payload_hash = _payload_hash("alice", "s1", MESSAGES)
        records = [{
            "id": "s1_00001", "text": "I live in Paris.", "kind": "message",
            "time": "", "role": "Alice", "source": "s1", "meta": {},
        }]
        meta = {
            "user_id": "alice",
            "sessions": {"s1": {"index": 1, "next": 2}},
            "next_index": 2,
            "applied": [],
            "applied_hashes": {},
            "pending": {"r1": {"session_id": "s1", "next": 2,
                               "records": records,
                               "payload_hash": payload_hash}},
        }
        atomic_write_text(store.meta_path, json.dumps(meta))
        assert store.stats()["pending"] == 1

        settled = await recover_all(root)
        assert settled == 1, settled
        assert [item["id"] for item in store.records()] == ["s1_00001"]
        meta_after = store._meta()
        assert not meta_after["pending"]
        assert "r1" in meta_after["applied"]

        # The retried request is now idempotent, not duplicated.
        assert await store.append("r1", "s1", MESSAGES) == 0
        assert len(store.records()) == 1


async def test_corrupt_meta_uses_backup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = UserStore("alice", Path(tmp))
        await store.append("r1", "s1", MESSAGES)
        good = store.meta_path.read_text(encoding="utf-8")
        # Simulate a torn write of the primary; the backup still has the WAL.
        store.meta_path.write_text('{"applied": [', encoding="utf-8")
        atomic_write_text(store.meta_path.with_name(META_BACKUP_NAME), good)
        meta = store._meta()
        assert "r1" in meta["applied"], meta
        # And the append path still sees the request as applied.
        assert await store.append("r1", "s1", MESSAGES) == 0


async def test_postagent_deadline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "inputs").mkdir()
        _write_memory(workspace / "inputs" / "memory.jsonl", [_record()])

        calls = {"n": 0}

        async def call_model(messages, schemas):
            calls["n"] += 1
            raise AssertionError("the deadline must stop the loop before the model call")

        result = await run_postagent(
            workspace=workspace, tools={}, call_model=call_model, schemas=None,
            config={"question": "Where does Alice live?", "max_steps": 4,
                    "deadline": time.monotonic() - 1})
        assert result.stop_reason == "deadline", result.stop_reason
        assert calls["n"] == 0


async def test_model_breaker_fails_fast() -> None:
    model = runner.ModelRunner({
        "generator": {"model": "stub", "base_url": "http://127.0.0.1:1/v1"},
        "llm_breaker_failures": 1, "max_retries": 1, "trust_env": False,
    })
    model._record_failure(llm.ModelUnavailable("endpoint down"))
    try:
        await model.call([{"role": "user", "content": "hi"}])
    except llm.ModelUnavailable as exc:
        assert "circuit open" in str(exc), exc
    else:
        raise AssertionError("an open breaker must raise before calling the endpoint")
    await model.aclose()


async def test_embedding_client_proxy_opt_out() -> None:
    """The embedding transport shares the proxy-aware OpenAI 3.x factory."""
    embedder = OpenAIEmbedder({
        "model": "stub-embedding",
        "base_url": "https://embedding.invalid/v1",
        "trust_env": False,
        "concurrency": 2,
    })
    client = embedder._get_client()
    assert client is not None
    await embedder.aclose()


async def test_search_cache_with_request_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        _write_memory(memory, [_record()])
        config = {"retrieval": {"mode": "coarse"}, "max_evidence": 8}
        first = await runner.run_question(
            memory_path=memory, question="Alice Paris", config=config,
            run_root=root / "runs", cache_dir=root / "index",
            request_id="req-1", user_id="alice")
        assert not first.get("cached")
        assert first["evidence"], first["summary"]
        assert (Path(first["run_dir"]) / "summary.json").exists()

        # A retry with the same id returns the finished run without rerunning.
        second = await runner.run_question(
            memory_path=memory, question="Alice Paris", config=config,
            run_root=root / "runs", cache_dir=root / "index",
            request_id="req-1", user_id="alice")
        assert second.get("cached") is True
        assert Path(second["run_dir"]) == Path(first["run_dir"])
        assert len(second["evidence"]) == len(first["evidence"])


class _DeadModel:
    """Model stub whose endpoint is down; all calls fail fast."""

    def __init__(self) -> None:
        self.calls = 0
        self.usage = llm.UsageTracker()
        self.schemas = None

    async def call(self, messages, schemas=None, usage=None, deadline=None):
        self.calls += 1
        raise llm.ModelUnavailable("stub endpoint down")

    async def text(self, messages, usage=None, deadline=None):
        self.calls += 1
        raise llm.ModelUnavailable("stub endpoint down")

    async def aclose(self) -> None:
        pass


async def test_search_resume_keeps_evidence() -> None:
    """An interrupted run continues from the evidence it already wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        _write_memory(memory, [_record()])
        question = "Where does Alice live?"
        options: list[str] = []
        payload_key = runner._payload_key(memory, question, options)
        run_id = runner._run_id(question, "alice\0req-1", payload_key)
        # Flat v2 layout: runs/<id>/evidence.jsonl (no ``qa`` level).
        qa = root / "runs" / run_id
        qa.mkdir(parents=True)
        claim = {"content": "Alice lives in Paris.",
                 "metadata": {"source": ["s1_00001"], "claim_type": "claim",
                              "score": 0.9}}
        (qa / "evidence.jsonl").write_text(json.dumps(claim) + "\n",
                                           encoding="utf-8")
        (qa / "request.json").write_text(
            json.dumps({"question": question, "options": []}), encoding="utf-8")

        config = {"retrieval": {"mode": "hybrid"}, "max_steps": 2,
                  "search_timeout": 5, "max_evidence": 8}
        result = await runner.run_question(
            memory_path=memory, question=question, config=config,
            run_root=root / "runs", cache_dir=root / "index",
            model=_DeadModel(), request_id="req-1", user_id="alice")
        assert result["summary"]["resumed"] is True, result["summary"]
        contents = [str(item.get("content") or "") for item in result["evidence"]]
        assert any("Alice lives in Paris." in text for text in contents), contents
        # The dead endpoint stops the agent but never the request.
        assert result["summary"]["stop_reason"].startswith("model_error"), \
            result["summary"]["stop_reason"]


async def test_legacy_run_layout_reopens() -> None:
    """A v1 ``runs/<id>/qa/`` workspace is reopened in place, not duplicated."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        _write_memory(memory, [_record()])
        question = "Where does Alice live?"
        payload_key = runner._payload_key(memory, question, [])
        run_id = runner._run_id(question, "alice\0req-legacy", payload_key)
        qa = root / "runs" / run_id / "qa"
        qa.mkdir(parents=True)
        (qa / "evidence.jsonl").write_text(json.dumps({
            "content": "Alice lives in Paris.",
            "metadata": {"source": ["s1_00001"]}}) + "\n", encoding="utf-8")
        (qa / "request.json").write_text(
            json.dumps({"question": question, "options": []}), encoding="utf-8")

        result = await runner.run_question(
            memory_path=memory, question=question,
            config={"retrieval": {"mode": "hybrid"}, "max_steps": 1,
                    "max_evidence": 8},
            run_root=root / "runs", cache_dir=root / "index",
            model=_DeadModel(), request_id="req-legacy", user_id="alice")
        assert Path(result["run_dir"]) == qa, result["run_dir"]
        assert result["summary"]["resumed"] is True
        assert not (root / "runs" / run_id / "request.json").exists()


async def test_shared_bin_and_snapshot_links() -> None:
    """Runs share one bin directory and hardlink the live memory snapshot."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "api_store"
        ensure_layout(store)
        user_dir = store / "users" / "alice"
        memory = user_dir / "memory.jsonl"
        _write_memory(memory, [_record()])
        result = await runner.run_question(
            memory_path=memory, question="Alice Paris",
            config={"retrieval": {"mode": "coarse"}, "max_evidence": 8},
            run_root=user_dir / "runs", cache_dir=user_dir / "cache",
            shared_dir=store / "shared", request_id="req-links",
            user_id="alice")
        qa = Path(result["run_dir"])
        assert qa.parent.name == "runs"
        bin_link = qa / "bin"
        assert bin_link.is_symlink(), "runs/*/bin must link to shared/bin"
        assert (bin_link / "append_matches.py").is_file()
        assert (Path(__file__).resolve().parents[1] / "scripts" / "coarse").is_file()
        snapshot = qa / "inputs" / "memory.jsonl"
        assert snapshot.stat().st_ino == memory.stat().st_ino, \
            "run snapshot must reuse the live file's inode"
        request = json.loads((qa / "request.json").read_text(encoding="utf-8"))
        assert request["snapshot"]["memory"]["link"] == "hardlink", request
        assert (store / "store.json").is_file()


async def test_search_dead_model_completes() -> None:
    """No evidence at all still returns coarse candidates instead of raising."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        _write_memory(memory, [_record()])
        config = {"retrieval": {"mode": "hybrid"}, "max_steps": 2,
                  "search_timeout": 5, "max_evidence": 8}
        result = await runner.run_question(
            memory_path=memory, question="Alice Paris", config=config,
            run_root=root / "runs", cache_dir=root / "index",
            model=_DeadModel())
        assert result["evidence"], result["summary"]
        assert result["summary"]["degraded"] is True


def test_search_response_contract() -> None:
    """The Search payload satisfies the competition contract invariants."""
    from asmem.api.server import build_search_items

    evidence = [
        {"content": "claim A",
         "metadata": {"source": ["s1_00001"], "anchor_id": "s1_00001",
                      "score": 0.1}},
        {"content": "claim B",
         "metadata": {"source": ["s1_00001"], "anchor_id": "s1_00001",
                      "score": 0.9}},
        {"content": "claim A", "metadata": {"source": ["s1_00001"]}},
        {"content": "   ", "metadata": {}},
        {"content": "query text", "metadata": {"source": ["query"]}},
    ]
    records = {"s1_00001": {"id": "s1_00001", "time": "2024-01-01T00:00:00Z"}}
    items = build_search_items(evidence, records, top_k=100)
    assert len(items) == 3, items
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids)), ids
    assert all(isinstance(item.content, str) and item.content.strip()
               for item in items)
    scores = [float(item.score) for item in items]
    assert all(left >= right for left, right in zip(scores, scores[1:])), scores
    assert items[0].created_at == "2024-01-01T00:00:00Z"
    assert items[2].created_at is None      # omitted by exclude_none

    limited = build_search_items(evidence, records, top_k=2)
    assert len(limited) == 2 and limited[0].score > limited[1].score

    parts = build_search_items(evidence, records, multimodal=True, top_k=1)
    assert isinstance(parts[0].content, list)
    assert parts[0].content[0] == {"type": "text", "text": "claim A"}

    assert build_search_items(evidence, records, top_k=0) == []


def test_deadline_signature_fallback() -> None:
    """Wrappers without ``deadline=`` must still work (bench ModelView, stubs)."""
    class NoDeadline:
        async def call(self, messages, schemas=None):
            return None

        async def text(self, messages):
            return ""

    assert runner._accepts_deadline(NoDeadline().call) is False
    assert runner._accepts_deadline(NoDeadline().text) is False
    assert runner._accepts_deadline(runner.ModelRunner.text) is True
    assert runner._accepts_deadline(runner.ModelRunner.call) is True


def main() -> None:
    asyncio.run(test_recover_pending())
    asyncio.run(test_corrupt_meta_uses_backup())
    asyncio.run(test_postagent_deadline())
    asyncio.run(test_model_breaker_fails_fast())
    asyncio.run(test_embedding_client_proxy_opt_out())
    asyncio.run(test_search_cache_with_request_id())
    asyncio.run(test_search_resume_keeps_evidence())
    asyncio.run(test_legacy_run_layout_reopens())
    asyncio.run(test_shared_bin_and_snapshot_links())
    asyncio.run(test_search_dead_model_completes())
    test_search_response_contract()
    test_deadline_signature_fallback()
    print("test_resilience: OK")


if __name__ == "__main__":
    main()
