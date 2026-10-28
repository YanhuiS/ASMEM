"""Pure-CPU self-test for the coarse retrieval layer.

    python scripts/test_retrieval.py

No model calls and no network: dense uses the deterministic hashing embedder.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem.io import write_jsonl  # noqa: E402
from asmem.retrieval.coarse import run_coarse  # noqa: E402
from asmem import llm  # noqa: E402
from asmem.bench.report import summarize_group  # noqa: E402
from asmem.retrieval.dense import EmbeddingCache, HashingEmbedder  # noqa: E402
from asmem.retrieval.gate import evaluate_gate  # noqa: E402
from asmem.retrieval.index import build_index  # noqa: E402
from asmem.retrieval.parents import (  # noqa: E402
    choose_resolution, parent_text, resolve_record,
)
from asmem.retrieval.postprocess import (  # noqa: E402
    annotate_resolved_dates, claim_mode, expand_with_neighbors, fallback_evidence,
    fit_evidence, fuse_evidence, postprocess_evidence,
)
from asmem.retrieval.tokenize import extract_phrases, tokenize  # noqa: E402
from asmem.schema import chunk_text, normalize_messages  # noqa: E402
from asmem.search import evidence as evidence_mod  # noqa: E402
from asmem.state import derive_state  # noqa: E402
from asmem.tables import build_tables_db  # noqa: E402
from asmem.search import prompts as search_prompts  # noqa: E402
from asmem.search import runner, toolcfg, verifier  # noqa: E402
from asmem.search.postagent import ModelReply, run_postagent  # noqa: E402
from asmem.search.tools import ToolResult  # noqa: E402


def test_tokenizer() -> None:
    tokens = tokenize("上海 weather is nice")
    assert "上海" in tokens and "海" in tokens and "weather" in tokens
    phrases = extract_phrases('Find "Ada Lovelace" in 2024-01-02')
    assert any("Ada Lovelace" in item for item in phrases)
    assert any("2024-01-02" in item for item in phrases)


def test_chunk_policy() -> None:
    text = " ".join(f"Sentence {index} about Paris." for index in range(120))
    # Legacy explicit character width still works.
    pieces = chunk_text(text, max_chars=200, overlap_chars=40)
    assert len(pieces) > 3
    assert all(len(piece) <= 240 for piece in pieces)
    assert set(pieces[0][-40:].split()) & set(pieces[1][:60].split())

    # Token windows: every unit respects the budget, no ASCII word is cut,
    # and consecutive units overlap by the configured token count.
    from asmem.schema import chunk_units_tokens, token_count
    units = chunk_units_tokens(text, 60, overlap_tokens=10, min_tokens=5)
    assert len(units) > 3
    assert all(unit["tokens"] <= 60 for unit in units)
    assert all(token_count(unit["text"]) <= 60 for unit in units)
    for previous, current in zip(units, units[1:]):
        assert current["token_start"] > previous["token_start"]
        assert current["token_start"] <= previous["token_end"]
        assert previous["token_end"] - current["token_start"] <= 10
    # A paragraph boundary near the window end is preferred.
    structured = ("A " * 100) + "\n\n" + ("B " * 100)
    first = chunk_units_tokens(structured, 120, overlap_tokens=10, min_tokens=10)[0]
    assert first["text"].rstrip().endswith("A")
    assert "B" not in first["text"]

    _, records = normalize_messages(
        [{"role": "user", "content": text}], session_id="s1", session_index=1,
        chunking={"version": 7, "message_tokens": 60, "passage_tokens": 40,
                  "record_tokens": 50, "overlap_tokens": 10, "min_tokens": 5})
    assert len(records) > 1
    assert all(record["meta"]["chunk_version"].startswith("v7:") for record in records)
    assert all(record["meta"]["parts"] == len(records) for record in records)
    assert all(record["meta"]["tokens"] <= 60 for record in records)


def test_parent_chunks() -> None:
    text = ("The code word is ORANGE. " + "Filler sentence. " * 200).strip()
    _, records = normalize_messages(
        [{"role": "user", "content": text, "timestamp": 1704067200000}],
        session_id="s1", session_index=1, max_chars=400,
    )
    assert len(records) > 1
    assert {record["meta"]["parent_id"] for record in records}.__len__() == 1
    assert records[0]["meta"]["parts"] == len(records)
    assert "ORANGE" in parent_text(records[0], records)
    mode = choose_resolution(
        records[0], route="summary", question="Summarize the document",
        config={"parent_max_chars": 8000})
    assert mode == "parent"
    resolved, mode = resolve_record(
        records[0], records, route="summary", question="Summarize",
        config={"parent_max_chars": 8000})
    assert mode == "parent" and "ORANGE" in resolved


def _cache_records() -> list[dict]:
    return [
        {"id": "s1_00001", "text": "Alice moved to Paris in 2023.",
         "kind": "message", "time": "2023-05-08T00:00:00Z", "role": "Alice",
         "source": "s1", "meta": {"parent_id": "p1", "part": 1, "parts": 1}},
        {"id": "s1_00002", "text": "Bob lives in Berlin.",
         "kind": "message", "time": "2023-05-09T00:00:00Z", "role": "Bob",
         "source": "s1", "meta": {"parent_id": "p2", "part": 1, "parts": 1}},
        {"id": "s1_00003", "text": "The archive code is ORANGE.",
         "kind": "message", "time": "2023-05-10T00:00:00Z", "role": "Alice",
         "source": "s1", "meta": {"parent_id": "p3", "part": 1, "parts": 1}},
    ]


async def test_index_cache_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, _cache_records())
        config = {"retrieval": {"dense": {"mode": "hash", "dim": 32}}}
        first = await build_index(memory, config, cache_dir=root / "cache")
        assert first.dense_enabled
        assert first.stats["embedded"] == 3
        await first.aclose()

        second = await build_index(memory, config, cache_dir=root / "cache")
        assert second.dense_enabled
        assert second.stats["embedded"] == 0
        assert second.stats["cache_hits"] == 3
        await second.aclose()


async def test_dense_window_encoding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        long_text = " ".join(f"fact {index} about Paris." for index in range(220))
        write_jsonl(memory, [{
            "id": "s1_00001", "text": long_text, "kind": "record",
            "time": "", "role": "", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        config = {"retrieval": {
            "dense": {"mode": "hash", "dim": 32,
                      "max_tokens": 80, "overlap_tokens": 20}}}
        first = await build_index(memory, config, cache_dir=root / "cache")
        assert first.dense_enabled
        assert first.stats["windows"] > 1
        assert len(first.vector_groups[0]) > 1
        ranked = await first.dense_rank("Paris fact 100")
        assert ranked and ranked[0][0] == 0
        await first.aclose()

        second = await build_index(memory, config, cache_dir=root / "cache")
        assert second.stats["embedded"] == 0
        assert second.stats["cache_hits"] == first.stats["windows"]
        await second.aclose()


async def test_index_cache_concurrent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, _cache_records())
        config = {"retrieval": {"dense": {"mode": "hash", "dim": 32}}}
        results = await asyncio.gather(*(
            build_index(memory, config, cache_dir=root / "cache")
            for _ in range(3)
        ))
        assert all(index.dense_enabled for index in results)
        # The first build embeds; the other two wait on the cache lock and read.
        assert sum(index.stats["embedded"] for index in results) == 3
        for index in results:
            await index.aclose()


def test_embedding_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = EmbeddingCache(Path(tmp), "test:v1")
        cache.put("hello", [0.1, 0.2, 0.3])
        cache.save()
        cache.close()
        again = EmbeddingCache(Path(tmp), "test:v1")
        got = again.get("hello")
        assert got is not None
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, [0.1, 0.2, 0.3]))
        assert again.get("missing") is None
        again.close()


async def test_bm25_and_coarse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        records = [
            {"id": "s1_00001", "text": "Alice moved to Paris in 2023.",
             "kind": "message", "time": "2023-05-08T00:00:00Z", "role": "Alice",
             "source": "s1", "meta": {"parent_id": "p1", "part": 1, "parts": 1}},
            {"id": "s1_00002", "text": "Bob lives in Berlin.",
             "kind": "message", "time": "2023-05-09T00:00:00Z", "role": "Bob",
             "source": "s1", "meta": {"parent_id": "p2", "part": 1, "parts": 1}},
            {"id": "s1_00003", "text": "The archive code is ORANGE.",
             "kind": "message", "time": "2023-05-10T00:00:00Z", "role": "Alice",
             "source": "s2", "meta": {"parent_id": "p3", "part": 1, "parts": 1}},
        ]
        write_jsonl(memory, records)
        config = {
            "retrieval": {
                "dense": {"mode": "hash", "dim": 64},
                "coarse": {"candidates": 5, "neighbor_window": 0},
            }
        }
        index = await build_index(memory, config)
        assert index.dense_enabled
        assert index.bm25.search(tokenize("Alice Paris"))
        result = await run_coarse(memory, "Where did Alice move?", config=config)
        assert result.candidates
        assert result.candidates[0]["text"].startswith("Alice moved")
        assert result.info["route"] in {"dialogue", "multi_hop"}
        assert any("bm25" in candidate["channels"]
                   for candidate in result.candidates)
        if result.index is not None:
            await result.index.aclose()


def test_postprocess_parent() -> None:
    text = ("The code word is ORANGE. " + "Filler. " * 200).strip()
    _, records = normalize_messages(
        [{"role": "user", "content": text}], session_id="s1", session_index=1,
        max_chars=400,
    )
    chunk = records[0]
    evidence = [{
        "content": "The code word is ORANGE.",
        "metadata": {"source": [chunk["id"]], "score": 0.9},
    }]
    processed, report = postprocess_evidence(
        evidence, records, question="Summarize the document", route="summary",
        config={"parent_max_chars": 8000, "max_evidence_chars": 8000})
    assert processed
    assert "ORANGE" in processed[0]["content"]
    assert processed[0]["metadata"]["resolution"] == "parent"
    assert report["out"] == 1

    forced, _ = postprocess_evidence(
        [{"content": "The code word is ORANGE.",
          "metadata": {"source": [chunk["id"]], "score": 0.9, "resolve": "chunk"}}],
        records, question="Summarize the document", route="summary",
        config={"parent_max_chars": 8000})
    assert forced[0]["metadata"]["resolution"] == "chunk"
    assert len(forced[0]["content"]) < 500

    parent_cited, _ = postprocess_evidence(
        [{"content": "The code word is ORANGE.",
          "metadata": {"source": [chunk["meta"]["parent_id"]], "score": 0.9}}],
        records, question="Summarize the document", route="summary",
        config={"parent_max_chars": 8000})
    assert parent_cited
    assert parent_cited[0]["metadata"]["resolution"] == "parent"
    assert parent_cited[0]["metadata"]["source"] == [chunk["id"]]


def test_neighbor_expansion() -> None:
    records = [
        {"id": "s1_00001", "text": "Alice introduced Bob.", "kind": "message",
         "time": "2023-01-01T00:00:00Z", "role": "Alice", "source": "s1",
         "meta": {"parent_id": "p1", "part": 1, "parts": 1}},
        {"id": "s1_00002", "text": "He moved to Rome.", "kind": "message",
         "time": "2023-01-02T00:00:00Z", "role": "Bob", "source": "s1",
         "meta": {"parent_id": "p2", "part": 1, "parts": 1}},
        {"id": "s1_00003", "text": "He works as a teacher.", "kind": "message",
         "time": "2023-01-03T00:00:00Z", "role": "Bob", "source": "s1",
         "meta": {"parent_id": "p3", "part": 1, "parts": 1}},
    ]
    item = {"content": "Bob moved to Rome.",
            "metadata": {"source": ["s1_00002"], "score": 0.9}}
    expanded = expand_with_neighbors(item, records, window=1, max_chars=400)
    assert expanded["metadata"]["context_ids"] == ["s1_00001", "s1_00003"]
    assert "Alice introduced Bob." in expanded["content"]
    assert "[context s1_00003]" in expanded["content"]
    # The cited source set is unchanged: context is not an anchor.
    assert expanded["metadata"]["source"] == ["s1_00002"]


def test_state_index() -> None:
    records = [
        {"id": "s1_00001", "text": "quarterback | sport | American football",
         "kind": "triple", "time": "2023-01-01T00:00:00Z",
         "meta": {"state_key": "quarterback|sport", "status": "active",
                  "mutation": "none", "object": "American football"}},
        {"id": "s2_00001", "text": "UPDATE: quarterback | sport | Muay Thai",
         "kind": "triple", "time": "2024-01-01T00:00:00Z",
         "meta": {"state_key": "quarterback|sport", "status": "active",
                  "mutation": "replace", "object": "Muay Thai"}},
    ]
    entries = derive_state(records)
    assert entries[0]["status"] == "superseded"
    assert entries[0]["valid_to"] == "2024-01-01"
    assert entries[0]["superseded_by"] == ["s2_00001"]
    assert entries[1]["status"] == "active"
    assert entries[1]["supersedes"] == ["s1_00001"]


async def test_state_rank() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [
            {"id": "s1_00001", "text": "quarterback | sport | American football",
             "kind": "triple", "time": "2023-01-01T00:00:00Z",
             "source": "s1", "meta": {"state_key": "quarterback|sport",
                                      "status": "active", "mutation": "none",
                                      "object": "American football"}},
            {"id": "s2_00001", "text": "UPDATE: quarterback | sport | Muay Thai",
             "kind": "triple", "time": "2024-01-01T00:00:00Z",
             "source": "s2", "meta": {"state_key": "quarterback|sport",
                                      "status": "active", "mutation": "replace",
                                      "object": "Muay Thai"}},
        ])
        index = await build_index(memory, {"retrieval": {"dense": {"mode": "off"}}})
        ranked = index.state_rank(tokenize("quarterback sport"))
        assert ranked and ranked[0][0] == 1  # Muay Thai is current
        assert len(index.state) == 2
        await index.aclose()


def test_tables_db() -> None:
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        records = [
            {"id": "r1", "text": "row 1: name: A School; closed: 1960",
             "kind": "row", "source": "t",
             "meta": {"row": 1, "columns": ["name", "closed"],
                      "data": ["A School", "1960"]}},
            {"id": "r2", "text": "row 2: name: B School; closed: 2006",
             "kind": "row", "source": "t",
             "meta": {"row": 2, "columns": ["name", "closed"],
                      "data": ["B School", "2006"]}},
        ]
        path = root / "tables.db"
        assert build_tables_db(records, path) == 2
        connection = sqlite3.connect(path)
        try:
            value = connection.execute(
                "SELECT name FROM rows WHERE closed = '2006'").fetchone()[0]
        finally:
            connection.close()
        assert value == "B School"


def test_token_usage() -> None:
    class Usage:
        def __init__(self, prompt: int, completion: int) -> None:
            self.prompt_tokens = prompt
            self.completion_tokens = completion
            self.total_tokens = prompt + completion

    tracker = llm.UsageTracker()
    tracker.add(Usage(10, 2), "postagent")
    tracker.add(Usage(20, 3), "verifier")
    data = tracker.as_dict()
    assert data["total_tokens"] == 35
    assert data["by_role"]["postagent"]["total_tokens"] == 12
    assert data["by_role"]["verifier"]["total_tokens"] == 23

    after = llm.UsageTracker()
    after.add(Usage(100, 10), "postagent")
    delta = llm.usage_delta(data, after.as_dict())
    assert delta["by_role"]["postagent"]["prompt_tokens"] == 90
    assert delta["total_tokens"] == 75

    summary = summarize_group([
        {"status": "ok", "search": {"sufficient": True, "steps": 2},
         "tokens": {"total_tokens": 100,
                    "embedding": {"total_tokens": 40},
                    "postagent": {"total_tokens": 50},
                    "verifier": {"total_tokens": 10}}},
        {"status": "ok", "search": {"sufficient": True, "steps": 3},
         "tokens": {"total_tokens": 200,
                    "embedding": {"total_tokens": 60},
                    "postagent": {"total_tokens": 100},
                    "verifier": {"total_tokens": 40}}},
    ])
    assert summary["tokens"]["total_tokens"] == 300
    assert summary["tokens"]["embedding"]["total_tokens"] == 100
    assert summary["tokens"]["avg_total_tokens"] == 150.0


def test_relative_resolution() -> None:
    resolved = search_prompts.resolve_relative_dates(
        "I went yesterday and painted the lake last year.", "2023-05-08T00:00:00Z")
    assert resolved["yesterday"] == "2023-05-07"
    assert resolved["last year"] == "2022"
    body = search_prompts.verifier_prompt("When did she go?", [{
        "content": "I went yesterday.",
        "metadata": {"source": ["s1_00001"],
                     "observed_at": "2023-05-08T00:00:00Z"}}])
    assert "[resolved: yesterday = 2023-05-07]" in body


def test_resolved_date_annotation() -> None:
    items = [{
        "content": "I went to the support group yesterday.",
        "metadata": {"source": ["s1_00001"],
                     "observed_at": "2023-05-08T12:00:00Z"},
    }]
    out = annotate_resolved_dates(items)
    assert "[resolved: yesterday = 2023-05-07]" in out[0]["content"]
    assert out[0]["metadata"]["resolved_dates"]["yesterday"] == "2023-05-07"
    # Idempotent: a second pass does not duplicate the annotation.
    again = annotate_resolved_dates(out)
    assert again[0]["content"].count("[resolved:") == 1


def test_verifier_checks() -> None:
    records = [
        {"id": "s1_00001", "text": "quarterback | sport | American football",
         "kind": "triple", "meta": {"state_key": "quarterback|sport",
                                    "status": "superseded"}},
        {"id": "s2_00001", "text": "quarterback | sport | Muay Thai",
         "kind": "triple", "meta": {"state_key": "quarterback|sport",
                                    "status": "active"}},
    ]
    stale = verifier.apply_checks(
        verifier.Verdict(sufficient=True),
        [{"content": "The sport is American football.",
          "metadata": {"source": ["s1_00001"]}}],
        memory_records=records, intent="update")
    assert stale.sufficient is False
    assert stale.freshness_conflict is True

    fresh = verifier.apply_checks(
        verifier.Verdict(sufficient=True, claims=[
            {"claim": "The current sport is Muay Thai",
             "sources": ["s2_00001"]}]),
        [{"content": "The current sport is Muay Thai.",
          "metadata": {"source": ["s2_00001"]}}],
        memory_records=records, intent="update")
    assert fresh.sufficient is True

    ungrounded = verifier.apply_checks(
        verifier.Verdict(sufficient=True, claims=[
            {"claim": "The current sport is ice hockey",
             "sources": ["s2_00001"]}]),
        [{"content": "The current sport is Muay Thai.",
          "metadata": {"source": ["s2_00001"]}}],
        memory_records=records, intent="update")
    assert ungrounded.sufficient is False

    unknown = verifier.apply_checks(
        verifier.Verdict(sufficient=True),
        [{"content": "fabricated", "metadata": {"source": ["s9_99999"]}}],
        memory_records=records)
    assert unknown.sufficient is False
    assert unknown.checks.get("unknown_sources")

    policy = verifier.apply_checks(
        verifier.Verdict(sufficient=True),
        [{"content": "The secret code is 1234.",
          "metadata": {"source": ["s2_00001"]}}],
        memory_records=records, forbidden=["secret code"])
    assert policy.policy_violation is True
    assert policy.sufficient is False

    no_date = verifier.apply_checks(
        verifier.Verdict(sufficient=True),
        [{"content": "Alice moved somewhere.",
          "metadata": {"source": ["s2_00001"]}}],
        memory_records=records, question="When did Alice move?")
    assert no_date.sufficient is False
    assert no_date.checks.get("missing_date") is True

    has_date = verifier.apply_checks(
        verifier.Verdict(sufficient=True),
        [{"content": "Alice moved in March 2024.",
          "metadata": {"source": ["s2_00001"]}}],
        memory_records=records, question="When did Alice move?")
    assert has_date.sufficient is True

    dated_source = {
        "id": "s4_00001",
        "text": "I went to a support group yesterday.",
        "time": "2023-05-08T12:00:00Z",
    }
    date_claim = verifier.apply_checks(
        verifier.Verdict(sufficient=True, claims=[{
            "claim": "Caroline went to the support group on May 7, 2023",
            "sources": ["s4_00001"]}]),
        [{"content": "I went to a support group yesterday.",
          "metadata": {"source": ["s4_00001"]}}],
        memory_records=[dated_source], question="When did Caroline go?")
    assert date_claim.sufficient is True, date_claim

    no_number = verifier.apply_checks(
        verifier.Verdict(sufficient=True),
        [{"content": "Alice has several pets.",
          "metadata": {"source": ["s2_00001"]}}],
        memory_records=records, question="How many pets does Alice have?")
    assert no_number.sufficient is False
    assert no_number.checks.get("missing_number") is True

    # A model may cite the evidence index ("1") instead of the record id; that
    # must not turn a correct verdict into "unknown source".
    sunrise = {
        "id": "s3_00001", "text": "Melanie painted a lake sunrise last year.",
        "kind": "message", "meta": {"state_key": ""},
    }
    indexed = verifier.apply_checks(
        verifier.Verdict(sufficient=False, missing=["specific date"], claims=[
            {"claim": "Melanie painted a lake sunrise last year",
             "sources": ["1"]}]),
        [{"content": "Melanie painted a lake sunrise last year.",
          "metadata": {"source": ["s3_00001"]}}],
        memory_records=records + [sunrise],
        question="When did Melanie paint a sunrise?")
    assert indexed.sufficient is True
    assert indexed.checks.get("rescued_date") is True
    assert indexed.claims[0]["sources"] == ["s3_00001"]


def test_fallback_metadata() -> None:
    records = [{
        "id": "s1_00001", "text": "Alice moved to Paris in 2023.",
        "kind": "message", "time": "2023-05-08T00:00:00Z", "role": "Alice",
        "source": "s1", "meta": {"parent_id": "p1", "part": 1, "parts": 1},
    }]
    candidates = [{"id": "s1_00001", "rank": 1, "text": records[0]["text"]}]
    out = fallback_evidence(
        candidates, records, question="Where did Alice move?",
        route="dialogue", config={}, limit=4)
    assert out[0]["metadata"]["observed_at"] == "2023-05-08T00:00:00Z"
    assert out[0]["metadata"]["speaker"] == "Alice"


def test_fuse_evidence() -> None:
    agent = [
        {"content": "Alice lives in Paris.",
         "metadata": {"source": ["a1"], "score": 0.9}},
        {"content": "She works as a teacher.",
         "metadata": {"source": ["a2"], "score": 0.8}},
    ]
    coarse = [
        {"content": "Alice lives in Paris.",
         "metadata": {"source": ["c1"], "score": 0.5}},
        {"content": "Bob lives in Berlin.",
         "metadata": {"source": ["c2"], "score": 0.4}},
        {"content": "Paris is in France.",
         "metadata": {"source": ["c3"], "score": 0.3}},
        {"content": "The archive code is ORANGE.",
         "metadata": {"source": ["c4"], "score": 0.2}},
    ]
    fused = fuse_evidence(agent, coarse, max_items=6, coarse_ratio=0.5)
    assert len(fused) <= 6
    contents = [item["content"] for item in fused]
    assert contents.count("Alice lives in Paris.") == 1
    first = fused[0]
    assert first["metadata"]["origin"] == "postagent"
    assert set(first["metadata"]["source"]) == {"a1", "c1"}
    # Both halves are present (the postagent wrote two; coarse backfills the rest).
    assert sum(1 for item in fused
               if item["metadata"]["origin"] == "postagent") == 2
    assert any(item["metadata"]["origin"] == "coarse" for item in fused)


def test_fit_evidence_keeps_novel_agent_evidence() -> None:
    """Raw-only datasets drop paraphrases but keep a new record the agent found.

    This is the LoCoMo regression: every postagent append was classified as a
    claim and dropped, so ``evidence_mix.postagent`` was 0 for 1382/1382
    queries and the search stage never affected the answer.
    """
    records = [
        {"id": "s1_00001", "text": "Alice lives in Paris.", "kind": "message",
         "time": "", "role": "Alice", "source": "s1",
         "meta": {"parent_id": "p1", "part": 1, "parts": 1}},
        {"id": "s1_00002", "text": "The code word is ORANGE.", "kind": "message",
         "time": "", "role": "Bob", "source": "s1",
         "meta": {"parent_id": "p2", "part": 1, "parts": 1}},
        {"id": "s1_00003", "text": "The code word is GREEN.", "kind": "message",
         "time": "", "role": "Bob", "source": "s1",
         "meta": {"parent_id": "p3", "part": 1, "parts": 1}},
    ]
    seeded = {
        "content": "Alice lives in Paris.",
        "metadata": {"source": ["s1_00001"], "score": 0.9, "origin": "coarse",
                     "claim_type": "raw", "seeded": True},
    }
    # A short verbatim record the agent appended (append_matches.py shape).
    short_copy = {
        "content": "The code word is ORANGE.",
        "metadata": {"source": ["s1_00002"], "score": 0.8,
                     "origin": "postagent", "claim_type": "copy"},
    }
    # A paraphrase of the already-seeded record: keep the raw seed, drop this.
    paraphrase = {
        "content": "Alice is a resident of Paris.",
        "metadata": {"source": ["s1_00001"], "score": 0.7,
                     "origin": "postagent", "claim_type": "claim"},
    }
    # A synthesized claim over a record the shortlist missed: keep it.
    novel_claim = {
        "content": "The code word is GREEN.",
        "metadata": {"source": ["s1_00003"], "score": 0.6,
                     "origin": "postagent", "claim_type": "claim"},
    }
    config = {"final_evidence_chars": 10000, "final_item_chars": 2500,
              "final_parent_cap": 2, "max_evidence": 64,
              "final_claim_cap": 12}
    assert claim_mode("locomo_refined", "fact") == "raw"
    out = fit_evidence([seeded, short_copy, paraphrase, novel_claim], records,
                       config=config, dataset="locomo_refined", intent="fact")
    contents = [item["content"] for item in out]
    assert any("ORANGE" in text for text in contents)
    assert any("GREEN" in text for text in contents)
    assert not any("resident of Paris" in text for text in contents)
    # Agent-added records are visible early, not after 20+ seeded candidates.
    assert "ORANGE" in contents[1] or "GREEN" in contents[1]


def test_postprocess_short_copy() -> None:
    """A short verbatim record is a copy, not a synthesized claim."""
    records = [{"id": "s1_00001", "text": "Alice lives in Paris.",
                "kind": "message", "time": "", "role": "Alice",
                "source": "s1",
                "meta": {"parent_id": "p1", "part": 1, "parts": 1}}]
    processed, _ = postprocess_evidence(
        [{"content": "Alice lives in Paris.",
          "metadata": {"source": ["s1_00001"]}}],
        records, question="Where does Alice live?", route="dialogue",
        config={"evidence_neighbor_window": 0})
    assert processed[0]["metadata"]["claim_type"] == "copy"


def test_evidence_shapes() -> None:
    """Accept the output shapes small models actually produce."""
    normalized, reason = evidence_mod.normalize(
        {"text": "Alice lives in Paris.", "source": "s1_00001"})
    assert normalized is not None and not reason
    assert normalized["metadata"]["source"] == ["s1_00001"]
    wrapped = evidence_mod.parse_text(json.dumps(
        {"evidence": [{"content": "Alice lives in Paris.",
                       "metadata": {"source": ["s1_00001"]}}]}))
    assert len(wrapped) == 1 and wrapped[0]["content"].startswith("Alice")
    top_level = evidence_mod.parse_text(
        '{"content": "Alice lives in Paris.", "sources": ["s1_00001"]}')
    assert evidence_mod.normalize(top_level[0])[0] is not None
    # A whole memory record copied into evidence: cite its id, not its session.
    copied, _ = evidence_mod.normalize(
        {"id": "s1_00002", "text": "Alice lives in Paris.", "source": "session_1"})
    assert copied is not None
    assert copied["metadata"]["source"] == ["s1_00002"]


async def test_postagent_auto_captures_grep_hit() -> None:
    """A real grep hit is appended even when the model forgets to write it."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "inputs").mkdir()
        write_jsonl(workspace / "inputs" / "memory.jsonl", [{
            "id": "s1_00001", "text": "The code word is ORANGE.",
            "kind": "message", "time": "", "role": "Bob", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])

        class FakeTool:
            async def run(self, args):
                return ToolResult(text="#1 s1_00001 (Bob) The code word is ORANGE.",
                                  ok=True)

        script = iter([
            '{"tool":"bash","args":{"command":"grep -in \'ORANGE\' '
            'inputs/memory.jsonl | head -5"}}',
            "done",
        ])

        async def call_model(messages, schemas):
            text = next(script)
            return ModelReply(text=text, calls=toolcfg.parse_tool_calls(text),
                              assistant_entry={"role": "assistant", "content": text})

        result = await run_postagent(
            workspace=workspace, tools={"bash": FakeTool()},
            call_model=call_model, schemas=None,
            config={"question": "What is the code word?", "max_steps": 2,
                    "auto_capture_hits": True, "auto_capture_limit": 2,
                    "audit_after_step": 0, "no_progress_patience": 0,
                    "write_nudge_after": 0})
        assert result.stop_reason == "sufficient", result.stop_reason
        contents = [
            item["content"] for item in
            evidence_mod.read_file(workspace / "evidence.jsonl")
        ]
        assert any("ORANGE" in text for text in contents), contents


async def test_postagent_auto_captures_query_hit() -> None:
    """A rewritten --query hit is appended when the model never writes it."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "inputs").mkdir()
        write_jsonl(workspace / "inputs" / "memory.jsonl", [{
            "id": "s4_00003",
            "text": "This necklace is a gift from my grandma in my home country, Sweden.",
            "kind": "message", "time": "", "role": "Caroline", "source": "s4",
            "meta": {"parent_id": "p4_00002", "part": 1, "parts": 1},
        }])

        class FakeTool:
            async def run(self, args):
                return ToolResult(
                    text="#1 s4_00003 [2023-06-27T10:37:00Z] (Caroline) why: bm25:country\n"
                         "  This necklace is a gift from my grandma in my home country, Sweden.",
                    ok=True)

        script = iter([
            '{"tool":"bash","args":{"command":"coarse '
            '--query \\"home country\\" --top 8"}}',
            "done",
        ])

        async def call_model(messages, schemas):
            text = next(script)
            return ModelReply(text=text, calls=toolcfg.parse_tool_calls(text),
                              assistant_entry={"role": "assistant", "content": text})

        result = await run_postagent(
            workspace=workspace, tools={"bash": FakeTool()},
            call_model=call_model, schemas=None,
            config={"question": "Where did Caroline move from?", "max_steps": 2,
                    "auto_capture_hits": True, "auto_capture_limit": 2,
                    "audit_after_step": 0, "no_progress_patience": 0,
                    "write_nudge_after": 0})
        assert result.stop_reason == "sufficient", result.stop_reason
        contents = [
            item["content"] for item in
            evidence_mod.read_file(workspace / "evidence.jsonl")
        ]
        assert any("Sweden" in text for text in contents), contents


async def test_runner_reuses_coarse_source() -> None:
    """The deterministic coarse stage can be loaded from a previous run."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [{
            "id": "s1_00001", "text": "Alice lives in Paris.",
            "kind": "message", "time": "", "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        source = root / "source_coarse"
        source.mkdir()
        write_jsonl(source / "candidates.jsonl", [{
            "id": "s1_00001", "rank": 1, "rrf": 0.02,
            "channels": {"bm25": 0.9}, "why": ["bm25:alice"],
            "neighbor_of": None, "text": "Alice lives in Paris.",
            "kind": "message", "time": "", "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
            "return_hint": "chunk",
        }])
        (source / "coarse.json").write_text(json.dumps({
            "query": "Where does Alice live?", "options": [],
            "mode": "hybrid", "route": "dialogue", "records": 1,
            "warnings": [],
        }), encoding="utf-8")
        config = runner.load_config()
        config["retrieval"]["dense"] = {"mode": "off"}
        config["retrieval"]["gate"]["enabled"] = False
        config["reuse_coarse_strict"] = False
        verdict = json.dumps({
            "sufficient": True, "reasoning": "ok", "missing": [],
            "claims": [{"claim": "Alice lives in Paris",
                        "sources": ["s1_00001"]}],
        })
        model = StubModel([verdict, verdict], allow_agent=False)
        out = await runner.run_question(
            memory_path=memory, question="Where does Alice live?",
            config=config, run_root=root / "runs", model=model,
            coarse_source=source)
        assert out["summary"]["coarse_reused"] is True
        assert out["summary"]["coarse"]["reused"] is True
        assert out["evidence"]
        qa = Path(out["run_dir"])
        assert (qa / "inputs" / "candidates.jsonl").exists()


def test_source_inputs_layouts() -> None:
    """Resolve v2 flat and v1 ``qa/`` source layouts from an old run_dir."""
    from asmem.bench.pipeline import Experiment

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        for tail, qa in (("u/q/ts", False), ("u2/q2/ts2", True)):
            inputs = source / "search" / tail / ("qa/inputs" if qa else "inputs")
            inputs.mkdir(parents=True)
            (inputs / "candidates.jsonl").write_text("", encoding="utf-8")
            (inputs / "coarse.json").write_text("{}", encoding="utf-8")
        flat = Experiment._source_inputs(
            source, "/old/host/run/search/u/q/ts")
        legacy = Experiment._source_inputs(
            source, "/old/host/run/search/u2/q2/ts2/qa")
        assert flat == source / "search/u/q/ts/inputs"
        assert legacy == source / "search/u2/q2/ts2/qa/inputs"


class StubModel:
    def __init__(self, script: list[str], *, allow_agent: bool = True) -> None:
        self.script = list(script)
        self.calls = 0
        self.schemas = None
        self.allow_agent = allow_agent

    async def call(self, messages, schemas=None, usage=None, deadline=None):
        if not self.allow_agent:
            raise AssertionError("agent should have been skipped by the coarse gate")
        self.calls += 1
        text = self.script.pop(0)
        return ModelReply(text=text, calls=toolcfg.parse_tool_calls(text),
                          assistant_entry={"role": "assistant", "content": text})

    async def text(self, messages, usage=None, deadline=None):
        self.calls += 1
        return self.script.pop(0)

    async def aclose(self):
        pass


def test_gate() -> None:
    candidate = {
        "id": "s1_00001", "rank": 1, "rrf": 0.02,
        "channels": {"bm25": 0.9},
        "text": "Alice lives in Paris and works as a teacher.",
        "neighbor_of": None,
    }
    config = {"retrieval": {"gate": {"enabled": True, "min_coverage": 0.6}}}
    assert evaluate_gate([candidate], "Where does Alice live and work?",
                         "dialogue", config).sufficient
    assert not evaluate_gate([candidate], "Who is the CEO of the company that "
                             "developed Windows Media Player?", "multi_hop",
                             config).sufficient
    assert not evaluate_gate([candidate], "Compare Alice and Bob", "dialogue",
                             config).sufficient
    assert not evaluate_gate([], "Where does Alice live?", "dialogue",
                             config).sufficient


async def test_runner_gate_skip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [{
            "id": "s1_00001", "text": "Alice lives in Paris and works as a teacher.",
            "kind": "message", "time": "", "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        config = runner.load_config()
        config["postagent_mode"] = "loop"
        config["retrieval"]["dense"] = {"mode": "off"}
        config["retrieval"]["coarse"]["candidates"] = 5
        config["retrieval"]["gate"]["verify"] = True
        verdict = json.dumps({
            "sufficient": True, "reasoning": "ok", "missing": [],
            "claims": [{"claim": "Alice lives in Paris and works as a teacher",
                        "sources": ["s1_00001"]}],
        })
        model = StubModel([verdict, verdict], allow_agent=False)
        out = await runner.run_question(
            memory_path=memory, question="Where does Alice live and work?",
            config=config, run_root=root / "runs", model=model)
        assert out["summary"]["gate_skipped"] is True
        assert out["summary"]["steps"] == 0
        assert out["summary"]["sufficient"] is True
        assert out["evidence"]


async def test_runner_fixed_pipeline_is_deterministic() -> None:
    """Fixed postagent must assemble evidence without planner/verifier calls."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [{
            "id": "s1_00001", "text": "Alice lives in Paris and works as a teacher.",
            "kind": "message", "time": "2023-01-01T00:00:00Z",
            "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        config = runner.load_config()
        config["retrieval"]["dense"] = {"mode": "off"}
        config["retrieval"]["coarse"]["candidates"] = 5
        config["retrieval"]["gate"]["enabled"] = True
        config["postagent_mode"] = "fixed"

        class NoSearchCalls:
            def __init__(self):
                self.calls = 0

            async def call(self, *args, **kwargs):
                raise AssertionError("fixed postagent must not call the planner")

            async def text(self, *args, **kwargs):
                raise AssertionError("fixed postagent must not call a verifier")

        out = await runner.run_question(
            memory_path=memory, question="Where does Alice live?",
            config=config, run_root=root / "runs", model=NoSearchCalls())
        again = await runner.run_question(
            memory_path=memory, question="Where does Alice live?",
            config=config, run_root=root / "runs_again", model=NoSearchCalls())
        summary = out["summary"]
        assert summary["stop_reason"] == "fixed_pipeline", summary
        assert summary["postagent_calls"] == 0
        assert summary["verifier_calls"] == 0
        assert summary["compile"]["mode"] == "fixed"
        assert out["evidence"]
        assert out["evidence"] == again["evidence"], "fixed evidence must be reproducible"


async def test_runner_pre_audit_skip() -> None:
    """A sufficient coarse shortlist skips the agent; the verifier decides."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [{
            "id": "s1_00001", "text": "Alice lives in Paris.",
            "kind": "message", "time": "", "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        config = runner.load_config()
        config["postagent_mode"] = "loop"
        config["pre_audit"] = True
        config["retrieval"]["dense"] = {"mode": "off"}
        config["retrieval"]["coarse"]["candidates"] = 5
        config["retrieval"]["gate"]["enabled"] = False
        verdict = json.dumps({
            "sufficient": True, "reasoning": "coarse shortlist answers it",
            "missing": [],
            "claims": [{"claim": "Alice lives in Paris",
                        "sources": ["s1_00001"]}],
        })
        model = StubModel([verdict, verdict], allow_agent=False)
        out = await runner.run_question(
            memory_path=memory, question="Where does Alice live?",
            config=config, run_root=root / "runs", model=model)
        summary = out["summary"]
        assert summary["steps"] == 0, summary["steps"]
        assert summary["postagent_calls"] == 0
        assert summary["stop_reason"] == "coarse_sufficient"
        assert summary["sufficient"] is True
        assert out["evidence"]


async def test_runner_eight_steps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [{
            "id": "s1_00001", "text": "Alice lives in Paris.",
            "kind": "message", "time": "2023-01-01T00:00:00Z",
            "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        config = runner.load_config()
        config["postagent_mode"] = "loop"
        config["pre_audit"] = True
        config["retrieval"]["dense"] = {"mode": "hash", "dim": 32}
        config["retrieval"]["coarse"]["candidates"] = 5
        config["retrieval"]["gate"]["enabled"] = False
        config["max_steps"] = 8
        record = json.dumps({
            "content": "Alice lives in Paris.",
            "metadata": {"source": ["s1_00001"], "score": 0.9},
        })
        append = f"cat >> evidence.jsonl <<'EOF'\n{record}\nEOF\n"
        verdict = json.dumps({
            "sufficient": True, "reasoning": "ok", "missing": [],
            "claims": [{"claim": "Alice lives in Paris",
                        "sources": ["s1_00001"]}],
        })
        pre_audit = json.dumps({
            "sufficient": False, "reasoning": "shortlist is not enough",
            "missing": ["the answer"], "claims": [],
        })
        model = StubModel([
            pre_audit,
            '{"tool":"bash","args":{"command":"jq -s \'sort_by(.rank)|.[0:3]\' inputs/candidates.jsonl | head -5"}}',
            json.dumps({"tool": "bash", "args": {"command": append}}),
            "done",
            verdict,
            verdict,
        ])
        out = await runner.run_question(
            memory_path=memory, question="Where does Alice live?",
            config=config, run_root=root / "runs", model=model)
        assert out["summary"]["steps"] <= 8
        assert out["summary"]["sufficient"] is True
        assert "Alice lives in Paris." in out["evidence"][0]["content"]
        assert out["summary"]["postagent_calls"] >= 2
        assert out["summary"]["verifier_calls"] >= 1
        qa = Path(out["run_dir"])
        helper = qa / "bin" / "append_matches.py"
        assert helper.exists() and helper.stat().st_mode & 0o111
        global_coarse = Path(__file__).resolve().parents[1] / "scripts" / "coarse"
        assert global_coarse.exists() and global_coarse.stat().st_mode & 0o111
        assert (qa / "inputs" / "retrieval.json").exists()
        steps_log = qa / "steps.jsonl"
        assert steps_log.exists()
        entries = [json.loads(line) for line in
                   steps_log.read_text(encoding="utf-8").splitlines() if line]
        assert any(entry.get("tool") == "bash" for entry in entries)


async def test_runner_force_finish() -> None:
    """Once the mid-loop audit passes, do not burn the remaining steps."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = root / "memory.jsonl"
        write_jsonl(memory, [{
            "id": "s1_00001", "text": "Alice lives in Paris.",
            "kind": "message", "time": "2023-01-01T00:00:00Z",
            "role": "Alice", "source": "s1",
            "meta": {"parent_id": "p1", "part": 1, "parts": 1},
        }])
        config = runner.load_config()
        config["postagent_mode"] = "loop"
        config["pre_audit"] = True
        config["retrieval"]["dense"] = {"mode": "off"}
        config["retrieval"]["coarse"]["candidates"] = 5
        config["retrieval"]["gate"]["enabled"] = False
        config["max_steps"] = 6
        config["audit_after_step"] = 1
        record = json.dumps({
            "content": "Alice lives in Paris.",
            "metadata": {"source": ["s1_00001"], "score": 0.9},
        })
        append = f"cat >> evidence.jsonl <<'EOF'\n{record}\nEOF\n"
        verdict = json.dumps({
            "sufficient": True, "reasoning": "ok", "missing": [],
            "claims": [{"claim": "Alice lives in Paris",
                        "sources": ["s1_00001"]}],
        })
        pre_audit = json.dumps({
            "sufficient": False, "reasoning": "shortlist is not enough",
            "missing": ["the answer"], "claims": [],
        })
        model = StubModel([
            pre_audit,
            json.dumps({"tool": "bash", "args": {"command": append}}),
            verdict,
            verdict,
        ])
        out = await runner.run_question(
            memory_path=memory, question="Where does Alice live?",
            config=config, run_root=root / "runs", model=model)
        summary = out["summary"]
        assert summary["steps"] == 1, summary["steps"]
        assert summary["stop_reason"] == "sufficient_after_check"
        assert summary["sufficient"] is True
        assert summary["postagent_calls"] == 1
        # Pre-audit + mid-loop audit + verification of the final answer-visible list.
        assert summary["verifier_calls"] == 3


def main() -> None:
    test_tokenizer()
    test_chunk_policy()
    test_parent_chunks()
    test_embedding_cache()
    asyncio.run(test_index_cache_reuse())
    asyncio.run(test_dense_window_encoding())
    asyncio.run(test_index_cache_concurrent())
    asyncio.run(test_bm25_and_coarse())
    test_postprocess_parent()
    test_neighbor_expansion()
    test_state_index()
    asyncio.run(test_state_rank())
    test_tables_db()
    test_resolved_date_annotation()
    test_token_usage()
    test_relative_resolution()
    test_verifier_checks()
    test_fallback_metadata()
    test_fuse_evidence()
    test_fit_evidence_keeps_novel_agent_evidence()
    test_postprocess_short_copy()
    test_evidence_shapes()
    asyncio.run(test_postagent_auto_captures_grep_hit())
    asyncio.run(test_postagent_auto_captures_query_hit())
    test_gate()
    asyncio.run(test_runner_gate_skip())
    asyncio.run(test_runner_fixed_pipeline_is_deterministic())
    asyncio.run(test_runner_pre_audit_skip())
    asyncio.run(test_runner_reuses_coarse_source())
    test_source_inputs_layouts()
    asyncio.run(test_runner_eight_steps())
    asyncio.run(test_runner_force_finish())
    print("test_retrieval: OK")


if __name__ == "__main__":
    main()
