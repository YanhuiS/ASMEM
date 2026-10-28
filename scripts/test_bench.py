"""Pure-CPU self-test for the benchmark pipeline (no model endpoint).

    python scripts/test_bench.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem import llm  # noqa: E402
from asmem.bench.answer import Answerer  # noqa: E402
from asmem.bench.judge import Judge, mcq_exact, token_f1  # noqa: E402
from asmem.bench.pipeline import mock_search  # noqa: E402
from asmem.bench.replay import FastMemory  # noqa: E402
from asmem.bench.report import summarize  # noqa: E402


class FakeModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def text(self, messages: list[dict], *, deadline=None) -> str:
        self.calls.append(messages)
        return self.reply


def test_fast_memory() -> None:
    memory = FastMemory()
    event = {
        "request_id": "r1",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Alice moved to Paris."}],
    }
    assert memory.add(event) == 1
    assert memory.add(event) == 0  # idempotent
    assert memory.records[0]["text"] == "Alice moved to Paris."
    assert memory.records[0]["id"].startswith("s1_")


def test_mock_search() -> None:
    records = [
        {"id": "s1_1", "text": "Alice lives in Paris."},
        {"id": "s1_2", "text": "Bob likes tea."},
    ]
    out = mock_search(records, "Where does Alice live?", top_k=1)
    assert out["evidence"][0]["metadata"]["source"] == ["s1_1"]


async def test_answer() -> None:
    model = FakeModel("Paris")
    answerer = Answerer(
        {"max_evidence_chars": 1000, "styles": {}},
        model,  # type: ignore[arg-type]
    )
    result = await answerer.answer(
        dataset="longmemeval_s",
        question="Where does Alice live?",
        evidence=[{"content": "Alice lives in Paris.",
                   "metadata": {"source": ["s1_1"]}}],
        gold={"answer": "Paris", "options": None},
    )
    assert result["answer"] == "Paris"
    assert "Alice lives in Paris." in model.calls[0][1]["content"]
    assert "Where does Alice live?" in model.calls[0][1]["content"]

    typed_model = FakeModel("I do not know.")
    typed_answerer = Answerer(
        {"max_evidence_chars": 1000, "styles": {},
         "type_styles": {"beam_100k/abstention": "ABSTENTION-GUARD"}},
        typed_model,  # type: ignore[arg-type]
    )
    await typed_answerer.answer(
        dataset="beam_500k", question="What did they decide?", evidence=[],
        gold={"answer": "Unanswerable", "answer_type": "abstention"},
    )
    assert "ABSTENTION-GUARD" in typed_model.calls[0][1]["content"]


async def test_acceptable_answers() -> None:
    model = FakeModel('{"correct": true, "reason": "alternative"}')
    judge = Judge({"mode": "judge", "judge": True, "rubric": False,
                   "locomo_official": False}, model)  # type: ignore[arg-type]
    result = await judge.evaluate(
        dataset="locomo_refined", question="When?",
        predicted="May 7, 2023",
        gold={"answer": "7 May 2023",
              "answer_meta": {"acceptable_answers": ["7 May 2023", "May 7, 2023"]}})
    assert result["metrics"]["token_f1"] == 1.0
    assert result["score"] == 1.0
    assert "Other acceptable gold answers" in model.calls[-1][1]["content"]


async def test_refusal_guard() -> None:
    model = FakeModel('{"label": "CORRECT"}')
    judge = Judge({"mode": "judge", "judge": True, "rubric": False,
                   "locomo_official": True}, model)  # type: ignore[arg-type]
    result = await judge.evaluate(
        dataset="locomo_refined", question="When?",
        predicted="The evidence does not provide information about this.",
        gold={"answer": "7 May 2023", "answer_meta": {}})
    assert result["metrics"]["judge_correct"] == 0.0
    assert result["metrics"].get("judge_refusal_guard") == 1.0
    assert not model.calls  # the guard must not ask the judge

    unanswerable_model = FakeModel('{"label": "CORRECT"}')
    unanswerable_judge = Judge(
        {"mode": "judge", "judge": True, "rubric": False,
         "locomo_official": True}, unanswerable_model)  # type: ignore[arg-type]
    unanswerable = await unanswerable_judge.evaluate(
        dataset="locomo_refined", question="When?",
        predicted="There is no information about this in the conversation.",
        gold={"answer": "There is no information about this.",
              "answer_meta": {}})
    assert unanswerable["metrics"]["judge_correct"] == 1.0


async def test_locomo_official_judge() -> None:
    model = FakeModel('{"label": "CORRECT"}')
    judge = Judge({"mode": "judge", "judge": True, "rubric": False,
                   "locomo_official": True}, model)  # type: ignore[arg-type]
    result = await judge.evaluate(
        dataset="locomo_refined", question="When?",
        predicted="May 7, 2023",
        gold={"answer": "7 May 2023",
              "answer_meta": {"acceptable_answers": ["7 May 2023", "May 7, 2023"]}})
    assert result["metrics"]["judge_correct"] == 1.0
    assert result["score"] == 1.0
    assert "Core principle" in model.calls[0][0]["content"]


async def test_judge() -> None:
    model = FakeModel('{"correct": true, "reason": "same"}')
    judge = Judge({"mode": "judge", "judge": True, "rubric": False}, model)  # type: ignore[arg-type]
    result = await judge.evaluate(
        dataset="longmemeval_s", question="Where?", predicted="Paris",
        gold={"answer": "Paris"})
    assert result["score"] == 1.0
    assert result["primary"] == "judge_correct"

    rubric_model = FakeModel('{"results": [true, false], "reason": "partial"}')
    rubric_judge = Judge({"mode": "rubric", "rubric": True}, rubric_model)  # type: ignore[arg-type]
    rubric_result = await rubric_judge.evaluate(
        dataset="clbench", question="Do X", predicted="X but not Y",
        gold={"rubrics": ["does X", "does Y"]})
    assert rubric_result["metrics"]["rubric_score"] == 0.5


class BatchRubricModel:
    """Returns one boolean per numbered rubric line it sees."""

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    async def text(self, messages: list[dict], *, deadline=None) -> str:
        self.calls += 1
        prompt = str(messages[-1].get("content") or "")
        count = len(re.findall(r"(?m)^\d+\. ", prompt))
        self.batch_sizes.append(count)
        return '{"results": [' + ",".join(["true"] * count) + "]}"


class OverflowOnceModel:
    def __init__(self) -> None:
        self.calls = 0
        self.config = {"max_tokens": 64}

    async def text(self, messages: list[dict], *, deadline=None) -> str:
        self.calls += 1
        if self.calls == 1:
            raise llm.ContextOverflow("maximum context length exceeded")
        return "Paris"


async def test_rubric_comments_and_batching() -> None:
    reply = ('```json\n{"results": [true, // one\n false, // two\n true,], '
             '"reason": "x"}\n```')
    judge = Judge({"mode": "rubric", "rubric": True}, FakeModel(reply))  # type: ignore[arg-type]
    result = await judge.evaluate(
        dataset="clbench", question="q", predicted="a",
        gold={"rubrics": ["a", "b", "c"]})
    assert result["metrics"]["rubric_score"] == 2 / 3

    batching = BatchRubricModel()
    judge = Judge(
        {"mode": "rubric", "rubric": True, "rubric_batch": 20},
        batching)  # type: ignore[arg-type]
    rubrics = [f"rubric {index}" for index in range(45)]
    result = await judge.evaluate(
        dataset="clbench", question="q", predicted="a", gold={"rubrics": rubrics})
    assert result["metrics"]["rubric_score"] == 1.0
    assert batching.batch_sizes == [20, 20, 5]


async def test_answer_context_overflow() -> None:
    model = OverflowOnceModel()
    answerer = Answerer(
        {"max_evidence_chars": 4000, "styles": {},
         "max_context_tokens": 600, "total_timeout": 5},
        model)  # type: ignore[arg-type]
    result = await answerer.answer(
        dataset="locomo_refined", question="Where does Alice live?",
        evidence=[{"content": "Alice lives in Paris. " * 200,
                   "metadata": {"source": ["s1_1"]}}],
        gold={"answer": "Paris", "options": None})
    assert result["answer"] == "Paris"
    assert model.calls == 2


def test_fit_messages() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "evidence " * 5000},
    ]
    # fit_messages keeps a 512-token floor; use a budget above it.
    fitted = llm.fit_messages(messages, 600, reserve_tokens=50)
    total = sum(llm.estimate_tokens(str(m["content"])) for m in fitted)
    assert total <= 600
    assert "truncated to fit the context window" in fitted[-1]["content"]
    assert fitted[0]["content"] == "system prompt"


def test_deterministic() -> None:
    assert token_f1("Paris", "Paris") == 1.0
    assert token_f1("London", "Paris") == 0.0
    score, letters = mcq_exact("(A, C)", ["A", "C"])
    assert score == 1.0 and letters == ["A", "C"]
    score, letters = mcq_exact("(C, A)", ["A", "C"])
    assert score == 1.0
    score, _ = mcq_exact("(C, A)", ["A", "C"], ordering=True)
    assert score == 0.0
    score, letters = mcq_exact("B. the right one", ["B"],
                               options=["A. no", "B. the right one"])
    assert score == 1.0 and letters == ["B"]


def test_report() -> None:
    results = [
        {"dataset": "d", "answer_type": "t", "status": "ok",
         "eval": {"score": 1.0, "metrics": {"token_f1": 1.0}},
         "search": {"sufficient": True, "steps": 2}},
        {"dataset": "d", "answer_type": "t", "status": "ok",
         "eval": {"score": 0.0, "metrics": {"token_f1": 0.0}},
         "search": {"sufficient": False, "steps": 4}},
    ]
    summary = summarize(results)
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["score"] == 0.5
    assert summary["overall"]["sufficient_rate"] == 0.5


def main() -> None:
    test_fast_memory()
    test_mock_search()
    asyncio.run(test_answer())
    asyncio.run(test_acceptable_answers())
    asyncio.run(test_refusal_guard())
    asyncio.run(test_locomo_official_judge())
    asyncio.run(test_judge())
    asyncio.run(test_rubric_comments_and_batching())
    asyncio.run(test_answer_context_overflow())
    test_fit_messages()
    test_deterministic()
    test_report()
    print("test_bench: OK")


if __name__ == "__main__":
    main()
