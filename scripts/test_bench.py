"""Pure-CPU self-test for the benchmark pipeline (no model endpoint).

    python scripts/test_bench.py
"""

from __future__ import annotations

import asyncio
import base64
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem import llm  # noqa: E402
from asmem.bench.answer import (  # noqa: E402
    Answerer,
    _parse_evidence_numbers,
    image_candidates,
    requested_image_count,
    resolved_candidates,
)
from asmem.bench.judge import Judge, mcq_exact, token_f1  # noqa: E402
from asmem.bench.pipeline import mock_search  # noqa: E402
from asmem.bench.replay import FastMemory  # noqa: E402
from asmem.bench.report import summarize  # noqa: E402
from asmem.search.evidence_loop import _locations, _month  # noqa: E402


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


def test_mem_gallery_helpers() -> None:
    assert requested_image_count("Which two images show the same landmark?") == 2
    assert requested_image_count("Which image shows the notebook?") == 1
    assert requested_image_count("What images did Julian upload?") is None
    assert _parse_evidence_numbers('{"evidence": [2, 4]}') == [2, 4]
    assert _parse_evidence_numbers("[3, 3, 1]") == [3, 1]

    evidence = [
        {"content": "[image: D3:IMG_001] [round D3:4]",
         "metadata": {"time": "2024-07-29T00:00:00Z", "source": ["s3_1"]}},
        {"content": "[image: D3:IMG_001]\n[caption] Abstract blue painting.\n"
                   "[round D3:4]\n[context s3_2] Here, this is my painting.",
         "metadata": {"time": "2024-07-29T00:00:00Z", "source": ["s3_2"]}},
    ]
    candidates = image_candidates(evidence)
    assert len(candidates) == 1
    assert candidates[0]["id"] == "D3:IMG_001"
    assert candidates[0]["caption"] == "Abstract blue painting."
    assert "this is my painting" in candidates[0]["context"]

    resolved = resolved_candidates([
        {"content": "[resolved: tomorrow = 2024-05-19] trip to Paris tomorrow"},
        {"content": "[resolved: tomorrow = 2024-05-19] trip to Paris tomorrow"},
        {"content": "no resolved date"},
    ])
    assert [item["date"] for item in resolved] == ["2024-05-19"]


async def test_mem_gallery_judge() -> None:
    model = FakeModel('{"correct": true, "reason": "same"}')
    judge = Judge({"mode": "judge", "judge": True, "rubric": False},
                  model)  # type: ignore[arg-type]
    result = await judge.evaluate(
        dataset="mem_gallery", question="Which images?",
        predicted="D2:IMG_004, D2:IMG_003",
        gold={"answer": "D2:IMG_003, D2:IMG_004",
              "answer_meta": {"point": "VS"}})
    assert result["primary"] == "mem_gallery_score"
    assert result["metrics"]["image_id_f1"] == 1.0
    assert result["metrics"]["image_id_exact"] == 1.0

    abstain = await judge.evaluate(
        dataset="mem_gallery", question="What is Amy's cat called?",
        predicted="Not mentioned.",
        gold={"answer": "Not mentioned.", "answer_meta": {"point": "AR"}})
    assert abstain["metrics"]["abstention_correct"] == 1.0
    assert abstain["score"] == 1.0

    conflict = await judge.evaluate(
        dataset="mem_gallery", question="Does it conflict?",
        predicted="Yes.",
        gold={"answer": "Yes.", "answer_meta": {"point": "CD"}})
    assert conflict["metrics"]["yesno_correct"] == 1.0


async def test_mem_gallery_answer_paths() -> None:
    evidence = [
        {"content": "[image: D2:IMG_003] [caption] Robotic arm. "
                   "[round D2:6]",
         "metadata": {"time": "2024-07-07T00:00:00Z", "source": ["s2_1"]}},
        {"content": "[image: D2:IMG_004] [caption] Another robotic arm. "
                   "[round D2:6]",
         "metadata": {"time": "2024-07-07T00:00:00Z", "source": ["s2_2"]}},
    ]
    model = FakeModel('{"evidence": [1, 2]}')
    answerer = Answerer({"max_evidence_chars": 8000, "styles": {}},
                        model)  # type: ignore[arg-type]
    picked = await answerer._answer_image_ids(
        "Which two images show a robotic arm?", evidence)
    assert picked == "D2:IMG_003, D2:IMG_004"

    conflict_model = FakeModel('{"verdict": "deny", "quote": "x"}')
    conflict = Answerer({"max_evidence_chars": 8000, "styles": {}},
                        conflict_model)  # type: ignore[arg-type]
    assert await conflict._answer_conflict(
        "Please determine whether the following information conflicts with "
        "the dialogue memory: they did not approve.", evidence) == "Yes."

    visual_model = FakeModel("Burj Khalifa")
    visual = Answerer({"max_evidence_chars": 8000, "styles": {}},
                      visual_model)  # type: ignore[arg-type]
    assert await visual._answer_visual_text(
        "Which landmark is this?\n\nThe image in the question shows: A tall tower.",
        evidence) == "Burj Khalifa"

    id_model = FakeModel("20240222_092257, email202411160004")
    id_answerer = Answerer({"max_evidence_chars": 8000, "styles": {}},
                           id_model)  # type: ignore[arg-type]
    id_evidence = [
        {"content": "[image: 20240222_092257] red train in rapeseed",
         "metadata": {}},
        {"content": "[email: email202411160004] hotel confirmation",
         "metadata": {}},
        {"content": "[image: 20240101_000000] unrelated", "metadata": {}},
    ]
    assert await id_answerer._answer_evidence_ids(
        "Find the red train photo and the hotel email.", id_evidence
    ) == "20240222_092257, email202411160004"

    atm_judge = Judge({"mode": "judge", "judge": True, "rubric": False},
                      FakeModel('{"correct": false}'))  # type: ignore[arg-type]
    atm_result = await atm_judge.evaluate(
        dataset="atm_bench", question="Find the photos.",
        predicted="20240222_092257, 20240222_092308",
        gold={"answer": "20240222_092257, 20240222_092308",
              "answer_meta": {"qtype": "list_recall"}})
    assert atm_result["primary"] == "atm_bench_score"
    assert atm_result["metrics"]["evidence_id_f1"] == 1.0

    count_model = FakeModel('{"count": 5}')
    counter = Answerer({"max_evidence_chars": 8000, "styles": {}},
                       count_model)  # type: ignore[arg-type]
    assert await counter._answer_count(
        "How many images show a robot?", evidence) == "5"

    abstain_model = FakeModel('{"answerable": false, "quote": "", "answer": ""}')
    abstain = Answerer({"max_evidence_chars": 8000, "styles": {}},
                       abstain_model)  # type: ignore[arg-type]
    assert await abstain._answer_abstain(
        "What is Amy's cat called?", evidence) == "Not mentioned."

    grounded_model = FakeModel(
        '{"answerable": true, "quote": "The robotic arm is blue.", '
        '"answer": "Blue"}')
    grounded = Answerer({"max_evidence_chars": 8000, "styles": {}},
                        grounded_model)  # type: ignore[arg-type]
    grounded_evidence = [{"content": "The robotic arm is blue.", "metadata": {}}]
    assert await grounded._answer_abstain(
        "What color is the robotic arm?", grounded_evidence) == "Blue"

    existence_model = FakeModel('{"stated": false, "quote": "never bought one"}')
    existence = Answerer({"max_evidence_chars": 8000, "styles": {}},
                         existence_model)  # type: ignore[arg-type]
    assert await existence._answer_existence(
        "Has Julian ever bought an autonomous vehicle?", evidence) == "No."

    order_model = FakeModel('{"first": 2, "second": 1}')
    order = Answerer({"max_evidence_chars": 8000, "styles": {}},
                     order_model)  # type: ignore[arg-type]
    order_evidence = [
        {"content": "Healthcare robots [round D2:2]", "metadata": {}},
        {"content": "Autonomous driving [round D2:4]", "metadata": {}},
    ]
    assert await order._answer_order(
        "Did Julian discuss autonomous driving before or after healthcare robots?",
        order_evidence) == "After."

    temporal_model = FakeModel('{"record": 2, "date": "2024-05-19"}')
    temporal = Answerer({"max_evidence_chars": 8000, "styles": {}},
                        temporal_model)  # type: ignore[arg-type]
    temporal_evidence = [
        {"content": "[resolved: today = 2024-05-21] Paris shopping.",
         "metadata": {}},
        {"content": "[resolved: tomorrow = 2024-05-19] trip to Paris tomorrow.",
         "metadata": {}},
    ]
    assert await temporal._answer_temporal(
        "When did Maria travel to Paris?", temporal_evidence) == "2024-05-19"


async def test_media_answer() -> None:
    """The answer message carries real image parts when media is configured."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAusB9Y9Z2ZkAAAAASUVORK5CYII=")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "D1_IMG_001.png").write_bytes(png)
        model = FakeModel("Paris")
        answerer = Answerer({"max_evidence_chars": 1000, "styles": {}},
                            model)  # type: ignore[arg-type]
        result = await answerer.answer(
            dataset="mem_gallery",
            question="Which image shows Paris?",
            evidence=[{"content": "[image: D1:IMG_001]\n[caption] Paris street.",
                       "metadata": {"source": ["s1"]}}],
            gold={"answer": "Paris", "answer_type": "TTL", "answer_meta": {}},
            media={"image_root": str(root), "max_images": 2, "max_edge": 128},
        )
        assert result["answer"] == "Paris"
        content = model.calls[0][1]["content"]
        assert isinstance(content, list)
        assert any(isinstance(part, dict) and part.get("type") == "image_url"
                   for part in content)


def test_multihop_helpers() -> None:
    items = [{"content": "[city] Glasgow, United Kingdom\n[caption] In November 2024"}]
    assert _locations(items) == ["Glasgow, United Kingdom"]
    assert _month(items, "How much for BMVC 2024?") == "November"
    assert _month(items, "What happened in May 2024?") == "May"


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
    test_mem_gallery_helpers()
    asyncio.run(test_mem_gallery_judge())
    asyncio.run(test_mem_gallery_answer_paths())
    asyncio.run(test_media_answer())
    test_multihop_helpers()
    print("test_bench: OK")


if __name__ == "__main__":
    main()
