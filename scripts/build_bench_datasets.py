#!/usr/bin/env python3
"""Convert the benchmark datasets under ``data/`` into AML-style event streams.

The output is one JSONL file per dataset, one line per user. Each line is a
**chronological event stream** -- ``add`` and ``search`` events stay in the
order the benchmark implies, so a search can happen before the next add. Every
``search`` event carries its reference answer (or rubric), which ``AML.jsonl``
does not have.

Supported datasets
------------------
``longmemeval_oracle`` / ``longmemeval_s`` / ``longmemeval_m``
    LongMemEval. One user per question. Every haystack session becomes one add,
    then the question is asked. Gold: ``answer``. Not interleaved: the question
    is asked after the whole haystack (the official setting).
``locomo_refined``
    LoCoMo-Refined. One user per conversation. Every session is one add; all
    questions are asked after the conversation (no question timestamps). Gold:
    one or more acceptable complete answers; evidence turn ids are preserved.
    Image/caption/query fields are encoded as text markers.
``beam_100k`` / ``beam_500k`` / ``beam_1m`` / ``beam_10m``
    BEAM. One user per chat. Each batch is one add. A probing question is asked
    right after the batch containing the last turn it references
    (``source_chat_ids``), which makes the stream genuinely add/search/add.
    Gold: category-specific answer field + ``rubric``.
``personamem_32k`` / ``personamem_128k``
    PersonaMem-v2 text benchmark. One user per persona. The chat history is cut
    into fixed-size segments (one add each). A query is asked after the segment
    that contains the end of its ``related_conversation_snippet`` when the
    preference is not updated / not forgotten / not sensitive; otherwise it is
    asked at the end. Gold: ``correct_answer``.
``clbench``
    CL-bench. Each row is one user: the context conversation becomes one add and
    the final user task becomes the query. Gold: ``rubrics`` (no single answer).
``scriptmem``
    ScriptMem. The release deliberately omits the source scripts, so this file
    is **search-only**: it has questions and gold MCQ answers but no add
    events. It cannot evaluate memory retrieval, only answer scoring.

Record shape (one line)::

    {"dataset": "...", "user_id": "...", "source_id": "...",
     "events": [
       {"step": 0, "type": "add", "time": "...", "request_id": "r_...",
        "session_id": "...", "session_index": 1,
        "messages": [{"role": "...", "content": "...", "timestamp": 171...}],
        "message_ids": ["session_1_1", ...], "written": 20,
        "applied": true, "applied_via": "planned", "duplicate": false,
        "retries": 0, "retry_times": [], "meta": {...}},
       {"step": 1, "type": "search", "query_id": "<user_id>:1", "time": "...",
        "query": "...", "query_raw": "...", "options": null, "top_k": 100,
        "run_id": null, "data_count": null, "evidence": [],
        "answer": "...", "answer_type": "...", "answer_letters": [...],
        "rubrics": [...], "answer_meta": {...}}
     ],
     "stats": {...}, "ground_truth": {"<query_id>": {...}}, "meta": {...}}

Usage::

    python scripts/build_bench_datasets.py                    # all -> data/processed/
    python scripts/build_bench_datasets.py --datasets beam_100k,clbench
    python scripts/build_bench_datasets.py --max-users 2 --out-dir /tmp/preview
    python scripts/build_bench_datasets.py --list
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = DATA / "processed"

TOP_K = 100
DEFAULT_MAX_CHARS = 3000          # matches schema's message chunk threshold
PERSONAMEM_TAR = DATA / "PersonaMem-V2" / "personamem_v2.tar.gz"

# Set by the CLI; PersonaMem's converter reads them.
PERSONAMEM_CHUNK = 20
PERSONAMEM_INTERLEAVE = True


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_id(dataset: str, user_id: str, session_id: str, index: int) -> str:
    return "r_" + sha256(f"{dataset}|{user_id}|{session_id}|{index}")[:64]


def iso(moment: datetime | None) -> str:
    if moment is None:
        return ""
    return moment.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_ms(moment: datetime | None) -> int | None:
    if moment is None:
        return None
    return int(moment.replace(tzinfo=timezone.utc).timestamp() * 1000)


def make_message(role: str, content: str, timestamp: int | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": role or "user", "content": content or ""}
    if timestamp is not None:
        message["timestamp"] = timestamp
    return message


def simple_message_ids(session_index: int, messages: list[dict[str, Any]]) -> list[str]:
    """Ids that add will assign, when every message stays one record.

    Long documents / JSON payloads are expanded by add, so no ids are claimed
    for those (the same convention as ``build_aml_dataset.py``).
    """
    if not messages:
        return []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or len(content) > DEFAULT_MAX_CHARS:
            return []
        if content.lstrip().startswith(("{", "[")):
            return []
    return [f"session_{session_index}_{n}" for n in range(1, len(messages) + 1)]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Event builder + validation
# ---------------------------------------------------------------------------

class UserBuilder:
    """Build one user's chronological add/search stream and its ground truth."""

    def __init__(self, dataset: str, user_id: str, meta: dict[str, Any] | None = None):
        self.dataset = dataset
        self.user_id = user_id
        self.meta = meta or {}
        self.events: list[dict[str, Any]] = []
        self.ground_truth: dict[str, dict[str, Any]] = {}

    # -- add ---------------------------------------------------------------
    def add(
        self,
        session_id: str,
        session_index: int,
        messages: list[dict[str, Any]],
        *,
        time: str = "",
        meta: dict[str, Any] | None = None,
        message_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        step = len(self.events)
        ids = simple_message_ids(session_index, messages) if message_ids is None else message_ids
        event = {
            "step": step,
            "type": "add",
            "time": time,
            "request_id": request_id(self.dataset, self.user_id, session_id, step),
            "session_id": session_id,
            "session_index": session_index,
            "messages": messages,
            "message_ids": ids,
            "written": len(messages),
            "applied": True,
            # planned replay: `written` is the number of input messages, not an
            # API response. Replaying the add may create more records.
            "applied_via": "planned",
            "duplicate": False,
            "retries": 0,
            "retry_times": [],
        }
        if meta:
            event["meta"] = meta
        self.events.append(event)
        return event

    # -- search ------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        query_raw: str | None = None,
        options: list[str] | None = None,
        time: str = "",
        answer: str | None = None,
        answer_type: str | None = None,
        answer_letters: list[str] | None = None,
        rubrics: list[str] | None = None,
        answer_meta: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        step = len(self.events)
        query_id = f"{self.user_id}:{step}"
        event = {
            "step": step,
            "type": "search",
            "query_id": query_id,
            "time": time,
            "query": query,
            "query_raw": query if query_raw is None else query_raw,
            "options": options,
            "top_k": TOP_K,
            "run_id": None,
            "data_count": None,
            "evidence": [],
            "answer": answer,
            "answer_type": answer_type,
            "answer_letters": answer_letters,
            "rubrics": rubrics or [],
            "answer_meta": answer_meta or {},
        }
        if meta:
            event["meta"] = meta
        self.events.append(event)

        gold = {
            "query": query,
            "answer": answer,
            "answer_type": answer_type,
            "answer_letters": answer_letters or [],
            "rubrics": rubrics or [],
            "options": options,
            "answer_meta": answer_meta or {},
        }
        self.ground_truth[query_id] = {
            key: value for key, value in gold.items() if value not in (None, "", [])
        }
        return event

    # -- finish ------------------------------------------------------------
    def record(self) -> dict[str, Any]:
        record = {
            "dataset": self.dataset,
            "user_id": self.user_id,
            # Suggested per-user store directory when replaying; the API will
            # still derive its own slug from user_id.
            "store_dir": f"{self.dataset}__{self.user_id}",
            "source_id": self.meta.get("source_id", self.user_id),
            "events": self.events,
            "stats": self._stats(),
            "ground_truth": self.ground_truth,
            "meta": self.meta,
        }
        validate_record(record)
        return record

    def _stats(self) -> dict[str, Any]:
        adds = [event for event in self.events if event["type"] == "add"]
        searches = [event for event in self.events if event["type"] == "search"]
        last_add_step = max((event["step"] for event in adds), default=-1)
        first_search_step = min((event["step"] for event in searches), default=None)
        return {
            "add_events": len(adds),
            "search_events": len(searches),
            "messages": sum(len(event["messages"]) for event in adds),
            "sessions": len({event["session_id"] for event in adds}),
            # True when at least one search ran before the stream finished adding.
            "streaming": bool(searches and adds and first_search_step < last_add_step),
            "last_add_step": last_add_step if adds else None,
            "first_search_step": first_search_step,
            "ground_truth_entries": len(self.ground_truth),
        }


def validate_record(record: dict[str, Any]) -> None:
    """Fail fast on malformed streams; the checks mirror ``docs/08-datasets.md``."""
    user_id = record["user_id"]
    events = record["events"]
    query_ids: set[str] = set()
    for index, event in enumerate(events):
        if event.get("step") != index:
            raise ValueError(f"{user_id}: step {event.get('step')} != index {index}")
        kind = event.get("type")
        if kind == "add":
            if not event.get("session_id"):
                raise ValueError(f"{user_id}: add without session_id at step {index}")
            if not event.get("messages"):
                raise ValueError(f"{user_id}: add without messages at step {index}")
            if event.get("session_index") is None:
                raise ValueError(f"{user_id}: add without session_index at step {index}")
        elif kind == "search":
            if not event.get("query"):
                raise ValueError(f"{user_id}: search without query at step {index}")
            query_id = event.get("query_id")
            if not query_id or query_id in query_ids:
                raise ValueError(f"{user_id}: duplicate/empty query_id at step {index}")
            query_ids.add(query_id)
            has_gold = bool(
                event.get("answer")
                or event.get("rubrics")
                or event.get("answer_letters")
                or (event.get("answer_meta") or {}).get("expected_compliance")
            )
            if not has_gold:
                raise ValueError(f"{user_id}: search without ground truth at step {index}")
        else:
            raise ValueError(f"{user_id}: unknown event type {kind!r} at step {index}")
    if set(record["ground_truth"]) != query_ids:
        raise ValueError(f"{user_id}: ground_truth keys do not match search events")


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

LONGMEMEVAL_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def parse_longmemeval_date(text: str) -> datetime | None:
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d (%a) %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except (TypeError, ValueError):
            continue
    return None


def convert_longmemeval(variant: str, max_users: int = 0) -> Iterator[dict[str, Any]]:
    path = DATA / "longmemeval-cleaned" / LONGMEMEVAL_FILES[variant]
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)

    for item_index, item in enumerate(items):
        if max_users and item_index >= max_users:
            break
        question_id = item["question_id"]
        user_id = f"lme-{variant}-{question_id}"
        question_date = item.get("question_date", "")
        question_moment = parse_longmemeval_date(question_date)

        sessions = list(zip(
            item.get("haystack_dates") or [],
            item.get("haystack_session_ids") or [],
            item.get("haystack_sessions") or [],
        ))
        # Chronological order is what a replay should see. Undated sessions keep
        # their original position at the end rather than being dropped.
        sessions.sort(key=lambda pair: (
            parse_longmemeval_date(pair[0]) or datetime.max,
            pair[2] is None,
        ))

        builder = UserBuilder(
            f"longmemeval_{variant}",
            user_id,
            {
                "source_id": question_id,
                "question_id": question_id,
                "question_type": item.get("question_type"),
                "question_date": question_date,
                "variant": variant,
                "sessions": len(sessions),
            },
        )

        session_id_by_source: dict[str, str] = {}
        for session_index, (date_text, source_id, messages) in enumerate(sessions, 1):
            session_moment = parse_longmemeval_date(date_text)
            session_id = f"lme-{variant}-{question_id}-s{session_index}"
            session_id_by_source[str(source_id)] = session_id
            converted = [
                make_message(
                    message.get("role", "user"),
                    message.get("content", ""),
                    to_ms(session_moment),
                )
                for message in messages
            ]
            builder.add(
                session_id,
                session_index,
                converted,
                time=iso(session_moment),
                meta={
                    "source_session_id": source_id,
                    "date": date_text,
                    # `has_answer` is benchmark bookkeeping; it is not memory and
                    # must not be passed to add.
                    "has_answer_messages": sum(
                        1 for message in messages if message.get("has_answer")
                    ),
                },
            )

        gold_sessions = [
            session_id_by_source.get(str(source_id), str(source_id))
            for source_id in item.get("answer_session_ids") or []
        ]
        builder.search(
            item["question"],
            query_raw=item["question"],
            time=iso(question_moment),
            answer=item.get("answer", ""),
            answer_type=item.get("question_type"),
            answer_meta={
                "question_id": question_id,
                "question_type": item.get("question_type"),
                "question_date": question_date,
                "answer_session_ids": item.get("answer_session_ids") or [],
                "gold_session_ids": gold_sessions,
            },
        )
        yield builder.record()


# ---------------------------------------------------------------------------
# LoCoMo-Refined
# ---------------------------------------------------------------------------

LOCOMO_PATH = DATA / "LoCoMo_refined" / "data" / "raw" / "locomo_refined.json"

#: The release uses numeric categories. These labels are descriptive only;
#: ``category`` itself is preserved verbatim in ``answer_meta``.
LOCOMO_CATEGORY_NAMES = {
    "1": "multi_hop",
    "2": "temporal",
    "3": "open_domain",
    "4": "single_hop",
}


def parse_locomo_time(value: Any) -> datetime | None:
    """``"1:56 pm on 8 May, 2023"`` -> datetime (None when unknown)."""
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"\b(am|pm)\b", lambda match: match.group(1).upper(), text)
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%d %B, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def normalize_dia_id(value: Any) -> str | None:
    """``D1:3`` / ``D30:05`` -> ``D1:3`` / ``D30:5``; other values -> None."""
    match = re.fullmatch(r"D(\d+):(\d+)", str(value or "").strip())
    if not match:
        return None
    return f"D{int(match.group(1))}:{int(match.group(2))}"


def locomo_evidence_ids(raw: Any) -> list[str]:
    """Evidence is usually a list of turn ids, but a few rows pack several
    ids into one string ('D8:6; D9:17') or use a leading zero ('D30:05')."""
    ids: list[str] = []
    items = [raw] if isinstance(raw, str) else (raw or [])
    for item in items:
        for match in re.findall(r"D\d+:\d+", str(item)):
            normalized = normalize_dia_id(match)
            if normalized:
                ids.append(normalized)
    return dedupe(ids)


def _locomo_images(raw: dict[str, Any]) -> list[str]:
    value = raw.get("img_url")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    text = str(value).strip()
    return [text] if text else []


def _locomo_message_text(raw: dict[str, Any]) -> str:
    """Keep the dialogue text and make image/caption/query searchable.

    ASMEM's search agent is text-only. The original LoCoMo messages can carry
    an image URL, a BLIP caption, and a search query; dropping them would make
    multimodal questions unanswerable, so they are encoded as text markers.
    """
    parts: list[str] = []
    text = str(raw.get("text") or "").strip()
    if text:
        parts.append(text)
    for image in _locomo_images(raw):
        parts.append(f"[image: {image}]")
    caption = str(raw.get("blip_caption") or "").strip()
    if caption:
        parts.append(f"[caption] {caption}")
    query = str(raw.get("query") or "").strip()
    if query:
        parts.append(f"[query] {query}")
    return "\n".join(parts)


def convert_locomo_refined(max_users: int = 0) -> Iterator[dict[str, Any]]:
    if not LOCOMO_PATH.exists():
        raise FileNotFoundError(LOCOMO_PATH)
    with open(LOCOMO_PATH, encoding="utf-8") as handle:
        samples = json.load(handle)

    for sample_index, sample in enumerate(samples):
        if max_users and sample_index >= max_users:
            break
        sample_id = str(sample.get("sample_id") or f"conv-{sample_index}")
        conversation = sample.get("conversation") or {}
        speaker_a = str(conversation.get("speaker_a") or "")
        speaker_b = str(conversation.get("speaker_b") or "")
        user_id = f"locomo-{sample_id}"

        builder = UserBuilder(
            "locomo_refined",
            user_id,
            {
                "source_id": sample_id,
                "sample_id": sample_id,
                "conversation_idx": sample_index,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "qa_count": len(sample.get("qa") or []),
                # Derived annotations are preserved for inspection but are
                # not added to memory; the benchmark memory is the dialogue.
                "annotations": {
                    "event_summary": sample.get("event_summary"),
                    "observation": sample.get("observation"),
                    "session_summary": sample.get("session_summary"),
                },
            },
        )

        message_lookup: dict[str, dict[str, Any]] = {}
        session_index = 1
        while f"session_{session_index}" in conversation:
            raw_messages = conversation.get(f"session_{session_index}") or []
            date_text = str(conversation.get(f"session_{session_index}_date_time") or "")
            moment = parse_locomo_time(date_text)
            messages: list[dict[str, Any]] = []
            for position, raw in enumerate(raw_messages, start=1):
                if not isinstance(raw, dict):
                    continue
                content = _locomo_message_text(raw)
                if not content:
                    continue
                speaker = str(raw.get("speaker") or "")
                # Preserve the real speaker name.  Mapping both people to
                # user/assistant loses the subject binding the verifier needs
                # ("I went..." cannot be tied to Caroline).  The Add adapter
                # accepts arbitrary role strings and keeps them.
                role = speaker or "user"
                messages.append(make_message(role, content, to_ms(moment)))

                dia_id = normalize_dia_id(raw.get("dia_id"))
                if dia_id:
                    message_index = int(dia_id.rsplit(":", 1)[1])
                    message_lookup[dia_id] = {
                        "dia_id": dia_id,
                        "session_index": session_index,
                        "message_index": message_index or position,
                        "speaker": speaker,
                        "text": str(raw.get("text") or "").strip(),
                        "images": _locomo_images(raw),
                        "blip_caption": str(raw.get("blip_caption") or "").strip(),
                        "query": str(raw.get("query") or "").strip(),
                        "has_multimodal_context": bool(
                            _locomo_images(raw)
                            or str(raw.get("blip_caption") or "").strip()
                            or str(raw.get("query") or "").strip()
                        ),
                    }
            if messages:
                builder.add(
                    f"locomo-{sample_id}-session-{session_index}",
                    session_index,
                    messages,
                    time=iso(moment),
                    meta={
                        "source_session_date_time": date_text,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                    },
                )
            session_index += 1

        for qa_index, qa in enumerate(sample.get("qa") or []):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            raw_answers = qa.get("answer")
            if isinstance(raw_answers, list):
                answers = [str(item).strip() for item in raw_answers if str(item).strip()]
            elif raw_answers in (None, ""):
                answers = []
            else:
                answers = [str(raw_answers).strip()]
            answers = dedupe(answers)
            evidence_ids = locomo_evidence_ids(qa.get("evidence"))
            evidence_messages = [
                message_lookup[item] for item in evidence_ids if item in message_lookup
            ]
            category = str(qa.get("category") or "")
            builder.search(
                question,
                query_raw=question,
                answer=answers[0] if answers else "",
                answer_type=f"category_{category}" if category else "category_unknown",
                answer_meta={
                    "acceptable_answers": answers,
                    "evidence": evidence_ids,
                    "evidence_messages": evidence_messages,
                    "is_multi_modality": bool(qa.get("is_multi_modality")),
                    "category": category,
                    "category_name": LOCOMO_CATEGORY_NAMES.get(category, ""),
                    "qa_index": qa_index,
                    "sample_id": sample_id,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            )

        yield builder.record()


# ---------------------------------------------------------------------------
# BEAM
# ---------------------------------------------------------------------------

BEAM_SIZES = {"100k": "100K", "500k": "500K", "1m": "1M", "10m": "10M"}

BEAM_ANSWER_FIELDS = {
    "abstention": ("ideal_response",),
    "contradiction_resolution": ("ideal_answer",),
    "event_ordering": ("answer",),
    "information_extraction": ("answer",),
    "instruction_following": ("expected_compliance",),
    "knowledge_update": ("answer",),
    "multi_session_reasoning": ("answer",),
    "preference_following": ("expected_compliance",),
    "summarization": ("ideal_summary",),
    "temporal_reasoning": ("answer",),
}


def parse_beam_date(text: Any) -> datetime | None:
    if not text:
        return None
    candidates = [str(text).strip()]
    for fmt in ("%B-%d-%Y", "%B %d, %Y", "%B %d %Y", "%Y-%m-%d",
                "%B-%d-%Y %H:%M", "%Y-%m-%dT%H:%M:%S"):
        for candidate in candidates:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def flatten_source_ids(value: Any) -> list[Any]:
    """``source_chat_ids`` is a list, a dict of lists, or a scalar."""
    if isinstance(value, dict):
        out: list[Any] = []
        for item in value.values():
            out.extend(flatten_source_ids(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_source_ids(item))
        return out
    if isinstance(value, (int, str)) and str(value) != "":
        return [value]
    return []


def beam_answer(question: dict[str, Any], category: str) -> str:
    for field in BEAM_ANSWER_FIELDS.get(category, ("answer",)):
        value = question.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value:
            return str(value)
    return ""


def normalize_beam_batches(raw: Any) -> list[tuple[str | None, dict[str, Any]]]:
    """Flatten the two on-disk layouts into ``[(plan_name, batch), ...]``.

    100K/500K/1M store ``[{batch_number, turns, ...}, ...]``. 10M stores
    ``[{"plan-1": [batch, ...]}, {"plan-2": [...]}, ...]``; its batch numbers
    restart inside each plan, so callers use the list position as the session
    index instead.
    """
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, dict) and "turns" in first:
        return [(None, batch) for batch in raw if isinstance(batch, dict)]
    out: list[tuple[str | None, dict[str, Any]]] = []
    for plan in raw:
        if not isinstance(plan, dict):
            continue
        for plan_name, batches in plan.items():
            if isinstance(batches, list):
                for batch in batches:
                    if isinstance(batch, dict):
                        out.append((plan_name, batch))
    return out


def convert_beam(size_key: str, max_users: int = 0) -> Iterator[dict[str, Any]]:
    size_dir = BEAM_SIZES[size_key]
    base = DATA / "BEAM" / "chats" / size_dir
    chat_dirs = sorted(
        (path for path in base.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for chat_index, chat_dir in enumerate(chat_dirs):
        if max_users and chat_index >= max_users:
            break
        chat_path = chat_dir / "chat.json"
        probe_path = chat_dir / "probing_questions" / "probing_questions.json"
        if not chat_path.exists() or not probe_path.exists():
            print(f"  skip {chat_dir}: missing chat/probing questions", file=sys.stderr)
            continue

        with open(chat_path, encoding="utf-8") as handle:
            batches = json.load(handle)
        with open(probe_path, encoding="utf-8") as handle:
            probes = json.load(handle)
        topic = {}
        topic_path = chat_dir / "topic.json"
        if topic_path.exists():
            with open(topic_path, encoding="utf-8") as handle:
                topic = json.load(handle)

        user_id = f"beam-{size_key}-{chat_dir.name}"
        raw_batches = normalize_beam_batches(batches)
        builder = UserBuilder(
            f"beam_{size_key}",
            user_id,
            {
                "source_id": chat_dir.name,
                "chat_size": size_dir,
                "topic": topic,
                "batches": len(raw_batches),
            },
        )

        # Index every message id (user and assistant) to the *session index* of
        # its batch, so a probe can be asked as soon as its last referenced turn
        # is in memory. 10M restarts batch_number inside every plan, hence the
        # position in the flattened list is the stable key.
        batch_of_id: dict[Any, int] = {}
        batch_payloads: list[
            tuple[int, str | None, dict[str, Any], list[dict[str, Any]], datetime | None]
        ] = []
        last_moment: datetime | None = None
        for plan_name, batch in raw_batches:
            declared_moment = parse_beam_date(batch.get("time_anchor"))
            first_message_moment: datetime | None = None
            messages: list[dict[str, Any]] = []
            for turn in batch.get("turns") or []:
                for message in turn or []:
                    moment = (
                        parse_beam_date(message.get("time_anchor"))
                        or declared_moment
                        or last_moment
                    )
                    if moment is not None:
                        if first_message_moment is None:
                            first_message_moment = moment
                        last_moment = moment
                    messages.append(make_message(
                        message.get("role", "user"),
                        message.get("content", ""),
                        to_ms(moment),
                    ))
            if not messages:
                continue
            session_index = len(batch_payloads) + 1
            for turn in batch.get("turns") or []:
                for message in turn or []:
                    message_id = message.get("id")
                    if message_id is not None:
                        batch_of_id[message_id] = session_index
            batch_moment = declared_moment or first_message_moment or last_moment
            batch_payloads.append(
                (session_index, plan_name, batch, messages, batch_moment)
            )

        last_batch = len(batch_payloads) or 1
        probes_by_batch: dict[int, list[tuple[int, int, str, dict[str, Any], list[Any]]]] = (
            collections.defaultdict(list)
        )
        def lookup_batch(value: Any) -> int | None:
            for key in (value, str(value)):
                if key in batch_of_id:
                    return batch_of_id[key]
            try:
                return batch_of_id.get(int(value))
            except (TypeError, ValueError):
                return None

        for category_index, (category, questions) in enumerate(probes.items()):
            for question_index, question in enumerate(questions):
                source_ids = flatten_source_ids(question.get("source_chat_ids"))
                mapped = [
                    batch for batch in (lookup_batch(value) for value in source_ids)
                    if batch is not None
                ]
                insert_at = max(mapped) if mapped else last_batch
                insert_at = min(insert_at, last_batch)
                probes_by_batch[insert_at].append(
                    (category_index, question_index, category, question, source_ids)
                )

        for session_index, plan_name, batch, messages, batch_moment in batch_payloads:
            builder.add(
                f"beam-{size_key}-{chat_dir.name}-batch-{session_index}",
                session_index,
                messages,
                time=iso(batch_moment),
                meta={
                    "batch_number": batch.get("batch_number"),
                    "plan": plan_name,
                    "time_anchor": batch.get("time_anchor"),
                    "turns": len(batch.get("turns") or []),
                },
            )
            for _, _, category, question, source_ids in sorted(
                probes_by_batch.get(session_index, []), key=lambda item: item[:2]
            ):
                answer = beam_answer(question, category)
                rubrics = question.get("rubric") or question.get("rubrics") or []
                if isinstance(rubrics, str):
                    rubrics = [rubrics]
                answer_meta = {
                    key: value
                    for key, value in question.items()
                    if key not in {"question", "rubric", "rubrics"}
                }
                answer_meta["category"] = category
                answer_meta["source_chat_ids"] = source_ids
                builder.search(
                    question.get("question", ""),
                    query_raw=question.get("question", ""),
                    time=iso(batch_moment),
                    answer=answer,
                    answer_type=category,
                    rubrics=rubrics,
                    answer_meta=answer_meta,
                    meta={
                        "asked_after_batch": session_index,
                        "plan": plan_name,
                        "source_chat_ids": source_ids,
                    },
                )

        yield builder.record()


# ---------------------------------------------------------------------------
# PersonaMem-v2
# ---------------------------------------------------------------------------

PERSONAMEM_LINK_FIELDS = {
    "32k": "chat_history_32k_link",
    "128k": "chat_history_128k_link",
}


def parse_personamem_query(raw: str) -> tuple[str, Any]:
    """``"{'role': 'user', 'content': '...'}"`` -> the content string."""
    text = (raw or "").strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:  # noqa: BLE001 - the field is a Python repr, not JSON
            continue
        if isinstance(value, dict):
            return str(value.get("content", "")), value
        if isinstance(value, str):
            return value, value
    match = re.search(r"""['"]content['"]\s*:\s*(['"])(.*)\1""", text, re.S)
    if match:
        return match.group(2), text
    return text, text


def parse_personamem_json(raw: Any, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (list, dict)):
        return raw
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw)
        except Exception:  # noqa: BLE001
            continue
    return default


def find_snippet_end(history: list[dict[str, Any]], snippet: list[dict[str, Any]]) -> int | None:
    """Index of the last history message that completes ``snippet``."""
    if not snippet:
        return None
    size = len(snippet)
    for index in range(len(history) - size, -1, -1):
        if all(
            history[index + offset].get("role") == snippet[offset].get("role")
            and history[index + offset].get("content") == snippet[offset].get("content")
            for offset in range(size)
        ):
            return index + size - 1
    # The CSV sometimes truncates / re-escapes long messages; fall back to the
    # last message's exact text, then to a substring match.
    last = snippet[-1]
    content = last.get("content") or ""
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == last.get("role") and history[index].get("content") == content:
            return index
    if content:
        for index in range(len(history) - 1, -1, -1):
            if content in (history[index].get("content") or ""):
                return index
    return None


def convert_personamem(
    variant: str,
    max_users: int = 0,
    chunk_messages: int = 20,
    interleave: bool = True,
) -> Iterator[dict[str, Any]]:
    if not PERSONAMEM_TAR.exists():
        raise FileNotFoundError(PERSONAMEM_TAR)
    link_field = PERSONAMEM_LINK_FIELDS[variant]

    with tarfile.open(PERSONAMEM_TAR, "r:gz") as archive:
        csv_member = archive.getmember("./benchmark/text/benchmark.csv")
        csv_file = archive.extractfile(csv_member)
        if csv_file is None:
            raise FileNotFoundError("benchmark/text/benchmark.csv in PersonaMem tar")
        rows = list(csv.DictReader(io.TextIOWrapper(csv_file, encoding="utf-8")))

    groups: "collections.OrderedDict[str, list[dict[str, Any]]]" = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row[link_field], []).append(row)

    needed = {f"./{link.lstrip('./')}" for link in groups}
    histories: dict[str, dict[str, Any]] = {}
    with tarfile.open(PERSONAMEM_TAR, "r:gz") as archive:
        for member in archive:
            if member.name in needed:
                handle = archive.extractfile(member)
                if handle is not None:
                    histories[member.name] = json.load(handle)
                if len(histories) == len(needed):
                    break
    missing = needed - set(histories)
    if missing:
        raise FileNotFoundError(f"missing PersonaMem chats: {sorted(missing)[:3]}")

    for group_index, (link, persona_rows) in enumerate(groups.items()):
        if max_users and group_index >= max_users:
            break
        member_name = f"./{link.lstrip('./')}"
        payload = histories[member_name]
        history = payload.get("chat_history") or payload
        persona_id = persona_rows[0].get("persona_id") or group_index
        user_id = f"personamem-{variant}-{persona_id}"

        builder = UserBuilder(
            f"personamem_{variant}",
            user_id,
            {
                "source_id": persona_id,
                "persona_id": persona_id,
                "variant": variant,
                "chat_link": link,
                "messages_in_history": len(history),
                "system_prompt_excluded": True,
                "chunk_messages": chunk_messages,
                "queries": len(persona_rows),
            },
        )

        # The system prompt contains the full persona and is generation setup,
        # not user memory; storing it would make every query trivially
        # answerable from the prompt for one, so it is excluded.
        chunks: list[list[tuple[int, dict[str, Any]]]] = []
        current: list[tuple[int, dict[str, Any]]] = []
        original_to_chunk: dict[int, int] = {}
        for original_index, message in enumerate(history):
            if (message.get("role") or "") == "system":
                continue
            current.append((original_index, message))
            if len(current) >= chunk_messages:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
        for chunk_index, chunk in enumerate(chunks):
            for original_index, _ in chunk:
                original_to_chunk[original_index] = chunk_index

        # Decide where each query is asked.
        by_chunk: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        at_end: list[dict[str, Any]] = []
        for row in persona_rows:
            query, query_raw = parse_personamem_query(row.get("user_query", ""))
            snippet = parse_personamem_json(row.get("related_conversation_snippet"), [])
            if not isinstance(snippet, list):
                snippet = []
            snippet = [
                item for item in snippet
                if isinstance(item, dict) and item.get("content")
            ]
            can_interleave = (
                interleave
                and str(row.get("updated", "")).lower() != "true"
                and row.get("pref_type") != "ask_to_forget"
                and str(row.get("sensitive_info", "")).lower() != "true"
            )
            insertion = None
            if can_interleave:
                snippet_end = find_snippet_end(history, snippet)
                if snippet_end is not None:
                    insertion = original_to_chunk.get(snippet_end)
            item = {"row": row, "query": query, "query_raw": query_raw, "insertion": insertion}
            if insertion is None:
                at_end.append(item)
            else:
                by_chunk[insertion].append(item)

        for chunk_index, chunk in enumerate(chunks):
            messages = [
                make_message(message.get("role", "user"), message.get("content", ""))
                for _, message in chunk
            ]
            builder.add(
                f"personamem-{variant}-{persona_id}-session-{chunk_index + 1}",
                chunk_index + 1,
                messages,
                meta={
                    "history_index_start": chunk[0][0],
                    "history_index_end": chunk[-1][0],
                    "chunk": chunk_index + 1,
                },
            )
            for item in by_chunk.get(chunk_index, []):
                _emit_personamem_search(builder, item, context="stream")

        for item in at_end:
            _emit_personamem_search(builder, item, context="end")

        yield builder.record()


def _emit_personamem_search(builder: UserBuilder, item: dict[str, Any], *, context: str) -> None:
    row = item["row"]
    incorrect = parse_personamem_json(row.get("incorrect_answers"), [])
    if not isinstance(incorrect, list):
        incorrect = []
    builder.search(
        item["query"],
        query_raw=item["query_raw"],
        answer=row.get("correct_answer", ""),
        answer_type=row.get("pref_type") or "freeform",
        answer_meta={
            "persona_id": row.get("persona_id"),
            "user_query_raw": row.get("user_query"),
            "incorrect_answers": incorrect,
            "preference": row.get("preference"),
            "topic_preference": row.get("topic_preference"),
            "topic_query": row.get("topic_query"),
            "pref_type": row.get("pref_type"),
            "updated": row.get("updated"),
            "prev_pref": row.get("prev_pref"),
            "who": row.get("who"),
            "sensitive_info": row.get("sensitive_info"),
            "conversation_scenario": row.get("conversation_scenario"),
            "related_conversation_snippet": row.get("related_conversation_snippet"),
            "distance_from_related_snippet_to_query_32k": row.get(
                "distance_from_related_snippet_to_query_32k"
            ),
            "distance_from_related_snippet_to_query_128k": row.get(
                "distance_from_related_snippet_to_query_128k"
            ),
        },
        meta={"asked_at": context},
    )


# ---------------------------------------------------------------------------
# CL-bench
# ---------------------------------------------------------------------------

_TASK_CUE = re.compile(
    r"""(?:
        \[?\s*(?:component\s*\d+\s*[:.]?\s*)?(?:final\s+)?(?:task|prompt\s*\d*|question)\s*\]?\s*[:.]?\b
        |please\b |can\s+you\b |could\s+you\b |would\s+you\b
        |i\s+need\b |i\s+want\b |i['’]d\s+like\b |using\s+only\b |using\b
        |write\b |create\b |draft\b |design\b |develop\b |explain\b |describe\b
        |provide\b |give\b |tell\b |list\b |identify\b |analyze\b |evaluate\b
        |summarize\b |predict\b |derive\b |calculate\b |exclude\b
        |need\b |help\b |hey\b |hi\b |hello\b |tell\s+me\b |show\s+me\b
        |for\s+each\b |make\s+sure\b |include\b |do\s+not\b |don['’]t\b
        |you\s+may\b |your\s+answer\b |respond\b
        |what\b |how\b |why\b |when\b |where\b |which\b |who\b
        |is\s+there\b |are\s+there\b |do\s+you\b |does\b |should\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

#: Phrases that almost certainly introduce the user's actual request rather
#: than an incidental sentence in a pasted document.
_STRONG_TASK = re.compile(
    r"""(?:
        i\s*(?:['’]d|would)\s+like\s+your\s+help
        |i\s+(?:want|need)\s+you\s+to
        |your\s+task\b |please\s+(?:produce|draft|create|write|provide)
        |can\s+you\s+(?:help|explain|summari[sz]e|describe|tell|show)
        |(?:tell|show)\s+me\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

#: Opening text that marks a pasted document/transcript, not the user request.
_CONTEXT_HEADER = re.compile(
    r"""^\s*(?:
        \[?\s*conversation\s+transcript\b
        |conversation\s+transcript\b
        |\[?\s*component\s*\d
        |===+\s*(?:component|context|file)\b
        |\#(?:\s|$)
        |documents?\s*:
        |context\s+documents?\s*:
        |here\s+(?:is|are)\s+the\s+(?:document|transcript|context)
        |file\s*:
        |extract\s+of\b
        |chapter\s+\d
        |\[turn\s*\d
        |message\s+from\s*:
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def paragraphs_with_offsets(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    def add(start: int, end: int) -> None:
        chunk = text[start:end]
        stripped = chunk.strip()
        if not stripped:
            return
        leading = len(chunk) - len(chunk.lstrip())
        trailing = len(chunk) - len(chunk.rstrip())
        out.append((start + leading, end - trailing, stripped))

    position = 0
    for match in re.finditer(r"\n\s*\n", text):
        add(position, match.start())
        position = match.end()
    add(position, len(text))
    return out


def looks_like_task(paragraph: str) -> bool:
    # Strip markdown/decorative leaders such as "=== Component 5: ...".
    stripped = paragraph.lstrip(" \t\r\n=#*->—")
    if _TASK_CUE.match(stripped):
        return True
    # A question mark only marks a short paragraph as the task. A 20k pasted
    # transcript also ends with "?", but it is context, not the request.
    return len(stripped) <= 500 and stripped.rstrip().endswith("?")


def _analysis_text(text: str) -> str:
    """Make literal ``\\n`` readable for paragraph splitting, keeping offsets.

    Some CL-bench rows encode newlines as the two characters ``\\`` + ``n``.
    Replacing them with newline + space keeps the string length identical, so
    offsets still point into the original text.
    """
    return text.replace("\\r", "\r ").replace("\\n", "\n ").replace("\\t", "\t ")


def _trim_to_sentence(text: str, start: int, limit: int = 1500) -> int:
    end = min(len(text), start + limit)
    if end == len(text):
        return end
    window = text[start:end]
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "),
              window.rfind(":\n"), window.rfind("\n"))
    return start + cut + 1 if cut > 100 else end


def extract_clbench_task(text: str) -> tuple[int, int, str]:
    """Return ``(start, end, method)`` for the task inside a CL-bench turn.

    The final user turn often contains both the task and the context document.
    The task can be the opening paragraph ("I have this article, what do you
    think? <article>") or the closing paragraph ("<rulebook> What do X?"), so
    both ends are considered and the more task-like / shorter candidate wins.
    """
    if len(text) <= 2000:
        return 0, len(text), "short_message"
    analysis = _analysis_text(text)
    paragraphs = paragraphs_with_offsets(analysis)
    if not paragraphs:
        return 0, len(text), "empty"

    # priority: 0 = a clear request at the opening, 1 = task-like,
    # 2 = fallback. Length breaks ties inside a priority.
    candidates: list[tuple[int, int, str, int]] = []

    def opening_priority(paragraph: str) -> int:
        cleaned = paragraph.lstrip(" \t\r\n=#*->—")
        if _TASK_CUE.match(cleaned) or _STRONG_TASK.search(cleaned[:600]):
            return 0
        return 1 if looks_like_task(paragraph) else 2

    # Opening paragraph: often the user's actual request, followed by the
    # document they pasted. Do not mistake "Component 2 — Conversation
    # Transcript" or an article title for the request.
    first_start, _, first_paragraph = paragraphs[0]
    first_cleaned = first_paragraph.lstrip(" \t\r\n=#*->—")
    is_header = bool(_CONTEXT_HEADER.match(first_cleaned[:160]))
    if not is_header and len(first_paragraph) <= 2000:
        priority = opening_priority(first_paragraph)
        if priority == 2 and len(first_paragraph) < 120:
            # A short intro ("Below are the instructions ...") can be followed
            # by the actual request in the next paragraph.
            for next_start, _, next_paragraph in paragraphs[1:3]:
                if len(next_paragraph) > 2000:
                    break
                next_priority = opening_priority(next_paragraph)
                if next_priority <= 1:
                    candidates.append((
                        first_start,
                        next_start + len(next_paragraph),
                        "opening_next_paragraph",
                        next_priority,
                    ))
                    break
        else:
            opening_end = first_start + len(first_paragraph)
            # A request can span a few short paragraphs ("... draft a deck." /
            # "Using only ..." / "For each slide ...") before the document.
            for next_start, _, next_paragraph in paragraphs[1:4]:
                cleaned = next_paragraph.lstrip(" \t\r\n=#*->—")
                if len(next_paragraph) > 2000 or not _TASK_CUE.match(cleaned):
                    break
                opening_end = next_start + len(next_paragraph)
            candidates.append((first_start, opening_end,
                               "opening_paragraph", priority))
    elif not is_header and (
        _TASK_CUE.match(first_cleaned) or _STRONG_TASK.search(first_cleaned[:600])
    ):
        candidates.append((first_start, _trim_to_sentence(text, first_start),
                           "opening_trimmed", opening_priority(first_paragraph)))

    # Closing paragraph: "<context> ... <question>". Prefer the first
    # task-like paragraph in the last three so constraints stay attached.
    end_candidate: tuple[int, int, str, int] | None = None
    for start, _, paragraph in paragraphs[max(0, len(paragraphs) - 3):]:
        if looks_like_task(paragraph):
            end_candidate = (start, len(text), "closing_task_like", 1)
            break
    if end_candidate is None:
        last_start, _, last_paragraph = paragraphs[-1]
        priority = 1 if looks_like_task(last_paragraph) else 2
        end_candidate = (last_start, len(text), "closing_paragraph", priority)
    candidates.append(end_candidate)

    # Some rows paste a transcript / article with no blank line before the
    # request ("<article> can you summarise ...?"). When the closing candidate
    # is a huge document, look for the last task-like *sentence* in its tail.
    if end_candidate[1] - end_candidate[0] > 4000:
        tail_start = max(0, len(text) - 3000)
        tail = analysis[tail_start:]
        boundaries = [0]
        boundaries += [match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n+", tail)]
        chosen = None
        for boundary in boundaries:
            raw = tail[boundary:]
            lead = len(raw) - len(raw.lstrip())
            segment = raw[lead:].lstrip(" \t\r\n=#*->—:;,.!?()[]'\"")
            if not segment:
                continue
            # The earliest cue sentence within 1200 chars of the end usually
            # starts the whole multi-sentence request.
            if len(tail) - boundary <= 1200 and _TASK_CUE.match(segment):
                chosen = boundary + lead + (len(raw[lead:]) - len(segment))
                break
        if chosen is not None:
            candidates.append((tail_start + chosen, len(text), "last_cue_sentence", 1))
        else:
            # No imperative cue: accept a final question only when it starts
            # like a sentence (a lowercase fragment such as "outlined by the
            # SAWCS?" is article text, not the user's request).
            question = tail.rfind("?")
            if question != -1:
                sentence_start = max(
                    tail.rfind(". ", 0, question),
                    tail.rfind("! ", 0, question),
                    tail.rfind("\n", 0, question),
                )
                sentence_start = sentence_start + 1 if sentence_start != -1 else 0
                segment = tail[sentence_start:question + 1].strip()
                if segment and segment[0].isupper():
                    candidates.append((
                        tail_start + sentence_start,
                        tail_start + question + 1,
                        "last_question",
                        1,
                    ))

    # A clear opening request wins over an incidental question inside the
    # pasted document; otherwise prefer task-like. When both ends are only
    # "fallback" quality, a closing paragraph of reasonable size beats a bare
    # document title at the opening.
    def rank(item: tuple[int, int, str, int]) -> tuple[int, int, int]:
        start, end, method, priority = item
        length = end - start
        closing_bonus = 0 if (
            priority == 2 and method.startswith("closing") and length <= 3000
        ) else 1
        return priority, closing_bonus, length

    best = min(candidates, key=rank)
    return best[0], best[1], best[2]


def convert_clbench(max_users: int = 0) -> Iterator[dict[str, Any]]:
    path = DATA / "CL-bench" / "CL-bench.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            if max_users and row_index >= max_users:
                break
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            task_id = metadata.get("task_id") or f"row-{row_index}"
            user_id = f"clbench-{task_id}"
            messages = row.get("messages") or []
            if not messages:
                continue
            last_user_index = max(
                (index for index, message in enumerate(messages) if message.get("role") == "user"),
                default=len(messages) - 1,
            )
            last_user = messages[last_user_index].get("content") or ""
            task_start, task_end, method = extract_clbench_task(last_user)
            query = last_user[task_start:task_end].strip() or last_user.strip()
            # Everything outside [task_start, task_end) is context. Keep the
            # two sides in one synthetic user message each so the add payload
            # stays faithful to the original ordering.
            prefix_parts = [
                part for part in (last_user[:task_start].strip(),
                                  last_user[task_end:].strip()) if part
            ]
            prefix = "\n\n".join(prefix_parts)

            context_messages = [
                make_message(message.get("role", "user"), message.get("content", ""))
                for index, message in enumerate(messages)
                if index != last_user_index
            ]
            if prefix:
                context_messages.append(make_message("user", prefix))
            self_contained = not context_messages
            if self_contained:
                # No separate context was shipped. Add the task text itself so
                # the stream is still replayable, and flag it.
                context_messages = [make_message("user", last_user)]

            builder = UserBuilder(
                "clbench",
                user_id,
                {
                    "source_id": task_id,
                    "task_id": task_id,
                    "context_id": metadata.get("context_id"),
                    "context_category": metadata.get("context_category"),
                    "sub_category": metadata.get("sub_category"),
                    "rubric_count": len(row.get("rubrics") or []),
                    "self_contained": self_contained,
                    "query_extraction": method,
                },
            )
            builder.add(
                f"clbench-{task_id}-context",
                1,
                context_messages,
                meta={
                    "task_id": task_id,
                    "context_id": metadata.get("context_id"),
                    "context_chars": sum(
                        len(message.get("content") or "") for message in context_messages
                    ),
                },
            )
            rubrics = row.get("rubrics") or []
            if isinstance(rubrics, str):
                rubrics = [rubrics]
            builder.search(
                query,
                query_raw=last_user if len(last_user) <= 4000 else query,
                answer=None,
                answer_type="rubric",
                rubrics=rubrics,
                answer_meta={
                    "task_id": task_id,
                    "context_id": metadata.get("context_id"),
                    "context_category": metadata.get("context_category"),
                    "sub_category": metadata.get("sub_category"),
                    "query_extraction": method,
                    "last_user_message_chars": len(last_user),
                    "context_chars": sum(
                        len(message.get("content") or "") for message in context_messages
                    ),
                    "self_contained": self_contained,
                },
            )
            yield builder.record()


# ---------------------------------------------------------------------------
# ScriptMem (questions only -- source scripts are not redistributed)
# ---------------------------------------------------------------------------

SCRIPTMEM_SOURCES = ("angry", "enemy", "friends", "man_earth")


def split_scriptmem_question(text: str, options: list[str]) -> str:
    if not options:
        return text.strip()
    marker = re.search(r"\n\s*A[\.\)]\s+", text)
    if marker:
        return text[:marker.start()].strip()
    first = options[0]
    if first and first in text:
        return text[:text.index(first)].strip()
    return text.strip()


def convert_scriptmem(max_users: int = 0) -> Iterator[dict[str, Any]]:
    path = DATA / "ScriptMem" / "data" / "public" / "questions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    grouped: "collections.OrderedDict[str, list[dict[str, Any]]]" = collections.OrderedDict()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped.setdefault(row.get("source") or "scriptmem", []).append(row)

    for group_index, (source, rows) in enumerate(grouped.items()):
        if max_users and group_index >= max_users:
            break
        builder = UserBuilder(
            "scriptmem",
            f"scriptmem-{source}",
            {
                "source_id": source,
                "source": source,
                "memory_available": False,
                "usable_for_memory_eval": False,
                "note": (
                    "ScriptMem does not redistribute the source scripts. This "
                    "stream is search-only: questions and gold answers are "
                    "present, add events are intentionally absent."
                ),
            },
        )
        for row in rows:
            options = row.get("option") or []
            if not isinstance(options, list):
                options = []
            query = split_scriptmem_question(row.get("question", ""), options)
            builder.search(
                query,
                query_raw=row.get("question", ""),
                options=options or None,
                answer=row.get("answer", ""),
                answer_type=row.get("qa_type"),
                answer_letters=row.get("answer_letters") or [],
                answer_meta={
                    "qa_id": row.get("qa_id"),
                    "question_id": row.get("question_id"),
                    "conversation_id": row.get("conversation_id"),
                    "source": source,
                    "sample_id": row.get("sample_id"),
                    "qa_index": row.get("qa_index"),
                    "qa_type": row.get("qa_type"),
                },
            )
        record = builder.record()
        record["stats"]["memory_available"] = False
        yield record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CONVERTERS = {
    "longmemeval_oracle": lambda limit: convert_longmemeval("oracle", limit),
    "longmemeval_s": lambda limit: convert_longmemeval("s", limit),
    "longmemeval_m": lambda limit: convert_longmemeval("m", limit),
    "locomo_refined": convert_locomo_refined,
    "beam_100k": lambda limit: convert_beam("100k", limit),
    "beam_500k": lambda limit: convert_beam("500k", limit),
    "beam_1m": lambda limit: convert_beam("1m", limit),
    "beam_10m": lambda limit: convert_beam("10m", limit),
    "personamem_32k": lambda limit: convert_personamem(
        "32k", limit, chunk_messages=PERSONAMEM_CHUNK,
        interleave=PERSONAMEM_INTERLEAVE),
    "personamem_128k": lambda limit: convert_personamem(
        "128k", limit, chunk_messages=PERSONAMEM_CHUNK,
        interleave=PERSONAMEM_INTERLEAVE),
    "clbench": convert_clbench,
    "scriptmem": convert_scriptmem,
}

DEFAULT_DATASETS = [
    "longmemeval_oracle", "longmemeval_s", "longmemeval_m", "locomo_refined",
    "beam_100k", "beam_500k", "beam_1m", "beam_10m",
    "personamem_32k", "personamem_128k",
    "clbench", "scriptmem",
]


def summarize_file(path: Path) -> dict[str, Any]:
    """Recompute a dataset summary from an existing output file."""
    users = adds = searches = messages = streaming_users = gt_entries = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            stats = record.get("stats") or {}
            users += 1
            adds += int(stats.get("add_events") or 0)
            searches += int(stats.get("search_events") or 0)
            messages += int(stats.get("messages") or 0)
            gt_entries += int(stats.get("ground_truth_entries") or 0)
            streaming_users += 1 if stats.get("streaming") else 0
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "users": users,
        "add_events": adds,
        "search_events": searches,
        "messages": messages,
        "ground_truth_entries": gt_entries,
        "streaming_users": streaming_users,
        "elapsed_seconds": None,
    }


def write_dataset(name: str, out_dir: Path, max_users: int, overwrite: bool) -> dict[str, Any]:
    out_path = out_dir / f"{name}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"  skip {name}: {out_path} exists (use --overwrite)", file=sys.stderr)
        return {"skipped": True, "path": str(out_path)}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    users = adds = searches = messages = streaming_users = 0
    gt_entries = 0
    started = datetime.now()

    with open(tmp_path, "w", encoding="utf-8") as out:
        for user_index, record in enumerate(CONVERTERS[name](max_users), 1):
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            users += 1
            stats = record["stats"]
            adds += stats["add_events"]
            searches += stats["search_events"]
            messages += stats["messages"]
            gt_entries += stats["ground_truth_entries"]
            streaming_users += 1 if stats["streaming"] else 0
            if user_index % 50 == 0:
                elapsed = (datetime.now() - started).total_seconds()
                print(
                    f"  {name}: {user_index} users, {adds} adds, {searches} searches "
                    f"({elapsed:.0f}s)",
                    file=sys.stderr,
                )

    os.replace(tmp_path, out_path)
    size = out_path.stat().st_size
    elapsed = (datetime.now() - started).total_seconds()
    summary = {
        "path": str(out_path),
        "bytes": size,
        "users": users,
        "add_events": adds,
        "search_events": searches,
        "messages": messages,
        "ground_truth_entries": gt_entries,
        "streaming_users": streaming_users,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        f"wrote {out_path}: {users} users, {adds} adds, {searches} searches, "
        f"{gt_entries} gold, {streaming_users} streaming users, "
        f"{size / 1e6:.1f} MB ({elapsed:.0f}s)",
        file=sys.stderr,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data/ benchmarks into AML-style add/search streams."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=f"comma list, or 'all'. choices: {', '.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-users", type=int, default=0,
                        help="debug: at most N users per dataset")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing output files")
    parser.add_argument("--personamem-chunk", type=int, default=20,
                        help="messages per PersonaMem add segment (default 20)")
    parser.add_argument("--no-personamem-interleave", action="store_true",
                        help="ask every PersonaMem query at the end instead of "
                             "after its related snippet")
    parser.add_argument("--manifest-only", action="store_true",
                        help="do not convert; recompute manifest entries from "
                             "existing output files")
    parser.add_argument("--list", action="store_true", help="list dataset names")
    args = parser.parse_args()

    global PERSONAMEM_CHUNK, PERSONAMEM_INTERLEAVE
    PERSONAMEM_CHUNK = max(1, args.personamem_chunk)
    PERSONAMEM_INTERLEAVE = not args.no_personamem_interleave

    if args.list:
        for name in DEFAULT_DATASETS:
            print(name)
        return

    if args.datasets.strip() == "all":
        datasets = DEFAULT_DATASETS
    else:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = [name for name in datasets if name not in CONVERTERS]
    if unknown:
        parser.error(f"unknown datasets: {unknown}; choices: {', '.join(DEFAULT_DATASETS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["schema"] = "AML-style chronological add/search events with ground truth"
    manifest.setdefault("datasets", {})

    for name in datasets:
        print(f"== {name} ==", file=sys.stderr)
        if args.manifest_only:
            path = out_dir / f"{name}.jsonl"
            if not path.exists():
                print(f"  skip {name}: {path} does not exist", file=sys.stderr)
                continue
            manifest["datasets"][name] = summarize_file(path)
            print(f"  summarized {path}", file=sys.stderr)
        else:
            manifest["datasets"][name] = write_dataset(
                name, out_dir, args.max_users, args.overwrite
            )

    manifest["datasets"] = {
        name: manifest["datasets"][name]
        for name in DEFAULT_DATASETS
        if name in manifest["datasets"]
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
