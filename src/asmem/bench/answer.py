"""Answer step: turn search evidence into a final answer."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import llm
from . import prompts
from .judge import normalize_text, parse_json_object, parse_letters

#: Small models often echo the prompt's trailing "Answer:" label.  It is not
#: part of the answer and makes the judge's inclusion check noisier.
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:final\s+answer|answer|response)\s*[:\-]\s*", re.IGNORECASE)
_IMAGE_MARKER = re.compile(r"\[image:\s*([A-Za-z0-9_.:-]+)\]")
#: ATM-Bench source ids: an email id or a media timestamp.
_SOURCE_ID = re.compile(
    r"\[(?:email|image|video):\s*([A-Za-z0-9_.:-]+)\]", re.IGNORECASE)
_ATM_ID = re.compile(r"\b(?:email\d+|\d{8}_\d{6})\b", re.IGNORECASE)
_CAPTION_LINE = re.compile(r"\[caption\]\s*(.*)", re.IGNORECASE)

#: Mem-Gallery visual answers are image ids (``D1:IMG_003``).  Asking Qwen3-8B
#: to emit the ids directly makes it enumerate every image in the evidence and
#: then loop; asking for the *evidence numbers* of the matching images and
#: mapping them back deterministically is stable and exact.
_IMAGE_ID_ANSWER = re.compile(
    r"\bimage[_\s-]?ids?\b", re.IGNORECASE)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_DATE = re.compile(
    r"\b(?:(early|mid|late)[-\s]?)?"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b(?:\s+(20\d{2}))?",
    re.IGNORECASE)


def _date_filters(question: str) -> list[Any]:
    """Predicates over an ISO timestamp for explicit dates in the question.

    Handles ``2024-07-07``, ``July 2024``, ``early-June`` and ``late-July``.
    """
    filters: list[Any] = []
    for match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", question or ""):
        target = match.group(0)

        def exact(time: str, target: str = target) -> bool:
            return time[:10] == target

        filters.append(exact)
    for match in _MONTH_DATE.finditer(question or ""):
        qualifier = (match.group(1) or "").lower()
        month = _MONTHS.get(match.group(2).lower())
        year = match.group(3)
        if not month:
            continue
        if year:
            prefix = f"{year}-{month:02d}"
            if qualifier in {"early", "mid", "late"}:
                days = {"early": (1, 10), "mid": (11, 20),
                        "late": (21, 31)}[qualifier]

                def month_range(time: str, prefix: str = prefix,
                                days: tuple[int, int] = days) -> bool:
                    if not time.startswith(prefix):
                        return False
                    day = int(time[8:10] or 0)
                    return days[0] <= day <= days[1]

                filters.append(month_range)
            else:
                def month_only(time: str, prefix: str = prefix) -> bool:
                    return time.startswith(prefix)

                filters.append(month_only)
        elif qualifier in {"early", "mid", "late"}:
            days = {"early": (1, 10), "mid": (11, 20),
                    "late": (21, 31)}[qualifier]

            def day_range(time: str, month: int = month,
                          days: tuple[int, int] = days) -> bool:
                return (time[5:7] == f"{month:02d}"
                        and days[0] <= int(time[8:10] or 0) <= days[1])

            filters.append(day_range)
        else:
            def any_year_month(time: str, month: int = month) -> bool:
                return time[5:7] == f"{month:02d}"

            filters.append(any_year_month)
    return filters
#: Contradiction questions that ask whether *two memory statements* are
#: incompatible, rather than whether an external claim conflicts with memory.
_BEFORE_AFTER = re.compile(r"\bbefore or after\b|\bafter or before\b",
                           re.IGNORECASE)
_EVER_QUESTION = re.compile(
    r"^\s*(?:has|have|did|does|was|were|is|are)\b[^.?!]*\bever\b",
    re.IGNORECASE)
_ROUND_RE = re.compile(r"\[round\s+D(\d+):(\d+)\]")
_EVENT_VERB = re.compile(
    r"^\s*(?:did|does|has|have|was|were|is|are)\b.*?\b"
    r"(?:discuss|mention|raise|ask|talk|say|state|express|show|buy|visit|"
    r"start|finish|learn|see)\b\s+(.*)$", re.IGNORECASE | re.DOTALL)


def _event_phrase(text: str) -> str:
    text = re.sub(r"^\s*(?:about|regarding)\s+", "", text or "",
                  flags=re.IGNORECASE)
    match = _EVENT_VERB.match(text)
    if match:
        text = match.group(1)
    return text.strip().strip("?").strip()

_INTERNAL_CONTRADICTION = re.compile(
    r"\b(?:"
    r"is there (?:a |any )?contradiction"
    r"|does .*? contradict (?:himself|herself|themselves)"
    r"|does the conversation contain (?:any )?(?:contradictory|conflicting) statements"
    r"|is there any instance where .*? contradictory"
    r"|contradicting earlier"
    r")", re.IGNORECASE)
#: An answer that is nothing but image id(s).  VR/TTL/KR questions expect text;
#: this shape means the model echoed an id instead of the requested value.
_IMAGE_ONLY = re.compile(
    r"^\s*D\d+:IMG_\d+(?:\s*[,;]\s*D\d+:IMG_\d+)*\s*\.?\s*$", re.IGNORECASE)
_QUERY_CAPTION = re.compile(
    r"The image in the question shows:\s*(.*)$", re.IGNORECASE | re.DOTALL)

_COUNT_WORDS = {
    "one": 1, "single": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3,
    "four": 4, "4": 4, "five": 5, "5": 5, "six": 6, "6": 6,
    "seven": 7, "7": 7, "eight": 8, "8": 8,
}


def requested_image_count(question: str) -> int | None:
    """Explicit image count in the question, else ``None``.

    ``Which two images ...`` -> 2.  ``Which image ...`` -> 1.  ``What images
    did ...`` -> ``None`` (the question asks for a set but not a size).
    """
    text = str(question or "").lower()
    match = re.search(
        r"\b(one|single|two|three|four|five|six|seven|eight|\d+)\b"
        r"(?:\s+\w+){0,4}\s+images?\b", text)
    if match:
        return _COUNT_WORDS.get(match.group(1))
    if re.search(r"\b(?:which|what)\s+(?:single\s+)?image\b", text):
        return 1
    return None


_RESOLVED_RE = re.compile(
    r"\[resolved:\s*(?P<expr>[^=\]]+?)\s*=\s*"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s*\]")


def resolved_candidates(evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Distinct ``[resolved: ...]`` records with their absolute date."""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in evidence:
        content = str(record.get("content") or "")
        match = _RESOLVED_RE.search(content)
        if not match:
            continue
        excerpt = re.sub(r"\s+", " ", content)[:260]
        key = (match.group("date"), excerpt[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append({"date": match.group("date"),
                    "expression": match.group("expr").strip(),
                    "text": excerpt})
    return out


_WEAK_TERMS = {
    "when", "what", "which", "who", "whom", "does", "did", "was", "were",
    "have", "has", "had", "the", "and", "for", "with", "from", "this",
    "that", "these", "those", "return", "format", "date", "yyyy", "please",
    "about", "into", "during", "after", "before", "them", "they", "their",
}


def _content_terms(text: str) -> set[str]:
    return {token for token in normalize_text(text).split()
            if len(token) > 3 and token not in _WEAK_TERMS}


@dataclass
class MediaContext:
    """Where the answerer finds the real images for a Mem-Gallery question."""

    image_root: str = ""
    query_image: str = ""
    max_images: int = 3
    max_edge: int = 512
    query_detail: str = "high"
    evidence_detail: str = "low"

    @classmethod
    def from_config(cls, value: dict[str, Any] | None) -> "MediaContext | None":
        if not isinstance(value, dict) or not value.get("image_root"):
            return None
        return cls(
            image_root=str(value.get("image_root") or ""),
            query_image=str(value.get("query_image") or ""),
            max_images=max(0, int(value.get("max_images", 3) or 0)),
            max_edge=max(128, int(value.get("max_edge", 512) or 512)),
            query_detail=str(value.get("query_detail") or "high"),
            evidence_detail=str(value.get("evidence_detail") or "low"),
        )


_IMAGE_URI_CACHE_DIR = Path(tempfile.gettempdir()) / "asmem_mg_images"
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def resolve_image_path(media: MediaContext, image_id: str) -> Path | None:
    """``D1:IMG_003`` -> ``<scenario>/D1_IMG_003.jpg`` (or .png/.webp)."""
    if not media.image_root or not image_id:
        return None
    root = Path(media.image_root)
    stem = str(image_id).replace(":", "_").replace("/", "_")
    for extension in _IMAGE_EXTENSIONS:
        candidate = root / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def image_data_uri(path: str | Path, *, max_edge: int = 512,
                   quality: int = 82) -> str:
    """A bounded inline image, resized once and cached on disk.

    The API token cost grows with resolution; a 512px long edge keeps a
    memory image cheap while still supporting visual matching.
    """
    source = Path(path)
    if not source.is_file():
        return ""
    _IMAGE_URI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        f"{source}|{max_edge}|{quality}".encode("utf-8")).hexdigest()[:20]
    cached = _IMAGE_URI_CACHE_DIR / f"{digest}.jpg"
    if not cached.exists():
        temp = cached.with_name(f".{cached.stem}.{os.getpid()}.tmp.jpg")
        try:
            subprocess.run(
                ["convert", str(source), "-auto-orient", "-resize",
                 f"{max_edge}x{max_edge}>", "-quality", str(quality), str(temp)],
                check=True, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(temp, cached)
        except (OSError, subprocess.SubprocessError):
            try:
                temp.unlink()
            except OSError:
                pass
            # Fall back to the original bytes when ImageMagick is unavailable.
            try:
                raw = source.read_bytes()
            except OSError:
                return ""
            mime = ("image/png" if source.suffix.lower() == ".png"
                    else "image/webp" if source.suffix.lower() == ".webp"
                    else "image/jpeg")
            return f"data:{mime};base64," + base64.b64encode(raw).decode()
    try:
        data = cached.read_bytes()
    except OSError:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _image_messages(text: str, images: list[tuple[str, str, str]],
                    *, max_edge: int = 512) -> list[dict[str, Any]]:
    """One user message with interleaved labelled images.

    ``images`` is ``[(label, path, detail)]``.  The label repeats the
    ``[image: id]`` marker from the evidence so the model can bind a picture
    to its record.
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for label, path, detail in images:
        uri = image_data_uri(path, max_edge=max_edge)
        if not uri:
            continue
        parts.append({"type": "text", "text": f"[attached image: {label}]"})
        parts.append({"type": "image_url",
                      "image_url": {"url": uri, "detail": detail}})
    return [{"role": "user", "content": parts}]


def _evidence_images(evidence: list[dict[str, Any]], media: MediaContext,
                     *, limit: int, detail: str) -> list[tuple[str, str, str]]:
    """Resolve memory images referenced by the evidence, in evidence order."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for record in evidence:
        for image_id in _IMAGE_MARKER.findall(str(record.get("content") or "")):
            image_id = image_id.strip()
            if image_id in seen:
                continue
            seen.add(image_id)
            path = resolve_image_path(media, image_id)
            if path is not None:
                out.append((image_id, str(path), detail))
                if len(out) >= limit:
                    return out
    return out


def _context_text(content: str) -> str:
    """The dialogue text around an image marker/caption, without markers."""
    text = re.sub(r"\[image:[^\]]*\]", " ", content or "")
    text = re.sub(r"\[caption\][^\n]*", " ", text)
    text = re.sub(r"\[round [^\]]*\]", " ", text)
    text = re.sub(r"\[context [^\]]*\]", " ", text)
    return " ".join(text.split())


def image_candidates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Image records in the evidence, merged across duplicate occurrences.

    One image can appear both as a bare caption record and inside the dialogue
    turn that uploaded it.  The bare record has the caption but not "this is my
    painting"; the turn has the context but may not repeat the caption.  Merge
    both, keeping the first evidence number for the id.
    """
    out: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(evidence, 1):
        content = str(record.get("content") or "")
        image_ids = list(dict.fromkeys(_IMAGE_MARKER.findall(content)))
        if not image_ids:
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        time = str(metadata.get("time") or metadata.get("observed_at") or "")
        source = ", ".join(str(item) for item in (metadata.get("source") or []))
        cap_match = _CAPTION_LINE.search(content)
        caption = ""
        if cap_match:
            caption = re.split(
                r"\s*\[(?:round|context|image)\b", cap_match.group(1))[0].strip()
        context = _context_text(content)
        round_match = _ROUND_RE.search(content)
        session = f"D{round_match.group(1)}" if round_match else ""
        for image_id in image_ids:
            entry = out.setdefault(image_id, {
                "evidence": index, "id": image_id, "caption": "",
                "context": "", "time": time, "source": source,
                "session": session,
            })
            if not entry["session"] and session:
                entry["session"] = session
            if len(caption) > len(entry["caption"]):
                entry["caption"] = caption
            if len(context) > len(entry["context"]):
                entry["context"] = context
            if not entry["time"] and time:
                entry["time"] = time
    return list(out.values())


def _quote_grounded(quote: str, evidence: list[dict[str, Any]]) -> bool:
    """True when a quoted sentence really occurs in the evidence."""
    quote_norm = normalize_text(quote)
    if not quote_norm:
        return False
    haystack = normalize_text(" ".join(
        str(record.get("content") or "") for record in evidence))
    if quote_norm in haystack:
        return True
    quote_tokens = quote_norm.split()
    if len(quote_tokens) < 3:
        return False
    hay_tokens = set(haystack.split())
    hits = sum(1 for token in quote_tokens if token in hay_tokens)
    return hits / len(quote_tokens) >= 0.8


def _parse_evidence_numbers(text: str) -> list[int]:
    """``[1, 3]`` / ``{"evidence": [1, 3]}`` / ``1, 3`` -> ``[1, 3]``."""
    value = str(text or "")
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    numbers: list[Any] = []
    if isinstance(parsed, dict):
        for key in ("evidence", "records", "indices", "index", "images"):
            if isinstance(parsed.get(key), list):
                numbers = parsed[key]
                break
    elif isinstance(parsed, list):
        numbers = parsed
    if not numbers:
        match = re.search(r"\[[^\]]*\]", value)
        numbers = re.findall(r"\d+", match.group(0) if match else value)
    out: list[int] = []
    for item in numbers:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return out


def clean_answer(text: str) -> str:
    value = str(text or "").strip()
    stripped = _ANSWER_PREFIX.sub("", value).strip()
    return stripped or value


def _letters(text: str) -> bool:
    return bool(parse_letters(text or ""))


class TextModel:
    """Small shared wrapper around the OpenAI async client (text completions)."""

    def __init__(self, config: dict[str, Any], *, role: str = "text") -> None:
        self.config = llm.merge_call_config(config)
        self.client = llm.make_client(config)
        self.calls = 0
        self.role = role
        self.usage = llm.UsageTracker()

    async def text(self, messages: list[dict[str, Any]],
                   *, deadline: float | None = None) -> str:
        self.calls += 1
        return await llm.chat_completion(
            self.client, self.config, messages,
            usage=llm.CURRENT_USAGE.get() or self.usage, role=self.role,
            deadline=deadline)

    async def aclose(self) -> None:
        await self.client.close()


def format_evidence(evidence: list[dict[str, Any]], max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for index, record in enumerate(evidence, 1):
        content = str(record.get("content") or "").strip()
        if not content:
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        sources = metadata.get("source") or []
        source_text = ", ".join(str(item) for item in sources) if sources else "unknown"
        time_text = str(metadata.get("time") or metadata.get("observed_at") or "").strip()
        speaker = str(metadata.get("speaker") or "").strip()
        header_parts = [source_text]
        if speaker:
            header_parts.append(f"speaker={speaker}")
        if time_text:
            header_parts.append(f"observed_at={time_text}")
        header = "(" + "; ".join(header_parts) + ")"
        line = f"[{index}] {header} {content}"
        if used + len(line) + 1 > max_chars and lines:
            lines.append(f"... ({len(evidence) - index + 1} more evidence records omitted)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "(no evidence records)"


def format_options(options: Any) -> str:
    if not options:
        return ""
    items = [str(item).strip() for item in options if str(item).strip()]
    if not items:
        return ""
    return "\n\nOptions:\n" + "\n".join(items)


class Answerer:
    def __init__(self, config: dict[str, Any], model: TextModel | None,
                 *, mock: bool = False) -> None:
        self.config = config
        self.model = model
        self.mock = mock

    async def answer(
        self,
        *,
        dataset: str,
        question: str,
        evidence: list[dict[str, Any]],
        gold: dict[str, Any],
        media: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.mock:
            text = gold.get("answer")
            if not text and gold.get("answer_letters"):
                text = "(" + ", ".join(gold["answer_letters"]) + ")"
            return {"answer": text or "MOCK ANSWER", "error": None, "mock": True}

        total_timeout = float(self.config.get("total_timeout", 0) or 0)
        deadline = time.monotonic() + total_timeout if total_timeout > 0 else None
        media_ctx = MediaContext.from_config(media)
        if dataset == "atm_bench" and str(
                gold.get("answer_type") or "").lower() == "list_recall":
            ids = await self._answer_evidence_ids(
                question, evidence, media=media_ctx, deadline=deadline)
            if ids:
                return {"answer": ids, "error": None,
                        "evidence_id_select": True}
        if dataset == "mem_gallery":
            point = str(gold.get("answer_type") or "").upper()
            if point == "TR" and _BEFORE_AFTER.search(question or ""):
                order = await self._answer_order(
                    question, evidence, deadline=deadline)
                if order:
                    return {"answer": order, "error": None,
                            "order_select": True}
            elif (point == "TR"
                    and re.search(r"\bwhen\b|\bwhat date\b|\bwhich day\b",
                                  question or "", re.IGNORECASE)):
                resolved = await self._answer_temporal(
                    question, evidence, deadline=deadline)
                if resolved:
                    return {"answer": resolved, "error": None,
                            "resolved_date": True}
            if (point in {"VR", "KR", "TTL"}
                    and re.match(r"\s*how\s+many\b", question or "",
                                 re.IGNORECASE)):
                counted = await self._answer_count(
                    question, evidence, deadline=deadline)
                if counted:
                    return {"answer": counted, "error": None,
                            "count_select": True}
            if point == "AR" and bool(self.config.get("ar_quote_check")):
                abstained = await self._answer_abstain(
                    question, evidence, deadline=deadline)
                return {"answer": abstained, "error": None,
                        "abstain_select": True}
            if point == "CD":
                if _EVER_QUESTION.match(question or ""):
                    exists = await self._answer_existence(
                        question, evidence, deadline=deadline)
                    if exists:
                        return {"answer": exists, "error": None,
                                "existence_select": True}
                verdict = await self._answer_conflict(
                    question, evidence, deadline=deadline)
                if verdict:
                    return {"answer": verdict, "error": None,
                            "conflict_select": True}
            elif (point == "VS" or _IMAGE_ID_ANSWER.search(question or "")):
                selected = await self._answer_image_ids(
                    question, evidence, media=media_ctx, deadline=deadline)
                if selected:
                    return {"answer": selected, "error": None,
                            "image_select": True}

        style = prompts.answer_style(dataset, self.config.get("styles"))
        type_styles = self.config.get("type_styles") or {}
        answer_type = str(gold.get("answer_type") or "").strip()
        if isinstance(type_styles, dict):
            candidates = [f"{dataset}/{answer_type}"]
            # BEAM variants share answer contracts.  Configs historically used
            # beam_100k/* keys; without this fallback beam_500k silently lost
            # the abstention and instruction-following guards.
            if dataset.startswith("beam_"):
                candidates.append(f"beam_100k/{answer_type}")
            candidates.append(answer_type)
            for key in candidates:
                extra = type_styles.get(key)
                if isinstance(extra, str) and extra.strip():
                    style = (style + "\n\n" + extra).strip()
                    break
        systems = self.config.get("system_by_dataset") or {}
        system = (systems.get(dataset) if isinstance(systems, dict) else None)
        system = system or self.config.get("system") or prompts.ANSWER_SYSTEM
        user_template = str(self.config.get("user") or prompts.ANSWER_USER)

        # Two independent guards:
        # * ``total_timeout`` bounds the whole answer stage (including retries),
        #   so one slow endpoint call cannot hold a QA open indefinitely.
        # * ``max_context_tokens`` preflight-shrinks the prompt; if the endpoint
        #   still reports a context overflow, halve the evidence and retry.
        input_budget = int(self.config.get("max_context_tokens", 0) or 0)
        reserve = int(getattr(self.model, "config", {}).get("max_tokens", 4096)) if self.model else 4096
        evidence_limit = int(self.config.get("max_evidence_chars", 16000))

        if self.model is None:
            return {"answer": "", "error": "answer model is not configured"}
        last_error: str | None = None
        for attempt in range(3):
            evidence_text = format_evidence(evidence, evidence_limit)
            user = user_template
            for key, value in {
                "{question}": question,
                "{evidence}": evidence_text,
                "{options}": format_options(gold.get("options")),
                "{style}": style,
                "{dataset}": dataset,
            }.items():
                user = user.replace(key, value)
            user_content: Any = user
            if media_ctx:
                images: list[tuple[str, str, str]] = []
                if media_ctx.query_image:
                    images.append(("query image", media_ctx.query_image,
                                   media_ctx.query_detail))
                remaining = max(0, media_ctx.max_images - len(images))
                images.extend(_evidence_images(
                    evidence, media_ctx, limit=remaining,
                    detail=media_ctx.evidence_detail))
                if images:
                    user_content = _image_messages(
                        user, images, max_edge=media_ctx.max_edge)[0]["content"]
            messages = [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": user_content},
            ]
            if input_budget > 0:
                messages = llm.fit_messages(
                    messages, input_budget, reserve_tokens=reserve)
            try:
                text = await self.model.text(messages, deadline=deadline)
                # MCQ robustness: a small model occasionally returns "" or "()".
                # One strict retry is cheap and cannot change a valid answer.
                if gold.get("options") and not _letters(text):
                    retry_messages = [
                        *messages,
                        {"role": "assistant", "content": text or "()"},
                        {"role": "user", "content":
                         "Return ONLY the option letter(s) in parentheses, e.g. (B) or "
                         "(A, C). Do not explain."},
                    ]
                    if input_budget > 0:
                        retry_messages = llm.fit_messages(
                            retry_messages, input_budget, reserve_tokens=reserve)
                    retry = await self.model.text(retry_messages, deadline=deadline)
                    if _letters(retry):
                        text = retry
                answer = clean_answer(text)
                point = str(gold.get("answer_type") or "").upper()
                if (dataset == "mem_gallery" and point in {"VR", "TTL", "KR"}
                        and _IMAGE_ONLY.match(answer or "")):
                    better = await self._answer_visual_text(
                        question, evidence, media=media_ctx, deadline=deadline)
                    if better and not _IMAGE_ONLY.match(better):
                        answer = better
                return {"answer": answer, "error": None}
            except llm.TruncatedCompletion as exc:
                # A long rubric answer that hit max_tokens is still usable;
                # retrying the same prompt would truncate again and cost a lot
                # of wall clock.  Keep the partial content.
                partial = clean_answer(getattr(exc, "content", "") or "")
                if partial:
                    return {"answer": partial, "error": None, "truncated": True}
                return {"answer": "", "error": "truncated completion"}
            except llm.ContextOverflow as exc:
                last_error = f"context overflow: {exc}"
                if evidence_limit <= 1000:
                    break
                evidence_limit = max(1000, evidence_limit // 2)
                continue
            except Exception as exc:  # noqa: BLE001 - a failed answer must not kill the run
                return {"answer": "", "error": f"{type(exc).__name__}: {exc}"}
        return {"answer": "", "error": last_error or "context window exceeded"}

    async def _answer_image_ids(self, question: str,
                                evidence: list[dict[str, Any]], *,
                                media: MediaContext | None = None,
                                deadline: float | None = None) -> str:
        """Pick image records by evidence number, then emit their ids.

        Qwen3-8B reliably selects the matching caption but, when asked to
        write image ids directly, keeps enumerating every id in the evidence
        until ``max_tokens``.  Selecting the 1-based evidence numbers and
        mapping them here is deterministic and cannot loop.
        """
        if self.model is None:
            return ""
        candidates = image_candidates(evidence)
        if not candidates:
            return ""
        # "Which image uploaded in 2024-07-07 ..." scopes the answer to that
        # session.  Enforce the date deterministically; otherwise a caption
        # that merely mentions a robotic arm in another session is selected.
        date_filters = _date_filters(question)
        if date_filters:
            scoped = [item for item in candidates
                      if any(check(item["time"]) for check in date_filters)]
            if scoped:
                candidates = scoped
        requested = requested_image_count(question)
        has_query_image = "the image in the question shows" in (question or "").lower()
        # A genuine set question ("what images ... are", "which pictures ...
        # show") expects several ids; "which picture ... is" expects one.
        plural_expected = bool(re.search(
            r"\b(?:images|pictures|photos)\b[^?.]{0,40}?"
            r"\b(?:are|were|did|show|shows|include|shared|uploaded|focus|"
            r"represent)\b", question or "", re.IGNORECASE))
        if (not date_filters and not has_query_image and requested is None
                and plural_expected):
            # "What images did Julian upload when talking about this topic?"
            # Answer images usually come from one session; pick the session
            # that contributes the most candidates (the top-ranked image alone
            # is often a distractor), so the selector is not flooded with
            # loosely related images from every session.
            by_session: dict[str, list[dict[str, Any]]] = {}
            for item in candidates:
                session = str(item.get("session") or "")
                if session:
                    by_session.setdefault(session, []).append(item)
            if len(by_session) > 1:
                session, group = max(
                    by_session.items(),
                    key=lambda pair: (len(pair[1]), -pair[1][0]["evidence"]))
                if len(group) >= 2:
                    candidates = group
        lines = []
        for item in candidates:
            date = f" date={item['time'][:10]}" if item["time"] else ""
            caption = item["caption"][:500].replace("\n", " ")
            context = item["context"][:220].replace("\n", " ")
            line = f"[{item['evidence']}] {item['id']}{date} caption: {caption}"
            if context:
                line += f" | dialogue: {context}"
            lines.append(line)
        if requested:
            count_rule = (
                f"The question asks for {requested} image(s); return exactly "
                f"{requested} number(s).")
        elif plural_expected:
            count_rule = (
                "The question asks for a set of images; return every image that "
                "clearly satisfies it (usually one to four), and do not add "
                "loosely related images.")
        else:
            count_rule = (
                "The question asks for one image; prefer the single best match.")
        system = (
            "You select images from a person's memory. Reply with one JSON "
            "object only, no prose.")
        user = (
            f"Question:\n{question}\n\n"
            "Image memories (number, id, date, caption):\n" + "\n".join(lines) +
            "\n\nReturn the number(s) of the image(s) that answer the question. "
            "Use only the numbers listed above. " + count_rule +
            ' Reply as {"evidence": [<numbers>]}.')
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if media is not None:
            images: list[tuple[str, str, str]] = []
            if media.query_image:
                images.append(("query image", media.query_image,
                               media.query_detail))
            for item in candidates:
                if len(images) >= max(1, media.max_images):
                    break
                path = resolve_image_path(media, str(item["id"]))
                if path is not None:
                    images.append((str(item["id"]), str(path),
                                   media.evidence_detail))
            if images:
                messages[1] = _image_messages(
                    user, images, max_edge=media.max_edge)[0]
        try:
            reply = await self.model.text(messages, deadline=deadline)
        except Exception:  # noqa: BLE001 - fall back to the normal answer path
            return ""
        numbers = _parse_evidence_numbers(reply)
        by_number = {item["evidence"]: item["id"] for item in candidates}
        ids = [by_number[number] for number in numbers if number in by_number]
        ids = list(dict.fromkeys(ids))
        if requested and len(ids) > requested:
            ids = ids[:requested]
        return ", ".join(ids)

    async def _answer_visual_text(self, question: str,
                                  evidence: list[dict[str, Any]], *,
                                  media: MediaContext | None = None,
                                  deadline: float | None = None) -> str:
        """Re-answer a visual text question that came back as an image id.

        "Which landmark is in this picture?" wants a name, not
        ``D2:IMG_002``.  The query caption is stated explicitly in the
        question; give it to the model and forbid an id answer.
        """
        if self.model is None:
            return ""
        match = _QUERY_CAPTION.search(question or "")
        caption = match.group(1).strip() if match else ""
        if not caption:
            return ""
        bare_question = _QUERY_CAPTION.sub("", question or "").strip()
        system = ("You answer questions about a person's dialogue memory. "
                  "Never answer with an image id unless the question asks for "
                  "one.")
        user = (
            f"Question:\n{bare_question}\n\n"
            f"The attached image is described as:\n{caption}\n\n"
            f"Dialogue evidence:\n{format_evidence(evidence, 14000)}\n\n"
            "Answer with the requested value (a name, place, date, count, or "
            "short description), not an image id. Return only the answer.")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if media is not None:
            images: list[tuple[str, str, str]] = []
            if media.query_image:
                images.append(("query image", media.query_image,
                               media.query_detail))
            remaining = max(0, media.max_images - len(images))
            images.extend(_evidence_images(
                evidence, media, limit=remaining,
                detail=media.evidence_detail))
            if images:
                messages[1] = _image_messages(
                    user, images, max_edge=media.max_edge)[0]
        try:
            reply = await self.model.text(messages, deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        return clean_answer(reply)

    async def _answer_order(self, question: str,
                            evidence: list[dict[str, Any]], *,
                            deadline: float | None = None) -> str:
        """Before/after questions, decided from the cited rounds.

        The model finds the two event records; the order comes from their
        ``[round Dx:y]`` markers (or observation time), so it cannot flip the
        comparison the way reading the answer in prose did.
        """
        if self.model is None:
            return ""
        system = ("You locate two events in memory. Reply with one JSON "
                  "object only.")
        events = ""
        parts = re.split(r"\bbefore or after\b", question or "",
                         maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            first_event = _event_phrase(parts[0])
            second_event = _event_phrase(parts[1])
            if first_event and second_event:
                events = (f"Event A: {first_event}\n"
                          f"Event B: {second_event}\n\n")
        user = (
            f"Question:\n{question}\n\n{events}"
            f"Memory evidence:\n{format_evidence(evidence, 12000)}\n\n"
            "Identify the record that states event A and the record that "
            "states event B. Return "
            '{"first": <A record number>, "second": <B record number>}.')
        try:
            reply = await self.model.text(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        value = parse_json_object(reply) or {}

        def position(number: Any) -> tuple[int, int, str] | None:
            try:
                index = int(number)
            except (TypeError, ValueError):
                return None
            if not 1 <= index <= len(evidence):
                return None
            record = evidence[index - 1]
            content = str(record.get("content") or "")
            match = _ROUND_RE.search(content)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            time = str(metadata.get("time") or metadata.get("observed_at") or "")
            if match:
                return (int(match.group(1)), int(match.group(2)), time)
            return (0, 0, time) if time else None

        first = position(value.get("first"))
        second = position(value.get("second"))
        if not first or not second:
            return ""
        if first[:2] and second[:2]:
            return "After." if first[:2] > second[:2] else "Before."
        if first[2] and second[2] and first[2] != second[2]:
            return "After." if first[2] > second[2] else "Before."
        return ""

    async def _answer_existence(self, question: str,
                                evidence: list[dict[str, Any]], *,
                                deadline: float | None = None) -> str:
        """"Has X ever ...?" — yes only when the memory states it."""
        if self.model is None:
            return ""
        system = ("You check whether a memory states an event. Reply with one "
                  "JSON object only.")
        user = (
            f"Question:\n{question}\n\n"
            f"Memory evidence:\n{format_evidence(evidence, 12000)}\n\n"
            "Does the memory explicitly state that this happened? A memory that "
            "says the opposite counts as not stated. Return "
            '{"stated": true or false, "quote": "<short quote>"}.')
        try:
            reply = await self.model.text(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        value = parse_json_object(reply) or {}
        stated = value.get("stated")
        if isinstance(stated, str):
            stated = stated.strip().lower() == "true"
        if stated is None:
            return ""
        return "Yes." if stated else "No."

    async def _answer_temporal(self, question: str,
                               evidence: list[dict[str, Any]], *,
                               deadline: float | None = None) -> str:
        """Resolve "When did ...?" through the ``[resolved]`` annotations.

        A record such as "trip to Paris tomorrow" carries
        ``[resolved: tomorrow = 2024-05-19]``; the session date (2024-05-18)
        is not the event date.  Showing the model only the resolved
        candidates turns a 50-record reading task into a small selection.
        """
        if self.model is None:
            return ""
        candidates = resolved_candidates(evidence)
        # Keep only records that mention a content term of the question; a
        # resolved date in an unrelated record must not answer "When ...?".
        question_terms = _content_terms(question)
        candidates = [
            item for item in candidates
            if question_terms & _content_terms(item["text"])
        ]
        if not candidates:
            return ""
        dates = sorted({item["date"] for item in candidates})
        if len(dates) == 1:
            return dates[0]
        # Keep the prompt bounded; candidates are already in evidence order.
        candidates = candidates[:24]
        lines = [
            f"[{index}] date={item['date']} (from '{item['expression']}') "
            f"excerpt: {item['text']}"
            for index, item in enumerate(candidates, 1)
        ]
        system = ("You extract event dates from memory. Reply with one JSON "
                  "object only.")
        user = (
            f"Question:\n{question}\n\n"
            "Memories whose relative date was resolved to an absolute date:\n" +
            "\n".join(lines) +
            "\n\nThe session date is when the conversation happened, not "
            "necessarily when the event happened. Pick the record that states "
            "when the asked event happened. Return "
            '{"record": <number>, "date": "YYYY-MM-DD"}.')
        try:
            reply = await self.model.text(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        value = parse_json_object(reply) or {}
        record = value.get("record")
        try:
            index = int(record)
        except (TypeError, ValueError):
            index = 0
        if 1 <= index <= len(candidates):
            return candidates[index - 1]["date"]
        date = str(value.get("date") or "").strip()
        return date if date in dates else ""

    async def _answer_evidence_ids(self, question: str,
                                   evidence: list[dict[str, Any]], *,
                                   media: MediaContext | None = None,
                                   deadline: float | None = None) -> str:
        """ATM-Bench list_recall: return the source ids of the matching records.

        The gold answers are evidence ids (``20240222_092257`` /
        ``email202411160004``), not descriptions.  Ask the model to select
        records from a compact numbered list and map them back to the ids that
        actually occur in the evidence.
        """
        if self.model is None:
            return ""
        records: list[dict[str, Any]] = []
        for index, record in enumerate(evidence, 1):
            content = str(record.get("content") or "")
            ids = list(dict.fromkeys(_SOURCE_ID.findall(content)))
            if not ids:
                continue
            preview = " ".join(
                re.sub(r"\[[^\]]*\]", " ", content).split())[:300]
            records.append({"evidence": index, "ids": ids,
                            "preview": preview, "content": content})
        if not records:
            return ""
        lines = [
            f"[{item['evidence']}] ids={', '.join(item['ids'])} | "
            f"{item['preview']}"
            for item in records[:40]
        ]
        system = ("You select the memory records that answer a question. "
                  "Reply with the source ids only, comma-separated.")
        user = (
            f"Question:\n{question}\n\n"
            "Memory records (number, source ids, preview):\n" +
            "\n".join(lines) +
            "\n\nReturn the source id(s) of every record that answers the "
            "question, exactly as written, comma-separated. Do not explain.")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if media is not None:
            images: list[tuple[str, str, str]] = []
            for item in records:
                if len(images) >= max(1, media.max_images):
                    break
                for image_id in item["ids"]:
                    path = resolve_image_path(media, image_id)
                    if path is not None:
                        images.append((image_id, str(path),
                                       media.evidence_detail))
                        break
            if images:
                messages[1] = _image_messages(
                    user, images, max_edge=media.max_edge)[0]
        try:
            reply = await self.model.text(messages, deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        valid = {image_id for item in records for image_id in item["ids"]}
        found = list(dict.fromkeys(_ATM_ID.findall(reply)))
        ids = [image_id for image_id in found if image_id in valid]
        return ", ".join(ids)

    async def _answer_count(self, question: str,
                            evidence: list[dict[str, Any]], *,
                            deadline: float | None = None) -> str:
        """Count image questions over a compact caption list.

        "How many foxglove images ...?" fails when the model has to count
        across 40+ evidence records.  Showing only the image ids, dates,
        captions and their dialogue, and asking for one integer, is much more
        reliable.
        """
        if self.model is None:
            return ""
        candidates = image_candidates(evidence)
        if not candidates:
            return ""
        date_filters = _date_filters(question)
        if date_filters:
            scoped = [item for item in candidates
                      if any(check(item["time"]) for check in date_filters)]
            if scoped:
                candidates = scoped
        lines = []
        for item in candidates[:40]:
            date = f" {item['time'][:10]}" if item["time"] else ""
            caption = item["caption"][:220].replace("\n", " ")
            context = item["context"][:160].replace("\n", " ")
            line = f"- {item['id']}{date}: {caption}"
            if context:
                line += f" | dialogue: {context}"
            lines.append(line)
        system = ("You count images from their captions and dialogue. "
                  "Reply with one JSON object only.")
        user = (
            f"Question:\n{question}\n\n"
            "Image memories:\n" + "\n".join(lines) +
            "\n\nCount the images that satisfy the question. Return "
            '{"count": <integer>}.')
        try:
            reply = await self.model.text(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        value = parse_json_object(reply) or {}
        count = value.get("count")
        try:
            number = int(count)
        except (TypeError, ValueError):
            match = re.search(r'"count"\s*:\s*(\d+)', reply)
            number = int(match.group(1)) if match else 0
        return str(number) if number > 0 else ""

    async def _answer_abstain(self, question: str,
                              evidence: list[dict[str, Any]], *,
                              deadline: float | None = None) -> str:
        """AR questions: answer only when a record explicitly states the fact.

        4o-mini otherwise treats a related fact as the answer (a Cairn Terrier
        for "Amy's cat", a related quote for "did X ever ...").  The quote is
        verified against the evidence before it is trusted.
        """
        if self.model is None:
            return "Not mentioned."
        system = ("You decide whether a memory explicitly answers a question. "
                  "Reply with one JSON object only.")
        user = (
            f"Question:\n{question}\n\n"
            f"Memory evidence:\n{format_evidence(evidence, 12000)}\n\n"
            "A record answers the question only if it explicitly states the "
            "asked fact. A related topic, a similar name, or a plausible "
            "inference is NOT an answer. Return "
            '{"answerable": true or false, "quote": "<exact sentence, or '
            'empty>", "answer": "<short answer, or empty>"}.')
        try:
            reply = await self.model.text(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                deadline=deadline)
        except Exception:  # noqa: BLE001
            return "Not mentioned."
        value = parse_json_object(reply) or {}
        answerable = value.get("answerable")
        if isinstance(answerable, str):
            answerable = answerable.strip().lower() == "true"
        quote = str(value.get("quote") or "").strip()
        answer = str(value.get("answer") or "").strip()
        if not answerable or not quote or not _quote_grounded(quote, evidence):
            return "Not mentioned."
        return answer or "Not mentioned."

    async def _answer_conflict(self, question: str,
                               evidence: list[dict[str, Any]], *,
                               deadline: float | None = None) -> str:
        """Contradiction questions: compare the claim with the memory.

        Two shapes need different reasoning:

        * internal consistency ("Is there a contradiction between X and Y?") —
          both statements can be true, so a coarse "conflict?" question is fine;
        * claim vs. memory ("... conflicts with the dialogue memory: X") — the
          model otherwise says "the memory does not mention X" and misses that
          negating an affirmed fact *is* the conflict.  Ask for an
          affirm/deny/not_mentioned verdict instead.
        """
        if self.model is None:
            return ""
        system = ("You compare a claim with a person's dialogue memory. "
                  "Reply with one JSON object only.")
        if _INTERNAL_CONTRADICTION.search(question or ""):
            user = (
                f"Memory evidence:\n{format_evidence(evidence, 12000)}\n\n"
                f"Question:\n{question}\n\n"
                "Decide whether the two statements are genuinely incompatible. "
                "Two statements that can both be true do not conflict. Return "
                '{"conflict": true or false, "memory": "<short quote>"}.')
        else:
            user = (
                f"Memory evidence:\n{format_evidence(evidence, 12000)}\n\n"
                f"Claim/question:\n{question}\n\n"
                "Does the memory affirm the claim (states the same thing), deny "
                "the claim (states the opposite), or not mention it? A memory "
                "that says 'solid golden-brown' denies a claim of a "
                "'white-and-liver' coat. Return "
                '{"verdict": "affirm" or "deny" or "not_mentioned", '
                '"quote": "<short quote>"}.')
        try:
            reply = await self.model.text(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                deadline=deadline)
        except Exception:  # noqa: BLE001
            return ""
        value = parse_json_object(reply) or {}
        verdict = str(value.get("verdict") or "").strip().lower()
        if verdict:
            return "Yes." if verdict == "deny" else "No."
        flag: Any = value.get("conflict")
        if flag is None:
            match = re.search(r'"conflict"\s*:\s*(true|false)', reply,
                              re.IGNORECASE)
            flag = (match.group(1).lower() == "true") if match else None
        if isinstance(flag, str):
            flag = flag.strip().lower() == "true"
        if flag is None:
            return ""
        return "Yes." if flag else "No."

