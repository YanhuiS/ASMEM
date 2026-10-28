"""One independent check: can the evidence answer the question?

The model verdict is only half of the audit.  ``apply_checks`` adds
deterministic rules that a small model gets wrong: fabricated claim sources,
policy leaks, and "current value" questions that cite only a superseded record.
A failed check can only downgrade ``sufficient`` to false; it never turns an
insufficient verdict into a passing one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from .. import llm
from ..retrieval.tokenize import extract_phrases, normalize_text, tokenize
from ..schema import normalize_time
from ..timeexpr import resolve_relative_dates
from . import prompts

Call = Callable[[list[dict[str, Any]]], Awaitable[str]]
_OBJECT = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)

#: Answer-shape guards.  A sufficient verdict for "when" must contain some
#: date expression; "how many" must contain a number.  These catch the most
#: common over-acceptance: the evidence is about the right topic but never
#: states the requested slot.
_QUESTION_DATE = re.compile(
    r"\b(when|what date|what day|which date|deadline|due date|since when)\b",
    re.IGNORECASE)
_MONTH = (
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_EXPR = re.compile(
    rf"\b(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|"
    rf"(?:{_MONTH})\s+\d{{1,2}}(?:,\s*\d{{4}})?|"
    rf"\d{{1,2}}\s+(?:{_MONTH})(?:\s+\d{{4}})?|(?:19|20)\d{{2}})\b",
    re.IGNORECASE)
_DATE_TOKEN = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|"
    r"nov|dec)\b|\b(?:yesterday|today|tomorrow|last week|this week|next week|"
    r"last month|this month|next month|last year|this year|next year)\b",
    re.IGNORECASE)
_QUESTION_NUMBER = re.compile(
    r"\b(how many|how much|total|count|number of)\b", re.IGNORECASE)
_NUMBER_TOKEN = re.compile(
    r"\b\d+(?:[.,]\d+)*\b|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|hundred|thousand|million|billion|half|twice|once)\b",
    re.IGNORECASE)
_MISSING_DATE = re.compile(r"\b(date|when|time|year|day|month)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
_STOP = frozenset({
    "the", "and", "for", "with", "that", "this", "was", "were", "has", "have",
    "had", "did", "does", "from", "what", "when", "where", "which", "who",
    "why", "how", "into", "about", "their", "there", "they", "them", "she",
    "her", "his", "him", "its", "you", "your", "our", "are", "not", "but",
})
#: Intents where a passing verdict must be backed by at least one grounded
#: claim.  Open-ended families (summary/list/preference/explain) and
#: self-contained/absence tasks are exempt.
_CLAIM_REQUIRED_INTENTS = {
    "fact", "yesno", "update", "conflict", "compare", "temporal",
    "duration", "order", "multi_hop", "aggregate",
}
_AGGREGATE_WORDS = re.compile(
    r"\b(total|sum|average|mean|count|combined|altogether|in all)\b", re.IGNORECASE)
_RELATIVE_WORDS = re.compile(
    r"\b(yesterday|today|tomorrow|last|this|next|ago)\b", re.IGNORECASE)


def _date_values(text: str) -> set[str]:
    """Normalize date-like spans to ``YYYY-MM-DD`` (years stay ``YYYY``)."""
    values: set[str] = set()
    for match in _DATE_EXPR.finditer(text or ""):
        raw = match.group(0).strip()
        if not re.search(r"\d{4}", raw):
            continue
        iso = normalize_time(raw)
        if iso:
            values.add(iso[:10])
        elif re.fullmatch(r"(19|20)\d{2}", raw):
            values.add(raw)
    return values


def _claim_grounded(
    claim: str,
    source_text: str,
    *,
    threshold: float = 0.35,
    aggregate: bool = False,
) -> bool:
    """Is a claim supported by the text of the records it cites?

    Lexical containment is a conservative proxy: a synthesis can paraphrase,
    but it cannot introduce a number or date that no cited record contains.
    Aggregate claims are allowed to compute new numbers from their sources.
    """
    claim = (claim or "").strip()
    if not claim:
        return False
    body = normalize_text(claim)
    source = normalize_text(source_text)
    if not source:
        return False
    claim_tokens = set(tokenize(claim))
    source_tokens = set(tokenize(source_text))
    claim_dates = _date_values(claim)
    source_dates = _date_values(source_text)
    dates_supported = bool(claim_dates) and claim_dates <= source_dates
    if claim_dates and not dates_supported and not aggregate:
        return False
    if dates_supported and (claim_tokens & source_tokens):
        # A supported date plus a shared subject token is enough: the claim is
        # the event's anchored statement, even if the wording is paraphrased.
        return True
    if claim_tokens:
        overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        if overlap < threshold:
            return False
    if _RELATIVE_WORDS.search(claim):
        # A relative date may resolve to a calendar value absent from the
        # source text; the verifier prompt carries the resolved annotation.
        return True
    numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", claim)
    if numbers and not aggregate:
        for value in numbers:
            if value.lower() not in source:
                return False
    return True


def _shares_subject(claim: str, question: str) -> bool:
    left = {word.lower() for word in _WORD.findall(claim)} - _STOP
    right = {word.lower() for word in _WORD.findall(question)} - _STOP
    return len(left & right) >= 1


def _index_sources(source: str, evidence: Sequence[dict[str, Any]]) -> list[str]:
    """Resolve a verifier citation like ``[1]`` / ``"1"`` to record ids.

    Small models often cite the evidence index instead of the record id that
    the prompt shows.  Treating that as a fabricated source would turn a
    correct verdict into a false negative.
    """
    match = re.fullmatch(r"\s*\[?#?(\d{1,3})\]?\s*", str(source))
    if match:
        position = int(match.group(1)) - 1
        if 0 <= position < len(evidence):
            return sorted(_sources(evidence[position]))
    return [str(source)]


@dataclass
class Verdict:
    sufficient: bool = False
    missing: list[str] = field(default_factory=list)
    reasoning: str = ""
    parsed: bool = True
    claims: list[dict[str, Any]] = field(default_factory=list)
    policy_violation: bool = False
    freshness_conflict: bool = False
    checks: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def _object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    for candidate in [stripped, *[m.group(0) for m in _OBJECT.finditer(text or "")]]:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def parse(text: str) -> Verdict:
    value = _object(text)
    if value is None:
        return Verdict(sufficient=False, parsed=False, raw=text or "")
    claims = []
    for item in value.get("claims") or []:
        if not isinstance(item, dict):
            continue
        sources = item.get("sources") or item.get("source") or []
        if isinstance(sources, str):
            sources = [sources]
        claims.append({
            "claim": str(item.get("claim") or "").strip(),
            "sources": [str(source) for source in sources if str(source).strip()],
            "kind": str(item.get("kind") or ""),
        })
    return Verdict(
        sufficient=bool(value.get("sufficient")),
        missing=[str(item) for item in value.get("missing") or [] if str(item).strip()],
        reasoning=str(value.get("reasoning") or ""),
        claims=claims,
        policy_violation=bool(value.get("policy_violation")),
        freshness_conflict=bool(value.get("freshness_conflict")),
        raw=text or "",
    )


def _sources(item: dict[str, Any]) -> set[str]:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    raw = metadata.get("source") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(value) for value in raw if str(value).strip()}


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("meta")
    return value if isinstance(value, dict) else {}


def apply_checks(
    verdict: Verdict,
    evidence: Sequence[dict[str, Any]],
    *,
    memory_records: Sequence[dict[str, Any]] | None = None,
    forbidden: Sequence[str] | None = None,
    intent: str = "",
    question: str = "",
    strict: bool = True,
    require_claims: bool = True,
    grounding_threshold: float = 0.35,
    min_key_coverage: float = 0.6,
) -> Verdict:
    """Deterministic rules applied after the model verdict.

    Downgrade rules run only on a ``sufficient`` verdict.  The one exception is
    the resolved-date rescue: a verifier that lists a date in ``missing`` while
    its own valid claim states that date is self-contradictory, and the rescue
    runs before the downgrade pass.
    """
    if not evidence:
        verdict.sufficient = False
        verdict.missing.append("no evidence records")
        verdict.checks["empty_evidence"] = True
        return verdict

    by_id = {str(record.get("id") or ""): record for record in (memory_records or [])}
    valid_ids = set(by_id) | {"query"}
    cited: set[str] = set()
    for item in evidence:
        cited |= _sources(item)

    unknown = sorted(source for source in cited if source not in valid_ids)
    if unknown:
        verdict.sufficient = False
        verdict.missing.append(
            "evidence cites unknown source ids: " + ", ".join(unknown[:5]))
        verdict.checks["unknown_sources"] = unknown[:10]

    def source_text(claim: dict[str, Any]) -> str:
        texts: list[str] = []
        for source in claim.get("sources") or []:
            if source == "query":
                texts.append(question)
                continue
            record = by_id.get(source)
            if record is not None:
                body = str(record.get("text") or "")
                texts.append(body)
                # A claim may state the calendar date that the verifier prompt
                # resolved from "yesterday"/"last year" + observed_at.  Add
                # that derived value so the grounding check can accept it.
                resolved = resolve_relative_dates(
                    body, str(record.get("time") or ""))
                if resolved:
                    texts.append(" ".join(resolved.values()))
        return "\n".join(texts)

    if verdict.claims:
        claims = list(verdict.claims)
        for claim in claims:
            normalized: list[str] = []
            for source in claim.get("sources") or []:
                for candidate in _index_sources(str(source), evidence):
                    if candidate in valid_ids and candidate not in normalized:
                        normalized.append(candidate)
            claim["sources"] = normalized
        valid_claims = [claim for claim in claims if claim["sources"]]
        grounded = [
            claim for claim in valid_claims
            if _claim_grounded(
                str(claim.get("claim") or ""), source_text(claim),
                threshold=grounding_threshold, aggregate=intent == "aggregate")
        ]
        verdict.claims = grounded
        verdict.checks["claims"] = len(grounded)
        verdict.checks["dropped_claims"] = len(claims) - len(grounded)
        verdict.checks["ungrounded_claims"] = len(valid_claims) - len(grounded)
        if verdict.sufficient and strict and grounded is not valid_claims:
            if not grounded:
                verdict.sufficient = False
                verdict.missing.append(
                    "no grounded claim supports the answer in the cited records")

    if (verdict.sufficient and strict and require_claims
            and intent in _CLAIM_REQUIRED_INTENTS and not verdict.claims):
        verdict.sufficient = False
        verdict.missing.append(
            "the verifier cited no supporting claim for a value question")
        verdict.checks["no_claims"] = True

    blob = "\n".join(str(item.get("content") or "") for item in evidence)
    if verdict.sufficient and strict and question and intent not in {
            "self_contained", "absence", "policy"}:
        keys = [
            key for key in extract_phrases(question)
            if len(key) >= 2 and key.lower() not in _STOP
        ]
        if keys:
            corpus = normalize_text(
                blob + " " + " ".join(
                    str(claim.get("claim") or "") for claim in verdict.claims))
            covered = [key for key in keys if normalize_text(key) in corpus]
            coverage = len(covered) / len(keys)
            verdict.checks["key_coverage"] = round(coverage, 3)
            if coverage < min_key_coverage:
                missing_keys = [key for key in keys if key not in covered]
                verdict.sufficient = False
                verdict.missing.append(
                    "the evidence does not cover: " + ", ".join(missing_keys[:5]))
                verdict.checks["missing_keys"] = missing_keys[:10]

    # A common self-contradiction: the verifier's own claim states a resolved
    # date, but its `missing` list still asks for that date.  If every missing
    # item is about time and a valid claim shares the question's subject and
    # contains a date, accept the resolved date instead of burning the
    # remaining postagent steps on a fact the evidence already establishes.
    if (not verdict.sufficient and not unknown and question
            and _QUESTION_DATE.search(question)):
        missing = [str(item) for item in verdict.missing if str(item).strip()]
        dated = [
            claim for claim in verdict.claims
            if claim.get("sources") and _DATE_TOKEN.search(str(claim.get("claim") or ""))
            and _shares_subject(str(claim.get("claim") or ""), question)
        ]
        if missing and all(_MISSING_DATE.search(item) for item in missing) and dated:
            verdict.sufficient = True
            verdict.missing = []
            verdict.reasoning += " [resolved date accepted from a cited claim]"
            verdict.checks["rescued_date"] = True

    if question:
        if _QUESTION_DATE.search(question) and not _DATE_TOKEN.search(blob):
            verdict.sufficient = False
            verdict.missing.append(
                "the question asks for a date, but the evidence states none")
            verdict.checks["missing_date"] = True
        if _QUESTION_NUMBER.search(question) and not _NUMBER_TOKEN.search(blob):
            verdict.sufficient = False
            verdict.missing.append(
                "the question asks for a number, but the evidence states none")
            verdict.checks["missing_number"] = True

    forbidden = [item for item in (forbidden or []) if item.strip()]
    if forbidden:
        leaked = [item for item in forbidden if item.lower() in blob.lower()]
        if leaked:
            verdict.policy_violation = True
            verdict.sufficient = False
            verdict.missing.append(
                "evidence discloses content the question forbids: "
                + ", ".join(leaked[:3]))
            verdict.checks["policy_leak"] = leaked[:5]

    # A "current value" question must not pass on a superseded record alone.
    if intent in {"update", "conflict"} and by_id:
        active: dict[str, dict[str, Any]] = {}
        for record in by_id.values():
            meta = _meta(record)
            key = str(meta.get("state_key") or "")
            if key and str(meta.get("status") or "active") == "active":
                active[key] = record
        stale: list[str] = []
        for source in cited:
            record = by_id.get(source)
            if record is None:
                continue
            meta = _meta(record)
            key = str(meta.get("state_key") or "")
            if not key or str(meta.get("status") or "") != "superseded":
                continue
            current = active.get(key)
            if current is not None and str(current.get("id") or "") not in cited:
                stale.append(key)
        if stale:
            verdict.freshness_conflict = True
            verdict.sufficient = False
            verdict.missing.append(
                "the current value is missing for: " + ", ".join(sorted(set(stale))[:3]))
            verdict.checks["stale_state"] = sorted(set(stale))[:5]

    verdict.checks.setdefault("cited_sources", len(cited))
    return verdict


def parse_gate(text: str) -> Verdict:
    """Parse the permissive gate verdict (``answerable`` rather than ``sufficient``)."""
    value = _object(text)
    if value is None:
        # A malformed gate reply is uncertain; run the loop rather than skip it.
        return Verdict(sufficient=False, parsed=False, raw=text or "",
                       missing=["gate verdict was not valid JSON"])
    answerable = value.get("answerable", value.get("sufficient", True))
    missing = value.get("missing") or []
    if isinstance(missing, str):
        missing = [missing]
    quote = str(
        value.get("value") or value.get("quote") or value.get("answer") or ""
    ).strip()
    sources = value.get("sources") or value.get("source") or []
    if isinstance(sources, str):
        sources = [sources]
    sources = [str(source) for source in sources if str(source).strip()]
    sufficient = bool(answerable)
    if sufficient and not quote:
        # Force the gate to extract a concrete value, not merely agree that
        # the topic is present.  A gate that cannot quote the value is
        # uncertain enough to run the loop.
        sufficient = False
        missing = missing or ["gate did not quote a specific value"]
    return Verdict(
        sufficient=sufficient,
        missing=[str(item) for item in missing if str(item).strip()],
        reasoning=str(value.get("reasoning") or ""),
        parsed=True,
        raw=text or "",
        checks={"gate_value": quote, "gate_sources": sources,
                "gate_slot": str(value.get("slot") or ""),
                "gate_answer": str(value.get("answer") or "")},
    )


_DURATION = re.compile(
    r"\b\d+\s*(?:day|week|month|year|hour)s?\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"several|few)\s+(?:day|week|month|year|hour)s?\b",
    re.IGNORECASE)
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
_GATE_NAMES = frozenset({
    "the", "this", "that", "what", "when", "where", "which", "who", "why",
    "how", "did", "does", "was", "were", "have", "has", "can", "could",
    "would", "should", "please", "tell", "give", "list", "name", "many",
})
_GATE_FILLER = frozenset({
    "make", "made", "take", "took", "get", "got", "give", "gave", "go",
    "went", "say", "said", "have", "had", "do", "did", "be", "was", "were",
    "is", "are", "also", "just", "really", "very", "much", "many", "some",
    "any", "own", "thing", "things", "the", "a", "an", "of", "in", "on",
    "to", "for", "with", "and", "or", "that", "this", "it", "its",
})


def _gate_shape_ok(question: str, quote: str,
                   records: Sequence[dict[str, Any]],
                   answer: str = "") -> bool:
    """Deterministic answer-shape/entity check on the gate's quote.

    The model is a good reader but a weak auditor: it accepts a date/name that
    belongs to another event.  These checks reject the common wrong shapes
    without another model call.
    """
    quote_tokens = set(tokenize(quote or ""))
    if not quote_tokens:
        return False
    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for record in records:
        tokens = set(tokenize(str(record.get("content") or "")))
        if not tokens:
            continue
        overlap = len(quote_tokens & tokens) / len(quote_tokens)
        if overlap > best_overlap:
            best, best_overlap = record, overlap
    if best is None or best_overlap < 0.5:
        return False
    body = str(best.get("content") or "")
    metadata = best.get("metadata") if isinstance(best.get("metadata"), dict) else {}
    speaker = str(metadata.get("speaker") or metadata.get("role") or "")
    combined = f"{speaker} {body}"

    q = question or ""
    lower = q.lower()
    if re.search(r"\bhow long\b", lower):
        # A duration question needs a duration, not merely a date ("last week").
        if not (_DURATION.search(quote or "") or _DURATION.search(answer or "")):
            return False
    elif re.search(r"\b(when|what date|what year|which month|what time)\b",
                   lower):
        if not (_DATE_TOKEN.search(quote or "")
                or _DATE_TOKEN.search(combined)):
            return False
    if re.search(r"\b(how many|how much|total|count|number of)\b", lower):
        if not _NUMBER_TOKEN.search(quote or ""):
            return False
    # "a few/several years ago" is deliberately vague; a specific year in the
    # answer is a calendar computation the record does not support.
    if re.search(r"\b(?:a\s+few|few|several|a\s+couple\s+of)\s+years?\s+ago\b",
                 quote or "", re.IGNORECASE):
        if re.search(r"\b(?:19|20)\d{2}\b", answer or ""):
            return False
    if re.search(r"\b(who|whose)\b", lower):
        names = [name for name in _NAME.findall(quote or "")
                 if name.lower() not in _GATE_NAMES]
        if not names:
            return False
    if re.search(r"\b(where|which city|which country|which place)\b", lower):
        if not _NAME.findall(quote or ""):
            return False
    if re.search(r"\b(which|what kind|what type)\b", lower):
        q_tokens = set(tokenize(q))
        new_tokens = {token for token in quote_tokens - q_tokens
                      if token not in _GATE_FILLER}
        if not new_tokens:
            return False
    # Person/event match: a named subject in the question must appear in the
    # cited record (speaker or content).  Document chunks without a speaker
    # are exempt: there is no role to compare.
    names = [name for name in _NAME.findall(q)
             if name.lower() not in _GATE_NAMES]
    if names and speaker:
        haystack = combined.lower()
        if not any(name.lower() in haystack for name in names):
            return False
    # First-person questions should not be answered from an assistant plan.
    if re.search(r"\b(i|i'm|i've|my|me|we|our)\b", lower) and speaker:
        if speaker.lower() in {"assistant", "system"}:
            return False
    return True


_VALUE_TOKEN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:19|20)\d{2}|\d+(?:[.,]\d+)*)\b")


def _values(text: str) -> set[str]:
    values = {match.group(0).lower()
              for match in _VALUE_TOKEN.finditer(text or "")}
    values |= {match.group(0).lower().strip()
               for match in _DURATION.finditer(text or "")}
    values |= {match.group(0).lower()
               for match in _DATE_TOKEN.finditer(text or "")}
    return values


def _answer_supported(answer: str, quote: str, question: str) -> bool:
    """Whether the gate's answer is actually backed by its quote.

    The model sometimes writes a plausible answer and attaches an unrelated
    quote ("would pursue writing" backed by a counseling sentence).  Reject
    when the answer introduces a number/date/duration absent from the quote,
    or when the answer's question predicate never appears in the quote.
    """
    answer = str(answer or "").strip()
    quote = str(quote or "").strip()
    if not answer or not quote:
        return False
    answer_values = _values(answer) - _values(question)
    value_backed = bool(answer_values) and bool(answer_values & _values(quote))
    if answer_values and not value_backed:
        return False
    question_tokens = set(tokenize(question)) - _GATE_FILLER
    answer_tokens = set(tokenize(answer))
    quote_tokens = set(tokenize(quote))
    shared = answer_tokens & question_tokens
    if shared and not (quote_tokens & shared):
        # A quote that supplies the asked value (a date/number/duration) is
        # enough even when it does not repeat the question's predicate.
        if value_backed:
            return True
        if not re.search(
                r"\b(yes|no|not|never|didn't|haven't|hasn't|isn't|aren't|"
                r"wasn't|weren't)\b",
                quote, re.IGNORECASE):
            return False
    return True


def _quote_grounded(quote: str, records: Sequence[dict[str, Any]],
                    threshold: float = 0.5) -> bool:
    """Whether the gate's quoted value actually occurs in the evidence."""
    quote_tokens = set(tokenize(quote or ""))
    if not quote_tokens:
        return False
    evidence_tokens: set[str] = set()
    for record in records:
        evidence_tokens |= set(tokenize(str(record.get("content") or "")))
    if not evidence_tokens:
        return False
    return len(quote_tokens & evidence_tokens) / len(quote_tokens) >= threshold


async def consistency_check(
    question: str,
    answer: str,
    quote: str,
    call: Call,
    *,
    max_context_tokens: int = 0,
) -> bool:
    """A second, stricter look at an accepted gate answer.

    The permissive gate is good at finding a value but sometimes attaches a
    quote about a different event.  This call compares the proposed answer to
    its quote; it is only used for single-value/ordering intents, not for
    long-document/rubric answers.
    """
    messages = [
        {"role": "system", "content": prompts.CONSISTENCY},
        {"role": "user", "content": (
            f"Question:\n{question}\n\nProposed answer:\n{answer}\n\n"
            f"Quote:\n{quote}")},
    ]
    if max_context_tokens > 0:
        messages = llm.fit_messages(
            messages, max_context_tokens, reserve_tokens=1024)
    try:
        value = _object(await call(messages))
    except Exception:  # noqa: BLE001 - uncertainty sends the query to the loop
        return False
    if value is None:
        return False
    return bool(value.get("consistent", True))


async def gate_verify(
    question: str,
    records: list[dict[str, Any]],
    call: Call,
    *,
    max_context_tokens: int = 0,
    check_consistency: bool = False,
    images: Sequence[str] | None = None,
) -> Verdict:
    """Permissive coarse-sufficiency gate.

    Unlike :func:`verify`, this does not apply the strict claim/grounding
    checks.  It answers one question: is the asked slot present in any
    wording?  Only a clear absence sends the query to the second retrieval
    loop.
    """
    if not records:
        return Verdict(sufficient=False, missing=["no coarse evidence"])
    messages = [
        {"role": "system", "content": prompts.GATE},
        {"role": "user", "content": prompts.verifier_prompt(question, records)},
    ]
    if images:
        # One query image, one bounded gate call.  With a text-only question
        # this branch is never reached; memory images are never attached.
        messages = llm.attach_images(messages, list(images), max_images=1)
    if max_context_tokens > 0:
        messages = llm.fit_messages(
            messages, max_context_tokens, reserve_tokens=1024)
    try:
        verdict = parse_gate(await call(messages))
    except Exception as exc:  # a failed gate must not skip the loop
        return Verdict(sufficient=False, parsed=False,
                       missing=[f"gate failed: {exc}"])
    if verdict.sufficient:
        quote = str(verdict.checks.get("gate_value") or "")
        if not _quote_grounded(quote, records):
            verdict.sufficient = False
            verdict.missing = [
                "the gate's quoted value is not present in the evidence"]
        elif not _gate_shape_ok(
                question, quote, records,
                answer=str(verdict.checks.get("gate_answer") or "")):
            verdict.sufficient = False
            verdict.missing = [
                "the gate's quote does not match the asked slot/entity"]
        elif not _answer_supported(
                str(verdict.checks.get("gate_answer") or ""), quote, question):
            verdict.sufficient = False
            verdict.missing = [
                "the gate's answer is not backed by its quoted text"]
        elif check_consistency and not await consistency_check(
                question, str(verdict.checks.get("gate_answer") or ""),
                quote, call, max_context_tokens=max_context_tokens):
            verdict.sufficient = False
            verdict.missing = [
                "the consistency check rejected the gate answer"]
    return verdict


async def verify(
    question: str,
    records: list[dict[str, Any]],
    call: Call,
    *,
    memory_records: Sequence[dict[str, Any]] | None = None,
    forbidden: Sequence[str] | None = None,
    intent: str = "",
    strict: bool = True,
    require_claims: bool = True,
    grounding_threshold: float = 0.35,
    min_key_coverage: float = 0.6,
    max_context_tokens: int = 0,
) -> Verdict:
    if not records:
        return Verdict(sufficient=False, missing=["no evidence records were written"])
    messages = [
        {"role": "system", "content": prompts.VERIFIER},
        {"role": "user", "content": prompts.verifier_prompt(question, records)},
    ]
    if max_context_tokens > 0:
        messages = llm.fit_messages(messages, max_context_tokens, reserve_tokens=1024)
    try:
        verdict = parse(await call(messages))
    except Exception as exc:  # conservative: an audit failure is not a pass
        return Verdict(sufficient=False, missing=[f"verifier failed: {exc}"],
                       parsed=False)
    return apply_checks(
        verdict, records, memory_records=memory_records,
        forbidden=forbidden, intent=intent, question=question,
        strict=strict, require_claims=require_claims,
        grounding_threshold=grounding_threshold,
        min_key_coverage=min_key_coverage)
