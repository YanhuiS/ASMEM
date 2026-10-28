"""Deterministic query routing.

No model call: the route only changes retrieval budgets, prompt hints, and
verification rules.  It never sends the query down a different code path that
would make failures hard to attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

_SELF_MARKERS = (
    "conversation transcript", "transcript:", "artifact", "artifacts:",
    "input:", "you are the", "your task", "given the following",
)
_POLICY = re.compile(
    r"\b(do not disclose|do not reveal|must not disclose|confidential|"
    r"do not share|sensitive|privacy|redact|please forget|do not remember|"
    r"don't remember|do not retain)\b", re.IGNORECASE)
_SUMMARY = re.compile(
    r"\b(summar(?:y|ize|ise)|overview|comprehensive|how has .* developed|"
    r"what has changed|timeline|recap)\b", re.IGNORECASE)
_AGGREGATION = re.compile(
    r"\b(how many|how much|total|sum|average|mean|maximum|minimum|max|min|"
    r"newest|oldest|latest|earliest|count|number of)\b", re.IGNORECASE)
_MULTI_HOP = re.compile(
    r"\b(of the|the .* of the|which .* of|who .* of|that .* of|"
    r"the .* who .* of)\b", re.IGNORECASE)
_CAPITAL = re.compile(r"\b[A-Z][A-Za-z0-9_.\-]{1,}\b")

# Intent cues.  They do not change the pipeline; they select a short, fixed
# instruction block for the agent (see ``search.prompts``).
_UPDATE = re.compile(
    r"\b(currently|latest|most recent|changed|updated|used to|no longer|"
    r"anymore|supersed(?:e|ed)|revised|new deadline|right now|as of now)\b",
    re.IGNORECASE)
_STILL = re.compile(
    r"\b(?:do|am|is|are|does|did)\s+(?:i|we|you|they|it)\s+still\b",
    re.IGNORECASE)
_CONFLICT = re.compile(
    r"\b(contradict|conflict|inconsistent|disagree|which .* (?:correct|true)|"
    r"both .* and|said .* but|mentioned .* but|never .* but)\b", re.IGNORECASE)
_ORDER = re.compile(
    r"\b(in order|in sequence|chronolog|order .* (?:happen|mention|discuss)|"
    r"first .* then|what came first|earliest to latest|which .* first)\b",
    re.IGNORECASE)
_COMPARE = re.compile(
    r"\b(compare|versus|\bvs\.?\b|difference between|"
    r"which (?:one|is|was) (?:faster|slower|better|higher|lower|more|less))\b",
    re.IGNORECASE)
_DURATION = re.compile(
    r"\b(how many (?:days|weeks|months|years|hours|minutes)|how long|"
    r"duration|time between|between .* and .*|elapsed)\b", re.IGNORECASE)
_TEMPORAL = re.compile(
    r"\b(when|what date|what day|deadline|due date|before|after|since|until)\b",
    re.IGNORECASE)
_PREFERENCE = re.compile(
    r"\b(recommend|suggest|advice|for me|my (?:preference|favorite|favourite)|"
    r"would i (?:like|enjoy)|do i (?:like|enjoy|prefer)|what do i (?:like|prefer)|"
    r"based on (?:my|what you know)|personaliz)\b", re.IGNORECASE)
_ABSENCE = re.compile(
    r"\b(do i have|have i ever|did i ever|is there any|are there any|"
    r"was there any|do you remember|can you confirm|is it true that)\b",
    re.IGNORECASE)
# Only first-person yes/no questions.  "Can you tell me…", "Do you remember…"
# and similar requests are NOT yes/no about the user; treating them as such
# made the agent look for a positive/negative pair instead of the asked fact.
_YESNO = re.compile(
    r"^\s*(?:have|has|did|do|does|is|are|was|were|can|could|will|would|am)\s+"
    r"(?:i|we)\b", re.IGNORECASE)
_EXPLAIN = re.compile(
    r"\b(what do .* do|what does .* do|how does .* work|how do .* work|"
    r"explain|what are (?:the )?(?:rules|mechanics|steps|types|requirements)|"
    r"describe|tell me about|what is the purpose)\b", re.IGNORECASE)
_LIST = re.compile(
    r"\b(list|what are some|which .* (?:activities|dishes|sources|options|items)|"
    r"name (?:all|the|some)|give me .* (?:list|ideas|examples))\b", re.IGNORECASE)


@dataclass
class Route:
    name: str = "dialogue"
    reasons: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    @property
    def self_contained(self) -> bool:
        return self.name == "self_contained"


def _forbidden(text: str) -> list[str]:
    """Extract the object of a simple ``do not disclose X`` clause."""
    match = re.search(
        r"(?:do not|must not|don't)\s+(?:disclose|reveal|share)\s+(.+?)(?:[.;\n]|$)",
        text, re.IGNORECASE)
    if not match:
        return []
    value = match.group(1).strip()
    return [value] if 2 <= len(value) <= 200 else []


def route_query(question: str, options: Sequence[str] | None = None) -> Route:
    text = question or ""
    lowered = text.lower()
    reasons: list[str] = []

    if options:
        reasons.append("options")
    if len(text) >= 1500 and any(marker in lowered for marker in _SELF_MARKERS):
        return Route("self_contained", ["long_query", "self_contained_marker"], [])
    if _POLICY.search(text):
        reasons.append("policy")
        route = Route("policy", reasons, _forbidden(text))
        return route
    if options:
        return Route("mcq", reasons, [])
    if _SUMMARY.search(text):
        reasons.append("summary")
        return Route("summary", reasons, [])
    if _AGGREGATION.search(text):
        reasons.append("aggregation")
        return Route("aggregation", reasons, [])
    names = [
        name for name in _CAPITAL.findall(text)
        if name.lower() not in {
            "what", "where", "when", "who", "why", "how", "which", "the",
            "does", "did", "is", "are", "was", "were", "can", "could",
            "would", "should", "please", "tell", "give",
        }
    ]
    if _MULTI_HOP.search(text) or len(set(names)) >= 2:
        reasons.append("multi_hop")
        return Route("multi_hop", reasons, [])
    return Route("dialogue", reasons, [])


def looks_like_table(text: str, *, min_rows: int = 3) -> bool:
    """True when the query itself carries a data table.

    A table row needs at least four tab/comma-separated fields and at least
    three numeric ones.  This is deliberately narrow: a question that merely
    says "the final terms above" is *not* self-contained, because the terms
    live in memory, not in the query.
    """
    rows = 0
    for line in (text or "").splitlines():
        fields = [item.strip() for item in re.split(r"[\t]", line)]
        if len(fields) < 4:
            fields = [item.strip() for item in line.split(",")]
        if len(fields) < 4:
            continue
        numeric = sum(1 for item in fields if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", item))
        if numeric >= 3:
            rows += 1
            if rows >= min_rows:
                return True
    return False


def detect_intent(question: str, route: Route | str | None = None,
                  options: Sequence[str] | None = None) -> str:
    """Classify *what kind of evidence* the agent must produce.

    The labels are stable strings consumed by ``search.prompts.intent_guidance``
    and by the evidence selector.  They are intentionally few: every label
    must change the instruction text, never the code path.
    """
    name = route.name if isinstance(route, Route) else str(route or "")
    text = question or ""
    if name == "self_contained":
        return "self_contained"
    if options:
        return "mcq"
    if name == "policy" or _POLICY.search(text):
        return "policy"
    if _CONFLICT.search(text):
        return "conflict"
    if _UPDATE.search(text) or _STILL.search(text):
        return "update"
    if _PREFERENCE.search(text):
        return "preference"
    if _YESNO.search(text):
        return "yesno"
    if _EXPLAIN.search(text):
        return "explain"
    if _LIST.search(text):
        return "list"
    if name == "summary":
        return "summary"
    if name == "aggregation" or _AGGREGATION.search(text):
        return "aggregate"
    if _ORDER.search(text):
        return "order"
    if _COMPARE.search(text):
        return "compare"
    if _DURATION.search(text):
        return "duration"
    if _TEMPORAL.search(text):
        return "temporal"
    if _ABSENCE.search(text):
        return "absence"
    if name == "multi_hop":
        return "multi_hop"
    return "fact"
