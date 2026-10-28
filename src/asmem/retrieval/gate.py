"""Decide whether coarse candidates are already enough to skip the agent.

The gate is deterministic and cheap.  It only allows simple, single-hop style
queries whose top candidate covers most query terms and is supported by at
least one real matching channel.  Multi-hop, summary, aggregation, policy,
MCQ, and obviously complex questions always go to the agent.

When ``gate.verify`` is true (default), the coarse evidence is also sent to the
existing verifier once.  If the verifier says it is insufficient, the runner
falls through to the agent; if it says sufficient, the agent loop is
skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .tokenize import terms, tokenize

_COMPLEX = re.compile(
    r"\b(compare|comparison|difference|why|how did|how has|before|after|since|"
    r"until|all of|every|each|summar(?:y|ize|ise)|overview|how many|how much|"
    r"which .* (?:first|last|latest|earliest)|relationship|because|caused|"
    r"conflict|contradict|timeline|deadline|due date|still|now|currently|"
    r"changed|updated|used to|no longer|anymore|instead|in order|sequence|"
    r"latest|most recent|forget|do not remember)\b",
    re.IGNORECASE,
)
_MATCH_CHANNELS = {"bm25", "dense", "exact", "state", "option"}
_DATE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}(?:,\s*\d{4})?\b", re.IGNORECASE)
_LONG_NUMBER = re.compile(r"\b\d{4,}\b")
_VALUE_CUES = re.compile(
    r"\b(when|date|deadline|how many|how much|total|key|id|number|price|cost)\b",
    re.IGNORECASE)


@dataclass
class GateResult:
    sufficient: bool = False
    reasons: list[str] = field(default_factory=list)
    coverage: float = 0.0
    channels: list[str] = field(default_factory=list)
    candidate_id: str = ""
    verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reasons": self.reasons,
            "coverage": round(self.coverage, 4),
            "channels": self.channels,
            "candidate_id": self.candidate_id,
            "verified": self.verified,
        }


def _config(config: dict[str, Any]) -> dict[str, Any]:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    gate = retrieval.get("gate") if isinstance(retrieval.get("gate"), dict) else {}
    return {
        "enabled": bool(gate.get("enabled", True)),
        "verify": bool(gate.get("verify", True)),
        "min_coverage": float(gate.get("min_coverage", 0.6)),
        "min_channels": int(gate.get("min_channels", 1)),
        "max_terms": int(gate.get("max_terms", 12)),
        "min_text_chars": int(gate.get("min_text_chars", 40)),
        "evidence_limit": int(gate.get("evidence_limit", 6)),
        "allow_routes": set(gate.get("allow_routes") or ["dialogue"]),
    }


def _conflicting_values(candidates: Sequence[dict[str, Any]], question: str) -> bool:
    """Cheap safety check: two candidate values for the same slot means no gate.

    A deadline question whose shortlist contains "April 1" and "April 5" is
    exactly the knowledge-update failure the gate used to cause.  We only run
    the check when the question asks for a value, and only flag clearly
    different dates / long numbers.
    """
    if not _VALUE_CUES.search(question or ""):
        return False
    dates: set[str] = set()
    numbers: set[str] = set()
    for candidate in list(candidates)[:8]:
        text = str(candidate.get("text") or "")
        dates.update(match.group(0).lower() for match in _DATE.finditer(text))
        numbers.update(match.group(0) for match in _LONG_NUMBER.finditer(text))
    return len(dates) >= 2 or len(numbers) >= 3


def evaluate_gate(
    candidates: Sequence[dict[str, Any]],
    question: str,
    route: str,
    config: dict[str, Any] | None = None,
) -> GateResult:
    cfg = _config(config or {})
    result = GateResult()
    if not cfg["enabled"]:
        result.reasons = ["gate_disabled"]
        return result
    if route not in cfg["allow_routes"]:
        result.reasons = [f"route:{route}"]
        return result
    if not candidates:
        result.reasons = ["no_candidates"]
        return result
    top = candidates[0]
    if top.get("neighbor_of"):
        result.reasons = ["top_is_neighbor"]
        return result
    text = str(top.get("text") or "")
    if len(text.strip()) < cfg["min_text_chars"]:
        result.reasons = ["top_text_too_short"]
        return result
    query_terms = set(tokenize(question))
    if not query_terms:
        result.reasons = ["no_query_terms"]
        return result
    if len(query_terms) > cfg["max_terms"]:
        result.reasons = [f"too_many_terms:{len(query_terms)}"]
        return result
    if _COMPLEX.search(question or ""):
        result.reasons = ["complex_question"]
        return result
    coverage = len(query_terms & terms(text)) / len(query_terms)
    result.coverage = coverage
    channels = [name for name in (top.get("channels") or {}) if name in _MATCH_CHANNELS]
    result.channels = channels
    result.candidate_id = str(top.get("id") or "")
    if coverage < cfg["min_coverage"]:
        result.reasons = [f"coverage:{coverage:.2f}<{cfg['min_coverage']:.2f}"]
        return result
    if len(channels) < cfg["min_channels"]:
        result.reasons = [f"channels:{len(channels)}<{cfg['min_channels']}"]
        return result
    if _conflicting_values(candidates, question):
        result.reasons = ["conflicting_values_in_shortlist"]
        return result
    result.sufficient = True
    result.reasons = [
        f"coverage:{coverage:.2f}",
        f"channels:{','.join(channels)}",
        f"route:{route}",
    ]
    return result


def gate_evidence_limit(config: dict[str, Any] | None = None) -> int:
    return max(1, _config(config or {})["evidence_limit"])


def gate_verify(config: dict[str, Any] | None = None) -> bool:
    return bool(_config(config or {})["verify"])
