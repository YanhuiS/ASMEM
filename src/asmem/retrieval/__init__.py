"""Deterministic coarse retrieval: BM25, dense, exact, RRF, candidates.

No generative model is called here.  Dense is optional and uses an embedding
encoder (local or OpenAI-compatible); if no encoder is configured the system
falls back to BM25 + exact + time + state.
"""

from .coarse import CoarseResult, run_coarse
from .gate import GateResult, evaluate_gate, gate_evidence_limit, gate_verify
from .index import RetrievalIndex, build_index
from .parents import choose_resolution, parent_text, resolve_record
from .postprocess import (
    annotate_resolved_dates,
    claim_mode,
    fit_evidence,
    postprocess_evidence,
    seed_evidence,
    select_final_evidence,
)
from .router import detect_intent, looks_like_table

__all__ = [
    "annotate_resolved_dates",
    "CoarseResult",
    "GateResult",
    "RetrievalIndex",
    "build_index",
    "choose_resolution",
    "claim_mode",
    "detect_intent",
    "evaluate_gate",
    "fit_evidence",
    "gate_evidence_limit",
    "gate_verify",
    "looks_like_table",
    "parent_text",
    "resolve_record",
    "postprocess_evidence",
    "run_coarse",
    "seed_evidence",
    "select_final_evidence",
]
