"""The classifier contract: a state string in, calibrated signals out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..features import RequestFeatures


@dataclass(frozen=True)
class Signals:
    # Probability over "simple" / "standard" / "complex".
    tier: dict[str, float]
    needs_reasoning: float
    needs_code: float
    source: str
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class Classifier(Protocol):
    name: str

    def classify(self, features: RequestFeatures, state: str) -> Signals: ...


# The typed questions sent to lev. Choice criteria are the option descriptions lev
# matches against the state; Noul questions return P(yes).
LEV_QUESTIONS: dict[str, dict] = {
    "tier": {
        "type": "choice",
        "instructions": "How capable a language model does this request need to be answered well?",
        "criteria": {
            "simple": "short factual answer, lookup, greeting, rewrite, translate, classify, extract a field, fix grammar",
            "standard": "multi-paragraph writing, summarizing a document, moderate code, explaining a concept, routine analysis",
            "complex": "multi-step reasoning, system or software architecture, hard math or proofs, debugging subtle issues, long agentic tool use, research synthesis",
        },
    },
    "needs_reasoning": {
        "type": "noul",
        "instructions": "Does answering this require multi-step reasoning or planning?",
    },
    "needs_code": {
        "type": "noul",
        "instructions": "Does this require writing or debugging non-trivial code?",
    },
}


def signals_from_answers(answers: dict, source: str, latency_ms: float) -> Signals:
    """Build Signals from a /v1/systemone `answers` mapping (dicts or pydantic models)."""

    def get(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    tier = get(answers["tier"], "probabilities")
    return Signals(
        tier={k: float(tier.get(k, 0.0)) for k in ("simple", "standard", "complex")},
        needs_reasoning=float(get(answers["needs_reasoning"], "noul")),
        needs_code=float(get(answers["needs_code"], "noul")),
        source=source,
        latency_ms=latency_ms,
    )
