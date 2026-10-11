"""Zero-dependency fallback: keyword and shape features mapped to the same Signals lev returns."""

from __future__ import annotations

import math
import re
import time

from ..features import RequestFeatures
from .base import Signals

_COMPLEX = re.compile(
    r"\b(architect\w*|design (a|an|the)|distributed|concurren\w*|race condition|prove|proof|derive|"
    r"theorem|optimi[sz]e|trade-?offs?|step[- ]by[- ]step|root cause|debug\w*|refactor\w*|"
    r"algorithm\w*|complexity|scalab\w*|migrat\w*|threat model|strategy|in-depth|comprehensive|"
    r"multi-step|research|evaluate|compare .* (and|vs)|plan (for|out))\b",
    re.I,
)
_SIMPLE = re.compile(
    r"^\s*(hi|hello|thanks|thank you)\b|\b(translate|fix (the )?(grammar|typo|spelling)|rephrase|"
    r"reword|one[- ](word|line|sentence)|yes or no|classify|categori[sz]e|extract|what is|who is|"
    r"when (is|was|did)|define|capital of|convert|format (this|as)|tl;?dr)\b",
    re.I,
)
_WRITING = re.compile(
    r"\b(write|draft|compose|email|essay|paragraphs?|blog|article|report|summari[sz]e|outline|cover letter)\b",
    re.I,
)
_REASONING = re.compile(r"\b(why|how would|explain|reason|analy[sz]e|plan|decide|should (i|we))\b", re.I)
_CODE = re.compile(
    r"```|\bdef |\bclass \w+|\bfunction\b|\bimport \w+|Traceback|stack trace|\bSQL\b|\bregex\b|"
    r"\b(python|typescript|javascript|java|golang|rust|c\+\+|kotlin|bash)\b|\bbug\b|\bcompile",
    re.I,
)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    exp = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


class HeuristicClassifier:
    name = "heuristic"

    def classify(self, features: RequestFeatures, state: str) -> Signals:
        start = time.perf_counter()
        text = features.last_user
        approx_tokens = len(text) / 4

        complex_hits = len(_COMPLEX.findall(text))
        simple_hits = len(_SIMPLE.findall(text))
        code_hits = len(_CODE.findall(text))
        reasoning_hits = len(_REASONING.findall(text))
        writing_hits = min(len(_WRITING.findall(text)), 2)

        length = min(approx_tokens / 400, 3.0)  # 0 for one-liners, 3 at ~1.2k tokens
        agentic = min(features.tool_count / 4, 1.0) + min(features.tool_results / 3, 1.0)

        scores = {
            "simple": 1.2 + 1.2 * simple_hits - 0.8 * length - 0.6 * code_hits - 0.8 * complex_hits
            - 0.5 * writing_hits - agentic,
            "standard": 0.6 + 0.5 * length + 0.3 * code_hits + 0.2 * reasoning_hits + 0.6 * writing_hits,
            "complex": -0.6 + 1.1 * complex_hits + 0.4 * length + 0.3 * code_hits + 0.8 * agentic,
        }
        if features.has_images:
            scores["simple"] -= 0.5

        return Signals(
            tier=_softmax(scores),
            needs_reasoning=1 - math.exp(-0.7 * (reasoning_hits + complex_hits)),
            needs_code=1 - math.exp(-0.9 * code_hits),
            source=self.name,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
