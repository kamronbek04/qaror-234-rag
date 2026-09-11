"""Evaluation metrics — pure functions, independent of models and I/O."""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.text.normalize import lexical_form

_DECIMAL_COMMA = re.compile(r"(\d),(\d)")
REFUSAL_KINDS = frozenset({"out_of_doc", "trap"})

TARGETS = {
    "refusal_recall": (">=", 0.95),
    "false_refusal_rate": ("<=", 0.10),
    "hit_at_5": (">=", 0.90),
    "fact_accuracy": (">=", 0.85),
}


@dataclass(frozen=True)
class E2EOutcome:
    item_id: str
    kind: str
    status: str
    correct: bool
    cited_expected: bool
    latency_ms: int


def hit_at_k(ranking: Sequence[str], expected: set[str], k: int) -> bool:
    return any(chunk_id in expected for chunk_id in ranking[:k])


def reciprocal_rank(ranking: Sequence[str], expected: set[str]) -> float:
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def suggest_threshold(in_doc_similarities: Iterable[float], margin: float = 0.03) -> float:
    """Just below the weakest in-document question, so the gate never refuses those."""
    return round(min(in_doc_similarities) - margin, 2)


def normalize_fact_text(text: str) -> str:
    return _DECIMAL_COMMA.sub(r"\1.\2", lexical_form(text))


def contains_fact(text: str, fact: str) -> bool:
    """True when any "|"-separated alternative of the fact occurs in the text."""
    haystack = normalize_fact_text(text)
    return any(normalize_fact_text(alternative) in haystack for alternative in fact.split("|"))


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def _mean(flags: Iterable[bool]) -> float | None:
    values = list(flags)
    return round(sum(values) / len(values), 3) if values else None


def e2e_metrics(outcomes: Sequence[E2EOutcome]) -> dict[str, float | None]:
    in_doc = [o for o in outcomes if o.kind == "in_doc"]
    refused = [o for o in outcomes if o.status == "not_found"]
    latencies = [o.latency_ms for o in outcomes]
    return {
        "refusal_recall": _mean(o.correct for o in outcomes if o.kind in REFUSAL_KINDS),
        "refusal_precision": _mean(o.kind in REFUSAL_KINDS for o in refused),
        "false_refusal_rate": _mean(o.status == "not_found" for o in in_doc),
        "fact_accuracy": _mean(o.correct for o in in_doc),
        "citation_accuracy": _mean(o.cited_expected for o in in_doc if o.status != "not_found"),
        "partial_accuracy": _mean(o.correct for o in outcomes if o.kind == "partial"),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }


def meets_target(name: str, value: float | None) -> bool | None:
    if name not in TARGETS or value is None:
        return None
    operator, target = TARGETS[name]
    return value >= target if operator == ">=" else value <= target
