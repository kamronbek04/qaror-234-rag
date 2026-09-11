from pathlib import Path

import pytest

from eval.dataset import DatasetError, load_dataset, validate_dataset
from eval.metrics import (
    E2EOutcome,
    contains_fact,
    e2e_metrics,
    hit_at_k,
    percentile,
    reciprocal_rank,
    suggest_threshold,
)

ROOT = Path(__file__).resolve().parents[2]


class TestRetrievalMetrics:
    def test_hit_at_k(self):
        assert hit_at_k(["a", "b", "c"], {"c"}, k=3) is True
        assert hit_at_k(["a", "b", "c"], {"c"}, k=2) is False

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 0.5
        assert reciprocal_rank(["a"], {"z"}) == 0.0

    def test_threshold_sits_below_the_weakest_in_document_score(self):
        assert suggest_threshold([0.61, 0.55, 0.72], margin=0.03) == pytest.approx(0.52)


class TestFactMatching:
    def test_facts_ignore_case_apostrophes_and_decimal_separator(self):
        assert contains_fact("Toʻlov miqdori 6,5 BXM", "6.5 bxm")
        assert contains_fact("Qo‘llanilmaydi, yoʻl qoʻyilmaydi", "yo'l qo'yilmaydi")
        assert not contains_fact("II toifa", "iii")


class TestE2EMetrics:
    def test_refusal_and_fact_metrics(self):
        outcomes = [
            E2EOutcome("d1", "in_doc", "answered", True, True, 1000),
            E2EOutcome("d2", "in_doc", "not_found", False, False, 50),
            E2EOutcome("o1", "out_of_doc", "not_found", True, False, 40),
            E2EOutcome("t1", "trap", "answered", False, False, 900),
            E2EOutcome("p1", "partial", "partial", True, True, 1200),
        ]

        metrics = e2e_metrics(outcomes)

        assert metrics["refusal_recall"] == 0.5
        assert metrics["refusal_precision"] == 0.5
        assert metrics["false_refusal_rate"] == 0.5
        assert metrics["fact_accuracy"] == 0.5
        assert metrics["citation_accuracy"] == 1.0
        assert metrics["partial_accuracy"] == 1.0
        assert metrics["latency_p50_ms"] == 900

    def test_percentile(self):
        assert percentile([10, 20, 30, 40], 50) == 20
        assert percentile([10, 20, 30, 40], 95) == 40


class TestDataset:
    def test_committed_dataset_meets_the_composition_rules(self):
        items = load_dataset(ROOT / "eval" / "dataset.jsonl")

        kinds = [item.kind for item in items]
        assert len(items) >= 40
        assert (kinds.count("out_of_doc") + kinds.count("trap")) / len(items) >= 0.25
        assert sum(item.non_canonical_spelling for item in items) >= 5

    def test_unknown_expected_chunks_are_reported(self):
        items = load_dataset(ROOT / "eval" / "dataset.jsonl")

        with pytest.raises(DatasetError, match="a1-r2"):
            validate_dataset(items, known_chunk_ids={"q-b7"})


def test_fact_alternatives_and_disclaimer_aware_forbidden_terms():
    from eval.dataset import EvalItem
    from eval.runner import judge

    alternatives = EvalItem("d", "in_doc", "q", ("x",), ("yidxp|yagona interaktiv portal",))
    trap = EvalItem("t", "trap", "q", (), (), ("jarima miqdori",))

    assert judge(alternatives, "answered", "Yagona interaktiv portal orqali", ["x"])[0]
    assert judge(trap, "partial", "Jarima miqdori haqida: Hujjatda bu haqida maʼlumot yoʻq.", [])[0]
    assert not judge(trap, "answered", "Jarima miqdori 10 BXM.", [])[0]
