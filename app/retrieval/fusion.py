"""Reciprocal Rank Fusion: combines rankings whose raw scores are not comparable."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def rank_by_fused_score(scores: Mapping[str, float], order: Mapping[str, int]) -> list[str]:
    """Best fused score first; ties broken by position in the document."""
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], order.get(chunk_id, 1 << 30)))
