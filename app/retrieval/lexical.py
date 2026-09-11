"""BM25 over normalized, stemmed Uzbek tokens."""

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from app.domain.models import Chunk
from app.text.stemmer import tokenize


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.search_text) or ["_"] for c in chunks])

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )
        return [(self._ids[i], float(scores[i])) for i in ranked[:limit]]
