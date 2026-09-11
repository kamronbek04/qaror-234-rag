"""Hybrid retrieval: dense + BM25 fused with RRF, explicit references, cross-reference expansion."""

import asyncio
import re
from collections import Counter

from app.domain.models import Chunk, RetrievalResult, ScoredChunk
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.fusion import rank_by_fused_score, reciprocal_rank_fusion
from app.retrieval.protocols import Embedder, LexicalIndex, VectorStore
from app.retrieval.references import ReferenceRouter
from app.text.normalize import canonicalize
from app.text.numbers import anchor_numbers, extract_numbers

_PART_SUFFIX = re.compile(r"-p\d+$")
MAX_PARTS_PER_ITEM = 2


def estimate_tokens(chunk: Chunk) -> int:
    """Conservative token estimate for Uzbek text (about three characters per token)."""
    return (len(chunk.breadcrumb) + len(chunk.text)) // 3 + 8


def _diverse(ranked: list[str], limit: int, taken: list[str]) -> list[str]:
    """Best-first selection that takes at most two parts of any one long item."""
    counts = Counter(_PART_SUFFIX.sub("", chunk_id) for chunk_id in taken)
    chosen: list[str] = []
    for chunk_id in ranked:
        if len(chosen) >= limit:
            break
        item = _PART_SUFFIX.sub("", chunk_id)
        if counts[item] < MAX_PARTS_PER_ITEM:
            counts[item] += 1
            chosen.append(chunk_id)
    return chosen


class HybridRetriever:
    def __init__(
        self,
        *,
        store: ChunkStore,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical: LexicalIndex,
        router: ReferenceRouter,
        top_k: int,
        candidates: int,
        expansion_max: int,
        token_budget: int,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._vectors = vector_store
        self._lexical = lexical
        self._router = router
        self._top_k = top_k
        self._candidates = candidates
        self._expansion_max = expansion_max
        self._token_budget = token_budget

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        limit = top_k or self._top_k
        text = canonicalize(query)
        [vector] = await self._embedder.embed([text])
        dense = await self._vectors.query(vector, self._candidates)
        lexical = await asyncio.to_thread(self._lexical.search, text, self._candidates)

        dense_scores = dict(dense)
        lexical_scores = dict(lexical)
        fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in lexical]])

        pinned = [i for i in self._router.route(text) if i in self._store][:limit]
        ranked = [
            i
            for i in rank_by_fused_score(fused, self._store.order)
            if i in self._store and i not in pinned
        ]
        selected_ids = pinned + _diverse(ranked, max(0, limit - len(pinned)), taken=pinned)

        def scored(chunk_id: str, **flags: bool) -> ScoredChunk:
            return ScoredChunk(
                chunk=self._store.get(chunk_id),
                dense_score=dense_scores.get(chunk_id),
                lexical_score=lexical_scores.get(chunk_id),
                fused_score=fused.get(chunk_id, 0.0),
                **flags,
            )

        selected = [scored(i, pinned=i in pinned) for i in selected_ids]
        expansions = [scored(i, expansion=True) for i in self._expansion_ids(selected_ids)]
        chunks = self._fit_budget(selected + expansions)
        for rank, item in enumerate(chunks, start=1):
            item.rank = rank

        anchors = anchor_numbers(text)
        return RetrievalResult(
            query=query,
            chunks=chunks,
            top_similarity=max(dense_scores.values(), default=0.0),
            reference_match=bool(pinned),
            lexical_anchor=any(
                anchors & extract_numbers(item.chunk.text) for item in chunks if not item.expansion
            ),
        )

    def _expansion_ids(self, selected_ids: list[str]) -> list[str]:
        seen = set(selected_ids)
        expansions: list[str] = []
        for chunk_id in selected_ids:
            for reference in self._store.get(chunk_id).references:
                if len(expansions) >= self._expansion_max:
                    return expansions
                if reference not in seen and reference in self._store:
                    seen.add(reference)
                    expansions.append(reference)
        return expansions

    def _fit_budget(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Keep chunks in rank order until the budget is used; the best chunk is always kept."""
        kept: list[ScoredChunk] = []
        used = 0
        for item in chunks:
            cost = estimate_tokens(item.chunk)
            if kept and used + cost > self._token_budget:
                break
            kept.append(item)
            used += cost
        return kept
