import pytest

from app.ingestion.chunker import StructuralChunker
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.lexical import BM25Index
from app.retrieval.references import ReferenceRouter
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from tests.fakes import FakeEmbedder


class FixedVectorStore:
    def __init__(self, hits):
        self.hits = hits

    async def query(self, embedding, limit):
        return self.hits[:limit]


class EmptyLexical:
    def search(self, query, limit):
        return []


@pytest.fixture(scope="module")
def store(document):
    return ChunkStore(StructuralChunker(max_chars=1500).chunk(document))


@pytest.fixture
async def retriever(store):
    embedder = FakeEmbedder()
    vectors = InMemoryVectorStore()
    chunks = store.all()
    await vectors.add(chunks, await embedder.embed([c.embedding_text for c in chunks]))
    return HybridRetriever(
        store=store,
        embedder=embedder,
        vector_store=vectors,
        lexical=BM25Index(chunks),
        router=ReferenceRouter.from_store(store),
        top_k=6,
        candidates=30,
        expansion_max=3,
        token_budget=3500,
    )


def ids(result):
    return [scored.chunk.id for scored in result.chunks]


async def test_dense_and_lexical_hits_are_both_kept(store, document):
    retriever = HybridRetriever(
        store=store,
        embedder=FakeEmbedder(),
        vector_store=FixedVectorStore([("a2-b6", 0.8)]),
        lexical=BM25Index(store.all()),
        router=ReferenceRouter.from_store(store),
        top_k=6,
        candidates=30,
        expansion_max=0,
        token_budget=3500,
    )

    result = await retriever.retrieve("Aeroport qurish uchun ekspertiza necha kun davom etadi?")

    assert {"a2-b6", "a1-r2"} <= set(ids(result))
    assert result.chunks[0].chunk.id in {"a2-b6", "a1-r2"}


async def test_exact_number_is_found_lexically(retriever):
    result = await retriever.retrieve("541-son qaror")

    assert "q-b6" in ids(result)[:3]
    assert result.reference_match is False


async def test_cyrillic_query_matches_latin_query(retriever):
    cyrillic = await retriever.retrieve("Аэропорт учун экспертиза муддати")
    latin = await retriever.retrieve("Aeroport uchun ekspertiza muddati")

    assert ids(cyrillic)[:5] == ids(latin)[:5]


async def test_explicit_reference_is_pinned_first(retriever):
    result = await retriever.retrieve("2-ilovaning 6-bandida nima deyilgan?")

    assert ids(result)[0] == "a2-b6"
    assert result.chunks[0].pinned is True
    assert result.reference_match is True


async def test_main_resolution_item_reference(retriever):
    assert ids(await retriever.retrieve("Qarorning 7-bandi"))[0] == "q-b7"


async def test_scores_are_reported_for_every_candidate(retriever):
    result = await retriever.retrieve("jamoatchilik eshituvi")

    assert len(result.chunks) >= 6
    assert all(s.fused_score > 0 for s in result.chunks if not s.expansion)
    assert result.top_similarity >= max(s.dense_score or 0 for s in result.chunks) > 0


async def test_results_are_deterministic(retriever):
    first = await retriever.retrieve("malaka sertifikatini berish tartibi")
    second = await retriever.retrieve("malaka sertifikatini berish tartibi")

    assert ids(first) == ids(second)


async def test_cross_referenced_item_is_added_as_expansion(store):
    retriever = HybridRetriever(
        store=store,
        embedder=FakeEmbedder(),
        vector_store=FixedVectorStore([("a2-b5", 0.9)]),
        lexical=EmptyLexical(),
        router=ReferenceRouter.from_store(store),
        top_k=1,
        candidates=30,
        expansion_max=3,
        token_budget=3500,
    )

    result = await retriever.retrieve("majburiy ekspertiza obyektlari")

    assert ids(result) == ["a2-b5", "a2-b4"]
    assert result.chunks[1].expansion is True
    assert result.top_similarity == pytest.approx(0.9)


async def test_context_budget_drops_lowest_ranked_chunks(store):
    retriever = HybridRetriever(
        store=store,
        embedder=FakeEmbedder(),
        vector_store=FixedVectorStore([("a2-b4", 0.9), ("a2-b5", 0.8), ("a2-b6", 0.7)]),
        lexical=EmptyLexical(),
        router=ReferenceRouter.from_store(store),
        top_k=3,
        candidates=30,
        expansion_max=0,
        token_budget=150,
    )

    result = await retriever.retrieve("savol")

    assert ids(result) == ["a2-b4"]


@pytest.mark.parametrize(
    ("question", "anchored"),
    [
        ("541-son qaror nima boʻldi?", True),
        ("234-son qaror nima haqida?", False),
        ("2-ilovaning 6-bandi", False),
        ("1990-yilgi qonun", False),
    ],
)
async def test_lexical_anchor_needs_a_multi_digit_number_found_in_the_results(
    store, question, anchored
):
    retriever = HybridRetriever(
        store=store,
        embedder=FakeEmbedder(),
        vector_store=FixedVectorStore([("q-b6", 0.3), ("q-b7", 0.2)]),
        lexical=EmptyLexical(),
        router=ReferenceRouter([]),
        top_k=2,
        candidates=30,
        expansion_max=0,
        token_budget=3500,
    )

    assert (await retriever.retrieve(question)).lexical_anchor is anchored


async def test_at_most_two_parts_of_one_item_are_selected(store):
    parts = [(f"a9-b1-p{n}", 0.9 - n / 100) for n in range(1, 6)]
    retriever = HybridRetriever(
        store=store,
        embedder=FakeEmbedder(),
        vector_store=FixedVectorStore([*parts, ("q-b6", 0.5)]),
        lexical=EmptyLexical(),
        router=ReferenceRouter([]),
        top_k=4,
        candidates=30,
        expansion_max=0,
        token_budget=10_000,
    )

    selected = ids(await retriever.retrieve("savol"))

    assert sum(chunk_id.startswith("a9-b1-") for chunk_id in selected) == 2
    assert "q-b6" in selected
