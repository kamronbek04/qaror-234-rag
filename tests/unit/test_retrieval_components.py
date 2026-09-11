from types import SimpleNamespace

import httpx
import ollama
import pytest

from app.core.errors import EmbeddingUnavailableError
from app.domain.models import Chunk, ChunkType
from app.retrieval.embedder import OllamaEmbedder
from app.retrieval.fusion import rank_by_fused_score, reciprocal_rank_fusion
from app.retrieval.lexical import BM25Index
from app.retrieval.references import ReferenceRouter
from app.retrieval.vector_store import ChromaVectorStore, InMemoryVectorStore


def chunk(chunk_id: str, text: str, order: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        type=ChunkType.ITEM,
        section="a2",
        breadcrumb="2-ilova",
        text=text,
        url="https://lex.uz/uz/docs/-8193120",
        order=order,
    )


class FakeOllamaClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.batches: list[list[str]] = []

    async def embed(self, model, input, **kwargs):
        if self.error:
            raise self.error
        self.batches.append(list(input))
        return SimpleNamespace(embeddings=[[3.0, 4.0] for _ in input])


class TestOllamaEmbedder:
    async def test_embeds_in_batches_and_normalizes(self):
        client = FakeOllamaClient()
        embedder = OllamaEmbedder(client, model="bge-m3", batch_size=2)

        vectors = await embedder.embed(["a", "b", "c", "d", "e"])

        assert [len(b) for b in client.batches] == [2, 2, 1]
        assert vectors[0] == pytest.approx([0.6, 0.8])

    @pytest.mark.parametrize(
        "error",
        [
            ollama.ResponseError("model 'bge-m3' not found", 404),
            httpx.ConnectError("connection refused"),
            ConnectionError("Failed to connect to Ollama"),
        ],
    )
    async def test_backend_failures_become_embedding_unavailable(self, error):
        embedder = OllamaEmbedder(FakeOllamaClient(error), model="bge-m3")

        with pytest.raises(EmbeddingUnavailableError):
            await embedder.embed(["savol"])


class TestVectorStores:
    async def test_chroma_round_trip_persists(self, tmp_path):
        store = ChromaVectorStore(tmp_path)
        await store.add(
            [chunk("a", "x"), chunk("b", "y"), chunk("c", "z")],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

        hits = await store.query([0.9, 0.1, 0.0], limit=2)

        assert [chunk_id for chunk_id, _ in hits] == ["a", "b"]
        assert hits[0][1] == pytest.approx(0.994, abs=1e-3)
        assert await ChromaVectorStore(tmp_path).count() == 3

    async def test_in_memory_store_uses_exact_cosine(self):
        store = InMemoryVectorStore()
        await store.add([chunk("a", "x"), chunk("b", "y")], [[1.0, 0.0], [0.0, 1.0]])

        assert await store.query([0.0, 2.0], limit=1) == [("b", pytest.approx(1.0))]


class TestBM25:
    def test_inflected_query_matches_other_form(self):
        index = BM25Index(
            [
                chunk("hit", "davlat ekologik ekspertizasi va ekspertizaning xulosasi"),
                chunk("miss", "jamoatchilik eshituvi tartibi"),
                chunk("other", "malaka sertifikati berish"),
            ]
        )

        results = index.search("ekspertizasidan", limit=3)

        assert results[0][0] == "hit"
        assert results[0][1] > 0
        assert "miss" not in [chunk_id for chunk_id, _ in results]

    def test_query_without_known_tokens_returns_nothing(self):
        index = BM25Index([chunk("a", "davlat ekologik ekspertizasi"), chunk("b", "boshqa")])

        assert index.search("va", limit=5) == []


class TestFusion:
    def test_rrf_rewards_agreement(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)

        assert rank_by_fused_score(scores, order={}) == ["b", "c", "a", "d"]

    def test_ties_follow_document_order(self):
        scores = reciprocal_rank_fusion([["late"], ["early"]], k=60)

        assert rank_by_fused_score(scores, order={"early": 1, "late": 9}) == ["early", "late"]


ROUTER_IDS = ("q-b6", "q-b7", "a1", "a1-r2", "a2", "a2-b6", "a2-b2-p1", "a2-b2-p2", "a9-b1-p1")


class TestReferenceRouter:
    @pytest.fixture
    def router(self):
        return ReferenceRouter(ROUTER_IDS, chapters={"a2-b6": "1-bob. Umumiy qoidalar"})

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("2-ilovaning 6-bandida nima deyilgan?", ["a2-b6"]),
            ("Qarorning 7-bandi", ["q-b7"]),
            ("1-ilova 2-qator", ["a1-r2"]),
            ("1-ilovaning 2-bandi", ["a1-r2"]),
            ("2-ilova nima haqida?", ["a2"]),
            ("2-ilovaning 2-bandi", ["a2-b2-p1", "a2-b2-p2"]),
            ("2-ilovaning 1-bobi", ["a2-b6"]),
        ],
    )
    def test_explicit_locations(self, router, query, expected):
        assert router.route(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "541-son qarorning 6-bandida nima deyilgan edi?",
            "Aeroport uchun ekspertiza muddati",
            "Nizomning 6-bandi",
        ],
    )
    def test_no_routing_for_other_acts_or_ambiguous_places(self, router, query):
        assert router.route(query) == []
