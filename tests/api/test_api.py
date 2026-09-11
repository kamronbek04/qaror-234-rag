import asyncio
from time import perf_counter

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import Container, IndexState
from app.core.errors import EmbeddingUnavailableError, LLMUnavailableError
from app.domain.models import REFUSAL_TEXT, RetrievalResult, ScoredChunk
from app.generation.guard import AnswerGuard
from app.ingestion.chunker import StructuralChunker
from app.ingestion.index_store import Manifest
from app.main import create_app
from app.retrieval.chunk_store import ChunkStore
from app.services.rag_service import RagService
from tests.fakes import FakeChatModel, StubRetriever, reply

AIRPORT = "Aeroportlar uchun muddat 25 ish kuni, toʻlov 25 BXM [a1-r2]."
MODELS = {"qwen2.5:7b", "bge-m3:latest"}


class FakeCatalog:
    def __init__(self, available=MODELS, error=None):
        self._available = set(available)
        self._error = error

    async def available(self):
        if self._error:
            raise self._error
        return self._available


class CountingRetriever(StubRetriever):
    def __init__(self, result, error=None):
        super().__init__(result)
        self.calls = 0
        self.error = error

    async def retrieve(self, query, top_k=None):
        self.calls += 1
        if self.error:
            raise self.error
        return await super().retrieve(query, top_k)


@pytest.fixture(scope="module")
def store(document):
    return ChunkStore(StructuralChunker(max_chars=1500).chunk(document))


@pytest.fixture
def settings():
    return Settings(_env_file=None)


def manifest() -> Manifest:
    return Manifest(
        fingerprint="abc123", created_at="2026-09-11T08:00:00+00:00", chunk_count=612,
        embedding_dim=1024, embed_model="bge-m3", chunk_strategy="structural",
        source_url="https://lex.uz/uz/docs/-8193120", source_sha256="0" * 64,
    )  # fmt: skip


def make_container(
    settings, store, chat=None, retriever_error=None, index=None, catalog=None, concurrency=2
):
    result = RetrievalResult(
        query="",
        chunks=[
            ScoredChunk(chunk=store.get("a1-r2"), dense_score=0.83, fused_score=0.03, rank=1),
            ScoredChunk(chunk=store.get("q-b7"), dense_score=0.61, fused_score=0.02, rank=2),
        ],
        top_similarity=0.83,
    )
    retriever = CountingRetriever(result, error=retriever_error)
    chat = chat or FakeChatModel(reply(answer=AIRPORT, citations=["a1-r2"]))
    rag = RagService(
        retriever=retriever, chat_model=chat, guard=AnswerGuard(), refusal_threshold=0.5,
        max_prompt_tokens=6000, max_concurrency=concurrency,
    )  # fmt: skip
    return Container(
        settings=settings,
        index=index or IndexState(ready=True, stale=False, manifest=manifest()),
        store=store,
        retriever=retriever,
        rag=rag,
        catalog=catalog or FakeCatalog(),
    )


def client_for(container):
    app = create_app(settings=container.settings, container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class TestAsk:
    async def test_covered_question(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            response = await client.post("/api/v1/ask", json={"question": "Ekolog-ekspert kim?"})

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "answered"
        assert body["found"] is True
        assert body["sources"][0]["url"].startswith("https://lex.uz/uz/docs/-8193120#")
        assert body["sources"][0]["chunk_id"] == "a1-r2"
        assert set(body["meta"]["timings_ms"]) == {"retrieval", "generation", "total"}
        assert body["meta"]["model"] == "fake-llm"
        assert "debug" not in body

    async def test_uncovered_question(self, settings, store):
        container = make_container(settings, store, chat=FakeChatModel(reply(status="not_found")))
        async with client_for(container) as client:
            response = await client.post("/api/v1/ask", json={"question": "QQS necha foiz?"})

        body = response.json()
        assert response.status_code == 200
        assert body["answer"] == REFUSAL_TEXT
        assert (body["status"], body["found"], body["sources"]) == ("not_found", False, [])

    async def test_debug_output_includes_retrieval_and_verification(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            response = await client.post("/api/v1/ask", json={"question": "savol", "debug": True})

        debug = response.json()["debug"]
        assert debug["retrieval"][0]["chunk_id"] == "a1-r2"
        assert debug["attempts"][0]["ok"] is True

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": "   "},
            {"question": "x" * 1001},
            {"question": "savol", "top_k": 0},
            {"question": "savol", "top_k": 21},
            {},
        ],
    )
    async def test_invalid_input_is_rejected_before_retrieval(self, settings, store, payload):
        container = make_container(settings, store)
        async with client_for(container) as client:
            response = await client.post("/api/v1/ask", json=payload)

        assert response.status_code == 422
        assert container.retriever.calls == 0

    async def test_llm_down_is_503(self, settings, store):
        container = make_container(settings, store, chat=FakeChatModel(LLMUnavailableError()))
        async with client_for(container) as client:
            response = await client.post("/api/v1/ask", json={"question": "savol"})

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "llm_unavailable"

    async def test_embedding_down_is_503(self, settings, store):
        container = make_container(settings, store, retriever_error=EmbeddingUnavailableError())
        async with client_for(container) as client:
            response = await client.post("/api/v1/ask", json={"question": "savol"})

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "embedding_unavailable"

    async def test_index_not_ready_is_503(self, settings, store):
        container = make_container(settings, store)
        container.rag = None
        container.retriever = None
        async with client_for(container) as client:
            response = await client.post("/api/v1/ask", json={"question": "savol"})

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "index_not_ready"

    async def test_request_id_round_trip(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            response = await client.post(
                "/api/v1/ask", json={"question": "savol"}, headers={"X-Request-ID": "demo-1"}
            )
            generated = await client.post("/api/v1/ask", json={"question": "savol"})

        assert response.headers["X-Request-ID"] == "demo-1"
        assert response.json()["meta"]["request_id"] == "demo-1"
        assert generated.headers["X-Request-ID"]


class TestSearchAndChunks:
    async def test_search_returns_scores_without_llm(self, settings, store):
        container = make_container(settings, store)
        async with client_for(container) as client:
            response = await client.post("/api/v1/search", json={"query": "aeroport", "top_k": 5})

        body = response.json()
        assert response.status_code == 200
        assert body["results"][0]["chunk_id"] == "a1-r2"
        assert {"dense_score", "lexical_score", "fused_score", "url"} <= set(body["results"][0])
        assert container.rag._chat.conversations == []

    async def test_chunk_lookup(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            found = await client.get("/api/v1/chunks/a2-b6")
            missing = await client.get("/api/v1/chunks/does-not-exist")

        assert found.status_code == 200
        assert found.json()["breadcrumb"].endswith("6-band")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "chunk_not_found"


class TestHealth:
    async def test_everything_ready(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            response = await client.get("/health")

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "ok"
        assert body["index"]["chunk_count"] == 612

    async def test_missing_model_is_named(self, settings, store):
        container = make_container(settings, store, catalog=FakeCatalog({"bge-m3:latest"}))
        async with client_for(container) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        assert "model_missing:qwen2.5:7b" in response.json()["problems"]

    async def test_unreachable_ollama(self, settings, store):
        container = make_container(
            settings, store, catalog=FakeCatalog(error=LLMUnavailableError())
        )
        async with client_for(container) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        assert response.json()["ollama"]["reachable"] is False

    async def test_stale_index(self, settings, store):
        stale = IndexState(ready=False, stale=True, manifest=manifest())
        async with client_for(make_container(settings, store, index=stale)) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        assert "index_stale" in response.json()["problems"]


class TestConcurrency:
    async def test_generations_are_bounded_and_health_stays_responsive(self, settings, store):
        chat = FakeChatModel(reply(answer=AIRPORT, citations=["a1-r2"]), delay=0.3)
        async with client_for(make_container(settings, store, chat=chat)) as client:
            asks = [
                asyncio.create_task(client.post("/api/v1/ask", json={"question": f"savol {i}"}))
                for i in range(5)
            ]
            await asyncio.sleep(0.05)
            started = perf_counter()
            health = await client.get("/health")
            health_seconds = perf_counter() - started
            responses = await asyncio.gather(*asks)

        assert all(r.status_code == 200 for r in responses)
        assert chat.max_active == 2
        assert health.status_code == 200
        assert health_seconds < 1.0


class TestPagesAndDocs:
    async def test_demo_page_is_self_contained(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            response = await client.get("/")

        html = response.text
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert 'id="ask-form"' in html
        assert "<script src=" not in html
        assert 'rel="stylesheet"' not in html

    async def test_openapi_has_examples(self, settings, store):
        async with client_for(make_container(settings, store)) as client:
            schema = (await client.get("/openapi.json")).json()

        ask_body = schema["paths"]["/api/v1/ask"]["post"]["requestBody"]["content"]
        assert ask_body["application/json"]["examples"]
        assert schema["paths"]["/api/v1/search"]["post"]["requestBody"]["content"][
            "application/json"
        ]["examples"]


def test_lifespan_builds_the_container_from_the_factory(settings, store):
    built = []

    async def factory(received_settings):
        built.append(received_settings)
        return make_container(received_settings, store)

    with TestClient(create_app(settings=settings, container_factory=factory)) as client:
        response = client.get("/health")

    assert built == [settings]
    assert response.status_code == 200
