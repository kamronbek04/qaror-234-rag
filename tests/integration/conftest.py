"""Fixtures that talk to a real Ollama server; the tests are skipped when it is not reachable."""

import httpx
import pytest
from ollama import AsyncClient

from app.core.config import load_settings
from app.ingestion.chunker import StructuralChunker
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.embedder import OllamaEmbedder
from app.retrieval.lexical import BM25Index
from app.retrieval.references import ReferenceRouter
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import InMemoryVectorStore


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def ollama_models(settings):
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        response.raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not reachable")
    return {m["name"] for m in response.json().get("models", [])}


@pytest.fixture(scope="session")
def require_embedding_model(settings, ollama_models):
    if not any(name.split(":")[0] == settings.embed_model.split(":")[0] for name in ollama_models):
        pytest.skip(f"embedding model {settings.embed_model} is not pulled")


@pytest.fixture(scope="session")
async def real_retriever(settings, require_embedding_model, document):
    embedder = OllamaEmbedder(
        AsyncClient(host=settings.ollama_base_url, timeout=settings.embed_timeout_s),
        model=settings.embed_model,
        batch_size=settings.embed_batch_size,
    )
    store = ChunkStore(StructuralChunker(settings.chunk_max_chars).chunk(document))
    chunks = store.all()
    vectors = InMemoryVectorStore()
    await vectors.add(chunks, await embedder.embed([c.embedding_text for c in chunks]))
    return HybridRetriever(
        store=store,
        embedder=embedder,
        vector_store=vectors,
        lexical=BM25Index(chunks),
        router=ReferenceRouter.from_store(store),
        top_k=settings.retrieval_top_k,
        candidates=settings.retrieval_candidates,
        expansion_max=settings.expansion_max,
        token_budget=settings.context_token_budget,
    )
