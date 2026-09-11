"""Composition root: the only place that knows which implementation backs each interface."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ollama import AsyncClient

from app.core.config import Settings
from app.core.errors import AppError, IndexNotReadyError
from app.generation.guard import AnswerGuard
from app.generation.ollama_chat import OllamaChatModel
from app.ingestion.index_store import IndexRepository, Manifest, expected_fingerprint
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.embedder import OllamaEmbedder
from app.retrieval.lexical import BM25Index
from app.retrieval.protocols import Retriever
from app.retrieval.references import ReferenceRouter
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore
from app.services.health import ModelCatalog, OllamaModelCatalog
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

_PROMPT_SAFETY_MARGIN = 256


@dataclass
class IndexState:
    ready: bool
    stale: bool
    manifest: Manifest | None = None
    error: str | None = None


@dataclass
class Container:
    settings: Settings
    index: IndexState
    store: ChunkStore | None = None
    retriever: Retriever | None = None
    rag: RagService | None = None
    catalog: ModelCatalog | None = None
    closers: list[Callable[[], None]] = field(default_factory=list)

    def require_rag(self) -> RagService:
        if self.rag is None:
            raise IndexNotReadyError()
        return self.rag

    def require_retriever(self) -> Retriever:
        if self.retriever is None:
            raise IndexNotReadyError()
        return self.retriever

    def require_store(self) -> ChunkStore:
        if self.store is None:
            raise IndexNotReadyError()
        return self.store

    def close(self) -> None:
        for close in self.closers:
            close()


def build_ollama_client(settings: Settings, timeout: float) -> AsyncClient:
    return AsyncClient(host=settings.ollama_base_url, timeout=timeout)


def build_embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        build_ollama_client(settings, settings.embed_timeout_s),
        model=settings.embed_model,
        batch_size=settings.embed_batch_size,
        keep_alive=settings.llm_keep_alive,
    )


def build_chat_model(settings: Settings, model: str | None = None) -> OllamaChatModel:
    return OllamaChatModel(
        build_ollama_client(settings, settings.llm_timeout_s),
        model=model or settings.llm_model,
        num_ctx=settings.llm_num_ctx,
        num_predict=settings.llm_num_predict,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        keep_alive=settings.llm_keep_alive,
        think=settings.llm_think,
    )


def build_rag_service(
    settings: Settings, retriever: Retriever, chat_model: OllamaChatModel
) -> RagService:
    return RagService(
        retriever=retriever,
        chat_model=chat_model,
        guard=AnswerGuard(),
        refusal_threshold=settings.refusal_threshold,
        max_prompt_tokens=settings.llm_num_ctx - settings.llm_num_predict - _PROMPT_SAFETY_MARGIN,
        max_concurrency=settings.llm_max_concurrency,
    )


def build_retriever(
    settings: Settings, store: ChunkStore, vectors: ChromaVectorStore, embedder: OllamaEmbedder
) -> HybridRetriever:
    return HybridRetriever(
        store=store,
        embedder=embedder,
        vector_store=vectors,
        lexical=BM25Index(store.all()),
        router=ReferenceRouter.from_store(store),
        top_k=settings.retrieval_top_k,
        candidates=settings.retrieval_candidates,
        expansion_max=settings.expansion_max,
        token_budget=settings.context_token_budget,
    )


async def build_container(settings: Settings) -> Container:
    """Open the current index (building it first when allowed) and wire every service."""
    from app.ingestion.pipeline import build_index  # imported lazily: heavy and CLI-shared

    catalog = OllamaModelCatalog(build_ollama_client(settings, timeout=5))
    embedder = build_embedder(settings)
    repository = IndexRepository(settings.index_dir)
    current = repository.current()
    stale = current is not None and current.manifest.fingerprint != expected_fingerprint(settings)
    error = None

    if (current is None or stale) and settings.auto_ingest:
        try:
            await build_index(settings, embedder=embedder)
            current, stale = repository.current(), False
        except AppError as exc:
            logger.error("indeksni qurib bo'lmadi", extra={"code": exc.code, "error": exc.message})
            error = exc.message

    if current is None or stale:
        state = IndexState(ready=False, stale=stale, manifest=current.manifest if current else None)
        state.error = error
        return Container(settings=settings, index=state, catalog=catalog)

    store = ChunkStore.load(current.chunks_path)
    vectors = ChromaVectorStore(current.chroma_path)
    retriever = build_retriever(settings, store, vectors, embedder)
    return Container(
        settings=settings,
        index=IndexState(ready=True, stale=False, manifest=current.manifest),
        store=store,
        retriever=retriever,
        rag=build_rag_service(settings, retriever, build_chat_model(settings)),
        catalog=catalog,
        closers=[vectors.close],
    )
