"""Interfaces the retrieval pipeline depends on; implementations are wired in the container."""

from collections.abc import Sequence
from typing import Protocol

from app.domain.models import Chunk, RetrievalResult


class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    async def query(self, embedding: Sequence[float], limit: int) -> list[tuple[str, float]]:
        """Nearest chunks as (chunk id, cosine similarity), best first."""
        ...

    async def count(self) -> int: ...


class LexicalIndex(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult: ...
