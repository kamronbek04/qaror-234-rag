"""Vector stores: persistent ChromaDB for the service, exact in-memory search for tests."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.domain.models import Chunk

_COLLECTION = "chunks"
_MAX_BATCH = 1000


class ChromaVectorStore:
    """Embedded ChromaDB with cosine distance; synchronous calls run off the event loop."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            _COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )

    def close(self) -> None:
        """Release the database files (Windows keeps open files locked)."""
        self._client.close()

    async def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        await asyncio.to_thread(self._add, list(chunks), [list(e) for e in embeddings])

    def _add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for start in range(0, len(chunks), _MAX_BATCH):
            batch = chunks[start : start + _MAX_BATCH]
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings[start : start + _MAX_BATCH],
                metadatas=[{"type": c.type.value, "section": c.section} for c in batch],
            )

    async def query(self, embedding: Sequence[float], limit: int) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._query, list(embedding), limit)

    def _query(self, embedding: list[float], limit: int) -> list[tuple[str, float]]:
        size = min(limit, self._collection.count())
        if size == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding], n_results=size, include=["distances"]
        )
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)


class InMemoryVectorStore:
    """Exact cosine search over a numpy matrix."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix = np.zeros((0, 0))

    async def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        rows = np.asarray(embeddings, dtype=float)
        rows = rows / np.clip(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12, None)
        self._matrix = rows if not self._ids else np.vstack([self._matrix, rows])
        self._ids.extend(c.id for c in chunks)

    async def query(self, embedding: Sequence[float], limit: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        vector = np.asarray(embedding, dtype=float)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        similarities = self._matrix @ vector
        best = sorted(range(len(self._ids)), key=lambda i: (-similarities[i], i))[:limit]
        return [(self._ids[i], float(similarities[i])) for i in best]

    async def count(self) -> int:
        return len(self._ids)
