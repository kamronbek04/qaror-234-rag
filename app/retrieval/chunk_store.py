"""In-memory chunk lookup, the single source of truth loaded from chunks.jsonl."""

from collections.abc import Iterable
from pathlib import Path

from app.core.errors import ChunkNotFoundError
from app.domain.models import Chunk


class ChunkStore:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._by_id = {c.id: c for c in self._chunks}
        self.order = {c.id: c.order for c in self._chunks}

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    def all(self) -> list[Chunk]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> Chunk:
        try:
            return self._by_id[chunk_id]
        except KeyError:
            raise ChunkNotFoundError(chunk_id) from None

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as file:
            for chunk in self._chunks:
                file.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        with path.open(encoding="utf-8") as file:
            return cls(Chunk.model_validate_json(line) for line in file if line.strip())
