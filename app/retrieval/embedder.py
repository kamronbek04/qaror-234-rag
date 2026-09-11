"""Dense embeddings through the local Ollama server."""

import math
from collections.abc import Sequence
from typing import Any

import httpx
import ollama

from app.core.errors import EmbeddingUnavailableError


class OllamaEmbedder:
    def __init__(
        self, client: Any, *, model: str, batch_size: int = 32, keep_alive: str = "30m"
    ) -> None:
        self._client = client
        self.model = model
        self.batch_size = batch_size
        self.keep_alive = keep_alive

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = await self._client.embed(
                    model=self.model, input=batch, keep_alive=self.keep_alive
                )
            except ollama.ResponseError as exc:
                raise EmbeddingUnavailableError(
                    f"Embedding modeli '{self.model}' xatolik qaytardi: {exc.error}"
                ) from exc
            except (httpx.HTTPError, ConnectionError, OSError) as exc:
                raise EmbeddingUnavailableError() from exc
            vectors.extend(_unit(vector) for vector in response.embeddings)
        return vectors


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
