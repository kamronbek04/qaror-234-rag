"""Deterministic stand-ins for Ollama used by unit and API tests."""

import hashlib
import math
from collections.abc import Sequence

from app.text.stemmer import tokenize

DIMENSIONS = 256


class FakeEmbedder:
    """Hashed bag-of-stems vectors: similar wording gives similar vectors, no model needed."""

    def __init__(self, model: str = "fake-embed") -> None:
        self.model = model
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in tokenize(text):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMENSIONS
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
