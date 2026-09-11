"""Deterministic stand-ins for Ollama used by unit and API tests."""

import asyncio
import hashlib
import json
import math
from collections.abc import Sequence

from app.domain.models import RetrievalResult
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


def reply(status: str = "answered", answer: str = "", citations: Sequence[str] = ()) -> str:
    return json.dumps({"status": status, "answer": answer, "citations": list(citations)})


class FakeChatModel:
    """Returns scripted replies in order and records every conversation it receives."""

    def __init__(self, *replies: str | Exception, delay: float = 0.0, model: str = "fake-llm"):
        self.model = model
        self.replies = list(replies)
        self.delay = delay
        self.conversations: list[list[dict]] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, messages: list[dict]) -> str:
        self.conversations.append(messages)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            result = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self.active -= 1


class StubRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        return self.result.model_copy(update={"query": query})
