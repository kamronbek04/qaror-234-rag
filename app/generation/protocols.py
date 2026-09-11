"""Interface of the chat model used for answer generation."""

from typing import Protocol


class ChatModel(Protocol):
    model: str

    async def complete(self, messages: list[dict]) -> str:
        """Raw model output for the conversation (expected to be LLMAnswer JSON)."""
        ...
