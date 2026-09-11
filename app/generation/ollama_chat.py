"""Chat completions through the local Ollama server with schema-constrained output."""

from typing import Any

import httpx
import ollama

from app.core.errors import LLMUnavailableError
from app.generation.prompts import LLMAnswer

_ANSWER_SCHEMA = LLMAnswer.model_json_schema()


class OllamaChatModel:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        seed: int,
        keep_alive: str,
        think: bool | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._options = {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,  # always explicit: the server default would truncate excerpts
            "num_predict": num_predict,
        }
        self._keep_alive = keep_alive
        self._think = think

    async def complete(self, messages: list[dict]) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "format": _ANSWER_SCHEMA,
            "options": dict(self._options),
            "keep_alive": self._keep_alive,
        }
        if self._think is not None:
            request["think"] = self._think
        try:
            response = await self._client.chat(**request)
        except ollama.ResponseError as exc:
            raise LLMUnavailableError(
                f"Til modeli '{self.model}' xatolik qaytardi: {exc.error}"
            ) from exc
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError() from exc
        return response.message.content or ""
