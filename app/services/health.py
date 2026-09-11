"""Readiness report: Ollama reachability, required models and index state."""

from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
import ollama
from pydantic import BaseModel

from app.core.errors import AppError, LLMUnavailableError

if TYPE_CHECKING:
    from app.core.container import Container


class OllamaStatus(BaseModel):
    url: str
    reachable: bool


class IndexStatus(BaseModel):
    ready: bool
    stale: bool
    fingerprint: str | None = None
    chunk_count: int | None = None
    built_at: str | None = None
    embed_model: str | None = None
    source_sha256: str | None = None
    source_snapshot_date: str
    error: str | None = None


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: OllamaStatus
    models: dict[str, bool]
    index: IndexStatus
    problems: list[str]


class ModelCatalog(Protocol):
    async def available(self) -> set[str]: ...


class OllamaModelCatalog:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def available(self) -> set[str]:
        try:
            response = await self._client.list()
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError() from exc
        return {model.model for model in response.models}


def has_model(available: set[str], name: str) -> bool:
    return name in available or (":" not in name and f"{name}:latest" in available)


async def check_health(container: "Container") -> HealthReport:
    settings = container.settings
    problems: list[str] = []
    try:
        available = await container.catalog.available() if container.catalog else set()
        reachable = container.catalog is not None
    except AppError:
        available, reachable = set(), False
    if not reachable:
        problems.append("ollama_unreachable")

    models = {
        name: has_model(available, name) for name in (settings.llm_model, settings.embed_model)
    }
    problems += [f"model_missing:{name}" for name, ok in models.items() if reachable and not ok]

    state = container.index
    if state.stale:
        problems.append("index_stale")
    elif not state.ready:
        problems.append("index_not_ready")

    manifest = state.manifest
    return HealthReport(
        status="ok" if not problems else "degraded",
        ollama=OllamaStatus(url=settings.ollama_base_url, reachable=reachable),
        models=models,
        index=IndexStatus(
            ready=state.ready,
            stale=state.stale,
            fingerprint=manifest.fingerprint if manifest else None,
            chunk_count=manifest.chunk_count if manifest else None,
            built_at=manifest.created_at if manifest else None,
            embed_model=manifest.embed_model if manifest else None,
            source_sha256=manifest.source_sha256 if manifest else None,
            source_snapshot_date=settings.source_snapshot_date,
            error=state.error,
        ),
        problems=problems,
    )
