"""Composition root: the only place that knows which implementation backs each interface."""

from ollama import AsyncClient

from app.core.config import Settings
from app.retrieval.embedder import OllamaEmbedder


def build_ollama_client(settings: Settings, timeout: float) -> AsyncClient:
    return AsyncClient(host=settings.ollama_base_url, timeout=timeout)


def build_embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        build_ollama_client(settings, settings.embed_timeout_s),
        model=settings.embed_model,
        batch_size=settings.embed_batch_size,
        keep_alive=settings.llm_keep_alive,
    )
