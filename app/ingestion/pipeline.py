"""Ingestion: snapshot → parse → validate → chunk → embed → versioned index."""

import hashlib
import logging
from datetime import UTC, datetime
from typing import Protocol

import httpx

from app.core.config import Settings
from app.domain.models import Chunk, Document
from app.ingestion.chunker import StructuralChunker
from app.ingestion.fixed_chunker import FixedSizeChunker
from app.ingestion.index_store import IndexRepository, Manifest, compute_fingerprint
from app.ingestion.lex_parser import parse_document, validate_document
from app.ingestion.source import load_snapshot, refresh_snapshot
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.protocols import Embedder
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


def make_chunker(settings: Settings) -> Chunker:
    if settings.chunk_strategy == "fixed":
        return FixedSizeChunker(settings.fixed_chunk_size, settings.fixed_chunk_overlap)
    return StructuralChunker(settings.chunk_max_chars)


def load_document(settings: Settings) -> Document:
    document = parse_document(load_snapshot(settings.source_html), url=settings.source_url)
    validate_document(document)
    return document


async def build_index(
    settings: Settings,
    *,
    embedder: Embedder,
    refresh: bool = False,
    force: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> Manifest:
    if refresh:
        await _refresh(settings, http_client)

    source = settings.source_html.read_bytes()
    document = load_document(settings)
    chunks = make_chunker(settings).chunk(document)
    fingerprint = compute_fingerprint(source, settings)

    repository = IndexRepository(settings.index_dir)
    current = repository.current()
    if current and current.manifest.fingerprint == fingerprint and not force:
        logger.info("indeks dolzarb, qayta qurilmaydi", extra={"fingerprint": fingerprint})
        return current.manifest

    logger.info("indeks qurilmoqda", extra={"fingerprint": fingerprint, "chunks": len(chunks)})
    vectors = await embedder.embed([c.embedding_text for c in chunks])
    path = repository.new_version_path(fingerprint)
    vector_store = ChromaVectorStore(path / "chroma")
    try:
        await vector_store.add(chunks, vectors)
    finally:
        vector_store.close()
    ChunkStore(chunks).save(path / "chunks.jsonl")
    manifest = Manifest(
        fingerprint=fingerprint,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        chunk_count=len(chunks),
        embedding_dim=len(vectors[0]) if vectors else 0,
        embed_model=settings.embed_model,
        chunk_strategy=settings.chunk_strategy,
        source_url=settings.source_url,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )
    repository.write_manifest(path, manifest)
    repository.publish(path)
    repository.prune(settings.index_keep_versions)
    logger.info("indeks tayyor", extra={"fingerprint": fingerprint, "path": str(path)})
    return manifest


async def _refresh(settings: Settings, client: httpx.AsyncClient | None) -> None:
    def validate(html: str) -> None:
        validate_document(parse_document(html, url=settings.source_url))

    if client is not None:
        await refresh_snapshot(
            settings.source_url, settings.source_html, client=client, validate=validate
        )
        return
    async with httpx.AsyncClient() as own_client:
        await refresh_snapshot(
            settings.source_url, settings.source_html, client=own_client, validate=validate
        )
