"""On-disk index layout: fingerprinted version directories and an atomic CURRENT pointer.

    data/index/
      CURRENT                       -> name of the version being served
      20260911T101500123456-9f2c…/  manifest.json, chunks.jsonl, chroma/

A version is built completely before CURRENT is switched, so readers never see a half-built
index, and the previous version stays on disk for rollback.
"""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from app.core.config import Settings

SCHEMA_VERSION = 1
_POINTER = "CURRENT"
_MANIFEST = "manifest.json"

logger = logging.getLogger(__name__)


class Manifest(BaseModel):
    fingerprint: str
    created_at: str
    chunk_count: int
    embedding_dim: int
    embed_model: str
    chunk_strategy: str
    source_url: str
    source_sha256: str
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class IndexVersion:
    path: Path
    manifest: Manifest

    @property
    def chunks_path(self) -> Path:
        return self.path / "chunks.jsonl"

    @property
    def chroma_path(self) -> Path:
        return self.path / "chroma"


def compute_fingerprint(source: bytes, settings: Settings) -> str:
    chunking: dict[str, object] = {"strategy": settings.chunk_strategy}
    if settings.chunk_strategy == "fixed":
        chunking |= {"size": settings.fixed_chunk_size, "overlap": settings.fixed_chunk_overlap}
    else:
        chunking |= {"max_chars": settings.chunk_max_chars}
    config = {"schema": SCHEMA_VERSION, "embed_model": settings.embed_model, "chunking": chunking}
    digest = hashlib.sha256(source)
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def expected_fingerprint(settings: Settings) -> str:
    return compute_fingerprint(settings.source_html.read_bytes(), settings)


class IndexRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    def current(self) -> IndexVersion | None:
        pointer = self.root / _POINTER
        if not pointer.exists():
            return None
        return self._version(self.root / pointer.read_text(encoding="utf-8").strip())

    def versions(self) -> list[IndexVersion]:
        if not self.root.exists():
            return []
        found = (self._version(p) for p in sorted(self.root.iterdir()) if p.is_dir())
        return [v for v in found if v is not None]

    def new_version_path(self, fingerprint: str) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = self.root / f"{stamp}-{fingerprint}"
        path.mkdir(parents=True)
        return path

    def write_manifest(self, path: Path, manifest: Manifest) -> None:
        (path / _MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def publish(self, path: Path) -> None:
        temporary = self.root / f"{_POINTER}.tmp"
        temporary.write_text(path.name, encoding="utf-8")
        os.replace(temporary, self.root / _POINTER)

    def prune(self, keep: int) -> None:
        """Delete all but the newest `keep` complete versions (the current one is always kept)."""
        current = self.current()
        versions = sorted(self.versions(), key=lambda v: v.path.name, reverse=True)
        keep_paths = {v.path for v in versions[:keep]} | ({current.path} if current else set())
        for candidate in self.root.iterdir():
            if candidate.is_dir() and candidate not in keep_paths:
                try:
                    shutil.rmtree(candidate)
                except OSError as exc:  # e.g. files still open by a running API on Windows
                    logger.warning(
                        "eski indeks o'chirilmadi",
                        extra={"path": str(candidate), "error": str(exc)},
                    )

    @staticmethod
    def _version(path: Path) -> IndexVersion | None:
        manifest_path = path / _MANIFEST
        if not manifest_path.exists():
            return None
        return IndexVersion(path, Manifest.model_validate_json(manifest_path.read_text("utf-8")))
