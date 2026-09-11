from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import SourceFetchError
from app.ingestion.index_store import IndexRepository, expected_fingerprint
from app.ingestion.pipeline import build_index, load_document, make_chunker
from tests.fakes import FakeEmbedder

SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "lex_8193120.html"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    fields = {
        "index_dir": tmp_path / "index",
        "source_html": SNAPSHOT,
        "embed_model": "fake-embed",
        "index_keep_versions": 2,
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)


async def test_build_publishes_a_complete_version(tmp_path):
    settings = make_settings(tmp_path)

    manifest = await build_index(settings, embedder=FakeEmbedder())

    current = IndexRepository(settings.index_dir).current()
    assert current is not None
    assert current.manifest == manifest
    assert manifest.chunk_count == len(make_chunker(settings).chunk(load_document(settings)))
    assert manifest.embedding_dim == 256
    assert manifest.fingerprint == expected_fingerprint(settings)
    assert (current.path / "chunks.jsonl").exists()
    assert (current.path / "chroma").is_dir()


async def test_unchanged_inputs_reuse_the_index_without_embedding(tmp_path):
    settings = make_settings(tmp_path)
    embedder = FakeEmbedder()
    first = await build_index(settings, embedder=embedder)
    calls = embedder.calls

    second = await build_index(settings, embedder=embedder)

    assert second == first
    assert embedder.calls == calls


async def test_changed_configuration_swaps_current_and_prunes_old_versions(tmp_path):
    repository = IndexRepository(tmp_path / "index")
    fingerprints = []
    for max_chars in (1500, 1200, 900):
        settings = make_settings(tmp_path, chunk_max_chars=max_chars)
        fingerprints.append((await build_index(settings, embedder=FakeEmbedder())).fingerprint)

    assert len(set(fingerprints)) == 3
    assert repository.current().manifest.fingerprint == fingerprints[-1]
    assert sorted(v.manifest.fingerprint for v in repository.versions()) == sorted(fingerprints[1:])


async def test_incomplete_version_is_not_served(tmp_path):
    repository = IndexRepository(tmp_path / "index")
    (tmp_path / "index" / "broken").mkdir(parents=True)
    (tmp_path / "index" / "CURRENT").write_text("broken", encoding="utf-8")

    assert repository.current() is None


async def test_failed_refresh_leaves_the_index_untouched(tmp_path):
    snapshot = tmp_path / "lex.html"
    snapshot.write_bytes(SNAPSHOT.read_bytes())
    settings = make_settings(tmp_path, source_html=snapshot)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html></html>"))

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SourceFetchError):
            await build_index(settings, embedder=FakeEmbedder(), refresh=True, http_client=client)

    assert IndexRepository(settings.index_dir).current() is None
    assert snapshot.read_bytes() == SNAPSHOT.read_bytes()
