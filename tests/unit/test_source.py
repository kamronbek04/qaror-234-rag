import httpx
import pytest

from app.core.errors import SourceFetchError
from app.ingestion.source import load_snapshot, refresh_snapshot

URL = "https://lex.uz/uz/docs/-8193120"


def client_returning(status: int, body: str) -> httpx.AsyncClient:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    return httpx.AsyncClient(transport=transport)


def accept_all(html: str) -> None:
    return None


def reject_all(html: str) -> None:
    raise ValueError("no appendices")


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "lex.html"
    path.write_text("OLD", encoding="utf-8")
    return path


async def test_refresh_replaces_snapshot_on_valid_page(snapshot):
    async with client_returning(200, "NEW PAGE") as client:
        await refresh_snapshot(URL, snapshot, client=client, validate=accept_all)

    assert load_snapshot(snapshot) == "NEW PAGE"


async def test_http_error_keeps_old_snapshot(snapshot):
    async with client_returning(503, "down") as client:
        with pytest.raises(SourceFetchError):
            await refresh_snapshot(URL, snapshot, client=client, validate=accept_all)

    assert load_snapshot(snapshot) == "OLD"


async def test_invalid_page_keeps_old_snapshot(snapshot):
    async with client_returning(200, "<html>captcha</html>") as client:
        with pytest.raises(SourceFetchError, match="no appendices"):
            await refresh_snapshot(URL, snapshot, client=client, validate=reject_all)

    assert load_snapshot(snapshot) == "OLD"


def test_missing_snapshot_is_reported(tmp_path):
    with pytest.raises(SourceFetchError):
        load_snapshot(tmp_path / "missing.html")
