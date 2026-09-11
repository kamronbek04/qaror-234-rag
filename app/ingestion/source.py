"""Loading the committed lex.uz snapshot and refreshing it from the live site."""

import os
from collections.abc import Callable
from pathlib import Path

import httpx

from app.core.errors import SourceFetchError

_USER_AGENT = "Mozilla/5.0 (compatible; qaror-234-rag/0.1)"


def load_snapshot(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SourceFetchError(f"Manba nusxasi topilmadi: {path}") from exc


async def refresh_snapshot(
    url: str,
    path: Path,
    *,
    client: httpx.AsyncClient,
    validate: Callable[[str], None],
) -> Path:
    """Download the page, validate it, and only then atomically replace the snapshot."""
    try:
        response = await client.get(
            url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True, timeout=60
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SourceFetchError(f"lex.uz'dan yuklab bo'lmadi: {exc}") from exc

    try:
        validate(response.text)
    except Exception as exc:
        raise SourceFetchError(f"Yuklangan sahifa tekshiruvdan o'tmadi: {exc}") from exc

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(response.content)
    os.replace(temporary, path)
    return path
