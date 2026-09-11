from pathlib import Path

import pytest

from app.ingestion.lex_parser import parse_document

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "raw" / "lex_8193120.html"


@pytest.fixture(scope="session")
def snapshot_html() -> str:
    return SNAPSHOT.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def document(snapshot_html):
    return parse_document(snapshot_html, url="https://lex.uz/uz/docs/-8193120")
