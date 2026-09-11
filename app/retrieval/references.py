"""Routes questions that name an exact place in the resolution ("2-ilovaning 6-bandi")."""

import re
from collections.abc import Iterable, Mapping

from app.retrieval.chunk_store import ChunkStore
from app.text.normalize import lexical_form

_APPENDIX = re.compile(r"\b(\d+)\s*-\s*ilova")
_ITEM = re.compile(r"\b(\d+(?:-\d+)?)\s*-\s*band")
_CHAPTER = re.compile(r"\b(\d+)\s*-\s*bob")
_ROW = re.compile(r"\b(\d+)\s*-\s*qator")
_ACT_NUMBER = re.compile(r"\b(\d+)\s*-\s*son")
_THIS_ACT = "234"


class ReferenceRouter:
    def __init__(self, chunk_ids: Iterable[str], chapters: Mapping[str, str] | None = None) -> None:
        self._ids = list(chunk_ids)
        self._known = set(self._ids)
        self._chapters = dict(chapters or {})

    @classmethod
    def from_store(cls, store: ChunkStore) -> "ReferenceRouter":
        chunks = store.all()
        return cls([c.id for c in chunks], {c.id: c.chapter for c in chunks if c.chapter})

    def route(self, query: str) -> list[str]:
        """Chunk ids the question points at explicitly, in document order; [] when none."""
        text = lexical_form(query)
        if any(number != _THIS_ACT for number in _ACT_NUMBER.findall(text)):
            return []  # the question is about another act (e.g. resolution No. 541)
        appendix = _first(_APPENDIX, text)
        item = _first(_ITEM, text)
        chapter = _first(_CHAPTER, text)
        row = _first(_ROW, text)

        if row and appendix in (None, "1"):
            return self._resolve(f"a1-r{row}")
        if appendix == "1" and item:
            return self._resolve(f"a1-r{item}")
        if appendix and item:
            return self._resolve(f"a{appendix}-b{item}")
        if appendix and chapter:
            prefix = f"a{appendix}-"
            return [
                chunk_id
                for chunk_id in self._ids
                if chunk_id.startswith(prefix)
                and self._chapters.get(chunk_id, "").startswith(f"{chapter}-bob")
            ]
        if appendix:
            return self._resolve(f"a{appendix}")
        if item and "nizom" not in text:
            return self._resolve(f"q-b{item}")
        return []

    def _resolve(self, logical_id: str) -> list[str]:
        if logical_id in self._known:
            return [logical_id]
        return [i for i in self._ids if i.startswith(logical_id + "-p")]


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None
