"""Deterministic verification of model answers against the excerpts they cite.

Nothing the model says reaches the user unless it passes these checks: the refusal text always
comes from code, citations must point at supplied excerpts, every number must be found in the
cited text, and neither spelled-out numbers nor currency words may appear unless the cited text
uses them too (this is how "25 BXM is 25 dollars" style conversions are caught).
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.domain.models import REFUSAL_TEXT, AnswerStatus, Chunk, ScoredChunk
from app.generation.prompts import LLMAnswer
from app.text.normalize import lexical_form
from app.text.numbers import DOCUMENT_NUMBER, extract_numbers

_CHUNK_ID = r"(?:q|a\d+|w)(?:-[a-z0-9]+)*"
_MARKER = re.compile(rf"\[\s*({_CHUNK_ID}(?:\s*,\s*{_CHUNK_ID})*)\s*\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")
_SPACES = re.compile(r"[ \t]{2,}")
_WORD = re.compile(r"[a-z']+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Spelled-out numbers ("bir" and "yuz" are left out: they are everyday words too).
_NUMBER_WORDS = frozenset(
    {
        "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz", "o'n",
        "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson",
        "ming", "million", "milliard",
    }
)  # fmt: skip
_CURRENCY_PREFIXES = ("dollar", "usd", "evro", "yevro", "euro", "rubl", "so'm")

PARTIAL_SUFFIX = f"Savolning qolgan qismi boʻyicha: {REFUSAL_TEXT}."


class Verdict(BaseModel):
    ok: bool
    status: AnswerStatus
    text: str
    reason: str | None = None
    citations: list[str] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    unsupported_terms: list[str] = Field(default_factory=list)


class AnswerGuard:
    def check(
        self, draft: LLMAnswer, excerpts: Sequence[ScoredChunk], question: str = ""
    ) -> Verdict:
        if draft.status == "not_found" or not draft.answer.strip():
            return _refusal("model_not_found")

        supplied = {item.chunk.id: item.chunk for item in excerpts}
        claimed = list(dict.fromkeys([*draft.citations, *_marker_ids(draft.answer)]))
        valid = [c for c in claimed if c in supplied]
        dropped = [c for c in claimed if c not in supplied]
        if not valid:
            return _refusal("no_valid_citations", dropped_citations=dropped)

        text = _clean_markers(draft.answer.strip(), set(valid))
        claims = _MARKER.sub(" ", text)
        source = "\n".join(_source_text(supplied[c]) for c in valid)
        numbers = sorted(
            extract_numbers(claims) - extract_numbers(source) - {DOCUMENT_NUMBER}, key=float
        )
        terms = _unsupported_terms(_without_disclaimers(claims), source)
        if numbers or terms:
            return _refusal(
                "unsupported_numbers" if numbers else "unsupported_terms",
                citations=valid,
                dropped_citations=dropped,
                unsupported_numbers=numbers,
                unsupported_terms=terms,
            )

        status = AnswerStatus.PARTIAL if draft.status == "partial" else AnswerStatus.ANSWERED
        if _unsupported_terms(question, source, number_words=False):
            status = AnswerStatus.PARTIAL  # the question asks for a unit the document never uses
        if status is AnswerStatus.PARTIAL and lexical_form(REFUSAL_TEXT) not in lexical_form(text):
            text = f"{text} {PARTIAL_SUFFIX}"
        return Verdict(
            ok=True, status=status, text=text, citations=valid, dropped_citations=dropped
        )


def _source_text(chunk: Chunk) -> str:
    return f"{chunk.breadcrumb}\n{chunk.text}"


def _without_disclaimers(claims: str) -> str:
    """Drop sentences that only say the document lacks something (they claim nothing)."""
    refusal = lexical_form(REFUSAL_TEXT)
    sentences = _SENTENCE_END.split(claims)
    return " ".join(s for s in sentences if refusal not in lexical_form(s))


def _unsupported_terms(claims: str, source: str, *, number_words: bool = True) -> list[str]:
    source_words = set(_WORD.findall(lexical_form(source)))
    found: list[str] = []
    for raw in _WORD.findall(lexical_form(claims)):
        word = raw.strip("'")
        if word in _NUMBER_WORDS:
            if not number_words:
                continue
            supported = word in source_words
        else:
            prefix = next((p for p in _CURRENCY_PREFIXES if word.startswith(p)), None)
            if prefix is None:
                continue
            supported = any(w.startswith(prefix) for w in source_words)
        if not supported and word not in found:
            found.append(word)
    return found


def _refusal(reason: str, **details: list[str]) -> Verdict:
    ok = reason == "model_not_found"
    return Verdict(
        ok=ok, status=AnswerStatus.NOT_FOUND, text=REFUSAL_TEXT, reason=reason, **details
    )


def _marker_ids(text: str) -> list[str]:
    return [i.strip() for match in _MARKER.findall(text) for i in match.split(",")]


def _clean_markers(text: str, valid: set[str]) -> str:
    def keep_valid(match: re.Match[str]) -> str:
        ids = [i.strip() for i in match.group(1).split(",") if i.strip() in valid]
        return f"[{', '.join(ids)}]" if ids else ""

    cleaned = _MARKER.sub(keep_valid, text)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    return _SPACES.sub(" ", cleaned).strip()
