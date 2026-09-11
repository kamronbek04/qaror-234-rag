"""Deterministic verification of model answers against the excerpts they cite.

Nothing the model says reaches the user unless it passes these checks: the refusal text always
comes from code, citations must point at supplied excerpts, and every number must be found in
the cited text.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.domain.models import REFUSAL_TEXT, AnswerStatus, ScoredChunk
from app.generation.prompts import LLMAnswer
from app.text.normalize import lexical_form
from app.text.numbers import extract_numbers

_CHUNK_ID = r"(?:q|a\d+|w)(?:-[a-z0-9]+)*"
_MARKER = re.compile(rf"\[\s*({_CHUNK_ID}(?:\s*,\s*{_CHUNK_ID})*)\s*\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")
_SPACES = re.compile(r"[ \t]{2,}")

# The resolution's own number may always be named, even when no excerpt repeats it.
ALWAYS_ALLOWED_NUMBERS = frozenset({"234"})
PARTIAL_SUFFIX = f"Savolning qolgan qismi boʻyicha: {REFUSAL_TEXT}."


class Verdict(BaseModel):
    ok: bool
    status: AnswerStatus
    text: str
    reason: str | None = None
    citations: list[str] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)


class AnswerGuard:
    def check(self, draft: LLMAnswer, excerpts: Sequence[ScoredChunk]) -> Verdict:
        if draft.status == "not_found" or not draft.answer.strip():
            return _refusal("model_not_found")

        supplied = {item.chunk.id: item.chunk for item in excerpts}
        claimed = list(dict.fromkeys([*draft.citations, *_marker_ids(draft.answer)]))
        valid = [c for c in claimed if c in supplied]
        dropped = [c for c in claimed if c not in supplied]
        if not valid:
            return _refusal("no_valid_citations", dropped_citations=dropped)

        text = _clean_markers(draft.answer.strip(), set(valid))
        allowed = set(ALWAYS_ALLOWED_NUMBERS)
        for chunk_id in valid:
            chunk = supplied[chunk_id]
            allowed |= extract_numbers(f"{chunk.breadcrumb}\n{chunk.text}")
        used = extract_numbers(_MARKER.sub(" ", text))
        unsupported = sorted(used - allowed, key=float)
        if unsupported:
            return _refusal(
                "unsupported_numbers",
                citations=valid,
                dropped_citations=dropped,
                unsupported_numbers=unsupported,
            )

        status = AnswerStatus.PARTIAL if draft.status == "partial" else AnswerStatus.ANSWERED
        if status is AnswerStatus.PARTIAL and lexical_form(REFUSAL_TEXT) not in lexical_form(text):
            text = f"{text} {PARTIAL_SUFFIX}"
        return Verdict(
            ok=True, status=status, text=text, citations=valid, dropped_citations=dropped
        )


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
