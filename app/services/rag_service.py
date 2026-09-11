"""The question-answering pipeline: retrieve → gate → generate → verify → respond."""

import asyncio
import logging
import re
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.domain.models import Answer, AnswerStatus, RetrievalResult, ScoredChunk
from app.generation.guard import AnswerGuard, Verdict
from app.generation.prompts import (
    LLMAnswer,
    build_messages,
    correction_feedback,
    format_reminder,
)
from app.generation.protocols import ChatModel
from app.retrieval.protocols import Retriever

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2
_CORRECTABLE = {"unsupported_numbers", "unsupported_terms"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class RagService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        chat_model: ChatModel,
        guard: AnswerGuard,
        refusal_threshold: float,
        max_prompt_tokens: int,
        max_concurrency: int,
    ) -> None:
        self._retriever = retriever
        self._chat = chat_model
        self._guard = guard
        self._threshold = refusal_threshold
        self._max_prompt_tokens = max_prompt_tokens
        self._generation_slots = asyncio.Semaphore(max_concurrency)

    @property
    def model(self) -> str:
        return self._chat.model

    async def ask(self, question: str, *, top_k: int | None = None, debug: bool = False) -> Answer:
        started = perf_counter()
        retrieval = await self._retriever.retrieve(question, top_k)
        retrieval_ms = _elapsed_ms(started)

        passed = (
            retrieval.reference_match
            or retrieval.lexical_anchor
            or retrieval.top_similarity >= self._threshold
        )
        trace: dict[str, Any] = {
            "gate": {
                "passed": passed,
                "top_similarity": round(retrieval.top_similarity, 4),
                "threshold": self._threshold,
                "reference_match": retrieval.reference_match,
                "lexical_anchor": retrieval.lexical_anchor,
            },
            "retrieval": [_describe(item) for item in retrieval.chunks],
            "attempts": [],
        }
        if not passed or not retrieval.chunks:
            return self._finish(Answer.refusal(), trace, debug, started, retrieval_ms, 0)

        generation_started = perf_counter()
        async with self._generation_slots:
            verdict, excerpts = await self._generate(question, retrieval, trace)
        generation_ms = _elapsed_ms(generation_started)

        if verdict is None or not verdict.ok or verdict.status is AnswerStatus.NOT_FOUND:
            answer = Answer.refusal()
        else:
            by_id = {item.chunk.id: item for item in excerpts}
            answer = Answer(
                text=verdict.text,
                status=verdict.status,
                citations=verdict.citations,
                sources=[by_id[c] for c in verdict.citations],
            )
        return self._finish(answer, trace, debug, started, retrieval_ms, generation_ms)

    async def _generate(
        self, question: str, retrieval: RetrievalResult, trace: dict[str, Any]
    ) -> tuple[Verdict | None, list[ScoredChunk]]:
        messages, excerpts = build_messages(question, retrieval.chunks, self._max_prompt_tokens)
        trace["prompt_excerpts"] = [item.chunk.id for item in excerpts]
        for _ in range(_MAX_ATTEMPTS):
            raw = await self._chat.complete(messages)
            draft = _parse(raw)
            if draft is None:
                trace["attempts"].append({"reason": "malformed_output", "raw": raw[:500]})
                messages = [*messages, {"role": "assistant", "content": raw}, format_reminder()]
                continue
            verdict = self._guard.check(draft, excerpts, question=question)
            trace["attempts"].append(
                {
                    "ok": verdict.ok,
                    "reason": verdict.reason,
                    "status": draft.status,
                    "citations": verdict.citations,
                    "dropped_citations": verdict.dropped_citations,
                    "unsupported_numbers": verdict.unsupported_numbers,
                    "unsupported_terms": verdict.unsupported_terms,
                    "raw": raw[:1000],
                }
            )
            if verdict.reason not in _CORRECTABLE:
                return verdict, excerpts
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                correction_feedback(verdict.unsupported_numbers, verdict.unsupported_terms),
            ]
        return None, excerpts

    def _finish(
        self,
        answer: Answer,
        trace: dict[str, Any],
        debug: bool,
        started: float,
        retrieval_ms: int,
        generation_ms: int,
    ) -> Answer:
        timings = {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": _elapsed_ms(started),
        }
        logger.info(
            "savolga javob berildi",
            extra={
                "status": answer.status.value,
                "gate_passed": trace["gate"]["passed"],
                "top_similarity": trace["gate"]["top_similarity"],
                "citations": answer.citations,
                "attempts": [a.get("reason") for a in trace["attempts"]],
                "timings_ms": timings,
            },
        )
        return answer.model_copy(
            update={"model": self.model, "timings_ms": timings, "debug": trace if debug else {}}
        )


def _parse(raw: str) -> LLMAnswer | None:
    for candidate in (raw, *(_JSON_OBJECT.findall(raw)[:1])):
        try:
            return LLMAnswer.model_validate_json(candidate)
        except ValidationError:
            continue
    return None


def _describe(item: ScoredChunk) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk.id,
        "rank": item.rank,
        "dense": None if item.dense_score is None else round(item.dense_score, 4),
        "lexical": None if item.lexical_score is None else round(item.lexical_score, 3),
        "fused": round(item.fused_score, 5),
        "pinned": item.pinned,
        "expansion": item.expansion,
    }


def _elapsed_ms(since: float) -> int:
    return int((perf_counter() - since) * 1000)
