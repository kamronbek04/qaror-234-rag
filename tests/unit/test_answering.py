import asyncio
from types import SimpleNamespace

import httpx
import ollama
import pytest

from app.core.errors import LLMUnavailableError
from app.domain.models import REFUSAL_TEXT, AnswerStatus, RetrievalResult, ScoredChunk
from app.generation.guard import AnswerGuard
from app.generation.ollama_chat import OllamaChatModel
from app.generation.prompts import LLMAnswer, build_messages
from app.ingestion.chunker import StructuralChunker
from app.retrieval.chunk_store import ChunkStore
from app.services.rag_service import RagService
from tests.fakes import FakeChatModel, StubRetriever, reply

AIRPORT = (
    "Aeroportlar uchun davlat ekologik ekspertizasini oʻtkazish muddati 25 ish kuni, "
    "toʻlov miqdori 25 BXM [a1-r2]."
)


@pytest.fixture(scope="module")
def store(document):
    return ChunkStore(StructuralChunker(max_chars=1500).chunk(document))


@pytest.fixture
def excerpts(store):
    return [
        ScoredChunk(chunk=store.get("a1-r2"), dense_score=0.82, fused_score=0.03),
        ScoredChunk(chunk=store.get("q-b7"), dense_score=0.61, fused_score=0.02),
    ]


def retrieval(excerpts, similarity=0.8, reference=False):
    return RetrievalResult(
        query="", chunks=excerpts, top_similarity=similarity, reference_match=reference
    )


def service(excerpts, chat, similarity=0.8, reference=False, threshold=0.4):
    return RagService(
        retriever=StubRetriever(retrieval(excerpts, similarity, reference)),
        chat_model=chat,
        guard=AnswerGuard(),
        refusal_threshold=threshold,
        max_prompt_tokens=6000,
        max_concurrency=2,
    )


class TestPrompts:
    def test_excerpts_are_labelled_and_question_is_delimited(self, excerpts):
        messages, used = build_messages("Aeroport ekspertizasi qancha?", excerpts, 6000)

        user = messages[-1]["content"]
        assert messages[0]["role"] == "system"
        assert "[a1-r2] 1-ilova" in user
        assert "25 BXM" in user
        assert "<question>\nAeroport ekspertizasi qancha?\n</question>" in user
        assert used == excerpts

    def test_few_shot_examples_show_both_outcomes(self, excerpts):
        messages, _ = build_messages("savol", excerpts, 6000)

        assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]
        assert any('"not_found"' in turn for turn in assistant_turns)
        assert any('"answered"' in turn for turn in assistant_turns)

    def test_budget_drops_lowest_ranked_excerpts(self, excerpts):
        _, used = build_messages("savol", excerpts, max_prompt_tokens=900)

        assert [e.chunk.id for e in used] == ["a1-r2"]

    def test_cyrillic_question_is_transliterated(self, excerpts):
        messages, _ = build_messages("Аэропорт экспертизаси", excerpts, 6000)

        assert "Aeroport ekspertizasi" in messages[-1]["content"]


class FakeOllamaChatClient:
    def __init__(self, error=None):
        self.error = error
        self.kwargs = None

    async def chat(self, **kwargs):
        if self.error:
            raise self.error
        self.kwargs = kwargs
        return SimpleNamespace(message=SimpleNamespace(content=reply(answer="ok", citations=["x"])))


class TestOllamaChatModel:
    def make(self, client, think=None):
        return OllamaChatModel(
            client, model="qwen2.5:7b", num_ctx=8192, num_predict=512,
            temperature=0.0, seed=42, keep_alive="30m", think=think,
        )  # fmt: skip

    async def test_request_carries_schema_and_deterministic_options(self):
        client = FakeOllamaChatClient()

        await self.make(client).complete([{"role": "user", "content": "savol"}])

        assert client.kwargs["format"] == LLMAnswer.model_json_schema()
        assert client.kwargs["options"] == {
            "temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 512,
        }  # fmt: skip
        assert "think" not in client.kwargs

    async def test_think_flag_is_sent_only_when_configured(self):
        client = FakeOllamaChatClient()

        await self.make(client, think=False).complete([])

        assert client.kwargs["think"] is False

    @pytest.mark.parametrize(
        "error",
        [
            ollama.ResponseError("model not found", 404),
            httpx.ReadTimeout("timed out"),
            ConnectionError("Failed to connect to Ollama"),
        ],
    )
    async def test_failures_become_llm_unavailable(self, error):
        with pytest.raises(LLMUnavailableError):
            await self.make(FakeOllamaChatClient(error)).complete([])


class TestGuard:
    guard = AnswerGuard()

    def check(self, excerpts, status="answered", answer=AIRPORT, citations=("a1-r2",)):
        draft = LLMAnswer(status=status, answer=answer, citations=list(citations))
        return self.guard.check(draft, excerpts)

    def test_supported_answer_passes(self, excerpts):
        verdict = self.check(excerpts)

        assert verdict.ok and verdict.status is AnswerStatus.ANSWERED
        assert verdict.citations == ["a1-r2"]
        assert verdict.text == AIRPORT

    def test_not_found_uses_the_exact_refusal(self, excerpts):
        verdict = self.check(excerpts, status="not_found", answer="Kechirasiz...", citations=())

        assert verdict.status is AnswerStatus.NOT_FOUND
        assert verdict.text == REFUSAL_TEXT

    def test_empty_answer_is_a_refusal(self, excerpts):
        assert self.check(excerpts, answer="  ").status is AnswerStatus.NOT_FOUND

    def test_fabricated_citation_is_rejected(self, excerpts):
        verdict = self.check(excerpts, answer="Aeroportlar 25 BXM [a7-b99].", citations=["a7-b99"])

        assert not verdict.ok
        assert verdict.reason == "no_valid_citations"

    def test_invalid_citations_are_dropped_and_markers_cleaned(self, excerpts):
        verdict = self.check(
            excerpts, answer="Toʻlov 25 BXM [a1-r2][a7-b99].", citations=["a1-r2", "a7-b99"]
        )

        assert verdict.ok
        assert verdict.citations == ["a1-r2"]
        assert verdict.dropped_citations == ["a7-b99"]
        assert verdict.text == "Toʻlov 25 BXM [a1-r2]."

    def test_inline_marker_counts_as_citation(self, excerpts):
        assert self.check(excerpts, citations=()).citations == ["a1-r2"]

    def test_wrong_number_is_rejected(self, excerpts):
        verdict = self.check(excerpts, answer="Muddat 30 ish kuni, toʻlov 25 BXM [a1-r2].")

        assert not verdict.ok
        assert verdict.reason == "unsupported_numbers"
        assert verdict.unsupported_numbers == ["30"]

    def test_decimal_separator_does_not_matter(self, store):
        excerpt = [ScoredChunk(chunk=store.get("a1-r134"))]
        answer = "Avtoservis uchun toʻlov 7.5 BXM, muddat 15 ish kuni [a1-r134]."

        verdict = self.guard.check(
            LLMAnswer(status="answered", answer=answer, citations=[]), excerpt
        )

        assert verdict.ok

    def test_invented_calendar_date_is_rejected(self, excerpts):
        verdict = self.check(
            excerpts, answer="Qaror 2026-yil 11-avgustda kuchga kiradi [q-b7].", citations=["q-b7"]
        )

        assert verdict.unsupported_numbers == ["11", "2026"]

    def test_resolution_number_is_always_allowed(self, excerpts):
        answer = "234-son qaror eʼlon qilingandan uch oy oʻtgach kuchga kiradi [q-b7]."

        assert self.check(excerpts, answer=answer, citations=["q-b7"]).ok

    def test_partial_answer_gets_the_refusal_sentence(self, excerpts):
        verdict = self.check(excerpts, status="partial", answer="Toʻlov 25 BXM [a1-r2].")

        assert verdict.status is AnswerStatus.PARTIAL
        assert verdict.text.endswith(REFUSAL_TEXT + ".")

    def test_partial_answer_with_refusal_sentence_is_kept(self, excerpts):
        answer = (
            "Toʻlov 25 BXM [a1-r2]. Dollardagi summasi haqida: Hujjatda bu haqida maʼlumot yoʻq."
        )

        assert self.check(excerpts, status="partial", answer=answer).text == answer


class TestRagService:
    async def test_gate_refuses_without_calling_the_model(self, excerpts):
        chat = FakeChatModel(reply(answer=AIRPORT, citations=["a1-r2"]))

        answer = await service(excerpts, chat, similarity=0.2).ask("Ob-havo qanday?", debug=True)

        assert answer.text == REFUSAL_TEXT
        assert answer.status is AnswerStatus.NOT_FOUND
        assert chat.conversations == []
        assert answer.debug["gate"]["passed"] is False

    async def test_explicit_reference_bypasses_the_gate(self, excerpts):
        chat = FakeChatModel(reply(answer=AIRPORT, citations=["a1-r2"]))

        answer = await service(excerpts, chat, similarity=0.2, reference=True).ask(
            "1-ilova 2-qator"
        )

        assert answer.status is AnswerStatus.ANSWERED

    async def test_correct_answer_passes_with_sources(self, excerpts):
        chat = FakeChatModel(reply(answer=AIRPORT, citations=["a1-r2"]))

        answer = await service(excerpts, chat).ask("Aeroport ekspertizasi qancha?")

        assert answer.status is AnswerStatus.ANSWERED
        assert answer.found is True
        assert [s.chunk.id for s in answer.sources] == ["a1-r2"]
        assert answer.model == "fake-llm"
        assert set(answer.timings_ms) == {"retrieval", "generation", "total"}

    async def test_wrong_number_retries_with_feedback_then_refuses(self, excerpts):
        wrong = reply(answer="Muddat 30 ish kuni [a1-r2].", citations=["a1-r2"])
        chat = FakeChatModel(wrong, wrong)

        answer = await service(excerpts, chat).ask("Aeroport ekspertizasi qancha?", debug=True)

        assert answer.text == REFUSAL_TEXT
        assert len(chat.conversations) == 2
        assert "30" in chat.conversations[1][-1]["content"]
        assert answer.debug["attempts"][-1]["reason"] == "unsupported_numbers"

    async def test_wrong_number_then_correct_answer_is_accepted(self, excerpts):
        chat = FakeChatModel(
            reply(answer="Muddat 30 ish kuni [a1-r2].", citations=["a1-r2"]),
            reply(answer=AIRPORT, citations=["a1-r2"]),
        )

        answer = await service(excerpts, chat).ask("Aeroport ekspertizasi qancha?")

        assert answer.status is AnswerStatus.ANSWERED
        assert answer.text == AIRPORT

    async def test_fabricated_citation_is_refused_without_retry(self, excerpts):
        chat = FakeChatModel(reply(answer="Toʻlov 25 BXM [a7-b99].", citations=["a7-b99"]))

        answer = await service(excerpts, chat).ask("savol")

        assert answer.text == REFUSAL_TEXT
        assert len(chat.conversations) == 1

    async def test_malformed_output_twice_is_refused(self, excerpts):
        chat = FakeChatModel("bu JSON emas", "{hali ham emas")

        answer = await service(excerpts, chat).ask("savol", debug=True)

        assert answer.text == REFUSAL_TEXT
        assert answer.debug["attempts"][-1]["reason"] == "malformed_output"

    async def test_model_not_found_status_is_refused(self, excerpts):
        chat = FakeChatModel(reply(status="not_found"))

        answer = await service(excerpts, chat).ask("QQS stavkasi necha foiz?")

        assert answer.text == REFUSAL_TEXT
        assert answer.sources == []

    async def test_backend_failure_propagates(self, excerpts):
        chat = FakeChatModel(LLMUnavailableError())

        with pytest.raises(LLMUnavailableError):
            await service(excerpts, chat).ask("savol")

    async def test_generation_concurrency_is_bounded(self, excerpts):
        chat = FakeChatModel(reply(answer=AIRPORT, citations=["a1-r2"]), delay=0.05)
        rag = service(excerpts, chat)

        answers = await asyncio.gather(*(rag.ask("savol") for _ in range(5)))

        assert all(a.status is AnswerStatus.ANSWERED for a in answers)
        assert chat.max_active == 2


class TestTermGuard:
    guard = AnswerGuard()

    def check(self, excerpts, answer, status="answered", citations=("a1-r2",)):
        draft = LLMAnswer(status=status, answer=answer, citations=list(citations))
        return self.guard.check(draft, excerpts)

    def test_currency_conversion_is_rejected(self, excerpts):
        verdict = self.check(excerpts, "Toʻlov 25 BXM. Bu dollarda 25 boʻladi [a1-r2].")

        assert not verdict.ok
        assert verdict.reason == "unsupported_terms"
        assert verdict.unsupported_terms == ["dollarda"]

    def test_numbers_spelled_out_are_rejected(self, excerpts):
        verdict = self.check(excerpts, "Muddat yigirma besh ish kuni [a1-r2].")

        assert verdict.reason == "unsupported_terms"
        assert verdict.unsupported_terms == ["yigirma", "besh"]

    def test_number_words_present_in_the_source_are_fine(self, excerpts):
        answer = "Qaror rasmiy eʼlon qilingan kundan eʼtiboran uch oy oʻtgach kuchga kiradi [q-b7]."

        assert self.check(excerpts, answer, citations=["q-b7"]).ok


async def test_unsupported_terms_retry_can_end_in_partial_answer(excerpts):
    chat = FakeChatModel(
        reply(answer="Toʻlov 25 BXM, dollarda 25 [a1-r2].", citations=["a1-r2"]),
        reply(status="partial", answer="Toʻlov 25 BXM [a1-r2].", citations=["a1-r2"]),
    )

    answer = await service(excerpts, chat).ask("Necha BXM va dollarda qancha?")

    assert answer.status is AnswerStatus.PARTIAL
    assert answer.text.endswith(REFUSAL_TEXT + ".")
    assert "dollarda" in chat.conversations[1][-1]["content"]


async def test_exact_number_in_question_passes_the_gate(excerpts):
    chat = FakeChatModel(
        reply(answer="Qaror uch oy oʻtgach kuchga kiradi [q-b7].", citations=["q-b7"])
    )
    rag = RagService(
        retriever=StubRetriever(
            RetrievalResult(query="", chunks=excerpts, top_similarity=0.3, lexical_anchor=True)
        ),
        chat_model=chat, guard=AnswerGuard(), refusal_threshold=0.5,
        max_prompt_tokens=6000, max_concurrency=2,
    )  # fmt: skip

    answer = await rag.ask("541-son qaror nima boʻldi?", debug=True)

    assert answer.debug["gate"] == {
        "passed": True, "top_similarity": 0.3, "threshold": 0.5,
        "reference_match": False, "lexical_anchor": True,
    }  # fmt: skip
    assert len(chat.conversations) == 1


def test_disclaimer_sentence_may_name_the_missing_currency_but_claims_may_not(excerpts):
    guard = AnswerGuard()
    disclaimer = "Toʻlov 25 BXM [a1-r2]. Dollardagi summasi: Hujjatda bu haqida maʼlumot yoʻq."
    claim = "Toʻlov 25 BXM, ya'ni 25 dollar [a1-r2]. Hujjatda bu haqida maʼlumot yoʻq."

    ok = guard.check(LLMAnswer(status="partial", answer=disclaimer, citations=[]), excerpts)
    bad = guard.check(LLMAnswer(status="partial", answer=claim, citations=[]), excerpts)

    assert ok.ok
    assert bad.unsupported_terms == ["dollar"]


def test_question_asking_for_a_currency_the_source_lacks_becomes_partial(excerpts):
    draft = LLMAnswer(status="answered", answer="Toʻlov 25 BXM [a1-r2].", citations=["a1-r2"])

    verdict = AnswerGuard().check(draft, excerpts, question="Necha BXM va dollarda qancha?")

    assert verdict.status is AnswerStatus.PARTIAL
    assert (
        verdict.text == f"Toʻlov 25 BXM [a1-r2]. Savolning qolgan qismi boʻyicha: {REFUSAL_TEXT}."
    )


def test_prompt_explains_how_to_read_ranges():
    from app.generation.prompts import SYSTEM_PROMPT

    assert "va undan ortiq" in SYSTEM_PROMPT
    assert "gacha" in SYSTEM_PROMPT
