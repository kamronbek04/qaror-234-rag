"""Spec scenarios of hybrid retrieval checked against the real embedding model."""

import pytest

pytestmark = [pytest.mark.ollama, pytest.mark.asyncio(loop_scope="session")]


def ids(result):
    return [scored.chunk.id for scored in result.chunks]


async def test_paraphrased_question_finds_the_table_row(real_retriever):
    result = await real_retriever.retrieve(
        "Aeroport qurish uchun ekologik ekspertiza necha kun davom etadi?"
    )

    assert "a1-r2" in ids(result)[:3]


async def test_exact_term_is_found(real_retriever):
    assert "q-b6" in ids(await real_retriever.retrieve("541-son qaror"))[:3]


async def test_cyrillic_query_matches_latin_query(real_retriever):
    cyrillic = await real_retriever.retrieve("Аэропорт учун экспертиза муддати")
    latin = await real_retriever.retrieve("Aeroport uchun ekspertiza muddati")

    assert ids(cyrillic)[:5] == ids(latin)[:5]


async def test_unrelated_question_has_low_relevance(real_retriever, settings):
    result = await real_retriever.retrieve("Toshkentda ertaga ob-havo qanday boʻladi?")

    assert result.top_similarity < settings.refusal_threshold
