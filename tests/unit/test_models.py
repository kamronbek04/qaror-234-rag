from app.domain.models import (
    Answer,
    AnswerStatus,
    Chunk,
    ChunkType,
    RetrievalResult,
    ScoredChunk,
)


def make_chunk(**overrides) -> Chunk:
    fields = {
        "id": "a2-b6",
        "type": ChunkType.ITEM,
        "section": "a2",
        "appendix": 2,
        "chapter": "1-bob. Umumiy qoidalar",
        "number": "6",
        "breadcrumb": "2-ilova › Nizom › 1-bob. Umumiy qoidalar › 6-band",
        "text": "6. Davlat ekologik ekspertizasi buyurtmachining mablagʻlari hisobidan.",
        "element_id": "-8205470",
        "url": "https://lex.uz/uz/docs/-8193120#-8205470",
        "order": 42,
    }
    fields.update(overrides)
    return Chunk(**fields)


def test_chunk_round_trips_through_json():
    chunk = make_chunk(references=["a2-b4"], metadata={"fee_bxm": 25.0})

    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_embedding_text_prefixes_breadcrumb_and_canonicalizes():
    chunk = make_chunk()

    assert chunk.embedding_text.startswith("2-ilova › Nizom › 1-bob. Umumiy qoidalar › 6-band\n")
    assert "mablag'lari" in chunk.embedding_text


def test_retrieval_result_and_answer_construct():
    scored = ScoredChunk(chunk=make_chunk(), dense_score=0.8, lexical_score=3.1, fused_score=0.03)
    result = RetrievalResult(query="savol", chunks=[scored], top_similarity=0.8)
    answer = Answer(
        text="javob", status=AnswerStatus.ANSWERED, citations=["a2-b6"], sources=[scored]
    )

    assert result.reference_match is False
    assert answer.found is True
    assert Answer.refusal().found is False
    assert Answer.refusal().text == "Hujjatda bu haqida ma'lumot yo'q"
