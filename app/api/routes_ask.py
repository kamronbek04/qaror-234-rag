"""POST /api/v1/ask — grounded question answering."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from app.api.deps import get_container
from app.api.schemas import AskRequest, AskResponse, ErrorResponse, Meta, Source
from app.core.container import Container
from app.core.logging import request_id_var

router = APIRouter(prefix="/api/v1", tags=["Savol-javob"])

ASK_EXAMPLES = {
    "toifa_va_tolov": {
        "summary": "Toifa, muddat va to'lov",
        "value": {
            "question": "Aeroport uchun davlat ekologik ekspertizasi muddati va to'lovi qancha?"
        },
    },
    "atama": {"summary": "Atama ta'rifi", "value": {"question": "Ekolog-ekspert kim?"}},
    "aniq_band": {
        "summary": "Aniq bandga murojaat",
        "value": {"question": "2-ilovaning 6-bandida nima deyilgan?"},
    },
    "hujjatda_yoq": {
        "summary": "Hujjatda yo'q savol",
        "value": {"question": "O'zbekistonda QQS stavkasi necha foiz?"},
    },
    "debug": {
        "summary": "Tafsilotlar bilan",
        "value": {"question": "Qaror qachon kuchga kiradi?", "top_k": 6, "debug": True},
    },
}


@router.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    summary="Savolga qaror matni asosida javob berish",
    responses={503: {"model": ErrorResponse, "description": "Ollama yoki indeks tayyor emas"}},
)
async def ask(
    request: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)],
    container: Annotated[Container, Depends(get_container)],
) -> AskResponse:
    answer = await container.require_rag().ask(
        request.question, top_k=request.top_k, debug=request.debug
    )
    return AskResponse(
        answer=answer.text,
        status=answer.status,
        found=answer.found,
        sources=[
            Source(
                chunk_id=item.chunk.id,
                breadcrumb=item.chunk.breadcrumb,
                quote=item.chunk.text,
                url=item.chunk.url,
                score=round(
                    item.dense_score if item.dense_score is not None else item.fused_score, 4
                ),
            )
            for item in answer.sources
        ],
        meta=Meta(
            model=answer.model or "",
            request_id=request_id_var.get() or "",
            timings_ms=answer.timings_ms,
        ),
        debug=answer.debug if request.debug else None,
    )
