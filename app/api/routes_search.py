"""POST /api/v1/search and GET /api/v1/chunks/{chunk_id} — retrieval inspection."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from app.api.deps import get_container
from app.api.schemas import ErrorResponse, SearchHit, SearchRequest, SearchResponse
from app.core.container import Container
from app.domain.models import Chunk

router = APIRouter(prefix="/api/v1", tags=["Qidiruv"])

SEARCH_EXAMPLES = {
    "mavzu": {"summary": "Mavzu bo'yicha", "value": {"query": "jamoatchilik eshituvi", "top_k": 5}},
    "raqam": {"summary": "Aniq atama yoki raqam", "value": {"query": "541-son qaror", "top_k": 3}},
    "kirill": {
        "summary": "Kirill yozuvida",
        "value": {"query": "Аэропорт учун экспертиза муддати"},
    },
}


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Faqat qidiruv (LLM chaqirilmaydi): bo'laklar va ularning ballari",
    responses={503: {"model": ErrorResponse}},
)
async def search(
    request: Annotated[SearchRequest, Body(openapi_examples=SEARCH_EXAMPLES)],
    container: Annotated[Container, Depends(get_container)],
) -> SearchResponse:
    result = await container.require_retriever().retrieve(request.query, request.top_k)
    return SearchResponse(
        query=request.query,
        top_similarity=round(result.top_similarity, 4),
        reference_match=result.reference_match,
        results=[
            SearchHit(
                chunk_id=item.chunk.id,
                type=item.chunk.type,
                breadcrumb=item.chunk.breadcrumb,
                text=item.chunk.text,
                url=item.chunk.url,
                dense_score=item.dense_score,
                lexical_score=item.lexical_score,
                fused_score=item.fused_score,
                rank=item.rank,
                pinned=item.pinned,
                expansion=item.expansion,
            )
            for item in result.chunks
        ],
    )


@router.get(
    "/chunks/{chunk_id}",
    response_model=Chunk,
    summary="Bitta bo'lakning to'liq matni va metama'lumoti",
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def get_chunk(
    chunk_id: str, container: Annotated[Container, Depends(get_container)]
) -> Chunk:
    return container.require_store().get(chunk_id)
