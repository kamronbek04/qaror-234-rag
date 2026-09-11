"""HTTP request and response contracts."""

from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

from app.domain.models import AnswerStatus, ChunkType

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Text = Field(description="Savol (o'zbek tilida, lotin yoki kirill yozuvida)")
    top_k: int | None = Field(None, ge=1, le=20, description="Kontekstga olinadigan bo'laklar")
    debug: bool = Field(False, description="Qidiruv va tekshiruv tafsilotlarini qaytarish")


class SearchRequest(BaseModel):
    query: Text
    top_k: int = Field(5, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str
    breadcrumb: str
    quote: str = Field(description="Manba matni (asl yozuvda)")
    url: str = Field(description="lex.uz'dagi aynan shu bandga havola")
    score: float


class Meta(BaseModel):
    model: str
    request_id: str
    timings_ms: dict[str, int]


class AskResponse(BaseModel):
    answer: str
    status: AnswerStatus
    found: bool
    sources: list[Source]
    meta: Meta
    debug: dict[str, Any] | None = None


class SearchHit(BaseModel):
    chunk_id: str
    type: ChunkType
    breadcrumb: str
    text: str
    url: str
    dense_score: float | None
    lexical_score: float | None
    fused_score: float
    rank: int
    pinned: bool
    expansion: bool


class SearchResponse(BaseModel):
    query: str
    top_similarity: float
    reference_match: bool
    results: list[SearchHit]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
