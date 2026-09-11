"""Domain models: the parsed document tree, retrieval units and answers."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from app.text.normalize import canonicalize

REFUSAL_TEXT = "Hujjatda bu haqida ma'lumot yo'q"


# --- Parsed document -------------------------------------------------------------------------


class Paragraph(BaseModel):
    text: str
    element_id: str | None = None


class Item(BaseModel):
    """A numbered item (band) with the unnumbered paragraphs that belong to it."""

    number: str
    text: str
    element_id: str | None = None
    sub_items: list[Paragraph] = Field(default_factory=list)


class Chapter(BaseModel):
    number: str | None
    title: str
    element_id: str | None = None
    items: list[Item] = Field(default_factory=list)


class Table(BaseModel):
    element_id: str | None = None
    caption: str | None = None
    rows: list[list[str]]
    after_item: str | None = None


class Footnote(BaseModel):
    text: str
    element_id: str | None = None


class Section(BaseModel):
    """The main resolution, one of its appendices, or an attachment of a regulation."""

    key: str
    label: str
    number: int | None = None
    title: str = ""
    doc_type: str | None = None
    element_id: str | None = None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    footnotes: list[Footnote] = Field(default_factory=list)
    attachments: list["Section"] = Field(default_factory=list)

    def all_items(self) -> list[Item]:
        return self.items + [item for chapter in self.chapters for item in chapter.items]

    def chapter_of(self, item: Item) -> Chapter | None:
        return next((c for c in self.chapters if any(i is item for i in c.items)), None)


class ActivityRow(BaseModel):
    """One row of the Appendix 1 table: activity type, category, review deadline and fee."""

    number: int
    category: str
    hazard: str
    category_title: str
    sector_number: int | None
    sector: str
    activity: str
    deadline: str
    deadline_days: int | None
    fee: str
    fee_bxm: float | None
    element_id: str | None = None


class Document(BaseModel):
    title: str
    url: str
    number: str | None = None
    date: str | None = None
    main: Section
    appendices: list[Section] = Field(default_factory=list)
    activity_rows: list[ActivityRow] = Field(default_factory=list)


# --- Retrieval -------------------------------------------------------------------------------


class ChunkType(StrEnum):
    OVERVIEW = "overview"
    ITEM = "item"
    PARAGRAPH = "paragraph"
    DEFINITION = "definition"
    TABLE_ROW = "table_row"
    SCHEME_STAGE = "scheme_stage"
    FOOTNOTE = "footnote"
    FORM = "form"
    WINDOW = "window"


class Chunk(BaseModel):
    """A self-contained, citable retrieval unit."""

    id: str
    type: ChunkType
    section: str
    appendix: int | None = None
    chapter: str | None = None
    number: str | None = None
    part: int | None = None
    breadcrumb: str
    text: str
    element_id: str | None = None
    url: str
    order: int
    references: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        return f"{canonicalize(self.breadcrumb)}\n{canonicalize(self.text)}"

    @property
    def search_text(self) -> str:
        return f"{self.breadcrumb}\n{self.text}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    rank: int = 0
    pinned: bool = False
    expansion: bool = False


class RetrievalResult(BaseModel):
    query: str
    chunks: list[ScoredChunk]
    top_similarity: float = 0.0
    reference_match: bool = False


# --- Answers ---------------------------------------------------------------------------------


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"


class Answer(BaseModel):
    text: str
    status: AnswerStatus
    citations: list[str] = Field(default_factory=list)
    sources: list[ScoredChunk] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def found(self) -> bool:
        return self.status is not AnswerStatus.NOT_FOUND

    @classmethod
    def refusal(cls, debug: dict[str, Any] | None = None) -> "Answer":
        return cls(text=REFUSAL_TEXT, status=AnswerStatus.NOT_FOUND, debug=debug or {})
