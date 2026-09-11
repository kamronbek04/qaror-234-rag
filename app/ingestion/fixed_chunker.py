"""Fixed-size character windows: the naive baseline structural chunking is measured against."""

from collections.abc import Iterator

from app.domain.models import Chunk, ChunkType, Document, Section


class FixedSizeChunker:
    def __init__(self, size: int = 1000, overlap: int = 200) -> None:
        if not 0 <= overlap < size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = render_plain_text(document)
        step = self.size - self.overlap
        starts = range(0, max(len(text) - self.overlap, 1), step)
        return [
            Chunk(
                id=f"w-{number}",
                type=ChunkType.WINDOW,
                section="doc",
                breadcrumb=f"Qaror matni{' › '}{number}-bo'lak",
                text=text[start : start + self.size],
                url=document.url,
                order=number - 1,
            )
            for number, start in enumerate(starts, start=1)
        ]


def render_plain_text(document: Document) -> str:
    """The document as one plain text, the way a naive HTML-to-text extraction would see it."""
    return "\n".join(line for line in _lines(document) if line)


def _lines(document: Document) -> Iterator[str]:
    yield document.title
    for section in [document.main, *document.appendices]:
        yield from _section_lines(section)


def _section_lines(section: Section) -> Iterator[str]:
    yield f"{section.label} {section.title}".strip()
    yield from (p.text for p in section.paragraphs)
    for item in section.items:
        yield item.text
        yield from (s.text for s in item.sub_items)
    for chapter in section.chapters:
        yield chapter.title
        for item in chapter.items:
            yield item.text
            yield from (s.text for s in item.sub_items)
    for table in section.tables:
        yield from (" | ".join(c for c in row if c) for row in table.rows)
    yield from (f.text for f in section.footnotes)
    for attachment in section.attachments:
        yield from _section_lines(attachment)
