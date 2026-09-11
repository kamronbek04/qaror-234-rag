"""Structure-aware chunking of the parsed resolution.

One chunk is one citable legal unit: a numbered item with its sub-items, a defined term, a row
of the Appendix 1 table, a stage of a workflow scheme, a footnote or a form. Every chunk carries
a breadcrumb (its position in the document) and a deep link to the lex.uz element.

Identifier grammar::

    q-p1, q-b7            main resolution preamble / item 7
    a2                    overview of appendix 2
    a2-b6, a2-b6-p2       item 6 of appendix 2 / its second part when split
    a2-b2-d5              fifth definition inside item 2
    a1-r2, a1-fn1         Appendix 1 table row 2 / footnote 1
    a2-x1-s3              stage 3 of the scheme in attachment 1 of appendix 2
    a2-x2-form            form template in attachment 2 of appendix 2
    a9-t1-s2              stage 2 of a scheme table quoted inside appendix 9
"""

import re
from collections.abc import Iterable
from typing import Any

from app.domain.models import (
    ActivityRow,
    Chunk,
    ChunkType,
    Document,
    Footnote,
    Item,
    Section,
    Table,
)

_ITEM_REF = re.compile(r"(?:ushbu|mazkur)\s+nizomning\s+(\d+(?:-\d+)?)-band", re.IGNORECASE)
_ATTACHMENT_REF = re.compile(r"(?:ushbu|mazkur)\s+nizomga\s+(\d+)-ilova", re.IGNORECASE)
_APPENDIX_REF = re.compile(r"(\d+)-ilovaga\s+muvofiq", re.IGNORECASE)
_STAGE = re.compile(r"^(\d+)-bosqich", re.IGNORECASE)
_BLANKS = re.compile(r"_{3,}")
_DEFINITION_MARKER = "tushuncha"
_DASH = " — "

SEPARATOR = " › "


class StructuralChunker:
    def __init__(self, max_chars: int = 1500) -> None:
        self.max_chars = max_chars

    def chunk(self, document: Document) -> list[Chunk]:
        emitter = _Emitter(document.url, self.max_chars)
        emitter.main_section(document.main)
        for appendix in document.appendices:
            emitter.appendix(appendix, document.activity_rows if appendix.number == 1 else [])
        return emitter.finish()


class _Emitter:
    def __init__(self, url: str, max_chars: int) -> None:
        self.url = url
        self.max_chars = max_chars
        self.chunks: list[Chunk] = []

    # --- public steps --------------------------------------------------------------------

    def main_section(self, section: Section) -> None:
        for number, paragraph in enumerate(section.paragraphs, start=1):
            self._add(
                id=f"q-p{number}",
                type=ChunkType.PARAGRAPH,
                section="q",
                breadcrumb=f"Qaror{SEPARATOR}Kirish qismi",
                text=paragraph.text,
                element_id=paragraph.element_id,
            )
        for item in section.all_items():
            references = [f"a{n}" for n in _APPENDIX_REF.findall(_item_text(item))]
            self._item(item, id_prefix="q", section=section, appendix=None,
                       breadcrumb="Qaror", references=references)  # fmt: skip

    def appendix(self, appendix: Section, activity_rows: list[ActivityRow]) -> None:
        heading = f"{appendix.label}{SEPARATOR}{_titled(appendix)}"
        self._overview(appendix, heading)
        for number, paragraph in enumerate(appendix.paragraphs, start=1):
            self._add(
                id=f"{appendix.key}-p{number}",
                type=ChunkType.PARAGRAPH,
                section=appendix.key,
                appendix=appendix.number,
                breadcrumb=heading,
                text=paragraph.text,
                element_id=paragraph.element_id,
            )
        for item in appendix.all_items():
            chapter = appendix.chapter_of(item)
            text = _item_text(item)
            references = [f"{appendix.key}-b{n}" for n in _ITEM_REF.findall(text)]
            references += [f"{appendix.key}-x{n}" for n in _ATTACHMENT_REF.findall(text)]
            self._item(
                item,
                id_prefix=appendix.key,
                section=appendix,
                appendix=appendix.number,
                breadcrumb=heading + (SEPARATOR + chapter.title if chapter else ""),
                chapter=chapter.title if chapter else None,
                references=references,
            )
        for row in activity_rows:
            self._activity_row(row, appendix, heading)
        tables = appendix.tables if not activity_rows else appendix.tables[1:]
        for number, table in enumerate(tables, start=1):
            place = heading + (f"{SEPARATOR}{table.after_item}-band" if table.after_item else "")
            title = table.caption or _titled(appendix)
            self._table(table, f"{appendix.key}-t{number}", appendix, place, title)
        self._footnotes(appendix.footnotes, appendix, appendix.key, heading)
        for attachment in appendix.attachments:
            title = _titled(attachment)
            place = f"{heading}{SEPARATOR}{attachment.label}" + (f": {title}" if title else "")
            for table in attachment.tables:
                self._table(table, attachment.key, appendix, place, title or attachment.label)
            self._footnotes(attachment.footnotes, appendix, attachment.key, place)

    def finish(self) -> list[Chunk]:
        """Turn logical references (item, attachment, appendix ids) into concrete chunk ids."""
        ids = [c.id for c in self.chunks]
        known = set(ids)

        def resolve(reference: str) -> str | None:
            if reference in known:
                return reference
            return next((i for i in ids if i.startswith(reference + "-")), None)

        for chunk in self.chunks:
            resolved = [resolve(r) for r in chunk.references]
            chunk.references = list(dict.fromkeys(r for r in resolved if r and r != chunk.id))
        return self.chunks

    # --- unit builders ---------------------------------------------------------------------

    def _add(self, **fields: Any) -> Chunk:
        element_id = fields.get("element_id")
        chunk = Chunk(
            url=f"{self.url}#{element_id}" if element_id else self.url,
            order=len(self.chunks),
            **fields,
        )
        self.chunks.append(chunk)
        return chunk

    def _overview(self, appendix: Section, heading: str) -> None:
        lines = [f"{appendix.label}: {_titled(appendix)}."]
        if appendix.chapters:
            lines.append("Boblar: " + "; ".join(c.title for c in appendix.chapters) + ".")
        if appendix.attachments:
            lines.append(
                "Ilovalar: "
                + "; ".join(
                    f"{a.label}" + (f" — {_titled(a)}" if _titled(a) else "")
                    for a in appendix.attachments
                )
                + "."
            )
        self._add(
            id=appendix.key,
            type=ChunkType.OVERVIEW,
            section=appendix.key,
            appendix=appendix.number,
            breadcrumb=heading,
            text="\n".join(lines),
            element_id=appendix.element_id,
        )

    def _item(
        self,
        item: Item,
        *,
        id_prefix: str,
        section: Section,
        appendix: int | None,
        breadcrumb: str,
        references: list[str],
        chapter: str | None = None,
    ) -> None:
        item_id = f"{id_prefix}-b{item.number}"
        crumb = f"{breadcrumb}{SEPARATOR}{item.number}-band"
        subs = [s.text for s in item.sub_items]
        common = {
            "type": ChunkType.ITEM,
            "section": section.key,
            "appendix": appendix,
            "chapter": chapter,
            "number": item.number,
            "element_id": item.element_id,
            "references": references,
        }
        parts = _pack(item.text, subs, self.max_chars)
        part_ids: list[str] = []
        if len(parts) == 1:
            self._add(id=item_id, breadcrumb=crumb, text=_join(item.text, parts[0]), **common)
            part_ids = [item_id] * len(subs)
        else:
            for number, part in enumerate(parts, start=1):
                part_id = f"{item_id}-p{number}"
                self._add(
                    id=part_id,
                    part=number,
                    breadcrumb=f"{crumb}{SEPARATOR}{number}-qism",
                    text=_join(item.text, part),
                    **common,
                )
                part_ids += [part_id] * len(part)
        if _DEFINITION_MARKER in item.text.lower():
            self._definitions(item, item_id, crumb, part_ids, common)

    def _definitions(
        self, item: Item, item_id: str, crumb: str, part_ids: list[str], common: dict
    ) -> None:
        number = 0
        for sub_item, parent_id in zip(item.sub_items, part_ids, strict=True):
            term = _defined_term(sub_item.text)
            if term is None:
                continue
            number += 1
            self._add(
                id=f"{item_id}-d{number}",
                type=ChunkType.DEFINITION,
                section=common["section"],
                appendix=common["appendix"],
                chapter=common["chapter"],
                number=item.number,
                breadcrumb=f"{crumb}{SEPARATOR}Atama: {term}",
                text=sub_item.text,
                element_id=sub_item.element_id,
                parent_id=parent_id,
                metadata={"term": term},
            )

    def _activity_row(self, row: ActivityRow, appendix: Section, heading: str) -> None:
        deadline = f"{row.deadline} ish kuni" if row.deadline_days is not None else row.deadline
        fee = f"{row.fee} BXM" if row.fee_bxm is not None else row.fee
        text = (
            f"{row.number}. {row.activity} "
            f"Toifasi: {row.category} toifa ({row.hazard}). "
            f"Soha: {row.sector}. "
            f"Davlat ekologik ekspertizasini oʻtkazish muddati: {deadline}. "
            f"Toʻlov miqdori: {fee}."
        )
        self._add(
            id=f"{appendix.key}-r{row.number}",
            type=ChunkType.TABLE_ROW,
            section=appendix.key,
            appendix=appendix.number,
            number=str(row.number),
            breadcrumb=f"{heading}{SEPARATOR}{row.category} toifa{SEPARATOR}{row.number}-qator",
            text=text,
            element_id=row.element_id,
            metadata={
                "category": row.category,
                "hazard": row.hazard,
                "sector": row.sector,
                "deadline_days": row.deadline_days,
                "fee": row.fee,
                "fee_bxm": row.fee_bxm,
            },
        )

    def _table(self, table: Table, key: str, appendix: Section, place: str, title: str) -> None:
        stages = _scheme_stages(table)
        if not stages:
            lines = [_BLANKS.sub("____", cell) for cell in _unique_cells(table.rows)]
            parts = _pack(title, lines, self.max_chars)
            for part_number, part in enumerate(parts, start=1):
                split = len(parts) > 1
                self._add(
                    id=f"{key}-form" + (f"-p{part_number}" if split else ""),
                    type=ChunkType.FORM,
                    section=key,
                    appendix=appendix.number,
                    part=part_number if split else None,
                    breadcrumb=place + (f"{SEPARATOR}{part_number}-qism" if split else ""),
                    text=_join(title, part),
                    element_id=table.element_id,
                )
            return
        for label, lines in stages:
            stage_number = _STAGE.match(label).group(1)  # type: ignore[union-attr]
            parts = _pack(f"{title}. {label}.", lines, self.max_chars)
            for part_number, part in enumerate(parts, start=1):
                suffix = f"-p{part_number}" if len(parts) > 1 else ""
                self._add(
                    id=f"{key}-s{stage_number}{suffix}",
                    type=ChunkType.SCHEME_STAGE,
                    section=key,
                    appendix=appendix.number,
                    number=stage_number,
                    part=part_number if len(parts) > 1 else None,
                    breadcrumb=f"{place}{SEPARATOR}{label}",
                    text=_join(f"{title}. {label}.", part),
                    element_id=table.element_id,
                )

    def _footnotes(
        self, footnotes: list[Footnote], appendix: Section, key: str, place: str
    ) -> None:
        number = 0
        for footnote in footnotes:
            if len(footnote.text) < 12 and footnote.text.rstrip().endswith(":"):
                continue  # a bare "Izoh:" heading
            number += 1
            self._add(
                id=f"{key}-fn{number}",
                type=ChunkType.FOOTNOTE,
                section=appendix.key,
                appendix=appendix.number,
                number=str(number),
                breadcrumb=f"{place}{SEPARATOR}Izoh",
                text=footnote.text,
                element_id=footnote.element_id,
            )


# --- helpers -----------------------------------------------------------------------------------


def _titled(section: Section) -> str:
    doc_type = section.doc_type
    if not doc_type or doc_type == "QARORI":
        return section.title
    return f"{section.title} {doc_type.lower()}".strip()


def _item_text(item: Item) -> str:
    return "\n".join([item.text, *(s.text for s in item.sub_items)])


def _join(lead: str, lines: Iterable[str]) -> str:
    return "\n".join([lead, *lines])


def _pack(lead: str, lines: list[str], max_chars: int) -> list[list[str]]:
    """Group lines into parts that fit max_chars together with the repeated lead."""
    if len(_join(lead, lines)) <= max_chars or not lines:
        return [lines]
    parts: list[list[str]] = [[]]
    for line in lines:
        candidate = [*parts[-1], line]
        if parts[-1] and len(_join(lead, candidate)) > max_chars:
            parts.append([line])
        else:
            parts[-1] = candidate
    return parts


def _defined_term(text: str) -> str | None:
    """The part before the first em dash that is not inside parentheses."""
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(_DASH, index):
            term = text[:index].strip()
            return term if 1 < len(term) <= 200 else None
    return None


def _scheme_stages(table: Table) -> list[tuple[str, list[str]]]:
    """Group scheme rows by stage and render each row with the column labels."""
    header: list[str] = []
    stages: list[tuple[str, list[list[str]]]] = []
    for row in table.rows:
        first = row[0] if row else ""
        if not header and first.lower().startswith("bosqich"):
            header = [c for c in row if c]
            continue
        rest = [c for c in row[1:] if c]
        if _STAGE.match(first) and (not stages or stages[-1][0] != first):
            stages.append((first, []))
        if stages and rest and (not stages[-1][1] or stages[-1][1][-1] != rest):
            stages[-1][1].append(rest)
    if not header:
        return []
    labels = header[1:]
    rendered = []
    for label, rows in stages:
        lines = []
        for cells in rows:
            if len(cells) == len(labels):
                lines.append(
                    " ".join(
                        f"{name}: {value.rstrip('.')}."
                        for name, value in zip(labels, cells, strict=True)
                    )
                )
            else:
                lines.append(" — ".join(cells))
        rendered.append((label, lines))
    return rendered


def _unique_cells(rows: list[list[str]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for cell in row:
            if cell:
                seen.setdefault(cell)
    return list(seen)
