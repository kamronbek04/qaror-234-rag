"""Parser for lex.uz document pages.

lex.uz renders every paragraph of an act as a flat sibling ``div`` inside ``#divCont`` whose
first CSS class says what the paragraph is (``ACT_TEXT``, ``TEXT_HEADER_DEFAULT``,
``TABLE_STD2`` …) and whose inner ``div[id]`` holds the text and a stable element id. A small
state machine over that flat list rebuilds the legal hierarchy.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from app.core.errors import ParseIntegrityError
from app.domain.models import (
    ActivityRow,
    Chapter,
    Document,
    Footnote,
    Item,
    Paragraph,
    Section,
    Table,
)

_UI_NOISE = ("Hujjatga taklif yuborish", "Audioni tinglash", "Hujjat elementidan havola olish")
_CLASSIFIER_NOTE = re.compile(r"\[\s*OKOZ:[^\]]*\]")
_WHITESPACE = re.compile(r"\s+")
_BLOCK_TAGS = {"br", "p", "div", "li", "tr", "td", "th", "table"}

_ITEM_NUMBER = re.compile(r"^(\d+)(?:-(\d+))?\.\s")
_CHAPTER = re.compile(r"^(\d+)-bob\b")
_APPENDIX_BANNER = re.compile(r"qaroriga\s+(\d+)-ILOVA", re.IGNORECASE)
_ATTACHMENT_BANNER = re.compile(r"nizomga\s+(?:(\d+)-)?ILOVA", re.IGNORECASE)
_ACT_META = re.compile(r"(\d{4})-yil\s+(\d{1,2})-([a-zʻ']+?)dagi\s+(\d+)-son", re.IGNORECASE)
_MONTHS = {
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6,
    "iyul": 7, "avgust": 8, "sentabr": 9, "oktabr": 10, "noyabr": 11, "dekabr": 12,
}  # fmt: skip

_CATEGORY_ROW = re.compile(r"^(III|II|IV|I)\.\s+(.*)$")
_SECTOR_ROW = re.compile(r"^(\d+)\.\s+(.*)$")
_HAZARD = re.compile(r"\(([^)]*xavfli)\)")

EXPECTED_APPENDICES = range(1, 10)
MIN_ACTIVITY_ROWS = 100


@dataclass
class _Element:
    css: str
    element_id: str | None
    text: str
    node: Tag


def parse_document(html: str, *, url: str) -> Document:
    builder = _DocumentBuilder(url)
    for element in _elements(html):
        builder.feed(element)
    return builder.build()


def validate_document(document: Document) -> None:
    """Raise ParseIntegrityError when the parse result does not look like resolution 234."""
    problems: list[str] = []
    found = {a.number for a in document.appendices}
    missing = [n for n in EXPECTED_APPENDICES if n not in found]
    if missing:
        problems.append("topilmagan ilovalar: " + ", ".join(f"{n}-ilova" for n in missing))
    if not document.main.all_items():
        problems.append("asosiy qarorda raqamlangan bandlar topilmadi")
    for appendix in document.appendices:
        if appendix.doc_type == "NIZOM" and not appendix.all_items():
            problems.append(f"{appendix.label}: nizom bandlari topilmadi")
    rows = document.activity_rows
    if len(rows) < MIN_ACTIVITY_ROWS:
        problems.append(f"1-ilova jadvalida {len(rows)} ta qator (kamida {MIN_ACTIVITY_ROWS})")
    elif [r.number for r in rows] != list(range(1, len(rows) + 1)):
        problems.append("1-ilova jadvali qatorlari ketma-ket emas")
    if problems:
        raise ParseIntegrityError("Hujjat tuzilmasi tekshiruvi: " + "; ".join(problems))


def iter_texts(document: Document) -> Iterator[str]:
    """Every piece of text in the parsed document (used by tests and integrity checks)."""
    yield document.title
    for section in [document.main, *document.appendices]:
        yield from _section_texts(section)
    for row in document.activity_rows:
        yield from (row.category_title, row.sector, row.activity, row.deadline, row.fee)


def _section_texts(section: Section) -> Iterator[str]:
    yield section.title
    yield from (p.text for p in section.paragraphs)
    yield from (c.title for c in section.chapters)
    for item in section.all_items():
        yield item.text
        yield from (s.text for s in item.sub_items)
    for table in section.tables:
        yield from (cell for row in table.rows for cell in row)
    yield from (f.text for f in section.footnotes)
    for attachment in section.attachments:
        yield from _section_texts(attachment)


# --- HTML extraction -------------------------------------------------------------------------


def _elements(html: str) -> list[_Element]:
    container = BeautifulSoup(html, "lxml").find(id="divCont")
    if not isinstance(container, Tag):
        raise ParseIntegrityError("Sahifada hujjat matni (#divCont) topilmadi")
    elements = []
    for block in container.find_all("div", recursive=False):
        classes = block.get("class") or []
        if "lx_elem" not in classes:
            continue
        content = block.find("div", id=True)
        if content is None:
            continue
        for hidden in content.select(".lx_no_select"):
            hidden.decompose()
        elements.append(_Element(classes[0], content.get("id"), _text(content), content))
    return elements


def _text(tag: Tag) -> str:
    pieces: list[str] = []
    for node in tag.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            pieces.append(str(node))
        elif isinstance(node, Tag) and node.name in _BLOCK_TAGS:
            pieces.append(" ")
    text = "".join(pieces)
    for noise in _UI_NOISE:
        text = text.replace(noise, " ")
    text = _CLASSIFIER_NOTE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _table_grid(tag: Tag) -> list[list[str]]:
    """Table rows with rowspan/colspan expanded so every row has its full set of cells."""
    grid: list[list[str]] = []
    carried: dict[int, list] = {}  # column -> [rows remaining, text]
    for tr in tag.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        row: list[str] = []
        column = 0
        index = 0
        while index < len(cells) or column <= max(carried, default=-1):
            if column in carried:
                remaining, text = carried[column]
                row.append(text)
                if remaining <= 1:
                    del carried[column]
                else:
                    carried[column][0] = remaining - 1
                column += 1
                continue
            if index >= len(cells):
                row.append("")
                column += 1
                continue
            cell = cells[index]
            index += 1
            text = _text(cell)
            rowspan = _span(cell.get("rowspan"))
            for offset in range(_span(cell.get("colspan"))):
                value = text if offset == 0 else ""
                row.append(value)
                if rowspan > 1:
                    carried[column] = [rowspan - 1, value]
                column += 1
        grid.append(row)
    return grid


def _span(value: object) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return 1


# --- Tree building ---------------------------------------------------------------------------


class _DocumentBuilder:
    def __init__(self, url: str) -> None:
        self.url = url
        self.title = ""
        self.number: str | None = None
        self.date: str | None = None
        self.main = Section(key="q", label="Qaror")
        self.appendices: list[Section] = []
        self.section = self.main
        self.appendix: Section | None = None
        self.chapter: Chapter | None = None
        self.item: Item | None = None
        self.pending_caption: str | None = None

    def feed(self, element: _Element) -> None:
        handlers = {
            "ACT_TITLE": self._on_act_title,
            "APPL_BANNER_LANDSCAPE_TITLE": self._on_banner,
            "ACT_TITLE_APPL": self._on_section_title,
            "ACT_FORM": self._on_form,
            "TEXT_HEADER_DEFAULT": self._on_chapter,
            "ACT_TEXT": self._on_text,
            "TEXT_CENTER": self._on_text,
            "TABLE_STD2": self._on_table,
            "FOOTNOTE": self._on_footnote,
        }
        handler = handlers.get(element.css)
        if handler and (element.text or element.css == "TABLE_STD2"):
            handler(element)

    def build(self) -> Document:
        appendix_one = next((a for a in self.appendices if a.number == 1), None)
        rows = (
            parse_activity_rows(appendix_one.tables[0])
            if appendix_one and appendix_one.tables
            else []
        )
        return Document(
            title=self.title,
            url=self.url,
            number=self.number,
            date=self.date,
            main=self.main,
            appendices=self.appendices,
            activity_rows=rows,
        )

    def _enter(self, section: Section) -> None:
        self.section = section
        self.chapter = None
        self.item = None
        self.pending_caption = None

    def _on_act_title(self, element: _Element) -> None:
        self.title = element.text

    def _on_banner(self, element: _Element) -> None:
        self._read_act_meta(element.text)
        if match := _APPENDIX_BANNER.search(element.text):
            number = int(match.group(1))
            appendix = Section(
                key=f"a{number}",
                label=f"{number}-ilova",
                number=number,
                element_id=element.element_id,
            )
            self.appendices.append(appendix)
            self.appendix = appendix
            self._enter(appendix)
        elif match := _ATTACHMENT_BANNER.search(element.text):
            parent = self.appendix or self.main
            number = int(match.group(1)) if match.group(1) else len(parent.attachments) + 1
            label = f"Nizomga {match.group(1)}-ilova" if match.group(1) else "Nizomga ilova"
            attachment = Section(
                key=f"{parent.key}-x{number}",
                label=label,
                number=number,
                element_id=element.element_id,
            )
            parent.attachments.append(attachment)
            self._enter(attachment)

    def _read_act_meta(self, text: str) -> None:
        if self.number or not (match := _ACT_META.search(text)):
            return
        year, day, month_word, number = match.groups()
        month = _MONTHS.get(month_word.lower())
        self.number = number
        if month:
            self.date = f"{year}-{month:02d}-{int(day):02d}"

    def _on_section_title(self, element: _Element) -> None:
        if not self.section.title:
            self.section.title = element.text
            self.section.element_id = self.section.element_id or element.element_id
        else:  # a heading quoted inside an amendment, e.g. a scheme being inserted
            self.pending_caption = element.text.strip('“”" ')

    def _on_form(self, element: _Element) -> None:
        if self.pending_caption:
            self.pending_caption = f"{self.pending_caption} {element.text.lower()}"
        elif self.section.doc_type is None:
            self.section.doc_type = element.text.upper()

    def _on_chapter(self, element: _Element) -> None:
        match = _CHAPTER.match(element.text)
        self.chapter = Chapter(
            number=match.group(1) if match else None,
            title=element.text,
            element_id=element.element_id,
        )
        self.section.chapters.append(self.chapter)
        self.item = None

    def _on_text(self, element: _Element) -> None:
        paragraph = Paragraph(text=element.text, element_id=element.element_id)
        match = _ITEM_NUMBER.match(element.text)
        if match and self._is_next_item(int(match.group(1)), match.group(2)):
            number = match.group(1) + (f"-{match.group(2)}" if match.group(2) else "")
            self.item = Item(number=number, text=element.text, element_id=element.element_id)
            (self.chapter.items if self.chapter else self.section.items).append(self.item)
        elif self.item is not None:
            self.item.sub_items.append(paragraph)
        else:
            self.section.paragraphs.append(paragraph)

    def _is_next_item(self, base: int, sub: str | None) -> bool:
        """Numbered paragraphs that break the sequence are quotations inside an item."""
        items = self.section.all_items()
        if not items:
            return base == 1 and sub is None
        previous_base, _, previous_sub = items[-1].number.partition("-")
        if sub is None:
            return base == int(previous_base) + 1
        return base == int(previous_base) and int(sub) == int(previous_sub or 0) + 1

    def _on_table(self, element: _Element) -> None:
        self.section.tables.append(
            Table(
                element_id=element.element_id,
                caption=self.pending_caption,
                rows=_table_grid(element.node),
                after_item=self.item.number if self.item else None,
            )
        )
        self.pending_caption = None

    def _on_footnote(self, element: _Element) -> None:
        self.section.footnotes.append(Footnote(text=element.text, element_id=element.element_id))


# --- Appendix 1 activity table ---------------------------------------------------------------


def parse_activity_rows(table: Table) -> list[ActivityRow]:
    rows: list[ActivityRow] = []
    category = hazard = category_title = ""
    sector = ""
    sector_number: int | None = None
    for cells in table.rows:
        filled = [c for c in cells if c]
        if not filled:
            continue
        if len(set(filled)) == 1:  # a heading spanning the whole row
            heading = filled[0]
            if match := _CATEGORY_ROW.match(heading):
                category, category_title = match.group(1), heading
                hazard_match = _HAZARD.search(heading)
                hazard = hazard_match.group(1) if hazard_match else ""
            elif match := _SECTOR_ROW.match(heading):
                sector_number, sector = int(match.group(1)), match.group(2).strip()
            continue
        number = cells[0].rstrip(".").strip()
        if not number.isdigit():
            continue  # column header
        deadline = cells[2] if len(cells) > 2 else ""
        fee = cells[3] if len(cells) > 3 else ""
        rows.append(
            ActivityRow(
                number=int(number),
                category=category,
                hazard=hazard,
                category_title=category_title,
                sector_number=sector_number,
                sector=sector,
                activity=cells[1],
                deadline=deadline,
                deadline_days=int(deadline) if deadline.isdigit() else None,
                fee=fee,
                fee_bxm=_as_float(fee),
                element_id=table.element_id,
            )
        )
    return rows


def _as_float(value: str) -> float | None:
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None
