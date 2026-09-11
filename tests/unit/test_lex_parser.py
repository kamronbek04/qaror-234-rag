import re

import pytest

from app.core.errors import ParseIntegrityError
from app.ingestion.lex_parser import iter_texts, parse_document, validate_document

UI_NOISE = (
    "Hujjatga taklif yuborish",
    "Audioni tinglash",
    "Hujjat elementidan havola olish",
    "OKOZ",
)


def appendix(document, number):
    return next(a for a in document.appendices if a.number == number)


def item(section, number):
    return next(i for i in section.all_items() if i.number == number)


class TestStructure:
    def test_document_metadata(self, document):
        assert document.title.startswith("Atrof-muhitga taʼsirni baholashning yangi mexanizmlarini")
        assert document.number == "234"
        assert document.date == "2026-05-11"

    def test_all_nine_appendices_are_recovered(self, document):
        assert [a.number for a in document.appendices] == list(range(1, 10))

    def test_main_resolution_has_preamble_and_eight_items(self, document):
        assert document.main.paragraphs[0].text.endswith("Vazirlar Mahkamasi qaror qiladi:")
        assert [i.number for i in document.main.all_items()] == [str(n) for n in range(1, 9)]

    def test_items_keep_lex_element_ids(self, document):
        assert item(document.main, "7").element_id == "-8205435"
        assert item(appendix(document, 2), "6").element_id == "-8205470"

    def test_regulation_title_type_and_chapters(self, document):
        regulation = appendix(document, 2)

        assert regulation.title == "Davlat ekologik ekspertizasini oʻtkazish tartibi toʻgʻrisida"
        assert regulation.doc_type == "NIZOM"
        assert len(regulation.chapters) == 8
        assert regulation.chapters[0].title == "1-bob. Umumiy qoidalar"

    def test_numbered_item_keeps_its_sub_items(self, document):
        objects = item(appendix(document, 2), "4")

        assert objects.text.startswith("4. Davlat ekologik ekspertizasi obyektlari quyidagilar")
        assert len(objects.sub_items) == 6
        assert objects.sub_items[1].text == "loyihaoldi va loyiha hujjatlari;"

    def test_regulation_attachments_are_nested(self, document):
        assert [x.key for x in appendix(document, 2).attachments] == ["a2-x1", "a2-x2"]
        assert [x.key for x in appendix(document, 7).attachments] == ["a7-x1", "a7-x2", "a7-x3"]
        assert appendix(document, 2).attachments[0].doc_type == "SXEMASI"

    def test_quoted_items_in_amendments_stay_inside_their_item(self, document):
        amendments = appendix(document, 9)

        assert [i.number for i in amendments.all_items()] == ["1", "2", "3"]
        assert any(
            s.text.startswith("4. Ekologik normativlar") for s in item(amendments, "1").sub_items
        )
        assert amendments.tables[0].after_item == "1"

    def test_no_interface_noise_in_any_text(self, document):
        for text in iter_texts(document):
            assert not any(noise in text for noise in UI_NOISE), text[:80]


class TestActivityTable:
    def test_all_rows_are_parsed_in_order(self, document):
        assert [r.number for r in document.activity_rows] == list(range(1, 222))

    @pytest.mark.parametrize(
        ("number", "category", "hazard", "days", "fee"),
        [
            (2, "I", "yuqori darajada xavfli", 25, "25"),
            (69, "II", "oʻrtacha darajada xavfli", 25, "15"),
            (134, "III", "past darajada xavfli", 15, "7,5"),
        ],
    )
    def test_row_values(self, document, number, category, hazard, days, fee):
        row = document.activity_rows[number - 1]

        assert (row.category, row.hazard, row.deadline_days, row.fee) == (
            category,
            hazard,
            days,
            fee,
        )

    def test_row_keeps_sector_and_activity(self, document):
        airport = document.activity_rows[1]

        assert airport.activity == "Aeroportlar."
        assert airport.sector == "Transport, elektrotexnika va yoʻl xoʻjaligi"

    def test_textual_fee_is_preserved(self, document):
        row = document.activity_rows[219]

        assert row.fee_bxm is None
        assert "30 foizi" in row.fee


class TestIntegrity:
    def test_snapshot_passes_integrity_checks(self, document):
        validate_document(document)

    def test_missing_appendix_headers_are_reported(self, snapshot_html):
        broken = re.sub(r"(qaroriga</a><br />)\s*\d-ILOVA", r"\1", snapshot_html)
        assert broken != snapshot_html

        with pytest.raises(ParseIntegrityError, match="ilova"):
            validate_document(parse_document(broken, url="https://lex.uz/uz/docs/-8193120"))
