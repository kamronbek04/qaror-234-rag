import pytest

from app.domain.models import ChunkType
from app.ingestion.chunker import StructuralChunker
from app.ingestion.fixed_chunker import FixedSizeChunker

MAX_CHARS = 1500


@pytest.fixture(scope="module")
def chunks(document):
    return StructuralChunker(max_chars=MAX_CHARS).chunk(document)


@pytest.fixture(scope="module")
def by_id(chunks):
    return {c.id: c for c in chunks}


class TestItems:
    def test_regulation_item_has_breadcrumb_text_and_deep_link(self, by_id):
        chunk = by_id["a2-b6"]

        assert chunk.type is ChunkType.ITEM
        assert chunk.breadcrumb == (
            "2-ilova › Davlat ekologik ekspertizasini oʻtkazish tartibi toʻgʻrisida nizom"
            " › 1-bob. Umumiy qoidalar › 6-band"
        )
        assert "buyurtmachining (tashabbuskorning) mablagʻlari hisobidan" in chunk.text
        assert chunk.url == "https://lex.uz/uz/docs/-8193120#-8205470"

    def test_main_resolution_item(self, by_id):
        chunk = by_id["q-b6"]

        assert chunk.breadcrumb == "Qaror › 6-band"
        assert "541-son" in chunk.text
        assert "oʻz kuchini yoʻqotgan" in chunk.text
        assert by_id["q-b7"].url.endswith("#-8205435")

    def test_preamble_is_a_paragraph_chunk(self, by_id):
        assert by_id["q-p1"].type is ChunkType.PARAGRAPH
        assert by_id["q-p1"].text.endswith("Vazirlar Mahkamasi qaror qiladi:")

    def test_item_text_keeps_sub_items(self, by_id):
        text = by_id["a2-b4"].text

        assert text.startswith("4. Davlat ekologik ekspertizasi obyektlari")
        assert "loyihaoldi va loyiha hujjatlari;" in text

    def test_appendix_overview_lists_chapters(self, by_id):
        overview = by_id["a2"]

        assert overview.type is ChunkType.OVERVIEW
        assert "8-bob. Yakunlovchi qoidalar" in overview.text


class TestDefinitions:
    def test_definition_chunk_for_ekolog_ekspert(self, chunks):
        definition = next(
            c
            for c in chunks
            if c.type is ChunkType.DEFINITION and c.metadata["term"] == "ekolog-ekspert"
        )

        assert "kamida uzluksiz uch yil ish stajiga ega" in definition.text
        assert definition.parent_id.startswith("a2-b2")
        assert definition.breadcrumb.endswith("› Atama: ekolog-ekspert")

    def test_term_with_dash_inside_parentheses(self, chunks):
        terms = {c.metadata["term"] for c in chunks if c.type is ChunkType.DEFINITION}

        assert (
            "atrof-muhitga taʼsirni baholash materiallarini va ekologik normativlar loyihalarini "
            "ishlab chiquvchi (keyingi oʻrinlarda — loyihani ishlab chiquvchi)"
        ) in terms

    def test_all_definitions_come_from_the_glossary_items(self, chunks):
        definitions = [c for c in chunks if c.type is ChunkType.DEFINITION]

        assert len(definitions) == 41  # 9 + 10 + 4 + 3 + 5 + 5 + 5 in appendices 2-8
        assert {c.parent_id.split("-p")[0] for c in definitions} == {
            f"a{n}-b2" for n in range(2, 9)
        }


class TestTableRows:
    def test_every_activity_row_is_a_chunk(self, chunks):
        assert sum(c.type is ChunkType.TABLE_ROW for c in chunks) == 221

    def test_airport_row_is_a_standalone_statement(self, by_id):
        row = by_id["a1-r2"]

        for fragment in (
            "Aeroportlar.",
            "I toifa (yuqori darajada xavfli)",
            "Transport, elektrotexnika va yoʻl xoʻjaligi",
            "25 ish kuni",
            "25 BXM",
        ):
            assert fragment in row.text
        assert row.metadata["deadline_days"] == 25
        assert row.metadata["fee_bxm"] == 25.0
        assert row.breadcrumb.endswith("› I toifa › 2-qator")

    def test_medium_hazard_row(self, by_id):
        row = by_id["a1-r69"]

        assert "II toifa (oʻrtacha darajada xavfli)" in row.text
        assert "25 ish kuni" in row.text
        assert "15 BXM" in row.text


class TestSchemesFootnotesForms:
    def test_footnote_about_unlisted_activities(self, chunks):
        footnotes = [c for c in chunks if c.type is ChunkType.FOOTNOTE and c.section == "a1"]

        assert any(
            "nazarda tutilmagan faoliyat turlari ham" in c.text and "ekspertlar kengashi" in c.text
            for c in footnotes
        )
        assert all(c.text != "Izoh:" for c in footnotes)

    def test_every_scheme_has_stage_chunks(self, chunks):
        schemes = {c.id.rsplit("-s", 1)[0] for c in chunks if c.type is ChunkType.SCHEME_STAGE}

        assert {"a2-x1", "a4-x1", "a5-x1", "a6-x1", "a7-x1", "a8-x1", "a9-t1"} <= schemes

    def test_scheme_stage_names_actor_and_action(self, by_id):
        stage = by_id["a2-x1-s1"]

        assert "1-bosqich" in stage.text
        assert "Buyurtmachi (tashabbuskor)" in stage.text
        assert "sxemasi" in stage.breadcrumb

    def test_application_form_is_split_into_bounded_parts(self, chunks):
        parts = [c for c in chunks if c.id.startswith("a2-x2-form")]

        assert parts
        assert all(c.type is ChunkType.FORM for c in parts)
        assert all(len(c.text) <= MAX_CHARS for c in parts)

    def test_no_chunk_exceeds_the_limit_without_a_single_oversized_line(self, chunks):
        for chunk in chunks:
            assert len(chunk.text) <= MAX_CHARS or chunk.text.count("\n") <= 1, chunk.id


class TestSizeBounds:
    def test_long_amendment_item_is_split_at_sub_items(self, chunks, document):
        amendment = next(a for a in document.appendices if a.number == 9).all_items()[0]
        parts = [c for c in chunks if c.id.startswith("a9-b1-p")]

        assert len(parts) >= 2
        for number, part in enumerate(parts, start=1):
            assert part.part == number
            assert part.text.startswith(amendment.text)
            assert part.breadcrumb.endswith(f"› 1-band › {number}-qism")
            assert len(part.text) <= MAX_CHARS or part.text.count("\n") == 1
        for sub_item in amendment.sub_items:
            assert sum(sub_item.text in p.text for p in parts) >= 1


class TestReferences:
    def test_reference_to_another_item(self, by_id):
        assert "a2-b4" in by_id["a2-b5"].references

    def test_main_resolution_references_appendices(self, by_id):
        assert {"a1", "a2", "a8"} <= set(by_id["q-b2"].references)

    def test_every_reference_resolves(self, chunks, by_id):
        for chunk in chunks:
            for reference in chunk.references:
                assert reference in by_id, (chunk.id, reference)


class TestDeterminism:
    def test_ids_are_unique(self, chunks):
        assert len({c.id for c in chunks}) == len(chunks)

    def test_chunking_twice_gives_identical_output(self, document, chunks):
        assert StructuralChunker(max_chars=MAX_CHARS).chunk(document) == chunks

    def test_order_follows_document(self, chunks):
        assert [c.order for c in chunks] == list(range(len(chunks)))


class TestFixedSizeBaseline:
    def test_windows_have_size_and_overlap(self, document):
        windows = FixedSizeChunker(size=400, overlap=100).chunk(document)

        assert all(c.type is ChunkType.WINDOW for c in windows)
        assert len(windows[0].text) == 400
        assert windows[1].text[:100] == windows[0].text[-100:]
        assert len({c.id for c in windows}) == len(windows)
