import pytest

from app.text.normalize import canonicalize, lexical_form, transliterate_cyrillic
from app.text.numbers import extract_numbers
from app.text.stemmer import stem, tokenize


class TestNormalize:
    def test_apostrophe_variants_are_equivalent(self):
        variants = ["oʻtkazish", "o'tkazish", "o`tkazish", "o‘tkazish", "o’tkazish"]

        assert {canonicalize(v) for v in variants} == {"o'tkazish"}

    def test_tutuq_belgisi_is_unified_too(self):
        assert canonicalize("taʼsir") == canonicalize("ta'sir") == "ta'sir"

    def test_cyrillic_is_transliterated_to_latin(self):
        assert canonicalize("Экологик экспертиза") == canonicalize("Ekologik ekspertiza")

    @pytest.mark.parametrize(
        ("cyrillic", "latin"),
        [
            ("Аэропорт учун экспертиза муддати", "Aeroport uchun ekspertiza muddati"),
            ("ўтказиш", "o'tkazish"),
            ("ғалла", "g'alla"),
            ("қарор", "qaror"),
            ("ҳудуд", "hudud"),
            ("шартнома", "shartnoma"),
            ("Ер ости", "Yer osti"),
            ("юридик", "yuridik"),
            ("Тошкент", "Toshkent"),
            ("маъмурий", "ma'muriy"),
        ],
    )
    def test_uzbek_cyrillic_letters(self, cyrillic, latin):
        assert transliterate_cyrillic(cyrillic) == latin

    def test_whitespace_collapses(self):
        assert (
            canonicalize("  davlat \n\t ekologik   ekspertizasi ") == "davlat ekologik ekspertizasi"
        )

    def test_lexical_form_is_lowercase_and_canonical(self):
        assert lexical_form("Davlat Ekologik EKSPERTIZASI oʻtkazish") == (
            "davlat ekologik ekspertizasi o'tkazish"
        )


class TestStemmer:
    def test_inflected_forms_share_one_stem(self):
        forms = ["ekspertiza", "ekspertizasi", "ekspertizasidan", "ekspertizaning", "ekspertizani"]

        assert {stem(f) for f in forms} == {"ekspertiza"}

    def test_plural_and_possessive_are_removed(self):
        assert stem("obyektlar") == stem("obyektlari") == stem("obyektlarining") == "obyekt"

    def test_consonant_alternation_is_restored(self):
        assert stem("vazirligi") == stem("vazirlik") == "vazirlik"

    def test_short_words_and_numbers_stay_intact(self):
        assert stem("va") == "va"
        assert stem("uch") == "uch"
        assert stem("25") == "25"

    def test_tokenize_splits_hyphens_drops_stopwords_and_stems(self):
        tokens = tokenize("2-ilovaning 6-bandida atrof-muhitga taʼsir va boshqalar")

        assert tokens == ["2", "ilova", "6", "band", "atrof", "muhit", "ta'sir", "boshqa"]

    def test_tokenize_accepts_cyrillic(self):
        assert tokenize("экспертизасидан") == ["ekspertiza"]


class TestNumbers:
    def test_decimal_comma_equals_decimal_point(self):
        assert extract_numbers("7,5 BXM") == extract_numbers("7.5 BXM") == {"7.5"}

    def test_numbers_attached_to_words(self):
        assert extract_numbers("2026-yil 11-may") == {"2026", "11"}

    def test_units(self):
        assert extract_numbers("110 kV va 0,6 MPa") == {"110", "0.6"}

    def test_trailing_zero_fraction_and_leading_zeros(self):
        assert extract_numbers("25,0 va 05") == {"25", "5"}

    def test_text_without_numbers(self):
        assert extract_numbers("uch oy o'tgach") == set()
