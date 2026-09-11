"""Tokenizer and light inflectional stemmer for Uzbek (Latin) used by the BM25 index.

Uzbek is agglutinative: "ekspertiza", "ekspertizasi", "ekspertizasidan" and "ekspertizaning"
are one lexeme. Stripping the frequent plural, possessive and case suffixes (longest first,
at most three passes) is enough to make them match without a full morphological analyser.
"""

import re

from app.text.normalize import lexical_form

# Longest first. First/second-person possessives are left out on purpose: legal text never
# uses them, and stripping them over-stems words such as "monitoring".
_SUFFIXES = (
    "larining", "laridan", "larida", "lariga", "larini",
    "gacha", "ning", "dagi", "lari",
    "lar", "dan",
    "si", "ni", "ga", "da",
    "i",
)  # fmt: skip
_DOUBLED_DATIVE = {"qa": "q", "ka": "k"}  # tuproqqa -> tuproq, bankka -> bank
_MIN_STEM = 3
_MAX_PASSES = 3

STOPWORDS = frozenset(
    {
        "va", "bilan", "uchun", "bo'yicha", "hamda", "yoki", "ham", "shu", "ushbu", "mazkur",
        "bu", "u", "ular", "qanday", "qancha", "nima", "necha", "nechta", "kim", "qaysi",
        "qachon", "qayerda", "nega", "esa", "deb", "agar", "lekin", "ammo", "o'z", "har",
        "mi", "haqida", "bormi",
    }
)  # fmt: skip

_TOKEN = re.compile(r"[a-z0-9']+")


def stem(token: str) -> str:
    if token.isdigit() or len(token) <= _MIN_STEM:
        return token
    word = token
    stripped = False
    for _ in range(_MAX_PASSES):
        suffix = _matching_suffix(word)
        if suffix is None:
            break
        word = word[: -len(suffix)]
        stripped = True
    if stripped and word.endswith("lig"):
        word = word[:-1] + "k"  # vazirligi -> vazirlik (k/g alternation)
    return word


def _matching_suffix(word: str) -> str | None:
    for dative, consonant in _DOUBLED_DATIVE.items():
        if word.endswith(consonant + dative) and len(word) - len(dative) >= _MIN_STEM:
            return dative
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return suffix
    return None


def tokenize(text: str) -> list[str]:
    """Lexical tokens: canonical lowercase, hyphen-split, stopwords removed, stemmed."""
    tokens = []
    for raw in _TOKEN.findall(lexical_form(text).replace("-", " ")):
        token = raw.strip("'")
        if token and token not in STOPWORDS:
            tokens.append(stem(token))
    return tokens
