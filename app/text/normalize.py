"""Text normalization shared by indexing and querying.

Uzbek Latin is written with several apostrophe look-alikes (oʻ, o', o`, o‘ …) and many users
still type in Cyrillic. Both sides of the search must see exactly the same canonical text.
"""

import re
import unicodedata

_APOSTROPHES = str.maketrans(
    {
        "ʻ": "'",  # ʻ modifier letter turned comma (oʻ, gʻ)
        "ʼ": "'",  # ʼ modifier letter apostrophe (tutuq belgisi)
        "‘": "'",  # ‘
        "’": "'",  # ’
        "‛": "'",  # ‛
        "`": "'",  # `
        "´": "'",  # ´
        "′": "'",  # ′
        "“": '"',  # “
        "”": '"',  # ”
        "„": '"',  # „
        "«": '"',  # «
        "»": '"',  # »
        " ": " ",  # no-break space
    }
)

_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sh", "ъ": "'", "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h",
}  # fmt: skip
_CYRILLIC_VOWELS = set("аеёиоуэюяўы")
_WHITESPACE = re.compile(r"\s+")


def transliterate_cyrillic(text: str) -> str:
    """Convert Uzbek Cyrillic to Uzbek Latin, keeping capitalisation of the first letter."""
    result: list[str] = []
    previous = ""
    for char in text:
        lower = char.lower()
        if lower == "е":
            starts_syllable = not previous.isalpha() or previous in _CYRILLIC_VOWELS | {"ъ", "ь"}
            latin = "ye" if starts_syllable else "e"
        else:
            latin = _CYRILLIC.get(lower)
        if latin is None:
            result.append(char)
        elif char != lower and latin:
            result.append(latin[0].upper() + latin[1:])
        else:
            result.append(latin)
        previous = lower
    return "".join(result)


def canonicalize(text: str) -> str:
    """Canonical form for embeddings and matching: unified apostrophes, Latin script."""
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    text = transliterate_cyrillic(text)
    return _WHITESPACE.sub(" ", text).strip()


def lexical_form(text: str) -> str:
    """Canonical form for lexical search: canonicalize and case-fold."""
    return canonicalize(text).casefold()
