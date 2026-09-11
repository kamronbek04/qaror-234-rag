"""Number extraction used to verify answers and to spot exact numeric anchors in questions."""

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# The resolution's own number: naming it is never a factual claim about its content.
DOCUMENT_NUMBER = "234"


def canonical_number(raw: str) -> str:
    value = raw.replace(",", ".")
    whole, _, fraction = value.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def extract_numbers(text: str) -> set[str]:
    """All numbers in the text in canonical form ("7,5" and "7.5" both become "7.5")."""
    return {canonical_number(match) for match in _NUMBER.findall(text)}


def anchor_numbers(text: str) -> set[str]:
    """Multi-digit numbers other than the resolution's own number (e.g. "541", "100", "0.6")."""
    return {
        n
        for n in extract_numbers(text)
        if n != DOCUMENT_NUMBER and len(n.replace(".", "").lstrip("0")) >= 2
    }
