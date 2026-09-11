"""Number extraction used to verify that answers only repeat numbers found in their sources."""

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def canonical_number(raw: str) -> str:
    value = raw.replace(",", ".")
    whole, _, fraction = value.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def extract_numbers(text: str) -> set[str]:
    """All numbers in the text in canonical form ("7,5" and "7.5" both become "7.5")."""
    return {canonical_number(match) for match in _NUMBER.findall(text)}
