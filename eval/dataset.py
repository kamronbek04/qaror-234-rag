"""Golden question set: loading and validation against the current index."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["in_doc", "out_of_doc", "trap", "partial"]
DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_APOSTROPHE_VARIANTS = "`‘’ʻʼ"


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class EvalItem:
    id: str
    kind: Kind
    question: str
    expected_chunks: tuple[str, ...] = field(default_factory=tuple)
    must_include: tuple[str, ...] = field(default_factory=tuple)
    forbidden: tuple[str, ...] = field(default_factory=tuple)

    @property
    def non_canonical_spelling(self) -> bool:
        """Cyrillic script or apostrophe look-alikes — checks normalization end to end."""
        return bool(_CYRILLIC.search(self.question)) or any(
            c in self.question for c in _APOSTROPHE_VARIANTS
        )


def load_dataset(path: Path = DATASET_PATH) -> list[EvalItem]:
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            items.append(
                EvalItem(
                    id=raw["id"],
                    kind=raw["kind"],
                    question=raw["question"],
                    expected_chunks=tuple(raw.get("expected_chunks", ())),
                    must_include=tuple(raw.get("must_include", ())),
                    forbidden=tuple(raw.get("forbidden", ())),
                )
            )
        except KeyError as exc:
            raise DatasetError(f"{path.name}:{number}: {exc.args[0]} maydoni yo'q") from exc
    return items


def validate_dataset(items: Iterable[EvalItem], known_chunk_ids: set[str]) -> None:
    unknown = sorted({c for item in items for c in item.expected_chunks} - known_chunk_ids)
    if unknown:
        raise DatasetError("Indeksda yo'q kutilgan bo'laklar: " + ", ".join(unknown))
