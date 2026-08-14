"""Puzzle sources. Currently the Kaggle sudoku-3m CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Puzzle:
    """One row of a puzzle dataset.

    ``difficulty`` is the source's own rating, kept so probe results can be
    stratified by it later.
    """

    id: str
    puzzle: str
    solution: str
    clues: int
    difficulty: float


def read_kaggle(path: str | Path, limit: int | None = None) -> Iterator[Puzzle]:
    """Stream rows of the sudoku-3m CSV without loading the file into memory."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for count, row in enumerate(csv.DictReader(handle)):
            if limit is not None and count >= limit:
                return
            yield Puzzle(
                id=row["id"],
                puzzle=row["puzzle"],
                solution=row["solution"],
                clues=int(row["clues"]),
                difficulty=float(row["difficulty"]),
            )
