"""Shared fixtures. Kaggle-backed ones skip when the gitignored CSV is absent."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CSV = REPO_ROOT / "data" / "sudoku-kaggle-dataset.csv"


def read_kaggle_rows(limit: int) -> list[dict[str, str]]:
    """Read the first rows without loading the whole 536 MB file."""
    rows: list[dict[str, str]] = []
    with KAGGLE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


@pytest.fixture(scope="session")
def kaggle_csv() -> Path:
    if not KAGGLE_CSV.exists():
        pytest.skip(f"{KAGGLE_CSV.name} not present")
    return KAGGLE_CSV


@pytest.fixture(scope="session")
def kaggle_rows(kaggle_csv) -> list[dict[str, str]]:
    return read_kaggle_rows(200)


@pytest.fixture
def easy_puzzle() -> tuple[str, str]:
    """A puzzle solvable by singles alone, with its solution."""
    return (
        "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
    )


@pytest.fixture
def hard_puzzle() -> str:
    """Norvig's hardest: 17 clues, a few hundred guesses, depth ~15.

    Its traces run past a thousand tokens, well beyond the paper's 250-token
    trim, so a puzzle like this would never reach the training set.
    """
    return "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"


@pytest.fixture
def row0_hidden_single() -> str:
    """A position where 9 has exactly one legal home left in row 0.

    Rows 1-8 carry a 9 in columns 0-7, one per box, so every cell of row 0 but
    the last has 9 eliminated by its column.
    """
    return (
        "........."
        "9........"
        "...9....."
        ".9......."
        "....9...."
        "......9.."
        "..9......"
        ".....9..."
        ".......9."
    )
