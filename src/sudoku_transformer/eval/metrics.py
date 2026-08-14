"""Scoring rollouts: per-cell and per-grid accuracy, and how they break down."""

from __future__ import annotations

from collections.abc import Callable

from ..sudoku.board import NUM_CELLS
from .rollout import RolloutResult


def cell_accuracy(results: list[RolloutResult]) -> float:
    """Fraction of the 81 cells filled correctly, averaged over puzzles.

    The paper's headline is 98.4% here. Clues count towards it, since they are
    part of the grid the model hands back.
    """
    if not results:
        return 0.0
    return sum(r.correct_cells for r in results) / (len(results) * NUM_CELLS)


def grid_accuracy(results: list[RolloutResult]) -> float:
    """Fraction of puzzles solved completely. The paper reports 97.5%."""
    if not results:
        return 0.0
    return sum(r.solved for r in results) / len(results)


def summarise(results: list[RolloutResult]) -> dict:
    """Headline metrics plus the diagnostics that explain a bad number."""
    if not results:
        return {"n": 0}

    n = len(results)
    return {
        "n": n,
        "cell_accuracy": cell_accuracy(results),
        "grid_accuracy": grid_accuracy(results),
        "finished_fraction": sum(r.finished for r in results) / n,
        "illegal_per_puzzle": sum(r.illegal for r in results) / n,
        "puzzles_with_illegal": sum(r.illegal > 0 for r in results) / n,
        "mean_generated": sum(r.generated for r in results) / n,
    }


def stratify(
    results: list[RolloutResult], key: Callable[[RolloutResult], object]
) -> dict[object, dict]:
    """Summarise separately per group, e.g. by clue count or difficulty band."""
    groups: dict[object, list[RolloutResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)
    return {
        name: summarise(group)
        for name, group in sorted(groups.items(), key=lambda kv: kv[0])
    }


def by_clues(results: list[RolloutResult]) -> dict[object, dict]:
    return stratify(results, lambda r: r.n_clues)


def by_difficulty(results: list[RolloutResult]) -> dict[object, dict]:
    """Grouped into the source dataset's integer difficulty bands."""
    return stratify(results, lambda r: int(r.difficulty))
