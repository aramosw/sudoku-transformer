"""Rolling the model out on puzzles and scoring the grids it produces."""

from .metrics import (
    by_clues,
    by_difficulty,
    cell_accuracy,
    grid_accuracy,
    stratify,
    summarise,
)
from .rollout import EvalConfig, RolloutResult, generate, rollout_split, simulate

__all__ = [
    "EvalConfig",
    "RolloutResult",
    "by_clues",
    "by_difficulty",
    "cell_accuracy",
    "generate",
    "grid_accuracy",
    "rollout_split",
    "simulate",
    "stratify",
    "summarise",
]
