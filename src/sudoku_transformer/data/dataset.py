"""Reading packed splits back off disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Split:
    """One split of a built dataset, arrays aligned row-for-row.

    ``tokens`` is (n, max_len) padded with [pad]; ``lengths`` gives the real
    extent of each row. ``n_clues`` doubles as the index of that row's
    [clues_end] token, which is what the loss mask keys off.
    """

    tokens: np.ndarray
    lengths: np.ndarray
    n_clues: np.ndarray
    difficulty: np.ndarray

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def max_len(self) -> int:
        return self.tokens.shape[1]


def load_split(out_dir: str | Path, name: str, mmap: bool = True) -> Split:
    """Load a split, memory-mapping the token array by default."""
    out_dir = Path(out_dir)
    mode = "r" if mmap else None
    return Split(
        tokens=np.load(out_dir / f"{name}_tokens.npy", mmap_mode=mode),
        lengths=np.load(out_dir / f"{name}_lengths.npy"),
        n_clues=np.load(out_dir / f"{name}_clues.npy"),
        difficulty=np.load(out_dir / f"{name}_difficulty.npy"),
    )


def load_manifest(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))


def target_mask(split: Split) -> np.ndarray:
    """Positions that count towards the loss, as a (n, max_len) bool array.

    Indexed by *target* position: entry j is True when token j is predicted and
    scored. Clues and the [clues_end] token itself are context, so scoring
    starts at j = n_clues + 1 and stops at the end of the real sequence. Under
    the usual shift this means input position i = j - 1 is the last one the
    model sees.
    """
    positions = np.arange(split.max_len)[None, :]
    after_clues = positions > split.n_clues[:, None]
    within = positions < split.lengths[:, None]
    return after_clues & within
