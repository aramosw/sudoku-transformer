"""Turn a puzzle source into packed token arrays on disk."""

from __future__ import annotations

import json
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sudoku.trace import trace_puzzle
from .sources import Puzzle, read_kaggle
from .tokenizer import encode
from .vocab import PAD, VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Inputs to a dataset build.

    ``test_fraction`` defaults to the paper's split, which held out 150k of
    2.85M puzzles. ``max_len`` is the paper's 250-token trim.
    """

    csv_path: Path
    out_dir: Path
    n_puzzles: int = 100_000
    test_fraction: float = 0.05
    max_len: int = 250
    seed: int = 0
    workers: int | None = None


def _puzzle_seed(master: int, index: int) -> int:
    """Derive a per-puzzle seed.

    Each puzzle gets its own generator so a build is reproducible regardless of
    worker count, and any single puzzle can be regenerated without replaying the
    ones before it.
    """
    return master * 1_000_003 + index


def _trace_one(args: tuple[str, int, int]) -> tuple[list[int] | None, int]:
    """Trace one puzzle, returning its tokens and untrimmed length.

    Tokens are None when the puzzle was unsolvable (length 0) or the trace
    overflowed max_len; the length is reported either way so the manifest can
    describe what was discarded.
    """
    puzzle, seed, max_len = args
    trace = trace_puzzle(puzzle, random.Random(seed))
    if not trace.solved:
        return None, 0
    tokens = encode(trace.events)
    if len(tokens) > max_len:
        return None, len(tokens)
    return tokens, len(tokens)


def _write_split(
    out_dir: Path, name: str, records: list[tuple[Puzzle, list[int]]], max_len: int
) -> None:
    n = len(records)
    tokens = np.full((n, max_len), PAD, dtype=np.uint16)
    lengths = np.zeros(n, dtype=np.uint16)
    n_clues = np.zeros(n, dtype=np.uint8)
    difficulty = np.zeros(n, dtype=np.float32)

    for i, (puzzle, encoded) in enumerate(records):
        tokens[i, : len(encoded)] = encoded
        lengths[i] = len(encoded)
        n_clues[i] = puzzle.clues
        difficulty[i] = puzzle.difficulty

    np.save(out_dir / f"{name}_tokens.npy", tokens)
    np.save(out_dir / f"{name}_lengths.npy", lengths)
    np.save(out_dir / f"{name}_clues.npy", n_clues)
    np.save(out_dir / f"{name}_difficulty.npy", difficulty)


def build(config: BuildConfig) -> dict:
    """Trace every puzzle, pack the survivors, and write the manifest.

    Returns the manifest so callers can report on the build without re-reading
    it. Puzzles whose traces overflow ``max_len`` are dropped, which biases the
    result away from hard puzzles: the manifest records by how much.
    """
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    puzzles = list(read_kaggle(config.csv_path, config.n_puzzles))
    args = [
        (p.puzzle, _puzzle_seed(config.seed, i), config.max_len)
        for i, p in enumerate(puzzles)
    ]
    workers = config.workers or max(1, (os.cpu_count() or 2) - 1)

    start = time.perf_counter()
    if workers == 1:
        results = [_trace_one(a) for a in args]
    else:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_trace_one, args, chunksize=256))
    elapsed = time.perf_counter() - start

    kept: list[tuple[Puzzle, list[int]]] = []
    all_lengths: list[int] = []
    dropped_unsolved = dropped_too_long = 0
    for puzzle, (encoded, length) in zip(puzzles, results, strict=True):
        if length:
            all_lengths.append(length)
        if encoded is None:
            if length:
                dropped_too_long += 1
            else:
                dropped_unsolved += 1
            continue
        kept.append((puzzle, encoded))

    rng = random.Random(config.seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    n_test = round(len(kept) * config.test_fraction)
    test = [kept[i] for i in order[:n_test]]
    train = [kept[i] for i in order[n_test:]]

    _write_split(out_dir, "train", train, config.max_len)
    _write_split(out_dir, "test", test, config.max_len)

    ordered = sorted(all_lengths)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(config.csv_path),
        "seed": config.seed,
        "max_len": config.max_len,
        "vocab_size": VOCAB_SIZE,
        "requested": len(puzzles),
        "kept": len(kept),
        "dropped_too_long": dropped_too_long,
        "dropped_unsolved": dropped_unsolved,
        "dropped_fraction": round(dropped_too_long / max(len(puzzles), 1), 5),
        "train": len(train),
        "test": len(test),
        "token_length": {
            "mean": round(statistics.mean(ordered), 2),
            "median": statistics.median(ordered),
            "p90": ordered[int(0.90 * len(ordered))],
            "p99": ordered[int(0.99 * len(ordered))],
            "max": ordered[-1],
        },
        "workers": workers,
        "elapsed_seconds": round(elapsed, 1),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
