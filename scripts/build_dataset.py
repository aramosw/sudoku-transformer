"""Build a tokenised trace dataset from the Kaggle sudoku-3m CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sudoku_transformer.data import BuildConfig, build

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=REPO_ROOT / "data" / "sudoku-kaggle-dataset.csv"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "traces")
    parser.add_argument("-n", "--n-puzzles", type=int, default=100_000)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    manifest = build(
        BuildConfig(
            csv_path=args.csv,
            out_dir=args.out,
            n_puzzles=args.n_puzzles,
            test_fraction=args.test_fraction,
            max_len=args.max_len,
            seed=args.seed,
            workers=args.workers,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
