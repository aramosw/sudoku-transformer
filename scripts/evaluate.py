"""Roll a trained checkpoint out on the test split and report accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sudoku_transformer.data.dataset import load_split
from sudoku_transformer.eval import (
    EvalConfig,
    by_clues,
    by_difficulty,
    rollout_split,
    summarise,
)
from sudoku_transformer.training.checkpoint import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "traces")
    parser.add_argument("--split", default="test")
    parser.add_argument("-n", "--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, device=args.device)["model"]
    cfg = EvalConfig(
        batch_size=args.batch_size, temperature=args.temperature, device=args.device
    )
    results = rollout_split(model, load_split(args.data, args.split), cfg, args.limit)

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "overall": summarise(results),
        "by_clues": {str(k): v for k, v in by_clues(results).items()},
        "by_difficulty": {str(k): v for k, v in by_difficulty(results).items()},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
