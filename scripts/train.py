"""Train the transformer on a built trace dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sudoku_transformer.data.dataset import load_split
from sudoku_transformer.model import ModelConfig, Transformer
from sudoku_transformer.training import TrainConfig, train

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "traces")
    parser.add_argument("--out", type=Path, default=None, help="run directory")
    parser.add_argument("--size", choices=("paper", "small"), default="small")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-tokens", type=int, default=5_000_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out or REPO_ROOT / "outputs" / "runs" / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    model = Transformer(getattr(ModelConfig, args.size)())
    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        warmup_tokens=args.warmup_tokens,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        device=args.device,
        seed=args.seed,
    )

    print(f"{args.size} model: {model.n_parameters():,} parameters -> {out_dir}")
    summary = train(
        model,
        load_split(args.data, "train"),
        load_split(args.data, "test"),
        cfg,
        out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
