"""Training loop: batching, masked loss, schedule, and the run itself."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.dataset import Split
from ..model.transformer import Transformer
from .checkpoint import save_checkpoint


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation settings.

    The paper's: AdamW at 1e-3, weight decay 0.1, batch size 512, six epochs,
    and a cosine schedule with 5M tokens of linear warmup. Gradient clipping and
    the decay exemption for biases and layer norms are standard practice it does
    not mention.
    """

    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_tokens: int = 5_000_000
    epochs: int = 6
    grad_clip: float = 1.0
    min_lr_fraction: float = 0.0
    bucket_by_length: bool = True
    eval_batches: int = 20
    eval_every: int = 500
    checkpoint_every: int = 2000
    log_every: int = 50
    max_steps: int | None = None
    seed: int = 0
    device: str = "cpu"


def lr_at(tokens_seen: int, total_tokens: int, cfg: TrainConfig) -> float:
    """Linear warmup over ``warmup_tokens``, then cosine decay across the run."""
    if tokens_seen < cfg.warmup_tokens:
        return cfg.lr * tokens_seen / max(cfg.warmup_tokens, 1)
    span = max(total_tokens - cfg.warmup_tokens, 1)
    progress = min((tokens_seen - cfg.warmup_tokens) / span, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * cosine)


def batch_indices(
    split: Split,
    batch_size: int,
    rng: np.random.Generator,
    bucket: bool = True,
    pool_factor: int = 50,
) -> list[np.ndarray]:
    """Group row indices into batches, optionally bucketing by trace length.

    Padding every row to 250 tokens wastes roughly two thirds of the compute
    when the median trace is 84, so rows are sorted by length inside shuffled
    pools: batches stay length-homogeneous while their contents still vary from
    epoch to epoch.
    """
    order = rng.permutation(len(split))
    if bucket:
        pool = batch_size * pool_factor
        chunks = [order[i : i + pool] for i in range(0, len(order), pool)]
        order = np.concatenate(
            [chunk[np.argsort(split.lengths[chunk], kind="stable")] for chunk in chunks]
        )
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def batch_tensors(
    split: Split, index: np.ndarray, device: str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Materialise one batch, trimmed to its longest trace.

    Returns the tokens and a boolean mask over *target* positions: True where a
    token is predicted and scored, which is everything after [clues_end] and
    before the padding.
    """
    index = np.sort(index)
    lengths = split.lengths[index].astype(np.int64)
    n_clues = split.n_clues[index].astype(np.int64)
    width = int(lengths.max())

    tokens = np.asarray(split.tokens[index][:, :width], dtype=np.int64)
    positions = np.arange(width)
    mask = (positions[None, :] > n_clues[:, None]) & (
        positions[None, :] < lengths[:, None]
    )

    return (
        torch.from_numpy(tokens).to(device),
        torch.from_numpy(mask).to(device),
    )


def masked_loss(
    model: Transformer, tokens: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Next-token cross-entropy over scored positions only.

    Position i predicts token i+1, so the target mask is the batch mask shifted
    left by one. Returns the mean loss and the count of correct predictions.
    """
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    target_mask = mask[:, 1:]

    logits = model(inputs)
    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    scored = target_mask.reshape(-1)
    loss = (flat * scored).sum() / scored.sum().clamp(min=1)

    correct = ((logits.argmax(-1) == targets) & target_mask).sum()
    return loss, correct


@torch.no_grad()
def evaluate(model: Transformer, split: Split, cfg: TrainConfig) -> dict:
    """Teacher-forced loss and token accuracy on a sample of a split."""
    model.eval()
    rng = np.random.default_rng(cfg.seed)
    batches = batch_indices(split, cfg.batch_size, rng, cfg.bucket_by_length)[
        : cfg.eval_batches
    ]

    total_loss = total_correct = total_tokens = 0.0
    for index in batches:
        tokens, mask = batch_tensors(split, index, cfg.device)
        loss, correct = masked_loss(model, tokens, mask)
        scored = int(mask[:, 1:].sum())
        total_loss += float(loss) * scored
        total_correct += int(correct)
        total_tokens += scored

    model.train()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
    }


def build_optimizer(model: Transformer, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only, not biases or layer norms."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def train(
    model: Transformer,
    train_split: Split,
    test_split: Split | None,
    cfg: TrainConfig,
    out_dir: str | Path,
) -> dict:
    """Run training, writing checkpoints and metrics into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "model": asdict(model.cfg)}, indent=2),
        encoding="utf-8",
    )
    metrics_path = out_dir / "metrics.jsonl"

    torch.manual_seed(cfg.seed)
    model.to(cfg.device).train()
    optimizer = build_optimizer(model, cfg)
    rng = np.random.default_rng(cfg.seed)

    total_tokens = int(train_split.lengths.sum()) * cfg.epochs
    tokens_seen = step = 0
    best_loss = float("inf")
    started = time.perf_counter()

    def log(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    for epoch in range(cfg.epochs):
        for index in batch_indices(
            train_split, cfg.batch_size, rng, cfg.bucket_by_length
        ):
            tokens, mask = batch_tensors(train_split, index, cfg.device)

            lr = lr_at(tokens_seen, total_tokens, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss, _ = masked_loss(model, tokens, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            tokens_seen += int(tokens.numel())
            step += 1

            if step % cfg.log_every == 0:
                log(
                    {
                        "step": step,
                        "epoch": epoch,
                        "tokens": tokens_seen,
                        "lr": lr,
                        "train_loss": float(loss.detach()),
                        "elapsed": round(time.perf_counter() - started, 1),
                    }
                )

            if test_split is not None and step % cfg.eval_every == 0:
                stats = evaluate(model, test_split, cfg)
                log(
                    {
                        "step": step,
                        "tokens": tokens_seen,
                        **{f"test_{k}": v for k, v in stats.items()},
                    }
                )
                if stats["loss"] < best_loss:
                    best_loss = stats["loss"]
                    save_checkpoint(
                        out_dir / "ckpt_best.pt",
                        model,
                        optimizer,
                        step=step,
                        tokens=tokens_seen,
                    )

            if step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    out_dir / "ckpt_last.pt",
                    model,
                    optimizer,
                    step=step,
                    tokens=tokens_seen,
                )

            if cfg.max_steps is not None and step >= cfg.max_steps:
                return _finish(
                    model,
                    optimizer,
                    test_split,
                    cfg,
                    out_dir,
                    step,
                    tokens_seen,
                    started,
                )

    return _finish(
        model, optimizer, test_split, cfg, out_dir, step, tokens_seen, started
    )


def _finish(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    test_split: Split | None,
    cfg: TrainConfig,
    out_dir: Path,
    step: int,
    tokens_seen: int,
    started: float,
) -> dict:
    save_checkpoint(
        out_dir / "ckpt_last.pt", model, optimizer, step=step, tokens=tokens_seen
    )
    summary = {
        "steps": step,
        "tokens": tokens_seen,
        "elapsed": round(time.perf_counter() - started, 1),
    }
    if test_split is not None:
        summary.update(
            {f"test_{k}": v for k, v in evaluate(model, test_split, cfg).items()}
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
