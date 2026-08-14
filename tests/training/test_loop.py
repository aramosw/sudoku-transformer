from __future__ import annotations

import numpy as np
import pytest
import torch

from sudoku_transformer.data.dataset import Split
from sudoku_transformer.data.vocab import CLUES_END, PAD, SUCCESS, VOCAB_SIZE
from sudoku_transformer.model import ModelConfig, Transformer
from sudoku_transformer.training.checkpoint import load_checkpoint, save_checkpoint
from sudoku_transformer.training.loop import (
    TrainConfig,
    batch_indices,
    batch_tensors,
    build_optimizer,
    evaluate,
    lr_at,
    masked_loss,
    train,
)

MAX_LEN = 24


def make_split(n: int = 32, seed: int = 0) -> Split:
    """A synthetic split with the same shape contract as a real one."""
    rng = np.random.default_rng(seed)
    tokens = np.full((n, MAX_LEN), PAD, dtype=np.uint16)
    lengths = np.zeros(n, dtype=np.uint16)
    n_clues = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        clues = int(rng.integers(3, 6))
        body = int(rng.integers(4, MAX_LEN - clues - 2))
        row = list(rng.integers(0, 729, clues)) + [CLUES_END]
        row += list(rng.integers(0, 729, body)) + [SUCCESS]
        tokens[i, : len(row)] = row
        lengths[i] = len(row)
        n_clues[i] = clues

    return Split(
        tokens=tokens,
        lengths=lengths,
        n_clues=n_clues,
        difficulty=rng.random(n).astype(np.float32),
    )


@pytest.fixture
def split() -> Split:
    return make_split()


@pytest.fixture
def tiny_model() -> Transformer:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=2, n_heads=2, d_model=64, d_mlp=128, n_ctx=MAX_LEN)
    return Transformer(cfg)


def test_warmup_is_linear_then_cosine_decays():
    cfg = TrainConfig(lr=1e-3, warmup_tokens=1000)
    assert lr_at(0, 10_000, cfg) == 0
    assert lr_at(500, 10_000, cfg) == pytest.approx(5e-4)
    assert lr_at(1000, 10_000, cfg) == pytest.approx(1e-3)

    mid = lr_at(5500, 10_000, cfg)
    assert 0 < mid < 1e-3
    assert lr_at(10_000, 10_000, cfg) == pytest.approx(0, abs=1e-12)


def test_schedule_never_exceeds_the_peak():
    cfg = TrainConfig(lr=1e-3, warmup_tokens=100)
    assert all(lr_at(t, 5000, cfg) <= cfg.lr + 1e-12 for t in range(0, 6000, 97))


def test_batching_covers_every_row_exactly_once(split):
    rng = np.random.default_rng(0)
    for bucket in (True, False):
        batches = batch_indices(split, batch_size=7, rng=rng, bucket=bucket)
        seen = np.concatenate(batches)
        assert sorted(seen.tolist()) == list(range(len(split)))


def test_bucketing_makes_batches_length_homogeneous():
    split = make_split(n=400)
    rng = np.random.default_rng(0)

    def padding_waste(bucket: bool) -> float:
        batches = batch_indices(
            split, batch_size=16, rng=rng, bucket=bucket, pool_factor=4
        )
        used = sum(int(split.lengths[b].sum()) for b in batches)
        allocated = sum(len(b) * int(split.lengths[b].max()) for b in batches)
        return 1 - used / allocated

    assert padding_waste(True) < padding_waste(False)


def test_batch_is_trimmed_to_its_longest_trace(split):
    index = np.arange(len(split))
    tokens, mask = batch_tensors(split, index)
    assert tokens.shape == (len(split), int(split.lengths.max()))
    assert mask.shape == tokens.shape


def test_mask_covers_exactly_the_scored_positions(split):
    index = np.arange(8)
    _, mask = batch_tensors(split, index)
    for row, i in enumerate(index):
        n_clues, length = int(split.n_clues[i]), int(split.lengths[i])
        assert not mask[row, : n_clues + 1].any()
        assert mask[row, n_clues + 1 : length].all()
        assert not mask[row, length:].any()


def test_loss_ignores_masked_positions(tiny_model, split):
    tokens, mask = batch_tensors(split, np.arange(8))
    baseline, _ = masked_loss(tiny_model, tokens, mask)

    scrambled = tokens.clone()
    unscored = ~mask
    scrambled[unscored] = (scrambled[unscored] + 1) % 729
    changed, _ = masked_loss(tiny_model, scrambled, mask)

    # Rewriting clues changes the context, so the loss moves; rewriting only the
    # padding past the end of each trace must not, since nothing attends to it.
    padding = torch.zeros_like(mask)
    for row, length in enumerate(split.lengths[:8]):
        padding[row, int(length) :] = True
    padded = tokens.clone()
    padded[padding] = (padded[padding] + 3) % 729
    same, _ = masked_loss(tiny_model, padded, mask)

    assert not torch.isclose(baseline, changed)
    assert torch.isclose(baseline, same, atol=1e-6)


def test_loss_is_finite_and_near_uniform_at_init(tiny_model, split):
    tokens, mask = batch_tensors(split, np.arange(len(split)))
    loss, correct = masked_loss(tiny_model, tokens, mask)
    assert torch.isfinite(loss)
    assert loss < np.log(VOCAB_SIZE) * 1.5
    assert 0 <= int(correct) <= int(mask[:, 1:].sum())


def test_optimizer_exempts_biases_and_norms_from_decay(tiny_model):
    optimizer = build_optimizer(tiny_model, TrainConfig())
    decayed, undecayed = optimizer.param_groups
    assert decayed["weight_decay"] > 0
    assert undecayed["weight_decay"] == 0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() < 2 for p in undecayed["params"])


def test_overfits_a_single_batch(tiny_model, split):
    """The loop's sanity check: memorising one batch must drive loss to ~0."""
    tokens, mask = batch_tensors(split, np.arange(8))
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)

    first = float(masked_loss(tiny_model, tokens, mask)[0].detach())
    final = first
    for _ in range(300):
        loss, _ = masked_loss(tiny_model, tokens, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())

    assert final < 0.05, f"loss stalled at {final:.3f} (started {first:.3f})"


def test_evaluate_reports_loss_and_accuracy(tiny_model, split):
    stats = evaluate(tiny_model, split, TrainConfig(batch_size=8, eval_batches=2))
    assert set(stats) == {"loss", "token_accuracy"}
    assert stats["loss"] > 0
    assert 0 <= stats["token_accuracy"] <= 1


def test_evaluate_leaves_the_model_training(tiny_model, split):
    tiny_model.train()
    evaluate(tiny_model, split, TrainConfig(batch_size=8, eval_batches=1))
    assert tiny_model.training


def test_train_writes_its_artifacts(tiny_model, split, tmp_path):
    cfg = TrainConfig(
        batch_size=8,
        epochs=1,
        max_steps=4,
        eval_every=2,
        checkpoint_every=4,
        log_every=1,
    )
    summary = train(tiny_model, split, split, cfg, tmp_path)

    assert summary["steps"] == 4
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "ckpt_last.pt").exists()

    records = [
        line for line in (tmp_path / "metrics.jsonl").read_text().splitlines() if line
    ]
    assert len(records) >= 4


def test_training_reduces_loss(split, tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=2, n_heads=2, d_model=64, d_mlp=128, n_ctx=MAX_LEN)
    model = Transformer(cfg)
    train_cfg = TrainConfig(
        batch_size=8, epochs=12, warmup_tokens=200, eval_every=10_000, log_every=10_000
    )

    before = evaluate(model, split, train_cfg)["loss"]
    train(model, split, None, train_cfg, tmp_path)
    after = evaluate(model, split, train_cfg)["loss"]
    assert after < before


def test_checkpoint_round_trip(tiny_model, tmp_path, split):
    optimizer = build_optimizer(tiny_model, TrainConfig())
    save_checkpoint(tmp_path / "ckpt.pt", tiny_model, optimizer, step=7, tokens=123)

    payload = load_checkpoint(tmp_path / "ckpt.pt")
    restored = payload["model"]
    assert payload["step"] == 7
    assert payload["tokens"] == 123
    assert restored.cfg == tiny_model.cfg

    tokens, _ = batch_tensors(split, np.arange(4))
    tiny_model.eval()
    restored.eval()
    assert torch.allclose(tiny_model(tokens), restored(tokens))
