from __future__ import annotations

import pytest
import torch

from sudoku_transformer.data.vocab import VOCAB_SIZE
from sudoku_transformer.model.cache import Tap
from sudoku_transformer.model.config import ModelConfig
from sudoku_transformer.model.transformer import Transformer


@pytest.fixture
def small_model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig.small(n_ctx=32)).eval()


@pytest.fixture
def tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (2, 16))


def test_paper_config_shape():
    cfg = ModelConfig.paper()
    assert (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_mlp) == (8, 8, 576, 3456)
    assert cfg.d_head == 72
    assert cfg.d_vocab == VOCAB_SIZE == 734
    assert cfg.n_ctx == 250


def test_config_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=100, n_heads=8)


def test_paper_parameter_count():
    """Guards against silent architecture drift."""
    cfg = ModelConfig.paper()
    model = Transformer(cfg)

    per_block = (
        4 * cfg.d_model  # two layer norms
        + cfg.d_model * 3 * cfg.d_model
        + 3 * cfg.d_model  # qkv
        + cfg.d_model * cfg.d_model
        + cfg.d_model  # attention output projection
        + cfg.d_model * cfg.d_mlp
        + cfg.d_mlp  # mlp up
        + cfg.d_mlp * cfg.d_model
        + cfg.d_model  # mlp down
    )
    expected = (
        cfg.d_vocab * cfg.d_model
        + cfg.n_ctx * cfg.d_model
        + cfg.n_layers * per_block
        + 2 * cfg.d_model
        + cfg.d_model * cfg.d_vocab
        + cfg.d_vocab
    )
    assert model.n_parameters() == expected
    assert 42_000_000 < expected < 44_000_000


def test_forward_shape(small_model, tokens):
    logits = small_model(tokens)
    assert logits.shape == (2, 16, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_rejects_sequences_beyond_context(small_model):
    with pytest.raises(ValueError, match="exceeds context"):
        small_model(torch.zeros(1, small_model.cfg.n_ctx + 1, dtype=torch.long))


def test_attention_is_causal(small_model, tokens):
    """Changing the last token must not move any earlier position's logits."""
    baseline = small_model(tokens)
    altered = tokens.clone()
    altered[:, -1] = (altered[:, -1] + 1) % VOCAB_SIZE

    changed = small_model(altered)
    assert torch.allclose(baseline[:, :-1], changed[:, :-1], atol=1e-5)
    assert not torch.allclose(baseline[:, -1], changed[:, -1])


def test_fused_and_explicit_attention_agree(small_model, tokens):
    """The tapped path recomputes attention by hand; it must match the kernel."""
    fused = small_model(tokens)
    explicit = small_model(tokens, Tap())
    assert torch.allclose(fused, explicit, atol=1e-5)


def test_attention_pattern_is_lower_triangular(small_model, tokens):
    _, cache = small_model.run_with_cache(tokens)
    pattern = cache.pattern(0)

    seq = tokens.shape[1]
    assert pattern.shape == (2, small_model.cfg.n_heads, seq, seq)
    assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
    assert (pattern.triu(diagonal=1) == 0).all()


def test_cache_holds_every_tap(small_model, tokens):
    _, cache = small_model.run_with_cache(tokens)
    batch, seq = tokens.shape
    cfg = small_model.cfg

    assert cache["embed"].shape == (batch, seq, cfg.d_model)
    assert cache["pos_embed"].shape == (seq, cfg.d_model)
    assert cache["logits"].shape == (batch, seq, cfg.d_vocab)
    for layer in range(cfg.n_layers):
        assert cache.resid_pre(layer).shape == (batch, seq, cfg.d_model)
        assert cache.resid_post(layer).shape == (batch, seq, cfg.d_model)
        assert cache.neurons(layer).shape == (batch, seq, cfg.d_mlp)
        assert cache[f"blocks.{layer}.attn.z"].shape == (
            batch,
            seq,
            cfg.n_heads,
            cfg.d_head,
        )


def test_cache_can_be_restricted(small_model, tokens):
    _, cache = small_model.run_with_cache(tokens, names=["blocks.2.resid_post"])
    assert list(cache) == ["blocks.2.resid_post"]


def test_residual_stream_accumulates(small_model, tokens):
    """resid_post = resid_mid + mlp_out, which is what makes probes decomposable."""
    _, cache = small_model.run_with_cache(tokens)
    reconstructed = cache["blocks.0.resid_mid"] + cache["blocks.0.mlp.out"]
    assert torch.allclose(cache.resid_post(0), reconstructed, atol=1e-5)

    reconstructed = cache.resid_pre(0) + cache["blocks.0.attn.out"]
    assert torch.allclose(cache["blocks.0.resid_mid"], reconstructed, atol=1e-5)


def test_edits_change_the_output(small_model, tokens):
    """Ablation works by rewriting an activation mid-forward."""
    baseline = small_model(tokens)

    def zero_head_zero(z):
        z = z.clone()
        z[:, :, 0] = 0
        return z

    ablated = small_model(tokens, Tap(edits={"blocks.0.attn.z": zero_head_zero}))
    assert not torch.allclose(baseline, ablated)


def test_edited_activation_is_what_gets_cached(small_model, tokens):
    def zero_it(x):
        return torch.zeros_like(x)

    _, cache = small_model.run_with_cache(tokens, edits={"blocks.1.mlp.post": zero_it})
    assert (cache.neurons(1) == 0).all()
    assert (cache["blocks.1.mlp.out"] == small_model.blocks[1].mlp.down.bias).all()


def test_gradients_flow(small_model, tokens):
    logits = small_model(tokens)
    logits.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in small_model.parameters()
    )


def test_cached_activations_are_detached(small_model, tokens):
    _, cache = small_model.run_with_cache(tokens)
    assert not cache.resid_post(0).requires_grad
