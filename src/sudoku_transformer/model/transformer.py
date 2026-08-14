"""The decoder-only transformer trained on solve traces."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cache import ActivationCache, Edit, Tap, apply_tap
from .config import ModelConfig


class Attention(nn.Module):
    """Causal multi-head attention.

    Runs fused scaled-dot-product attention when nothing is tapped and an
    explicit softmax otherwise, since the fused kernel cannot hand back the
    attention pattern. The two paths are checked against each other in tests.
    """

    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: Tensor, tap: Tap | None) -> Tensor:
        batch, seq, _ = x.shape
        heads, d_head = self.cfg.n_heads, self.cfg.d_head

        q, k, v = self.qkv(x).split(self.cfg.d_model, dim=-1)
        q = q.view(batch, seq, heads, d_head).transpose(1, 2)
        k = k.view(batch, seq, heads, d_head).transpose(1, 2)
        v = v.view(batch, seq, heads, d_head).transpose(1, 2)

        if tap is None:
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_head)
            scores = scores.masked_fill(~self.causal_mask[:seq, :seq], float("-inf"))
            pattern = tap(f"blocks.{self.layer}.attn.pattern", scores.softmax(dim=-1))
            z = pattern @ v

        z = apply_tap(tap, f"blocks.{self.layer}.attn.z", z.transpose(1, 2))
        out = self.proj(z.reshape(batch, seq, self.cfg.d_model))
        return apply_tap(tap, f"blocks.{self.layer}.attn.out", out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.up = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: Tensor, tap: Tap | None) -> Tensor:
        pre = apply_tap(tap, f"blocks.{self.layer}.mlp.pre", self.up(x))
        post = apply_tap(tap, f"blocks.{self.layer}.mlp.post", F.gelu(pre))
        return apply_tap(tap, f"blocks.{self.layer}.mlp.out", self.down(post))


class Block(nn.Module):
    """Pre-layer-norm transformer block."""

    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.mlp = MLP(cfg, layer)

    def forward(self, x: Tensor, tap: Tap | None) -> Tensor:
        x = apply_tap(tap, f"blocks.{self.layer}.resid_pre", x)
        x = apply_tap(
            tap, f"blocks.{self.layer}.resid_mid", x + self.attn(self.ln1(x), tap)
        )
        return apply_tap(
            tap, f"blocks.{self.layer}.resid_post", x + self.mlp(self.ln2(x), tap)
        )


class Transformer(nn.Module):
    """Decoder-only transformer over the 734-token trace vocabulary.

    Padding needs no attention mask: causal attention means a real position
    never sees the trailing [pad] tokens, and the loss mask drops the positions
    that would.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.cfg.init_std)

    def _scale_residual_projections(self) -> None:
        """Shrink the two projections that write into the residual stream.

        Without this the stream's variance grows with depth and 8 layers at the
        paper's 1e-3 learning rate is unstable early on. Standard GPT-2 practice
        rather than anything the paper states.
        """
        scale = math.sqrt(2 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.proj.weight.div_(scale)
                block.mlp.down.weight.div_(scale)

    def forward(self, tokens: Tensor, tap: Tap | None = None) -> Tensor:
        _, seq = tokens.shape
        if seq > self.cfg.n_ctx:
            raise ValueError(f"sequence of {seq} exceeds context {self.cfg.n_ctx}")

        embedded = apply_tap(tap, "embed", self.embed(tokens))
        positions = self.pos_embed(torch.arange(seq, device=tokens.device))
        x = embedded + apply_tap(tap, "pos_embed", positions)

        for block in self.blocks:
            x = block(x, tap)

        x = apply_tap(tap, "resid_final", x)
        x = apply_tap(tap, "ln_final", self.ln_final(x))
        return apply_tap(tap, "logits", self.unembed(x))

    def run_with_cache(
        self,
        tokens: Tensor,
        names: list[str] | None = None,
        edits: dict[str, Edit] | None = None,
    ) -> tuple[Tensor, ActivationCache]:
        cache = ActivationCache()
        logits = self(tokens, Tap(cache=cache, edits=edits, names=names))
        return logits, cache

    def n_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )
