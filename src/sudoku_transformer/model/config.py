"""Model shape and the two configurations we train."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.vocab import VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Decoder-only transformer shape.

    Defaults are the paper's: 8 layers, 8 heads, d_model 576, d_mlp 3456 (a 6x
    expansion), pre-layer-norm, GELU, no dropout. Positional encoding and weight
    tying are not specified there; we use learned absolute positions and an
    untied unembedding, which keeps logit attribution readable in phase 7.
    """

    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 576
    d_mlp: int = 3456
    n_ctx: int = 250
    d_vocab: int = VOCAB_SIZE
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by {self.n_heads} heads"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def paper(cls, **overrides) -> ModelConfig:
        return cls(**overrides)

    @classmethod
    def small(cls, **overrides) -> ModelConfig:
        """A CPU-sized version for debugging the pipeline end to end."""
        return cls(n_layers=4, n_heads=4, d_model=256, d_mlp=1024, **overrides)
