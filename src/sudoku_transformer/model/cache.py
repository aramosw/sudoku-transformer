"""Named taps into the forward pass, for reading and for intervening.

Probing needs to read the residual stream; the circuit work needs to *change*
it mid-forward -- mean-ablating a head, subtracting a probe direction -- so a
tap both records and rewrites. Building this in now rather than when phase 7
asks for it is the difference between adding a keyword argument and rewriting
forward().

Tap names:

    embed, pos_embed
    blocks.{i}.resid_pre
    blocks.{i}.attn.pattern     (batch, head, query, key), post-softmax
    blocks.{i}.attn.z           (batch, pos, head, d_head), per-head output
    blocks.{i}.attn.out         (batch, pos, d_model)
    blocks.{i}.resid_mid
    blocks.{i}.mlp.pre          (batch, pos, d_mlp), before GELU
    blocks.{i}.mlp.post         (batch, pos, d_mlp), the neurons
    blocks.{i}.mlp.out
    blocks.{i}.resid_post
    resid_final, ln_final, logits
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import Tensor

Edit = Callable[[Tensor], Tensor]


class ActivationCache(dict[str, Tensor]):
    """Activations captured during a forward pass, keyed by tap name."""

    def resid_pre(self, layer: int) -> Tensor:
        return self[f"blocks.{layer}.resid_pre"]

    def resid_post(self, layer: int) -> Tensor:
        return self[f"blocks.{layer}.resid_post"]

    def neurons(self, layer: int) -> Tensor:
        """Post-GELU MLP activations, the unit the naked-single circuit lives in."""
        return self[f"blocks.{layer}.mlp.post"]

    def pattern(self, layer: int) -> Tensor:
        return self[f"blocks.{layer}.attn.pattern"]


class Tap:
    """Records activations, applies edits, or both.

    ``names`` restricts what is stored; leaving it None keeps everything, which
    is convenient but holds the whole forward pass in memory. Edits run before
    the value is recorded, so a cache taken alongside an edit reflects the
    modified activation.
    """

    __slots__ = ("cache", "edits", "names")

    def __init__(
        self,
        cache: ActivationCache | None = None,
        edits: dict[str, Edit] | None = None,
        names: Iterable[str] | None = None,
    ) -> None:
        self.cache = cache
        self.edits = edits
        self.names = frozenset(names) if names is not None else None

    def __call__(self, name: str, tensor: Tensor) -> Tensor:
        if self.edits is not None and name in self.edits:
            tensor = self.edits[name](tensor)
        if self.cache is not None and (self.names is None or name in self.names):
            self.cache[name] = tensor.detach()
        return tensor


def apply_tap(tap: Tap | None, name: str, tensor: Tensor) -> Tensor:
    return tensor if tap is None else tap(name, tensor)
