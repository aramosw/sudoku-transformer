"""Saving and restoring training state."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..model.config import ModelConfig
from ..model.transformer import Transformer


def save_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer | None = None,
    **state: Any,
) -> None:
    """Write model weights, optimiser state and the model config to one file.

    The config travels with the weights so a checkpoint can be reloaded without
    knowing which configuration produced it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model.cfg),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            **state,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: Transformer | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load a checkpoint, building the model from its stored config if needed."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)

    if model is None:
        model = Transformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    payload["model"] = model

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
