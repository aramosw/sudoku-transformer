"""Autoregressive generation from a clue prefix, and the grid it produces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import Split
from ..data.tokenizer import decode, puzzle_from_tokens
from ..data.vocab import (
    CLUES_END,
    PAD,
    POP,
    PUSH,
    SUCCESS,
    decode_placement,
    is_placement,
)
from ..model.transformer import Transformer
from ..sudoku.board import PEERS
from ..sudoku.trace import replay


@dataclass(frozen=True, slots=True)
class EvalConfig:
    batch_size: int = 64
    max_new_tokens: int | None = None
    temperature: float = 0.0
    device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """One puzzle attempted end to end."""

    puzzle: str
    truth: str
    grid: str
    finished: bool
    illegal: int
    generated: int
    n_clues: int
    difficulty: float

    @property
    def correct_cells(self) -> int:
        return sum(a == b for a, b in zip(self.grid, self.truth))

    @property
    def solved(self) -> bool:
        return self.grid == self.truth


def simulate(puzzle: str, generated: list[int]) -> tuple[str, int]:
    """Apply generated tokens to a puzzle, returning the final grid.

    Deliberately lenient: an illegal placement is counted and then applied
    anyway, last write winning, because a model that has not learned the rules
    still needs a grid to be scored against. Push and pop are honoured with a
    snapshot stack, so a trace that backtracks correctly rewinds.
    """
    grid = [0 if c == "." else int(c) for c in puzzle]
    stack: list[list[int]] = []
    illegal = 0

    for token in generated:
        if is_placement(token):
            cell, digit = decode_placement(token)
            if grid[cell] != 0 or any(grid[peer] == digit for peer in PEERS[cell]):
                illegal += 1
            grid[cell] = digit
        elif token == PUSH:
            stack.append(grid.copy())
        elif token == POP:
            if stack:
                grid = stack.pop()
            else:
                illegal += 1
        elif token in (SUCCESS, PAD, CLUES_END):
            break

    return "".join(str(d) if d else "." for d in grid), illegal


@torch.no_grad()
def generate(
    model: Transformer, prefixes: list[list[int]], cfg: EvalConfig
) -> list[list[int]]:
    """Continue each prefix until [success], the context limit, or the token cap.

    Prefixes have different lengths because puzzles have different clue counts,
    so each row carries its own cursor and the batch is only ever run out to the
    furthest one. Causal attention means a row never sees past its own cursor.
    """
    was_training = model.training
    model.eval()
    device = cfg.device
    width = model.cfg.n_ctx
    rng = torch.Generator(device=device).manual_seed(cfg.seed)

    n = len(prefixes)
    tokens = torch.full((n, width), PAD, dtype=torch.long, device=device)
    for row, prefix in enumerate(prefixes):
        tokens[row, : len(prefix)] = torch.tensor(prefix, device=device)

    cursors = np.array([len(p) for p in prefixes])
    starts = cursors.copy()
    done = np.zeros(n, dtype=bool)
    cap = cfg.max_new_tokens if cfg.max_new_tokens is not None else width

    while not done.all():
        pos = int(cursors.max())
        if pos >= width:
            break

        logits = model(tokens[:, :pos])
        last = torch.from_numpy(cursors - 1).to(device)
        step = logits[torch.arange(n, device=device), last]

        if cfg.temperature > 0:
            probs = torch.softmax(step / cfg.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        else:
            nxt = step.argmax(dim=-1)

        active = torch.from_numpy(~done).to(device)
        rows = torch.nonzero(active, as_tuple=True)[0]
        tokens[rows, torch.from_numpy(cursors).to(device)[rows]] = nxt[rows]

        chosen = nxt.cpu().numpy()
        cursors = np.where(~done, cursors + 1, cursors)
        done |= (chosen == SUCCESS) | (chosen == PAD)
        done |= cursors - starts >= cap
        done |= cursors >= width

    model.train(was_training)
    return [tokens[row, starts[row] : cursors[row]].tolist() for row in range(n)]


def rollout_split(
    model: Transformer, split: Split, cfg: EvalConfig, limit: int | None = None
) -> list[RolloutResult]:
    """Roll the model out on a split and score it against the stored traces.

    Ground truth comes from replaying each stored trace rather than from a
    separate solutions file: the trace already ends in the solved grid.
    """
    total = len(split) if limit is None else min(limit, len(split))
    results: list[RolloutResult] = []

    for start in range(0, total, cfg.batch_size):
        index = np.arange(start, min(start + cfg.batch_size, total))
        rows = [
            [int(t) for t in split.tokens[i][: int(split.lengths[i])]] for i in index
        ]
        prefixes = [row[: int(split.n_clues[i]) + 1] for row, i in zip(rows, index)]

        for row, prefix, generated, i in zip(
            rows, prefixes, generate(model, prefixes, cfg), index
        ):
            puzzle = puzzle_from_tokens(row)
            grid, illegal = simulate(puzzle, generated)
            results.append(
                RolloutResult(
                    puzzle=puzzle,
                    truth=replay(decode(row)).to_string(),
                    grid=grid,
                    finished=bool(generated and generated[-1] == SUCCESS),
                    illegal=illegal,
                    generated=len(generated),
                    n_clues=int(split.n_clues[i]),
                    difficulty=float(split.difficulty[i]),
                )
            )

    return results
