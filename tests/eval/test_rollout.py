from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from sudoku_transformer.data.dataset import Split
from sudoku_transformer.data.tokenizer import decode, encode, puzzle_from_tokens
from sudoku_transformer.data.vocab import (
    PAD,
    POP,
    PUSH,
    SUCCESS,
    VOCAB_SIZE,
    placement_id,
)
from sudoku_transformer.eval.metrics import (
    by_clues,
    cell_accuracy,
    grid_accuracy,
    stratify,
    summarise,
)
from sudoku_transformer.eval.rollout import (
    EvalConfig,
    generate,
    rollout_split,
    simulate,
)
from sudoku_transformer.model import ModelConfig, Transformer
from sudoku_transformer.sudoku.trace import replay, trace_puzzle

MAX_LEN = 160


class OracleModel(nn.Module):
    """A model that always predicts the next token of a known sequence.

    Standing in for a perfectly trained network, so the rollout, the grid
    reconstruction and the metrics can be tested without training anything.
    """

    def __init__(self, sequences: list[list[int]], n_ctx: int = MAX_LEN) -> None:
        super().__init__()
        self.sequences = sequences
        self.cfg = ModelConfig(n_ctx=n_ctx)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        rows, width = tokens.shape
        logits = torch.zeros(rows, width, VOCAB_SIZE)
        for row, sequence in enumerate(self.sequences):
            for pos in range(width):
                nxt = sequence[pos + 1] if pos + 1 < len(sequence) else PAD
                logits[row, pos, nxt] = 10.0
        return logits


def traced(puzzle: str, seed: int = 0):
    trace = trace_puzzle(puzzle, random.Random(seed))
    return trace, encode(trace.events)


def make_split(rows: list[list[int]]) -> Split:
    tokens = np.full((len(rows), MAX_LEN), PAD, dtype=np.uint16)
    lengths = np.zeros(len(rows), dtype=np.uint16)
    n_clues = np.zeros(len(rows), dtype=np.uint8)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = row
        lengths[i] = len(row)
        n_clues[i] = 81 - puzzle_from_tokens(row).count(".")
    return Split(
        tokens=tokens,
        lengths=lengths,
        n_clues=n_clues,
        difficulty=np.zeros(len(rows), dtype=np.float32),
    )


def test_simulate_reproduces_a_real_trace(easy_puzzle):
    puzzle, solution = easy_puzzle
    trace, tokens = traced(puzzle)
    continuation = tokens[81 - puzzle.count(".") + 1 :]

    grid, illegal = simulate(puzzle, continuation)
    assert grid == solution
    assert illegal == 0


def test_simulate_honours_push_and_pop():
    puzzle = "." * 81
    tokens = [placement_id(0, 1), PUSH, placement_id(1, 2), POP, placement_id(1, 3)]
    grid, illegal = simulate(puzzle, tokens)

    assert grid[0] == "1"
    assert grid[1] == "3"
    assert illegal == 0


def test_simulate_counts_illegal_placements():
    puzzle = "." * 81
    same_row = [placement_id(0, 5), placement_id(1, 5)]
    _, illegal = simulate(puzzle, same_row)
    assert illegal == 1

    overwrite = [placement_id(0, 5), placement_id(0, 6)]
    grid, illegal = simulate(puzzle, overwrite)
    assert illegal == 1
    assert grid[0] == "6"


def test_simulate_counts_an_unmatched_pop():
    _, illegal = simulate("." * 81, [POP])
    assert illegal == 1


def test_simulate_stops_at_success():
    tokens = [placement_id(0, 1), SUCCESS, placement_id(1, 2)]
    grid, _ = simulate("." * 81, tokens)
    assert grid[1] == "."


def test_oracle_model_rolls_out_the_true_trace(easy_puzzle):
    puzzle, solution = easy_puzzle
    _, tokens = traced(puzzle)
    n_clues = 81 - puzzle.count(".")

    model = OracleModel([tokens])
    generated = generate(model, [tokens[: n_clues + 1]], EvalConfig())[0]

    assert generated == tokens[n_clues + 1 :]
    assert generated[-1] == SUCCESS
    assert simulate(puzzle, generated)[0] == solution


def test_generate_handles_mixed_prefix_lengths(kaggle_rows):
    """Puzzles have different clue counts, so rows start generating at different positions."""
    rows = [traced(r["puzzle"], seed=i)[1] for i, r in enumerate(kaggle_rows[:6])]
    rows = [row for row in rows if len(row) <= MAX_LEN]
    prefixes = [row[: 81 - puzzle_from_tokens(row).count(".") + 1] for row in rows]
    assert len({len(p) for p in prefixes}) > 1, "need prefixes of differing length"

    generated = generate(OracleModel(rows), prefixes, EvalConfig())
    for row, prefix, output in zip(rows, prefixes, generated):
        assert output == row[len(prefix) :]


def test_generate_respects_the_token_cap(easy_puzzle):
    puzzle, _ = easy_puzzle
    _, tokens = traced(puzzle)
    n_clues = 81 - puzzle.count(".")

    model = OracleModel([tokens])
    generated = generate(model, [tokens[: n_clues + 1]], EvalConfig(max_new_tokens=5))[
        0
    ]
    assert len(generated) == 5


def test_generate_leaves_training_mode_untouched():
    torch.manual_seed(0)
    model = Transformer(ModelConfig.small(n_ctx=MAX_LEN))
    model.train()
    generate(model, [[1, 2, 3]], EvalConfig(max_new_tokens=2))
    assert model.training


def test_untrained_model_produces_a_scorable_grid(kaggle_rows):
    """Even a random model must yield a grid, not an exception."""
    torch.manual_seed(0)
    rows = [traced(r["puzzle"], seed=i)[1] for i, r in enumerate(kaggle_rows[:4])]
    rows = [row for row in rows if len(row) <= MAX_LEN]

    model = Transformer(ModelConfig.small(n_ctx=MAX_LEN))
    results = rollout_split(
        model, make_split(rows), EvalConfig(batch_size=4, max_new_tokens=12)
    )

    assert len(results) == len(rows)
    assert all(len(r.grid) == 81 for r in results)
    assert all(0 <= r.correct_cells <= 81 for r in results)


def test_rollout_split_scores_a_perfect_model(kaggle_rows):
    rows = [traced(r["puzzle"], seed=i)[1] for i, r in enumerate(kaggle_rows[:5])]
    rows = [row for row in rows if len(row) <= MAX_LEN]
    split = make_split(rows)

    results = rollout_split(OracleModel(rows), split, EvalConfig(batch_size=8))

    assert all(r.solved for r in results)
    assert all(r.finished for r in results)
    assert all(r.illegal == 0 for r in results)
    assert cell_accuracy(results) == 1.0
    assert grid_accuracy(results) == 1.0


def test_truth_comes_from_replaying_the_stored_trace(kaggle_rows):
    rows = [traced(r["puzzle"], seed=i)[1] for i, r in enumerate(kaggle_rows[:3])]
    rows = [row for row in rows if len(row) <= MAX_LEN]
    results = rollout_split(OracleModel(rows), make_split(rows), EvalConfig())

    for result, row in zip(results, rows):
        assert result.truth == replay(decode(row)).to_string()


def test_summarise_reports_the_headline_metrics(kaggle_rows):
    rows = [traced(r["puzzle"], seed=i)[1] for i, r in enumerate(kaggle_rows[:4])]
    rows = [row for row in rows if len(row) <= MAX_LEN]
    stats = summarise(rollout_split(OracleModel(rows), make_split(rows), EvalConfig()))

    assert stats["n"] == len(rows)
    assert stats["cell_accuracy"] == 1.0
    assert stats["grid_accuracy"] == 1.0
    assert stats["finished_fraction"] == 1.0
    assert stats["illegal_per_puzzle"] == 0


def test_summarise_handles_no_results():
    assert summarise([]) == {"n": 0}
    assert cell_accuracy([]) == 0.0
    assert grid_accuracy([]) == 0.0


def test_stratification_partitions_the_results(kaggle_rows):
    rows = [traced(r["puzzle"], seed=i)[1] for i, r in enumerate(kaggle_rows[:8])]
    rows = [row for row in rows if len(row) <= MAX_LEN]
    results = rollout_split(OracleModel(rows), make_split(rows), EvalConfig())

    groups = by_clues(results)
    assert sum(g["n"] for g in groups.values()) == len(results)
    assert all(isinstance(k, int) for k in groups)

    single = stratify(results, lambda r: "all")
    assert single["all"]["n"] == len(results)
