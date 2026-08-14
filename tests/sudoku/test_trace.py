from __future__ import annotations

import random

import pytest

from sudoku_transformer.sudoku.solver import Reason
from sudoku_transformer.sudoku.trace import (
    CluesEnd,
    InvalidTrace,
    Place,
    Pop,
    Push,
    Success,
    replay,
    trace_puzzle,
)


def test_trace_starts_with_clues_then_clues_end(easy_puzzle):
    puzzle, _ = easy_puzzle
    trace = trace_puzzle(puzzle)
    n_clues = 81 - puzzle.count(".")
    head, marker = trace.events[:n_clues], trace.events[n_clues]

    assert all(isinstance(e, Place) and e.reason is Reason.CLUE for e in head)
    assert isinstance(marker, CluesEnd)
    assert [e.cell for e in head] == [i for i, c in enumerate(puzzle) if c != "."]
    assert all(puzzle[e.cell] == str(e.digit) for e in head)


def test_trace_ends_with_success(easy_puzzle):
    puzzle, solution = easy_puzzle
    trace = trace_puzzle(puzzle)
    assert isinstance(trace.events[-1], Success)
    assert trace.solved
    assert trace.solution == solution


def test_easy_puzzle_needs_no_search(easy_puzzle):
    trace = trace_puzzle(easy_puzzle[0])
    assert trace.n_guesses == 0
    assert trace.n_backtracks == 0
    assert trace.max_depth == 0


def test_hard_puzzle_pushes_and_pops(hard_puzzle):
    trace = trace_puzzle(hard_puzzle, random.Random(0))
    assert trace.solved
    assert trace.n_guesses > 0
    assert trace.n_backtracks < trace.n_guesses


def test_replay_reconstructs_the_solution(easy_puzzle, hard_puzzle):
    for puzzle in (easy_puzzle[0], hard_puzzle):
        for seed in range(4):
            trace = trace_puzzle(puzzle, random.Random(seed))
            board = replay(trace.events)
            assert board.is_solved
            assert board.to_string() == trace.solution


def test_replay_of_kaggle_sample(kaggle_rows):
    for row in kaggle_rows[:50]:
        trace = trace_puzzle(row["puzzle"], random.Random(row["id"]))
        assert trace.solution == row["solution"]
        assert replay(trace.events).to_string() == row["solution"]


def test_every_placement_is_legal_when_emitted(hard_puzzle):
    for seed in range(8):
        replay(trace_puzzle(hard_puzzle, random.Random(seed)).events)


def test_pushes_and_pops_are_balanced(hard_puzzle):
    trace = trace_puzzle(hard_puzzle, random.Random(1))
    depth = 0
    for event in trace.events:
        if isinstance(event, Push):
            depth += 1
        elif isinstance(event, Pop):
            depth -= 1
            assert depth >= 0
    assert depth == trace.n_guesses - trace.n_backtracks >= 0


def test_rollback_is_load_bearing(hard_puzzle):
    """Strip the Pops and the trace stops replaying: the rollback does real work."""
    trace = trace_puzzle(hard_puzzle, random.Random(2))
    assert trace.n_backtracks > 0
    replay(trace.events)

    with pytest.raises(InvalidTrace):
        replay([e for e in trace.events if not isinstance(e, Pop)])


def test_seeded_traces_are_reproducible(hard_puzzle):
    a = trace_puzzle(hard_puzzle, random.Random(7))
    b = trace_puzzle(hard_puzzle, random.Random(7))
    assert a.events == b.events


def test_different_seeds_give_different_traces(hard_puzzle):
    shapes = {
        tuple(
            type(e).__name__ for e in trace_puzzle(hard_puzzle, random.Random(s)).events
        )
        for s in range(8)
    }
    assert len(shapes) > 1


def test_unseeded_trace_is_deterministic(hard_puzzle):
    assert trace_puzzle(hard_puzzle).events == trace_puzzle(hard_puzzle).events


def test_token_count_is_event_count(easy_puzzle):
    puzzle, _ = easy_puzzle
    trace = trace_puzzle(puzzle)
    assert trace.n_tokens == len(trace.events)
    assert trace.n_tokens == (81 - puzzle.count(".")) + 1 + puzzle.count(".") + 1


def test_placements_can_be_filtered_by_reason(easy_puzzle):
    puzzle, _ = easy_puzzle
    trace = trace_puzzle(puzzle)
    assert len(trace.placements(Reason.CLUE)) == 81 - puzzle.count(".")
    assert len(trace.placements()) == len(trace.events) - 2


def test_replay_rejects_a_tampered_trace(easy_puzzle):
    events = list(trace_puzzle(easy_puzzle[0]).events)
    last = next(i for i, e in reversed(list(enumerate(events))) if isinstance(e, Place))
    events[last] = Place(events[last].cell, 1 + events[last].digit % 9, Reason.GUESS)
    with pytest.raises(InvalidTrace):
        replay(events)


def test_replay_rejects_pop_without_push():
    with pytest.raises(InvalidTrace, match="no matching push"):
        replay([CluesEnd(), Pop()])


def test_replay_requires_clues_end():
    with pytest.raises(InvalidTrace, match="clues_end"):
        replay([Place(0, 1, Reason.CLUE)])


def test_trace_rejects_contradictory_clues():
    with pytest.raises(ValueError, match="contradicts"):
        trace_puzzle("55" + "." * 79)


def test_trace_rejects_wrong_length():
    with pytest.raises(ValueError, match="81"):
        trace_puzzle("123")
