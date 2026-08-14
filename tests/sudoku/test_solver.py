from __future__ import annotations

import random

from sudoku_transformer.sudoku.board import Board
from sudoku_transformer.sudoku.solver import (
    Reason,
    count_solutions,
    hidden_singles,
    naked_singles,
    search,
    select_branch_cell,
    solve,
)


def test_naked_single_detection():
    board = Board.empty()
    for cell, digit in zip(range(1, 9), range(1, 9), strict=True):
        board.place(cell, digit)
    assert (0, 9) in naked_singles(board)


def test_hidden_single_detection(row0_hidden_single):
    board = Board.from_string(row0_hidden_single)
    assert (8, 9) in hidden_singles(board)
    assert board.n_candidates(8) > 1
    assert (8, 9) not in naked_singles(board)


def test_hidden_singles_are_deduplicated(row0_hidden_single):
    found = hidden_singles(Board.from_string(row0_hidden_single))
    assert len(found) == len(set(found))


def test_mrv_picks_the_most_constrained_cell():
    board = Board.empty()
    for cell, digit in zip(range(1, 8), range(1, 8), strict=True):
        board.place(cell, digit)
    assert board.n_candidates(0) == 2
    assert select_branch_cell(board, None) == 0


def test_solves_an_easy_puzzle(easy_puzzle):
    puzzle, solution = easy_puzzle
    assert solve(puzzle) == solution


def test_solves_a_hard_puzzle(hard_puzzle):
    solution = solve(hard_puzzle)
    assert solution is not None
    assert Board.from_string(solution).is_solved
    assert all(c == "." or c == s for c, s in zip(hard_puzzle, solution, strict=True))


def test_unsolvable_board_returns_none(row0_hidden_single):
    board = Board.from_string(row0_hidden_single)
    assert board.place(8, 1)
    assert solve(board) is None


def test_randomisation_changes_the_search_but_not_the_answer(hard_puzzle):
    answers = {solve(hard_puzzle, random.Random(seed)) for seed in range(8)}
    assert len(answers) == 1
    assert answers.pop() == solve(hard_puzzle)


def test_seeded_solves_are_reproducible(hard_puzzle):
    assert solve(hard_puzzle, random.Random(0)) == solve(hard_puzzle, random.Random(0))


def test_count_solutions_on_a_unique_puzzle(easy_puzzle):
    puzzle, _ = easy_puzzle
    assert count_solutions(puzzle) == 1


def test_count_solutions_stops_at_the_limit():
    assert count_solutions("." * 81, limit=2) == 2


def test_search_reports_failure_on_a_dead_board(row0_hidden_single):
    board = Board.from_string(row0_hidden_single)
    board.place(8, 1)
    assert not search(board)


def test_recorder_sees_every_placement(easy_puzzle):
    puzzle, solution = easy_puzzle

    class Counter:
        def __init__(self):
            self.reasons = []
            self.pushes = self.pops = self.successes = 0

        def on_place(self, cell, digit, reason):
            self.reasons.append(reason)

        def on_push(self):
            self.pushes += 1

        def on_pop(self):
            self.pops += 1

        def on_success(self):
            self.successes += 1

    board = Board.from_string(puzzle)
    counter = Counter()
    assert search(board, None, counter)
    assert counter.pushes == 0
    assert len(counter.reasons) == puzzle.count(".")
    assert set(counter.reasons) <= {Reason.NAKED_SINGLE, Reason.HIDDEN_SINGLE}
    assert board.to_string() == solution


def test_solves_kaggle_sample(kaggle_rows):
    for row in kaggle_rows[:50]:
        assert solve(row["puzzle"]) == row["solution"]


def test_kaggle_puzzles_are_uniquely_solvable(kaggle_rows):
    for row in kaggle_rows[:25]:
        assert count_solutions(row["puzzle"]) == 1
