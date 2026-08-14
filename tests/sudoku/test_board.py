from __future__ import annotations

import pytest

from sudoku_transformer.sudoku.board import (
    ALL_DIGITS,
    NUM_CELLS,
    PEERS,
    UNITS,
    Board,
    box_of,
    col_of,
    row_of,
)


def test_unit_structure():
    assert len(UNITS) == 27
    assert all(len(unit) == 9 for unit in UNITS)
    appearances = [sum(cell in unit for unit in UNITS) for cell in range(NUM_CELLS)]
    assert appearances == [3] * NUM_CELLS


def test_every_cell_has_twenty_peers():
    assert all(len(peers) == 20 for peers in PEERS)
    assert all(len(set(peers)) == 20 for peers in PEERS)
    assert all(cell not in PEERS[cell] for cell in range(NUM_CELLS))


def test_peers_are_symmetric():
    for cell, peers in enumerate(PEERS):
        for peer in peers:
            assert cell in PEERS[peer]


def test_coordinate_helpers():
    assert (row_of(0), col_of(0), box_of(0)) == (0, 0, 0)
    assert (row_of(80), col_of(80), box_of(80)) == (8, 8, 8)
    assert box_of(30) == 4


def test_empty_board():
    board = Board.empty()
    assert board.n_empty == NUM_CELLS
    assert not board.is_solved
    assert all(mask == ALL_DIGITS for mask in board.cands)


def test_place_eliminates_from_peers():
    board = Board.empty()
    assert board.place(0, 5)
    assert board.grid[0] == 5
    assert board.cands[0] == 0
    assert board.n_empty == NUM_CELLS - 1
    assert all(not board.has_candidate(peer, 5) for peer in PEERS[0])
    assert board.has_candidate(80, 5)


def test_place_rejects_illegal_moves():
    board = Board.empty()
    board.place(0, 5)
    assert not board.place(0, 3)
    assert not board.place(1, 5)


def test_place_reports_starved_peer():
    board = Board.empty()
    for cell, digit in zip(range(1, 9), range(1, 9), strict=True):
        assert board.place(cell, digit)
    assert board.candidates(0) == (9,)
    assert not board.place(9, 9)
    assert board.cands[0] == 0


def test_round_trip_string(easy_puzzle):
    puzzle, _ = easy_puzzle
    assert Board.from_string(puzzle).to_string() == puzzle


def test_from_string_accepts_zero_for_empty(easy_puzzle):
    puzzle, _ = easy_puzzle
    assert Board.from_string(puzzle.replace(".", "0")).to_string() == puzzle


def test_from_string_rejects_bad_input():
    with pytest.raises(ValueError):
        Board.from_string("123")
    with pytest.raises(ValueError):
        Board.from_string("x" + "." * 80)
    with pytest.raises(ValueError, match="contradicts"):
        Board.from_string("55" + "." * 79)


def test_copy_and_restore_are_independent():
    board = Board.empty()
    board.place(0, 1)
    snapshot = board.copy()
    board.place(1, 2)
    assert board.grid[1] == 2

    board.restore(snapshot)
    assert board.grid[1] == 0
    assert board.grid[0] == 1
    assert board.n_empty == snapshot.n_empty

    board.place(1, 3)
    assert snapshot.grid[1] == 0


def test_cell_level_contradiction():
    assert not Board.empty().is_contradictory()
    board = Board.empty()
    for cell, digit in zip(range(1, 9), range(1, 9), strict=True):
        assert board.place(cell, digit)
    assert board.candidates(0) == (9,)
    board.place(9, 9)
    assert board.is_contradictory()


def test_unit_level_contradiction_without_starved_cell(row0_hidden_single):
    board = Board.from_string(row0_hidden_single)
    assert not board.is_contradictory()
    assert board.has_candidate(8, 9)
    assert all(not board.has_candidate(cell, 9) for cell in range(8))

    assert board.place(8, 1)
    assert board.is_contradictory()
    assert all(board.cands[c] != 0 for c in range(NUM_CELLS) if board.grid[c] == 0)
