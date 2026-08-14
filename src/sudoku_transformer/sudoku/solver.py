"""Norvig-style deduction rules and the randomised search built on them."""

from __future__ import annotations

import random
from enum import StrEnum
from typing import Protocol

from .board import ALL_DIGITS, BIT, N, UNITS, Board

Placement = tuple[int, int]


class Reason(StrEnum):
    """Why a digit was placed. Metadata for probing; not part of the vocabulary."""

    CLUE = "clue"
    NAKED_SINGLE = "naked_single"
    HIDDEN_SINGLE = "hidden_single"
    GUESS = "guess"


class Recorder(Protocol):
    """Observes the solver's decisions. Implemented by the tracer."""

    def on_place(self, cell: int, digit: int, reason: Reason) -> None: ...
    def on_push(self) -> None: ...
    def on_pop(self) -> None: ...
    def on_success(self) -> None: ...


class NullRecorder:
    """Discards everything, for when only the answer is wanted."""

    __slots__ = ()

    def on_place(self, cell: int, digit: int, reason: Reason) -> None:
        pass

    def on_push(self) -> None:
        pass

    def on_pop(self) -> None:
        pass

    def on_success(self) -> None:
        pass


def naked_singles(board: Board) -> list[Placement]:
    """Empty cells with exactly one remaining candidate, in cell order."""
    out: list[Placement] = []
    grid, cands = board.grid, board.cands
    for cell, mask in enumerate(cands):
        if grid[cell] == 0 and mask.bit_count() == 1:
            out.append((cell, mask.bit_length()))
    return out


def hidden_singles(board: Board) -> list[Placement]:
    """Digits with exactly one possible home left in some unit, de-duplicated."""
    found: set[Placement] = set()
    grid, cands = board.grid, board.cands
    for unit in UNITS:
        placed = 0
        for cell in unit:
            if grid[cell]:
                placed |= BIT[grid[cell]]
        missing = ALL_DIGITS & ~placed
        if not missing:
            continue
        for digit in range(1, N + 1):
            bit = BIT[digit]
            if not (missing & bit):
                continue
            home = -1
            for cell in unit:
                if grid[cell] == 0 and cands[cell] & bit:
                    if home >= 0:
                        home = -1
                        break
                    home = cell
            if home >= 0:
                found.add((home, digit))
    return sorted(found)


def deduce(board: Board, rng: random.Random | None, rec: Recorder) -> bool:
    """Apply both single-cell rules until neither fires. False on contradiction.

    Each round identifies *all* naked singles from one position, shuffles them,
    and plays the whole batch before rescanning; hidden singles are only sought
    once no naked single is available. The paper does not pin this down, but
    recomputing after every placement would let each deduction cascade into the
    next, making the random order a forced one. Batch entries are rechecked as
    they are applied because earlier ones can invalidate later ones.
    """
    while True:
        if board.is_contradictory():
            return False

        reason = Reason.NAKED_SINGLE
        batch = naked_singles(board)
        if not batch:
            reason = Reason.HIDDEN_SINGLE
            batch = hidden_singles(board)
        if not batch:
            return True

        if rng is not None:
            rng.shuffle(batch)

        for cell, digit in batch:
            placed = board.grid[cell]
            if placed:
                if placed == digit:
                    continue
                return False
            if not board.has_candidate(cell, digit):
                return False
            ok = board.place(cell, digit)
            rec.on_place(cell, digit, reason)
            if not ok:
                return False


def select_branch_cell(board: Board, rng: random.Random | None) -> int:
    """Pick an empty cell with the fewest candidates, ties broken randomly."""
    best = N + 1
    tied: list[int] = []
    for cell in range(len(board.grid)):
        if board.grid[cell]:
            continue
        count = board.cands[cell].bit_count()
        if count < best:
            best = count
            tied = [cell]
        elif count == best:
            tied.append(cell)
    if rng is None or len(tied) == 1:
        return tied[0]
    return rng.choice(tied)


def search(
    board: Board, rng: random.Random | None = None, rec: Recorder | None = None
) -> bool:
    """Solve the board in place, reporting every decision to the recorder.

    True if solved; on False the board is spent and the caller must restore it.
    """
    rec = rec if rec is not None else NullRecorder()

    if not deduce(board, rng, rec):
        return False
    if board.is_solved:
        return True

    cell = select_branch_cell(board, rng)
    digits = list(board.candidates(cell))
    if rng is not None:
        rng.shuffle(digits)

    for digit in digits:
        snapshot = board.copy()
        rec.on_push()
        ok = board.place(cell, digit)
        rec.on_place(cell, digit, Reason.GUESS)
        if ok and search(board, rng, rec):
            return True
        rec.on_pop()
        board.restore(snapshot)
    return False


def solve(puzzle: str | Board, rng: random.Random | None = None) -> str | None:
    """Return the solved grid as an 81-character string, or None if unsolvable."""
    board = Board.from_string(puzzle) if isinstance(puzzle, str) else puzzle.copy()
    if not search(board, rng):
        return None
    return board.to_string()


def count_solutions(puzzle: str | Board, limit: int = 2) -> int:
    """Count solutions, stopping at ``limit``. A limit of 2 answers uniqueness."""
    board = Board.from_string(puzzle) if isinstance(puzzle, str) else puzzle.copy()
    return _count(board, limit)


def _count(board: Board, limit: int) -> int:
    if not deduce(board, None, NullRecorder()):
        return 0
    if board.is_solved:
        return 1
    cell = select_branch_cell(board, None)
    total = 0
    for digit in board.candidates(cell):
        branch = board.copy()
        if branch.place(cell, digit):
            total += _count(branch, limit - total)
            if total >= limit:
                return total
    return total
