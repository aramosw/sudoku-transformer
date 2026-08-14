"""Solve traces: the event stream a puzzle produces on its way to being solved.

Events are the paper's ``[clues] [clues_end] [deductions] [push] ... [pop] ...
[success]`` sequence before tokenisation. A Place carries why the digit went in,
which the 729 placement tokens cannot express but the probing phase needs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .board import EMPTY_CHARS, NUM_CELLS, Board
from .solver import Reason, search


class InvalidTrace(ValueError):
    """Raised when a trace does not replay against a fresh board."""


@dataclass(frozen=True, slots=True)
class Place:
    """A digit going into a cell.

    The reason is None for events decoded from tokens, which cannot carry it.
    """

    cell: int
    digit: int
    reason: Reason | None = None


@dataclass(frozen=True, slots=True)
class CluesEnd:
    """End of the given clues. Loss is masked to everything after this."""


@dataclass(frozen=True, slots=True)
class Push:
    """Start of a guessed branch."""


@dataclass(frozen=True, slots=True)
class Pop:
    """The branch opened by the matching Push failed; state rolls back."""


@dataclass(frozen=True, slots=True)
class Success:
    """The grid is complete."""


Event = Place | CluesEnd | Push | Pop | Success


@dataclass(slots=True)
class Trace:
    """A puzzle, its solution if one was found, and the full event stream."""

    puzzle: str
    solution: str | None
    events: list[Event] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.solution is not None

    @property
    def n_tokens(self) -> int:
        return len(self.events)

    @property
    def n_guesses(self) -> int:
        return sum(1 for e in self.events if isinstance(e, Push))

    @property
    def n_backtracks(self) -> int:
        return sum(1 for e in self.events if isinstance(e, Pop))

    @property
    def max_depth(self) -> int:
        depth = best = 0
        for event in self.events:
            if isinstance(event, Push):
                depth += 1
                best = max(best, depth)
            elif isinstance(event, Pop):
                depth -= 1
        return best

    def placements(self, reason: Reason | None = None) -> list[Place]:
        return [
            e
            for e in self.events
            if isinstance(e, Place) and (reason is None or e.reason is reason)
        ]


class _TraceRecorder:
    __slots__ = ("events",)

    def __init__(self, events: list[Event]) -> None:
        self.events = events

    def on_place(self, cell: int, digit: int, reason: Reason) -> None:
        self.events.append(Place(cell, digit, reason))

    def on_push(self) -> None:
        self.events.append(Push())

    def on_pop(self) -> None:
        self.events.append(Pop())

    def on_success(self) -> None:
        self.events.append(Success())


def trace_puzzle(puzzle: str, rng: random.Random | None = None) -> Trace:
    """Solve a puzzle and return the full event stream, failed branches included.

    Pass an rng for the randomisation the paper applies at every choice point;
    without one the solve is deterministic.
    """
    if len(puzzle) != NUM_CELLS:
        raise ValueError(f"expected {NUM_CELLS} characters, got {len(puzzle)}")

    events: list[Event] = []
    rec = _TraceRecorder(events)
    board = Board.empty()

    for cell, char in enumerate(puzzle):
        if char in EMPTY_CHARS:
            continue
        digit = int(char)
        if not board.place(cell, digit):
            raise ValueError(f"clue {digit} at cell {cell} contradicts an earlier clue")
        rec.on_place(cell, digit, Reason.CLUE)
    events.append(CluesEnd())

    solved = search(board, rng, rec)
    if solved:
        rec.on_success()
    return Trace(
        puzzle=puzzle, solution=board.to_string() if solved else None, events=events
    )


def replay(events: list[Event]) -> Board:
    """Re-run an event stream against a fresh board, checking it as we go.

    Every placement must have been legal at the moment it was emitted, every Pop
    must unwind to its Push, and a Success must find a complete grid.
    """
    board = Board.empty()
    stack: list[Board] = []
    seen_clues_end = False

    for index, event in enumerate(events):
        match event:
            case Place(cell=cell, digit=digit):
                if board.grid[cell] != 0:
                    raise InvalidTrace(
                        f"event {index}: cell {cell} already holds {board.grid[cell]}"
                    )
                if not board.has_candidate(cell, digit):
                    raise InvalidTrace(
                        f"event {index}: digit {digit} is not a candidate for cell {cell}"
                    )
                board.place(cell, digit)
            case CluesEnd():
                if seen_clues_end:
                    raise InvalidTrace(f"event {index}: second clues_end")
                seen_clues_end = True
            case Push():
                stack.append(board.copy())
            case Pop():
                if not stack:
                    raise InvalidTrace(f"event {index}: pop with no matching push")
                board.restore(stack.pop())
            case Success():
                if not board.is_solved:
                    raise InvalidTrace(f"event {index}: success on an incomplete grid")

    if not seen_clues_end:
        raise InvalidTrace("trace contains no clues_end")
    return board
