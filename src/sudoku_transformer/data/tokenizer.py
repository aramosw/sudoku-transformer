"""Conversion between solve-trace events and token ids.

Encoding is lossy in exactly one way: a Place's Reason has no token, since all
four reasons share the same 729 placements. Decoded events carry reason=None.
"""

from __future__ import annotations

from ..sudoku.board import NUM_CELLS
from ..sudoku.trace import CluesEnd, Event, Place, Pop, Push, Success, Trace
from .vocab import (
    CLUES_END,
    PAD,
    POP,
    PUSH,
    SUCCESS,
    decode_placement,
    is_placement,
    placement_id,
)

_SPECIAL_EVENTS = {CLUES_END: CluesEnd, PUSH: Push, POP: Pop, SUCCESS: Success}


def encode(events: list[Event]) -> list[int]:
    tokens = []
    for event in events:
        match event:
            case Place(cell=cell, digit=digit):
                tokens.append(placement_id(cell, digit))
            case CluesEnd():
                tokens.append(CLUES_END)
            case Push():
                tokens.append(PUSH)
            case Pop():
                tokens.append(POP)
            case Success():
                tokens.append(SUCCESS)
    return tokens


def decode(tokens: list[int]) -> list[Event]:
    """Turn token ids back into events, ignoring padding."""
    events: list[Event] = []
    for token in tokens:
        if token == PAD:
            continue
        if is_placement(token):
            cell, digit = decode_placement(token)
            events.append(Place(cell, digit))
        elif token in _SPECIAL_EVENTS:
            events.append(_SPECIAL_EVENTS[token]())
        else:
            raise ValueError(f"{token} is outside the vocabulary")
    return events


def encode_trace(trace: Trace, max_len: int | None = None) -> list[int] | None:
    """Encode a solved trace, or None if it failed or overflows ``max_len``."""
    if not trace.solved:
        return None
    tokens = encode(trace.events)
    if max_len is not None and len(tokens) > max_len:
        return None
    return tokens


def clues_end_index(tokens: list[int]) -> int:
    """Position of the [clues_end] token, which is also the number of clues."""
    return tokens.index(CLUES_END)


def puzzle_from_tokens(tokens: list[int]) -> str:
    """Recover the 81-character puzzle from the clue prefix of a trace.

    The clues are stored in the token stream itself, so nothing else needs to
    keep a copy of the puzzle alongside the packed arrays.
    """
    grid = ["."] * NUM_CELLS
    for token in tokens[: clues_end_index(tokens)]:
        cell, digit = decode_placement(token)
        grid[cell] = str(digit)
    return "".join(grid)
