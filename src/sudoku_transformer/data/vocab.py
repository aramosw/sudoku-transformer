"""The 734-token vocabulary: 729 cell/digit placements plus five specials.

Placements occupy ids 0..728 as ``cell * 9 + (digit - 1)``, so ``token < 729``
is the test for "is this a move", which rollout uses to mask illegal output.
The specials follow.
"""

from __future__ import annotations

from ..sudoku.board import N as NUM_DIGITS
from ..sudoku.board import NUM_CELLS, col_of, row_of

NUM_PLACEMENTS = NUM_CELLS * NUM_DIGITS

CLUES_END = NUM_PLACEMENTS
PUSH = NUM_PLACEMENTS + 1
POP = NUM_PLACEMENTS + 2
SUCCESS = NUM_PLACEMENTS + 3
PAD = NUM_PLACEMENTS + 4

VOCAB_SIZE = NUM_PLACEMENTS + 5

SPECIAL_NAMES = {
    CLUES_END: "[clues_end]",
    PUSH: "[push]",
    POP: "[pop]",
    SUCCESS: "[success]",
    PAD: "[pad]",
}


def placement_id(cell: int, digit: int) -> int:
    return cell * NUM_DIGITS + digit - 1


def decode_placement(token: int) -> tuple[int, int]:
    """Split a placement token back into its cell and digit."""
    if not is_placement(token):
        raise ValueError(f"{token} is not a placement token")
    return divmod(token, NUM_DIGITS)[0], token % NUM_DIGITS + 1


def is_placement(token: int) -> bool:
    return 0 <= token < NUM_PLACEMENTS


def token_name(token: int) -> str:
    """Render a token the way the paper writes it, for debugging and logs."""
    if is_placement(token):
        cell, digit = decode_placement(token)
        return f"[R{row_of(cell)}C{col_of(cell)}={digit}]"
    if token in SPECIAL_NAMES:
        return SPECIAL_NAMES[token]
    raise ValueError(f"{token} is outside the vocabulary")
