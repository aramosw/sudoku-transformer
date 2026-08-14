from __future__ import annotations

import pytest

from sudoku_transformer.data.vocab import (
    CLUES_END,
    NUM_PLACEMENTS,
    PAD,
    POP,
    PUSH,
    SUCCESS,
    VOCAB_SIZE,
    decode_placement,
    is_placement,
    placement_id,
    token_name,
)


def test_vocabulary_size():
    assert NUM_PLACEMENTS == 729
    assert VOCAB_SIZE == 734


def test_specials_are_distinct_and_above_the_placements():
    specials = [CLUES_END, PUSH, POP, SUCCESS, PAD]
    assert len(set(specials)) == 5
    assert all(token >= NUM_PLACEMENTS for token in specials)
    assert max(specials) == VOCAB_SIZE - 1


def test_placement_ids_are_a_bijection():
    seen = {placement_id(cell, digit) for cell in range(81) for digit in range(1, 10)}
    assert seen == set(range(NUM_PLACEMENTS))


def test_placement_round_trip():
    for cell in range(81):
        for digit in range(1, 10):
            assert decode_placement(placement_id(cell, digit)) == (cell, digit)


def test_is_placement_separates_moves_from_specials():
    assert is_placement(0)
    assert is_placement(NUM_PLACEMENTS - 1)
    assert not is_placement(CLUES_END)
    assert not is_placement(PAD)


def test_decode_placement_rejects_specials():
    with pytest.raises(ValueError):
        decode_placement(CLUES_END)


def test_token_names():
    assert token_name(placement_id(0, 5)) == "[R0C0=5]"
    assert token_name(placement_id(80, 9)) == "[R8C8=9]"
    assert token_name(CLUES_END) == "[clues_end]"
    with pytest.raises(ValueError):
        token_name(VOCAB_SIZE)
