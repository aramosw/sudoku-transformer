from __future__ import annotations

import random

import pytest

from sudoku_transformer.data.tokenizer import (
    clues_end_index,
    decode,
    encode,
    encode_trace,
    puzzle_from_tokens,
)
from sudoku_transformer.data.vocab import CLUES_END, PAD, POP, PUSH, SUCCESS, VOCAB_SIZE
from sudoku_transformer.sudoku.trace import (
    CluesEnd,
    Place,
    Pop,
    Push,
    Success,
    replay,
    trace_puzzle,
)


def test_encodes_each_event_type():
    events = [Place(0, 5), CluesEnd(), Push(), Pop(), Success()]
    assert encode(events) == [4, CLUES_END, PUSH, POP, SUCCESS]


def test_tokens_stay_inside_the_vocabulary(easy_puzzle, hard_puzzle):
    for puzzle in (easy_puzzle[0], hard_puzzle):
        tokens = encode(trace_puzzle(puzzle, random.Random(0)).events)
        assert all(0 <= token < VOCAB_SIZE for token in tokens)


def test_round_trip_through_tokens(hard_puzzle):
    events = trace_puzzle(hard_puzzle, random.Random(3)).events
    tokens = encode(events)
    assert encode(decode(tokens)) == tokens


def test_decode_drops_the_reason(easy_puzzle):
    events = trace_puzzle(easy_puzzle[0]).events
    decoded = decode(encode(events))
    assert all(e.reason is None for e in decoded if isinstance(e, Place))
    assert [(e.cell, e.digit) for e in decoded if isinstance(e, Place)] == [
        (e.cell, e.digit) for e in events if isinstance(e, Place)
    ]


def test_decoded_events_still_replay(hard_puzzle):
    trace = trace_puzzle(hard_puzzle, random.Random(4))
    assert replay(decode(encode(trace.events))).to_string() == trace.solution


def test_decode_ignores_padding(easy_puzzle):
    tokens = encode(trace_puzzle(easy_puzzle[0]).events)
    assert decode(tokens + [PAD] * 20) == decode(tokens)


def test_decode_rejects_unknown_tokens():
    with pytest.raises(ValueError):
        decode([VOCAB_SIZE])


def test_clues_end_index_equals_the_clue_count(easy_puzzle):
    puzzle, _ = easy_puzzle
    tokens = encode(trace_puzzle(puzzle).events)
    assert clues_end_index(tokens) == 81 - puzzle.count(".")


def test_puzzle_is_recoverable_from_its_tokens(easy_puzzle, hard_puzzle):
    for puzzle in (easy_puzzle[0], hard_puzzle):
        tokens = encode(trace_puzzle(puzzle, random.Random(1)).events)
        assert puzzle_from_tokens(tokens) == puzzle


def test_encode_trace_enforces_the_length_limit(hard_puzzle, easy_puzzle):
    long_trace = trace_puzzle(hard_puzzle, random.Random(0))
    assert long_trace.n_tokens > 250
    assert encode_trace(long_trace, max_len=250) is None
    assert encode_trace(long_trace) is not None

    short_trace = trace_puzzle(easy_puzzle[0])
    assert encode_trace(short_trace, max_len=250) == encode(short_trace.events)


def test_kaggle_traces_round_trip(kaggle_rows):
    for row in kaggle_rows[:50]:
        trace = trace_puzzle(row["puzzle"], random.Random(row["id"]))
        tokens = encode(trace.events)
        assert puzzle_from_tokens(tokens) == row["puzzle"]
        assert replay(decode(tokens)).to_string() == row["solution"]
