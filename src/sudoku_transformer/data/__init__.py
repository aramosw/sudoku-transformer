"""Turning solve traces into token arrays: vocabulary, tokeniser, dataset build."""

from .build import BuildConfig, build
from .dataset import Split, load_manifest, load_split, target_mask
from .sources import Puzzle, read_kaggle
from .tokenizer import clues_end_index, decode, encode, encode_trace, puzzle_from_tokens
from .vocab import (
    CLUES_END,
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

__all__ = [
    "BuildConfig",
    "CLUES_END",
    "PAD",
    "POP",
    "PUSH",
    "Puzzle",
    "SUCCESS",
    "Split",
    "VOCAB_SIZE",
    "build",
    "clues_end_index",
    "decode",
    "decode_placement",
    "encode",
    "encode_trace",
    "is_placement",
    "load_manifest",
    "load_split",
    "placement_id",
    "puzzle_from_tokens",
    "read_kaggle",
    "target_mask",
    "token_name",
]
