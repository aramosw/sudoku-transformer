from __future__ import annotations

import numpy as np
import pytest

from sudoku_transformer.data.build import BuildConfig, build
from sudoku_transformer.data.dataset import load_manifest, load_split, target_mask
from sudoku_transformer.data.tokenizer import (
    clues_end_index,
    decode,
    puzzle_from_tokens,
)
from sudoku_transformer.data.vocab import PAD, SUCCESS
from sudoku_transformer.sudoku.trace import replay


@pytest.fixture(scope="module")
def built(tmp_path_factory, kaggle_csv) -> tuple:
    out_dir = tmp_path_factory.mktemp("traces")
    manifest = build(
        BuildConfig(
            csv_path=kaggle_csv,
            out_dir=out_dir,
            n_puzzles=300,
            test_fraction=0.1,
            seed=0,
            workers=1,
        )
    )
    return out_dir, manifest


def test_manifest_accounts_for_every_puzzle(built):
    _, manifest = built
    assert manifest["requested"] == 300
    assert (
        manifest["kept"] + manifest["dropped_too_long"] + manifest["dropped_unsolved"]
        == 300
    )
    assert manifest["dropped_unsolved"] == 0
    assert manifest["train"] + manifest["test"] == manifest["kept"]


def test_manifest_is_written_to_disk(built):
    out_dir, manifest = built
    assert load_manifest(out_dir) == manifest


def test_split_shapes_line_up(built):
    out_dir, manifest = built
    for name in ("train", "test"):
        split = load_split(out_dir, name)
        assert len(split) == manifest[name]
        assert split.tokens.shape == (manifest[name], manifest["max_len"])
        assert (
            len(split.lengths)
            == len(split.n_clues)
            == len(split.difficulty)
            == len(split)
        )


def test_rows_are_padded_beyond_their_length(built):
    out_dir, _ = built
    split = load_split(out_dir, "train")
    for row, length in zip(split.tokens[:50], split.lengths[:50], strict=True):
        assert row[length - 1] == SUCCESS
        assert np.all(row[length:] == PAD)


def test_clue_count_matches_the_clues_end_position(built):
    out_dir, _ = built
    split = load_split(out_dir, "train")
    for row, length, n_clues in zip(
        split.tokens[:50], split.lengths[:50], split.n_clues[:50], strict=True
    ):
        assert clues_end_index(list(row[:length])) == n_clues


def test_stored_traces_replay_to_a_solved_grid(built):
    out_dir, _ = built
    split = load_split(out_dir, "train")
    for row, length in zip(split.tokens[:50], split.lengths[:50], strict=True):
        tokens = [int(t) for t in row[:length]]
        board = replay(decode(tokens))
        assert board.is_solved
        puzzle = puzzle_from_tokens(tokens)
        assert all(
            c == "." or c == s for c, s in zip(puzzle, board.to_string(), strict=True)
        )


def test_target_mask_starts_after_clues_end(built):
    out_dir, _ = built
    split = load_split(out_dir, "train")
    mask = target_mask(split)

    assert mask.shape == split.tokens.shape
    for i in range(50):
        n_clues, length = int(split.n_clues[i]), int(split.lengths[i])
        assert not mask[i, : n_clues + 1].any()
        assert mask[i, n_clues + 1 : length].all()
        assert not mask[i, length:].any()
        assert mask[i].sum() == length - n_clues - 1


def test_build_is_reproducible(tmp_path, built, kaggle_csv):
    out_dir, manifest = built
    again = build(
        BuildConfig(
            csv_path=kaggle_csv,
            out_dir=tmp_path / "again",
            n_puzzles=300,
            test_fraction=0.1,
            seed=0,
            workers=1,
        )
    )
    assert again["kept"] == manifest["kept"]
    assert np.array_equal(
        load_split(out_dir, "train").tokens,
        load_split(tmp_path / "again", "train").tokens,
    )


def test_different_seed_gives_different_traces(tmp_path, built, kaggle_csv):
    out_dir, _ = built
    build(
        BuildConfig(
            csv_path=kaggle_csv,
            out_dir=tmp_path / "seed1",
            n_puzzles=300,
            test_fraction=0.1,
            seed=1,
            workers=1,
        )
    )
    assert not np.array_equal(
        load_split(out_dir, "train").tokens,
        load_split(tmp_path / "seed1", "train").tokens,
    )
