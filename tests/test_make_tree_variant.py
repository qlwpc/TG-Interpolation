"""Tests for datatools/make_tree_variant.py — LIN variant stream generation.

Generalizes the dev-only prototype ``datatools/process_bbc.py`` into a
parameterized, streaming generator for the paper's causal-attention tree
linearization variants (Table 1/3):

  * ``noont``     — LIN1−ONT: drop every opening non-terminal
  * ``compress``  — LIN1−merge: collapse each run of closing NTs into one
                    generic closing token (``<X)>``)
  * ``triplecnt`` — LIN3: repeat every closing NT 3×

Tests use synthetic token ids: opening NTs 100–102, closing NTs 103–105,
terminals 10–29, EOS=1, BOS=0.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from datatools.parse_pretrain_data.make_tree_variant import (
    NTRanges,
    chunk_bounds,
    nt_ranges_from_tokenizer,
    transform_tokens,
)

OPEN = NTRanges(open_lo=100, open_hi=103, close_lo=103, close_hi=106, generic_close=105)
EOS = 1


def _sentence(terminals: list[int], opens: list[int], closes: list[int]) -> list[int]:
    """A LIN1-shaped sentence: opens... terminals... closes... EOS."""
    return opens + terminals + closes + [EOS]


def _is_close(t: int) -> bool:
    return OPEN.close_lo <= t < OPEN.close_hi


# ---------------------------------------------------------------------------
# transform_tokens — per-chunk pure function
# ---------------------------------------------------------------------------

class TestTransformTokens:
    def test_noont_drops_only_opening(self):
        tokens = np.asarray(_sentence([10, 11], opens=[100, 101], closes=[103, 104]), dtype=np.uint16)
        out = transform_tokens(tokens, OPEN, "noont", fixed_token=105)
        assert out.tolist() == [10, 11, 103, 104, EOS]

    def test_noont_keeps_terminals_and_closing_intact(self):
        tokens = np.asarray([10, 103, 11, 104, EOS], dtype=np.uint16)
        out = transform_tokens(tokens, OPEN, "noont", fixed_token=105)
        assert out.tolist() == [10, 103, 11, 104, EOS]

    def test_compress_collapses_run_to_generic(self):
        # run of three closing NTs → single <X)> (105); single closing → 105 too
        tokens = np.asarray([100, 10, 103, 103, 104, 11, 104, EOS], dtype=np.uint16)
        out = transform_tokens(tokens, OPEN, "compress", fixed_token=105)
        assert out.tolist() == [100, 10, 105, 11, 105, EOS]

    def test_compress_preserves_non_closing_runs(self):
        # consecutive terminals must NOT be merged
        tokens = np.asarray([10, 11, 12, EOS], dtype=np.uint16)
        out = transform_tokens(tokens, OPEN, "compress", fixed_token=105)
        assert out.tolist() == [10, 11, 12, EOS]

    def test_triplecnt_repeats_each_closing_three_times(self):
        tokens = np.asarray([100, 10, 103, 104, 11, EOS], dtype=np.uint16)
        out = transform_tokens(tokens, OPEN, "triplecnt", fixed_token=105)
        assert out.tolist() == [100, 10, 103, 103, 103, 104, 104, 104, 11, EOS]

    def test_empty_input(self):
        empty = np.asarray([], dtype=np.uint16)
        for mode in ("noont", "compress", "triplecnt"):
            out = transform_tokens(empty, OPEN, mode, fixed_token=105)
            assert out.size == 0

    def test_unknown_mode_raises(self):
        tokens = np.asarray([10], dtype=np.uint16)
        with pytest.raises(ValueError, match="unknown variant"):
            transform_tokens(tokens, OPEN, "shuffle", fixed_token=105)


# ---------------------------------------------------------------------------
# chunk_bounds — boundaries must never split a closing-NT run
# ---------------------------------------------------------------------------

class TestChunkBounds:
    def test_boundary_moves_off_run(self):
        # tokens: [10, C, C, C, 11] with nominal boundary at index 2 (mid-run)
        tokens = np.asarray([10, 103, 103, 103, 11], dtype=np.uint16)
        bounds = list(chunk_bounds(tokens, chunk_size=2, ranges=OPEN))
        # first chunk must end at the run start (index 1), not mid-run
        assert bounds[0] == (0, 1)

    def test_boundaries_cover_everything(self):
        rng = np.random.default_rng(0)
        tokens = rng.integers(10, 106, size=997).astype(np.uint16)
        bounds = list(chunk_bounds(tokens, chunk_size=64, ranges=OPEN))
        assert bounds[0][0] == 0
        assert bounds[-1][1] == len(tokens)
        for (_, end_a), (start_b, _) in zip(bounds, bounds[1:]):
            assert end_a == start_b

    def test_run_longer_than_chunk_size(self):
        # run of 5 closing NTs with chunk_size=2: the boundary must extend
        # forward past the run instead of collapsing to an empty chunk
        # (regression: the backward walk used to loop forever here)
        tokens = np.asarray([10, 103, 103, 103, 103, 103, 11], dtype=np.uint16)
        bounds = list(chunk_bounds(tokens, chunk_size=2, ranges=OPEN))
        assert bounds[0] == (0, 1)
        assert bounds[1] == (1, 6)  # covers the whole run
        assert bounds[-1] == (6, 7)
        assert [e - s for s, e in bounds].count(0) == 0  # no empty chunks

    def test_no_split_runs_end_to_end(self):
        rng = np.random.default_rng(1)
        tokens = rng.integers(10, 106, size=2000).astype(np.uint16)
        for start, end in chunk_bounds(tokens, chunk_size=32, ranges=OPEN):
            if start > 0 and end < len(tokens):
                inside_run = _is_close(tokens[end - 1]) and _is_close(tokens[end])
                assert not inside_run


# ---------------------------------------------------------------------------
# nt_ranges_from_tokenizer — derive ranges from a tokenizer JSON
# ---------------------------------------------------------------------------

class TestNtRangesFromTokenizer:
    def _write_fake_tokenizer(self, tmp_path):
        added = (
            [{"id": 50256, "content": "<|beginoftext|>"}, {"id": 50257, "content": "<|pad|>"}]
            + [{"id": 50268 + i, "content": f"<({name}>"} for i, name in enumerate(["S", "NP", "VP", "X"])]
            + [{"id": 50272 + i, "content": f"{name})>"} for i, name in enumerate(["S", "NP", "VP", "X"])]
        )
        path = tmp_path / "tok.json"
        path.write_text(json.dumps({"added_tokens": added}))
        return path

    def test_ranges_derived(self, tmp_path):
        path = self._write_fake_tokenizer(tmp_path)
        r = nt_ranges_from_tokenizer(path)
        assert (r.open_lo, r.open_hi) == (50268, 50272)
        assert (r.close_lo, r.close_hi) == (50272, 50276)
        assert r.generic_close == 50275  # <X)>

    def test_real_tokenizer_json(self):
        """Against the repo's actual TG_GPT2_tokenizer.json (if present):
        opening NTs 50268–50293, closing 50294–50319, <X)> = 50319."""
        from pathlib import Path

        tok = Path("dataset/bbc-news/TG_GPT2_tokenizer.json")
        if not tok.exists():
            pytest.skip("TG_GPT2_tokenizer.json not on disk")
        r = nt_ranges_from_tokenizer(tok)
        assert (r.open_lo, r.open_hi) == (50268, 50294)
        assert (r.close_lo, r.close_hi) == (50294, 50320)
        assert r.generic_close == 50319


# ---------------------------------------------------------------------------
# transform_file — end-to-end with forced chunking
# ---------------------------------------------------------------------------

class TestTransformFile:
    def test_end_to_end_chunked_compress(self, tmp_path):
        from datatools.parse_pretrain_data.make_tree_variant import transform_file

        # 3 sentences; a closing run of 2 in sentence 2
        tokens = np.asarray(
            [0]
            + _sentence([10, 11], opens=[100], closes=[103])
            + _sentence([12], opens=[101, 102], closes=[103, 104])
            + _sentence([13], opens=[100], closes=[105])
            + [EOS],
            dtype=np.uint16,
        )
        src = tmp_path / "train.npy"
        np.save(src, tokens)

        out_path = transform_file(
            src,
            tmp_path / "out" / "train.npy",
            ranges=OPEN,
            variant="compress",
            fixed_token=105,
            chunk_size=4,  # force many chunks, some landing mid-run
        )
        out = np.load(out_path)
        # s1: [100,10,11,105,EOS]  s2: [101,102,12,105,EOS]  s3: [100,13,105,EOS]
        assert out.tolist() == [
            0, 100, 10, 11, 105, EOS,
            101, 102, 12, 105, EOS,
            100, 13, 105, EOS,
            EOS,
        ]

    def test_end_to_end_noont(self, tmp_path):
        from datatools.parse_pretrain_data.make_tree_variant import transform_file

        tokens = np.asarray(
            _sentence([10, 11, 12], opens=[100, 101, 102], closes=[103, 104, 105]),
            dtype=np.uint16,
        )
        src = tmp_path / "train.npy"
        np.save(src, tokens)
        out_path = transform_file(
            src, tmp_path / "out" / "train.npy", ranges=OPEN, variant="noont",
            fixed_token=105, chunk_size=3,
        )
        out = np.load(out_path)
        assert out.tolist() == [10, 11, 12, 103, 104, 105, EOS]
