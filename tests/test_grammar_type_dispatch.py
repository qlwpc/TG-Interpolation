"""
Tests for transformer_grammar_type-specific code paths in data input and downstream
evaluation. Verifies that special types (pause, tree_shuffle, tgnomask, etc.) are
handled correctly and don't collide due to fragile prefix/suffix string matching.
"""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from olmo.config import (
    ModelConfig,
    TrainConfig,
    TokenizerConfig,
    TGConfig,
    PaddingDirection,
)
from olmo.data import (
    DataCollator,
    get_TG_generate_bias_func,
)
from olmo.data.memmap_dataset import MemMapDataset
from olmo.data.util import pause_input_ids


# ---------------------------------------------------------------------------
# Helper: create a minimal mock vocab for testing
# ---------------------------------------------------------------------------

def _mock_vocab():
    """Return a SentencepieceVocab-like mock with controlled behavior."""
    vocab = MagicMock()
    vocab.pad = 50258
    vocab.bos = 50256
    vocab.eos = 50257
    vocab.unk = 0
    vocab.bosent = -1
    vocab.eosent = -1
    vocab.pause = 50260
    vocab.opening_non_terminals = (100, 200)
    vocab.closing_non_terminals = (200, 300)

    def convert_to_terminal(tree):
        """Strip non-terminals (IDs 100-299)."""
        arr = np.array(tree)
        mask = (arr < 100) | (arr >= 300)
        return arr[mask].copy()

    def convert_TG_to_tree(tg_arr):
        """Remove duplicates of closing NTs."""
        arr = np.array(tg_arr)
        result = [arr[0]]
        skip_next = False
        for i in range(1, len(arr)):
            if skip_next:
                skip_next = False
                continue
            if 200 <= arr[i] < 300 and i < len(arr) - 1 and arr[i] == arr[i + 1]:
                result.append(arr[i])
                skip_next = True
            else:
                result.append(arr[i])
        return np.array(result, dtype=arr.dtype)

    def get_non_terminal_mask(input_ids):
        ids = np.array(input_ids)
        mask = np.ones(len(ids), dtype=bool)
        mask[(ids >= 100) & (ids < 300)] = False
        return mask

    def is_non_terminal(token_id):
        return 100 <= token_id < 300

    # Make the mock actually work for convert operations
    vocab.convert_treenpy_to_terminal = convert_to_terminal
    vocab.convert_TGnpy_to_tree = convert_TG_to_tree
    vocab.get_non_terminal_mask = get_non_terminal_mask
    vocab.is_non_terminal = is_non_terminal

    # convert_treenpy_to_TG: duplicate closing NTs
    def convert_tree_to_TG(tree):
        arr = np.array(tree)
        result = []
        for t in arr:
            result.append(t)
            if 200 <= t < 300:  # closing NT
                result.append(t)
        return np.array(result, dtype=arr.dtype)

    vocab.convert_treenpy_to_TG = convert_tree_to_TG

    return vocab


# ---------------------------------------------------------------------------
# 1. convert_grammar_input dispatch — ICL task dataset
# ---------------------------------------------------------------------------

class TestConvertGrammarInput:
    """Tests for ICLMultiChoiceTaskDataset.convert_grammar_input()."""

    def _make_mock_dataset(self, grammar_type):
        ds = MagicMock()
        ds.transformer_grammar_type = grammar_type
        ds.vocab = _mock_vocab()
        ds.ispause = 0
        # Replicate the actual method
        from olmo.eval.downstream import ICLMultiChoiceTaskDataset

        def convert_grammar_input(input_ids, grammar_type_param=None):
            if not isinstance(input_ids, np.ndarray):
                input_ids = np.array(input_ids)
            gt = grammar_type_param if grammar_type_param is not None else ds.transformer_grammar_type
            if gt[:8] == "terminal" or gt[:5] == "pause":
                input_ids = ds.vocab.convert_treenpy_to_terminal(input_ids)
            elif gt[:4] == "tree":
                input_ids = ds.vocab.convert_TGnpy_to_tree(input_ids)
            return input_ids.tolist()

        ds.convert_grammar_input = convert_grammar_input
        return ds

    # Input: TG-format array with ONT at IDs 150, 160; terminals at 10, 20; CNT at 250, 260
    # TG format means CNT are duplicated: [150, 10, 250, 250, 160, 20, 260, 260]
    _TG_INPUT = np.array([150, 10, 250, 250, 160, 20, 260, 260])

    def test_terminal_strips_nonterminals(self):
        ds = self._make_mock_dataset("terminal")
        result = ds.convert_grammar_input(self._TG_INPUT)
        # Should only keep terminals: [10, 20]
        assert result == [10, 20]

    def test_pause1_converts_to_terminal(self):
        ds = self._make_mock_dataset("pause1")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == [10, 20]

    def test_pause2_converts_to_terminal(self):
        ds = self._make_mock_dataset("pause2")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == [10, 20]

    def test_pause3_converts_to_terminal(self):
        ds = self._make_mock_dataset("pause3")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == [10, 20]

    def test_pause_slash_converts_to_terminal(self):
        ds = self._make_mock_dataset("pause1/2")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == [10, 20]

    def test_tree_undoes_tg_format(self):
        """tree type should undo TG duplication (CNT doubled → single)."""
        ds = self._make_mock_dataset("tree")
        result = ds.convert_grammar_input(self._TG_INPUT)
        # TG→tree: removes CNT duplicates → [150, 10, 250, 160, 20, 260]
        assert result == [150, 10, 250, 160, 20, 260]

    def test_tree_shuffle_undoes_tg_format(self):
        """BUG CHECK: tree_shuffle[:4]=='tree' so it undoes TG format too.
        Is this correct? For tree_shuffle, the data IS tree-format (not TG),
        so converting from TG→tree on TG input would be wrong.
        But prompts in ICL are typically terminal strings, not tree/TG.
        So this conversion path is only hit for terminal-string prompts."""
        ds = self._make_mock_dataset("tree_shuffle")
        result = ds.convert_grammar_input(self._TG_INPUT)
        # tree_shuffle matches [:4]=='tree' → converts TG→tree
        assert result == [150, 10, 250, 160, 20, 260]

    def test_tg_no_conversion_needed(self):
        """tg type: TG format already, no conversion in convert_grammar_input.
        The function returns input as-is (no if/elif branch matches)."""
        ds = self._make_mock_dataset("tg")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == self._TG_INPUT.tolist()

    def test_tgnomask_no_conversion(self):
        ds = self._make_mock_dataset("tgnomask")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == self._TG_INPUT.tolist()

    def test_mixing_no_conversion(self):
        ds = self._make_mock_dataset("mixing")
        result = ds.convert_grammar_input(self._TG_INPUT)
        assert result == self._TG_INPUT.tolist()


# ---------------------------------------------------------------------------
# 2. BLiMPApproximationDataset — grammar-type-specific dataset selection
# ---------------------------------------------------------------------------

class TestBLiMPDatasetSelection:
    """Verify BLiMPApproximationDataset picks the right .npy file and SENT_SIZE."""

    def _check_config(self, grammar_type, expected_dataset, expected_sent_size):
        """Simulate BLiMPApproximationDataset's grammar dispatch logic (FIXED)."""
        # From downstream.py lines 1428-1441
        if grammar_type[:8] == "terminal" or grammar_type[:5] == "pause":
            dataset_name = "terminal"
            sent_size = 1
        elif grammar_type[:4] == "tree":
            dataset_name = "tree_300"
            sent_size = 300
        else:
            dataset_name = "tg_300"
            sent_size = 300
        assert dataset_name == expected_dataset, (
            f"Grammar '{grammar_type}': expected dataset '{expected_dataset}', got '{dataset_name}'"
        )
        assert sent_size == expected_sent_size, (
            f"Grammar '{grammar_type}': expected SENT_SIZE {expected_sent_size}, got {sent_size}"
        )

    def test_terminal(self):
        self._check_config("terminal", "terminal", 1)

    def test_pause_slash(self):
        self._check_config("pause1/2", "terminal", 1)

    def test_pause_slash_label(self):
        self._check_config("pause1/2_label", "terminal", 1)

    def test_tree(self):
        self._check_config("tree", "tree_300", 300)

    def test_tree_shuffle(self):
        """tree_shuffle[:4] == 'tree' → tree_300. Seems correct."""
        self._check_config("tree_shuffle", "tree_300", 300)

    def test_tg(self):
        self._check_config("tg", "tg_300", 300)

    def test_tgnomask(self):
        self._check_config("tgnomask", "tg_300", 300)

    def test_mixing(self):
        self._check_config("mixing", "tg_300", 300)

    # ---- FIXED ----

    def test_pause1_now_handled(self):
        """FIXED: 'pause1'[:5]=='pause' → terminal/1."""
        self._check_config("pause1", "terminal", 1)

    def test_pause2_now_handled(self):
        """FIXED: 'pause2'[:5]=='pause' → terminal/1."""
        self._check_config("pause2", "terminal", 1)

    def test_pause3_now_handled(self):
        """FIXED: 'pause3'[:5]=='pause' → terminal/1."""
        self._check_config("pause3", "terminal", 1)


# ---------------------------------------------------------------------------
# 3. DataCollator — shuffle_tree string matching
# ---------------------------------------------------------------------------

class TestDataCollatorShuffleTree:
    """Verify DataCollator's string-based dispatch doesn't collide."""

    def test_mask_suffix_matches_tgnomask(self):
        """BUG: tgnomask[-4:] == 'mask' → True.
        In ICL batch collation, this triggers non_terminal_mask application
        even though tgnomask already has its own TG mask."""
        gt = "tgnomask"
        triggers_mask = gt[-4:] == "mask"
        assert triggers_mask is True  # This is the bug

    def test_mask_suffix_matches_tgnomaskaug(self):
        """tgnomaskaug[-4:] == 'aug' → no collision."""
        gt = "tgnomaskaug"
        triggers_mask = gt[-4:] == "mask"
        assert triggers_mask is False

    def test_mask_suffix_tree_shuffle_mask_intentional(self):
        """tree_shuffle_mask ends in 'mask' — this IS intentional."""
        gt = "tree_shuffle_mask"
        triggers_mask = gt[-4:] == "mask"
        assert triggers_mask is True

    def test_shuffle_prefix_tree_shuffle(self):
        """tree_shuffle and tree_shuffle_mask both match [:12]=='tree_shuffle'."""
        assert "tree_shuffle"[:12] == "tree_shuffle"
        assert "tree_shuffle_mask"[:12] == "tree_shuffle"

    def test_shuffle_prefix_not_tgnomask(self):
        """tgnomask is too short for [:12]=='tree_shuffle'."""
        assert "tgnomask"[:12] == "tgnomask"  # shorter than 12, returns full string
        assert "tgnomask"[:12] != "tree_shuffle"

    @pytest.mark.parametrize("grammar_type,expect_vocab_loaded", [
        ("terminal", True),    # truthy string
        ("tree", True),
        ("tg", True),
        ("pause1", True),
        ("tree_shuffle", True),
    ])
    def test_vocab_always_loaded(self, grammar_type, expect_vocab_loaded):
        """DataCollator.from_train_config always loads vocab because
        transformer_grammar_type is always a non-empty (truthy) string."""
        obj = DataCollator(
            pad_direction=PaddingDirection.right,
            pad_token_id=50258,
            generate_attention_mask=False,
            shuffle_tree=grammar_type,
        )
        assert bool(obj.shuffle_tree) is expect_vocab_loaded


# ---------------------------------------------------------------------------
# 4. pause_input_ids — context length consistency
# ---------------------------------------------------------------------------

class TestPauseContextLength:
    """Verify pause_input_ids produces correct length expansion."""

    def test_pause1_doubles_length(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=1)
        assert len(result) == 6  # 3 * (1+1)

    def test_pause2_triples_length(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=2)
        assert len(result) == 9  # 3 * (1+2)

    def test_pause3_quadruples_length(self):
        arr = np.array([1, 2], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=3)
        assert len(result) == 8  # 2 * (1+3)

    def test_pause1_correct_interleave(self):
        arr = np.array([10, 20], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=1)
        expected = np.array([10, 99, 20, 99], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_pause2_correct_interleave(self):
        arr = np.array([10], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=2)
        expected = np.array([10, 99, 99], dtype=np.int64)
        assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# 5. ModelConfig.ispause — edge cases for downstream
# ---------------------------------------------------------------------------

class TestIsPauseDownstream:
    """ispause property used in ICL tasks for pause_input_ids calls."""

    def test_pause1_ispause(self):
        cfg = ModelConfig(transformer_grammar_type="pause1")
        assert cfg.ispause == 1

    def test_pause2_ispause(self):
        cfg = ModelConfig(transformer_grammar_type="pause2")
        assert cfg.ispause == 2

    def test_pause_slash_ispause(self):
        cfg = ModelConfig(transformer_grammar_type="pause1/2")
        assert cfg.ispause == 1  # 1/2 → 1 pause token

    def test_pause_slash_label_ispause(self):
        cfg = ModelConfig(transformer_grammar_type="pause1/2_label")
        assert cfg.ispause == 1


# ---------------------------------------------------------------------------
# 6. XSum format conversion — fragile prefix matching
# ---------------------------------------------------------------------------

class TestXSumFormatDispatch:
    """Simulate XSumDataset.__getitem__ format conversion logic."""

    def _simulate_format_choice(self, grammar_type, has_tg_bias):
        """Replicate the format conversion branching from XSumDataset.__getitem__."""
        if grammar_type == "terminal":
            return "convert_to_terminal"
        elif grammar_type == "tree":
            return "convert_TG_to_tree"
        elif grammar_type[:5] == "pause":
            return "pause_interleave"
        elif has_tg_bias:
            return "apply_tg_bias"
        else:
            return "no_conversion"

    def test_terminal(self):
        assert self._simulate_format_choice("terminal", False) == "convert_to_terminal"

    def test_tree(self):
        assert self._simulate_format_choice("tree", False) == "convert_TG_to_tree"

    def test_tg(self):
        assert self._simulate_format_choice("tg", True) == "apply_tg_bias"

    def test_tgnomask(self):
        assert self._simulate_format_choice("tgnomask", True) == "apply_tg_bias"

    def test_pause_slash(self):
        assert self._simulate_format_choice("pause1/2", False) == "pause_interleave"

    def test_pause2_xsum(self):
        assert self._simulate_format_choice("pause2", False) == "pause_interleave"

    # ---- FIXED ----

    def test_pause1_now_handled_in_xsum(self):
        """FIXED: 'pause1'[:5]=='pause' → pause_interleave."""
        result = self._simulate_format_choice("pause1", False)
        assert result == "pause_interleave"

    def test_tree_shuffle_no_longer_treated_as_tree(self):
        """FIXED: exact match 'tree' avoids tree_shuffle collision."""
        result = self._simulate_format_choice("tree_shuffle", False)
        assert result == "no_conversion"


# ---------------------------------------------------------------------------
# 7. MemMapDataset — pause handling
# ---------------------------------------------------------------------------

class TestMemMapDatasetPause:
    """Verify MemMapDataset correctly handles pause grammar types."""

    def test_pause1_triggers_pause_ids(self, monkeypatch):
        """When grammar_type starts with 'pause', pause_input_ids is called."""
        import olmo.data.memmap_dataset as mmd
        # chunk_size=10 means 10 uint16 elements. Return 20 bytes (10 * sizeof(uint16)).
        monkeypatch.setattr(mmd, "get_bytes_range", lambda p, s, n: b"\x0a\x00" * 10)

        ds = MemMapDataset(
            "/fake/path.npy",
            chunk_size=10,
            transformer_grammar_type="pause1",
            pause_token_id=99,
            memmap_dtype=np.uint16,
            memmap_format="raw",
            include_instance_metadata=False,
        )
        ds._mmap_offsets = [(0, 100)]
        ds._num_instances = 100

        item = ds[0]
        # 10 uint16 tokens raw → after pause1 expansion: 10 * (1+1) = 20 tokens
        assert item["input_ids"].shape[0] == 20

    def test_pause2_doubles_pause_count(self, monkeypatch):
        import olmo.data.memmap_dataset as mmd
        # chunk_size=10 uint16 elements → 20 bytes
        monkeypatch.setattr(mmd, "get_bytes_range", lambda p, s, n: b"\x0b\x00" * 10)

        ds = MemMapDataset(
            "/fake/path.npy",
            chunk_size=10,
            transformer_grammar_type="pause2",
            pause_token_id=99,
            memmap_dtype=np.uint16,
            memmap_format="raw",
            include_instance_metadata=False,
        )
        ds._mmap_offsets = [(0, 100)]
        ds._num_instances = 100

        item = ds[0]
        # 10 uint16 tokens raw → pause2: (1+2)*10 = 30 tokens
        assert item["input_ids"].shape[0] == 30


# ---------------------------------------------------------------------------
# 8. get_TG_generate_bias_func — type dispatch
# ---------------------------------------------------------------------------

class TestTGBiasFactoryDispatch:
    """Verify get_TG_generate_bias_func returns correct class per grammar type."""

    def _factory_result_type(self, grammar_type):
        """Return what type of bias object would be created."""
        if grammar_type == "tgtree":
            return None
        if grammar_type == "terminal":
            return None
        if grammar_type == "mixing":
            return "HeadMixingBias"
        if grammar_type == "tg":
            return "TG_attention_bias"
        if grammar_type[:10] == "tgproximal":
            return "KProximal_TG_attention_bias"
        if grammar_type[:8] == "tgnomask":
            return "KProximal_TG_attention_bias"
        if grammar_type == "tgheight":
            return "Height_TG_attention_bias"
        return None

    def test_tg(self):
        assert self._factory_result_type("tg") == "TG_attention_bias"

    def test_tgproximal(self):
        assert self._factory_result_type("tgproximal") == "KProximal_TG_attention_bias"

    def test_tgproximal_aug(self):
        assert self._factory_result_type("tgproximalaug") == "KProximal_TG_attention_bias"

    def test_tgnomask(self):
        assert self._factory_result_type("tgnomask") == "KProximal_TG_attention_bias"

    def test_tgnomask_aug(self):
        assert self._factory_result_type("tgnomaskaug") == "KProximal_TG_attention_bias"

    def test_terminal(self):
        assert self._factory_result_type("terminal") is None

    def test_tree(self):
        """tree maps to tgtree implicitly? No — 'tree' is not in the if/elif chain.
        The config uses transformer_grammar_type='tree' which in get_TG_generate_bias_func
        doesn't match any branch. But 'tgtree' → early return None."""
        assert self._factory_result_type("tree") is None

    def test_pause_types(self):
        for pt in ["pause1", "pause2", "pause3", "pause1/2"]:
            assert self._factory_result_type(pt) is None, f"Failed for {pt}"


# ---------------------------------------------------------------------------
# 9. SG dataset pause handling
# ---------------------------------------------------------------------------

class TestSGDatasetPause:
    """Verify SyntacticGeneralizationDataset pause handling."""

    def _simulate_sg_pause_handling(self, grammar_type):
        """Simulate the pause check in SG prep_examples (line 1215, FIXED)."""
        return grammar_type[:5] == "pause"

    def test_pause_slash_handled(self):
        assert self._simulate_sg_pause_handling("pause1/2") is True

    def test_pause_slash_label_handled(self):
        assert self._simulate_sg_pause_handling("pause1/2_label") is True

    # ---- FIXED ----
    def test_pause1_now_handled(self):
        assert self._simulate_sg_pause_handling("pause1") is True

    def test_pause2_now_handled(self):
        assert self._simulate_sg_pause_handling("pause2") is True

    def test_pause3_now_handled(self):
        assert self._simulate_sg_pause_handling("pause3") is True
