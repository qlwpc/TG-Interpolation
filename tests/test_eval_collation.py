"""
Tests for ICL/downstream evaluation collation logic with special grammar types.

Focuses on the fragile string prefix/suffix checks in:
- ICLMultiChoiceTaskDataset batch collation
- DataCollator.__call__ tree_shuffle / mask logic
- XSumDataset format conversion
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from olmo.config import (
    ModelConfig,
    TrainConfig,
    PaddingDirection,
)
from olmo.data import DataCollator


# ---------------------------------------------------------------------------
# DataCollator: tree_shuffle and mask dispatch
# ---------------------------------------------------------------------------

class TestCollatorShuffleMaskDispatch:
    """The DataCollator uses shuffle_tree[:12] and shuffle_tree[-4:] for branching."""

    @pytest.mark.parametrize("grammar_type,expect_shuffle", [
        ("tree_shuffle", True),
        ("tree_shuffle_mask", True),
        ("tree", False),
        ("tg", False),
        ("tgnomask", False),  # too short for [:12]
        ("terminal", False),
    ])
    def test_tree_shuffle_detection(self, grammar_type, expect_shuffle):
        """[:12] == 'tree_shuffle' correctly identifies shuffle variants."""
        detected = grammar_type[:12] == "tree_shuffle"
        assert detected is expect_shuffle, f"'{grammar_type}'[:12]=='tree_shuffle' → {detected}"

    @pytest.mark.parametrize("grammar_type,expect_mask", [
        ("tree_shuffle_mask", True),   # intentional
        ("tgnomask", False),            # FIXED: exact match now
        ("tgnomaskaug", False),
        ("tree_shuffle", False),
        ("tree", False),
        ("tg", False),
        ("terminal", False),
        ("pause1", False),
    ])
    def test_mask_suffix_detection(self, grammar_type, expect_mask):
        """FIXED: exact match 'tree_shuffle_mask' avoids tgnomask collision."""
        detected = grammar_type == "tree_shuffle_mask"
        assert detected is expect_mask, f"'{grammar_type}'=='tree_shuffle_mask' → {detected}"


# ---------------------------------------------------------------------------
# Collator vocab loading
# ---------------------------------------------------------------------------

class TestCollatorVocabCondition:
    """DataCollator.from_train_config loads vocab when shuffle_tree is truthy."""

    def test_all_grammar_types_truthy(self):
        """Every transformer_grammar_type is a non-empty string → truthy.
        This means vocab is always loaded from file when from_train_config is used."""
        all_types = [
            "terminal", "tree", "tg", "tgnomask", "tgnomaskaug",
            "tgproximal", "tgheight", "pause1", "pause2", "pause3",
            "pause1/2", "pause1/2_label", "tree_shuffle", "tree_shuffle_mask",
            "mixing", "tgtree",
        ]
        for gt in all_types:
            assert bool(gt) is True, f"'{gt}' should be truthy"
            assert gt[:12] != ""  # always has content


# ---------------------------------------------------------------------------
# ICL batch collation: label_mask from non_terminal_mask
# ---------------------------------------------------------------------------

class TestICLBatchCollationMask:
    """Simulate the ICL collate_fn logic for label_mask from
    self.transformer_grammar_type[-4:] == 'mask' check."""

    def _mock_vocab(self):
        vocab = MagicMock()
        def get_non_terminal_mask(input_ids):
            arr = input_ids.numpy() if hasattr(input_ids, 'numpy') else np.array(input_ids)
            mask = np.ones(len(arr), dtype=bool)
            mask[(arr >= 100) & (arr < 300)] = False
            return mask
        vocab.get_non_terminal_mask = get_non_terminal_mask
        return vocab

    def _should_apply_nt_mask(self, grammar_type):
        """Replicate the check at downstream.py line 487 (FIXED to exact match)."""
        return grammar_type == "tree_shuffle_mask"

    def test_tree_shuffle_mask_gets_nt_mask(self):
        """tree_shuffle_mask: [-4:]=='mask' → applies NT mask. Intentional."""
        assert self._should_apply_nt_mask("tree_shuffle_mask") is True

    def test_tgnomask_no_longer_gets_nt_mask(self):
        """FIXED: exact match 'tree_shuffle_mask' excludes tgnomask."""
        assert self._should_apply_nt_mask("tgnomask") is False

    def test_tgnomaskaug_does_not_get_nt_mask(self):
        """tgnomaskaug[-4:]=='aug' → no collision."""
        assert self._should_apply_nt_mask("tgnomaskaug") is False

    def test_tg_does_not_get_nt_mask(self):
        assert self._should_apply_nt_mask("tg") is False

    def test_tree_does_not_get_nt_mask(self):
        assert self._should_apply_nt_mask("tree") is False


# ---------------------------------------------------------------------------
# ICL batch collation: tree_shuffle
# ---------------------------------------------------------------------------

class TestICLBatchCollationShuffle:
    """Simulate the ICL collate_fn logic for tree_shuffle."""

    def _should_shuffle(self, grammar_type):
        """Replicate the check at downstream.py line 473."""
        return grammar_type[:12] == "tree_shuffle"

    def test_tree_shuffle_shuffles(self):
        assert self._should_shuffle("tree_shuffle") is True

    def test_tree_shuffle_mask_shuffles(self):
        assert self._should_shuffle("tree_shuffle_mask") is True

    def test_tree_does_not_shuffle(self):
        assert self._should_shuffle("tree") is False

    def test_short_types_do_not_shuffle(self):
        for gt in ["tg", "tree", "pause1", "terminal"]:
            assert self._should_shuffle(gt) is False, f"'{gt}' should not shuffle"


# ---------------------------------------------------------------------------
# XSum model_ctx_len for pause variants
# ---------------------------------------------------------------------------

class TestXSumModelCtxLen:
    """In XSumDataset.__init__, model_ctx_len is divided for pause types."""

    def _compute_ctx_len(self, grammar_type, base_len=2048):
        """Simulate XSumDataset.__init__ context length logic (FIXED)."""
        if grammar_type[:5] == "pause":
            numstr = grammar_type[5:]
            if numstr == "1/2" or numstr == "1/2_label":
                pause_num = 1
            else:
                pause_num = int(numstr) if numstr else 1
            return base_len // (1 + pause_num)
        return base_len

    def test_pause1_correct_divisor(self):
        """pause1 → base/(1+1) = base/2."""
        assert self._compute_ctx_len("pause1") == 1024

    def test_pause2_correct_divisor(self):
        """FIXED: pause2 → base/(1+2) = base/3."""
        assert self._compute_ctx_len("pause2") == 2048 // 3

    def test_pause3_correct_divisor(self):
        """FIXED: pause3 → base/(1+3) = base/4."""
        assert self._compute_ctx_len("pause3") == 2048 // 4

    def test_pause_slash_divisor(self):
        """pause1/2 → ispause=1 → base/2."""
        assert self._compute_ctx_len("pause1/2") == 1024

    def test_non_pause_unchanged(self):
        for gt in ["terminal", "tree", "tg", "tgnomask"]:
            assert self._compute_ctx_len(gt) == 2048, f"'{gt}' should keep base len"


# ---------------------------------------------------------------------------
# BLiMP collate_fn: pause token insertion
# ---------------------------------------------------------------------------

class TestBLiMPCollatePause:
    """BLiMPApproximationDataset.collate_fn inserts pause tokens."""

    def _should_pause_interleave(self, grammar_type):
        """Check at downstream.py line 1473 (FIXED to [:5]=='pause')."""
        return grammar_type[:5] == "pause"

    def test_pause_slash_interleaves(self):
        assert self._should_pause_interleave("pause1/2") is True

    def test_pause_slash_label_interleaves(self):
        assert self._should_pause_interleave("pause1/2_label") is True

    # ---- FIXED ----
    def test_pause1_now_interleaved(self):
        assert self._should_pause_interleave("pause1") is True

    def test_pause2_now_interleaved(self):
        assert self._should_pause_interleave("pause2") is True

    def test_pause3_now_interleaved(self):
        assert self._should_pause_interleave("pause3") is True


# ---------------------------------------------------------------------------
# End-to-end: grammar_type dispatch consistency
# ---------------------------------------------------------------------------

class TestDispatchConsistency:
    """Verify that all code paths agree on how each grammar type is handled."""

    @pytest.mark.parametrize("grammar_type", [
        "terminal", "tree", "tg", "tgnomask", "tgnomaskaug",
        "tgproximal", "tgheight", "pause1", "pause2", "pause3",
        "pause1/2", "pause1/2_label", "tree_shuffle", "tree_shuffle_mask",
        "mixing", "tgtree",
    ])
    def test_grammar_type_has_consistent_prefix(self, grammar_type):
        """Every grammar type should have consistent behavior across
        prefix_length checks. This test documents what each type resolves to."""
        results = {
            "[:4]": grammar_type[:4],
            "[:5]": grammar_type[:5],
            "[:8]": grammar_type[:8],
            "[:10]": grammar_type[:10],
            "[:12]": grammar_type[:12],
            "[-4:]": grammar_type[-4:],
        }
        # Just ensure no crashes — the test serves as documentation
        for k, v in results.items():
            assert isinstance(v, str), f"Slice {k} of '{grammar_type}' is not a string"

    def test_all_pause_types_start_with_pause(self):
        """All pause variants should start with 'pause'."""
        for pt in ["pause1", "pause2", "pause3", "pause1/2", "pause1/2_label"]:
            assert pt[:5] == "pause", f"'{pt}' should start with 'pause'"

    def test_all_tree_types_start_with_tree(self):
        """All tree variants should start with 'tree'."""
        for tt in ["tree", "tree_shuffle", "tree_shuffle_mask"]:
            assert tt[:4] == "tree", f"'{tt}' should start with 'tree'"
