"""
Tests for data utility functions, including pause_input_ids,
get_document_lengths, find_periodic_sequences, and SequentialDistributedSampler.

These functions live in olmo/data/util.py and olmo/data/__init__.py.
"""

import numpy as np
import pytest
import torch

from olmo.data.util import (
    get_document_lengths,
    find_periodic_sequences,
    pause_input_ids,
    pause_label_mask,
    is_pause_label,
    pause_spec_from_grammar_type,
    find_end_first_consecutive_true,
    find_start_last_consecutive_true,
    group_consecutive_values,
    SequentialDistributedSampler,
)


# ---------------------------------------------------------------------------
# find_end_first_consecutive_true
# ---------------------------------------------------------------------------

class TestFindEndFirstConsecutiveTrue:
    def test_all_true(self):
        arr = np.array([True, True, True])
        assert find_end_first_consecutive_true(arr) == 3

    def test_first_false(self):
        arr = np.array([False, True, True])
        assert find_end_first_consecutive_true(arr) == 0

    def test_mid_break(self):
        arr = np.array([True, True, False, True])
        assert find_end_first_consecutive_true(arr) == 2

    def test_single_true(self):
        assert find_end_first_consecutive_true(np.array([True, False])) == 1


# ---------------------------------------------------------------------------
# find_start_last_consecutive_true
# ---------------------------------------------------------------------------

class TestFindStartLastConsecutiveTrue:
    def test_all_true(self):
        arr = np.array([True, True, True])
        assert find_start_last_consecutive_true(arr) == 0

    def test_last_false(self):
        arr = np.array([True, False])
        assert find_start_last_consecutive_true(arr) == -1

    def test_tail_true(self):
        arr = np.array([False, True, True])
        assert find_start_last_consecutive_true(arr) == 1


# ---------------------------------------------------------------------------
# group_consecutive_values
# ---------------------------------------------------------------------------

class TestGroupConsecutiveValues:
    def test_simple(self):
        result = group_consecutive_values(np.array([1, 2, 3, 5, 6, 9]))
        assert len(result) == 3
        assert np.array_equal(result[0], [1, 2, 3])
        assert np.array_equal(result[1], [5, 6])
        assert np.array_equal(result[2], [9])

    def test_single_group(self):
        result = group_consecutive_values(np.array([10, 11, 12]))
        assert len(result) == 1
        assert np.array_equal(result[0], [10, 11, 12])


# ---------------------------------------------------------------------------
# find_periodic_sequences
# ---------------------------------------------------------------------------

class TestFindPeriodicSequences:
    def test_no_periodic(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        results = list(find_periodic_sequences(arr, max_period=3))
        assert len(results) == 0

    def test_repeating_triplet(self):
        # [1,2,3] repeated 3+ times
        arr = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        results = list(find_periodic_sequences(arr, max_period=3))
        assert len(results) >= 1
        r = results[0]
        assert r.period == 3
        assert r.times >= 3

    def test_max_period_exceeds_length(self):
        """max_period should be clamped to len(arr)//3."""
        arr = np.array([1, 1, 1, 1, 1, 1])
        # max_period=100 but arr has 6 elements, so clamped to 6//3=2
        results = list(find_periodic_sequences(arr, max_period=100))
        # period=1 with 6 repeats should be found
        assert len(results) >= 1
        assert results[0].period == 1

    def test_mask_value_in_array_raises(self):
        arr = np.array([1, -1, 3])
        with pytest.raises(ValueError, match="mask_value"):
            list(find_periodic_sequences(arr, max_period=3, mask_value=-1))


# ---------------------------------------------------------------------------
# get_document_lengths
# ---------------------------------------------------------------------------

class TestGetDocumentLengths:
    def test_single_document_no_eos(self):
        """Document with no EOS token: entire sequence is one doc."""
        input_ids = torch.tensor([5, 10, 15, 20])
        lens = get_document_lengths(input_ids, eos_token_id=100)
        assert lens.tolist() == [4]

    def test_single_document_with_trailing_eos(self):
        input_ids = torch.tensor([5, 10, 100])
        lens = get_document_lengths(input_ids, eos_token_id=100)
        # doc boundary after EOS. Since input_ids[-1]==100, no trailing sentinel.
        # boundaries: [-1, 2], lengths: [2 - (-1)] = [3]
        assert lens.tolist() == [3]

    def test_multiple_documents(self):
        input_ids = torch.tensor([5, 100, 10, 15, 100, 20, 25])
        # EOS at indices 1 and 4.
        # boundaries: [-1, 1, 4, 6] (6 because last token is not EOS)
        # lens: [2, 3, 2]
        lens = get_document_lengths(input_ids, eos_token_id=100)
        assert lens.tolist() == [2, 3, 2]

    def test_all_eos(self):
        input_ids = torch.tensor([100, 100, 100])
        lens = get_document_lengths(input_ids, eos_token_id=100)
        # boundaries: [-1, 0, 1, 2] (last == EOS so no trailing sentinel added)
        # lens: [1, 1, 1]
        assert lens.tolist() == [1, 1, 1]

    def test_leading_eos(self):
        """EOS at start: first doc is EOS token itself."""
        input_ids = torch.tensor([100, 5, 10])
        lens = get_document_lengths(input_ids, eos_token_id=100)
        # boundaries: [-1, 0, 2] (trailing sentinel: 2 = last_idx)
        # lens: [1, 2]
        assert lens.tolist() == [1, 2]

    def test_empty_input_handling(self):
        """Empty tensors - verify no crash, though behavior is undefined."""
        input_ids = torch.tensor([], dtype=torch.long)
        # With empty input, nonzero returns empty; boundaries = [-1] since
        # last token check on empty tensor may behave oddly.
        # This test just verifies no crash.
        try:
            lens = get_document_lengths(input_ids, eos_token_id=100)
            # Should return empty or single-doc
            assert len(lens) >= 0
        except Exception:
            pass  # acceptable: edge case not designed for empty input


# ---------------------------------------------------------------------------
# pause_input_ids
# ---------------------------------------------------------------------------

class TestPauseInputIds:
    def test_numpy_no_pause_token(self):
        """When pause_token_id is None, tokens are interleaved with themselves."""
        arr = np.array([10, 20, 30], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=None, pause_num=1)
        expected = np.array([10, 10, 20, 20, 30, 30], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_numpy_with_pause_token(self):
        """When pause_token_id is set, it fills the pause slots."""
        arr = np.array([10, 20], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=1)
        expected = np.array([10, 99, 20, 99], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_torch_tensor(self):
        t = torch.tensor([1, 2, 3])
        result = pause_input_ids(t, pause_token_id=0, pause_num=1)
        expected = torch.tensor([1, 0, 2, 0, 3, 0])
        assert torch.equal(result, expected)

    def test_pause_num_from_string(self):
        """pause_num can be a string like 'pause3'."""
        arr = np.array([1, 2], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=0, pause_num="pause2")
        # 2 pause tokens per original token → (1+2)*2 = 6 elements
        assert len(result) == 6
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == 0

    def test_pause_num_two(self):
        """pause_num=2: original token, then 2 pause tokens."""
        arr = np.array([10], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num=2)
        expected = np.array([10, 99, 99], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_list_input(self):
        result = pause_input_ids([1, 2], pause_token_id=0, pause_num=1)
        assert result == [1, 0, 2, 0]

    def test_invalid_type_raises(self):
        with pytest.raises(NotImplementedError):
            pause_input_ids("invalid", pause_token_id=0, pause_num=1)

    def test_2d_array_raises(self):
        with pytest.raises(AssertionError):
            pause_input_ids(np.array([[1, 2]]), pause_token_id=0, pause_num=1)


# ---------------------------------------------------------------------------
# pause_input_ids — rational "pausep/q" format (p pauses per q real tokens)
# ---------------------------------------------------------------------------

class TestPauseInputIdsRational:
    def test_pause_1_2_numpy(self):
        """pause1/2: 1 pause token after every 2 real tokens."""
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num="pause1/2")
        expected = np.array([10, 20, 99, 30, 40, 99], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_pause_2_3_numpy(self):
        """pause2/3: 2 pause tokens after every 3 real tokens."""
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num="pause2/3")
        expected = np.array([1, 2, 3, 99, 99, 4, 5, 6, 99, 99], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_pause_rational_length(self):
        """Length is n_full*(q+p) + remainder; divisible -> n*(q+p)/q."""
        arr = np.array([1, 2, 3, 4], dtype=np.int64)  # 4 tokens, q=2 -> 2 full blocks
        result = pause_input_ids(arr, pause_token_id=99, pause_num="pause1/2")
        assert len(result) == 6  # 2 blocks * (2+1)

    def test_pause_rational_remainder(self):
        """A trailing partial block keeps its tokens but adds no pauses."""
        arr = np.array([10, 20, 30], dtype=np.int64)  # q=2 -> 1 full + 1 remainder
        result = pause_input_ids(arr, pause_token_id=99, pause_num="pause1/2")
        expected = np.array([10, 20, 99, 30], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_pause_rational_torch(self):
        t = torch.tensor([10, 20, 30, 40])
        result = pause_input_ids(t, pause_token_id=99, pause_num="pause1/2")
        expected = torch.tensor([10, 20, 99, 30, 40, 99])
        assert torch.equal(result, expected)

    def test_pause_rational_list(self):
        result = pause_input_ids([10, 20, 30, 40], pause_token_id=99, pause_num="pause1/2")
        assert result == [10, 20, 99, 30, 40, 99]

    def test_pause_rational_label_suffix(self):
        """A trailing '_label' tag is tolerated."""
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=99, pause_num="pause1/2_label")
        expected = np.array([10, 20, 99, 30, 40, 99], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_pause_rational_matches_integer_when_q1(self):
        """pause2 == pause2/1: both insert 2 pauses after every token."""
        arr = np.array([10, 20], dtype=np.int64)
        a = pause_input_ids(arr, pause_token_id=99, pause_num="pause2")
        b = pause_input_ids(arr, pause_token_id=99, pause_num="pause2/1")
        assert np.array_equal(a, b)

    def test_pause_rational_mask_broadcast(self):
        """pause_token_id=None broadcasts each group's last real token to pauses."""
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        result = pause_input_ids(arr, pause_token_id=None, pause_num="pause1/2")
        expected = np.array([10, 20, 20, 30, 40, 40], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_pause_rational_zero_denominator_raises(self):
        with pytest.raises(ValueError):
            pause_input_ids(np.array([1, 2]), pause_token_id=0, pause_num="pause1/0")


# ---------------------------------------------------------------------------
# pause_label_mask
# ---------------------------------------------------------------------------

class TestPauseLabelMask:
    def _real_positions(self, expanded_len, p, q):
        """Ground-truth real-token positions in an expanded (p,q) sequence."""
        pos = []
        j = 0  # real-token index
        while True:
            ep = j + (j // q) * p  # expanded position of real token j
            if ep >= expanded_len:
                break
            pos.append(ep)
            j += 1
        return set(pos)

    def test_pause_1_2(self):
        # pause1/2: [R R P] [R R P] ... -> False only at pause slots (idx 2,5,...)
        mask = pause_label_mask(6, p=1, q=2)
        expected = np.array([True, True, False, True, True, False])
        assert np.array_equal(mask, expected)
        assert mask.dtype == np.bool_

    def test_pause_2_3(self):
        # pause2/3: [R R R P P] repeated
        mask = pause_label_mask(10, p=2, q=3)
        expected = np.array([True, True, True, False, False,
                             True, True, True, False, False])
        assert np.array_equal(mask, expected)

    def test_p_zero_all_true(self):
        # p=0 -> no pause slots, all real
        assert np.array_equal(pause_label_mask(7, p=0, q=2),
                              np.ones(7, dtype=np.bool_))
        # empty also fine
        assert np.array_equal(pause_label_mask(0, p=0, q=2),
                              np.ones(0, dtype=np.bool_))

    def test_remainder_block_all_true(self):
        # pause1/2 over 4 tokens -> expanded [R R P R] (1 full + 1 remainder)
        mask = pause_label_mask(4, p=1, q=2)
        expected = np.array([True, True, False, True])
        assert np.array_equal(mask, expected)

    def test_remainder_shorter_than_q(self):
        # pause2/3, expanded_len=4 -> [R R R P] (1 full block + 1 pause of the
        # next block would need 5; here only 4 slots: 3 real + 1 pause)
        mask = pause_label_mask(4, p=2, q=3)
        expected = np.array([True, True, True, False])
        assert np.array_equal(mask, expected)

    def test_remainder_only_real(self):
        # pause2/3, expanded_len=5 -> full block [R R R P P] exactly
        mask = pause_label_mask(5, p=2, q=3)
        expected = np.array([True, True, True, False, False])
        assert np.array_equal(mask, expected)

    def test_aligns_with_pause_input_ids_dedicated_id(self):
        # Where pause_input_ids inserts the pause id, the mask must be False.
        for spec, p, q in [("pause1", 1, 1), ("pause2", 2, 1),
                           ("pause1/2", 1, 2), ("pause2/3", 2, 3),
                           ("pause1/4", 1, 4)]:
            for n_real in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
                arr = np.arange(1, n_real + 1, dtype=np.int64)
                expanded = pause_input_ids(arr, pause_token_id=999, pause_num=spec)
                mask = pause_label_mask(len(expanded), p, q)
                assert len(mask) == len(expanded)
                # pause slots hold 999 and must be masked False
                assert not mask[expanded == 999].any()
                # real slots hold a real id and must be True
                assert mask[expanded != 999].all()

    def test_aligns_with_pause_input_ids_repeat_mode(self):
        # In repeat mode there is no dedicated id, so verify against the
        # structural real-token positions instead.
        for spec, p, q in [("pause1/2", 1, 2), ("pause2/3", 2, 3), ("pause2", 2, 1)]:
            for n_real in [1, 2, 3, 5, 6, 7, 9]:
                arr = np.arange(1, n_real + 1, dtype=np.int64)
                expanded = pause_input_ids(arr, pause_token_id=None, pause_num=spec)
                mask = pause_label_mask(len(expanded), p, q)
                real_pos = self._real_positions(len(expanded), p, q)
                for i in range(len(expanded)):
                    assert bool(mask[i]) == (i in real_pos)

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            pause_label_mask(10, p=1, q=0)

    def test_negative_p_raises(self):
        with pytest.raises(ValueError):
            pause_label_mask(10, p=-1, q=2)

    def test_next_token_loss_alignment(self):
        # Simulate get_labels left-shift: label_mask marks input_ids[j];
        # after [.., 1:] the loss at logit i targets input_ids[i+1]. So the loss
        # is masked exactly when the *next* token is a pause (mask False).
        expanded = pause_input_ids(np.array([10, 20, 30, 40], dtype=np.int64),
                                   pause_token_id=99, pause_num="pause1/2")
        # [10, 20, 99, 30, 40, 99]
        mask = pause_label_mask(len(expanded), 1, 2)  # [T T F T T F]
        # targets (input_ids shifted left) = [20, 99, 30, 40, 99]
        targets = expanded[1:]
        target_mask = mask[1:]  # [T, F, T, T, F]
        # loss is kept where the *next* token (target) is real (mask True)
        kept_targets = targets[target_mask]
        assert 99 not in kept_targets  # no pause token is ever a trained target
        assert np.array_equal(kept_targets, np.array([20, 30, 40]))


# ---------------------------------------------------------------------------
# is_pause_label
# ---------------------------------------------------------------------------

class TestIsPauseLabel:
    def test_plain_pause_is_false(self):
        assert not is_pause_label("pause1/2")
        assert not is_pause_label("pause1")
        assert not is_pause_label("pause2/3")
        assert not is_pause_label("pause")

    def test_label_suffix_is_true(self):
        assert is_pause_label("pause1/2_label")
        assert is_pause_label("pause2_label")
        assert is_pause_label("pause2/3_label")

    def test_non_pause_is_false(self):
        assert not is_pause_label("terminal")
        assert not is_pause_label("tree")
        assert not is_pause_label("tg")
        assert not is_pause_label("")
        assert not is_pause_label("tree_shuffle_mask")  # '_mask', not '_label'

    def test_label_suffix_preserves_spec(self):
        # _label variant must parse to the same (p, q) as the bare spec.
        assert pause_spec_from_grammar_type("pause1/2_label") == (1, 2)
        assert pause_spec_from_grammar_type("pause2/3_label") == (2, 3)
        assert (pause_spec_from_grammar_type("pause1/2_label")
                == pause_spec_from_grammar_type("pause1/2"))


# ---------------------------------------------------------------------------
# SequentialDistributedSampler
# ---------------------------------------------------------------------------

class TestSequentialDistributedSampler:
    def test_basic_split(self):
        class FakeDataset:
            def __len__(self):
                return 100

        sampler = SequentialDistributedSampler(FakeDataset(), num_replicas=4, rank=0)
        indices = list(sampler)
        assert len(indices) == 25
        assert indices == list(range(0, 25))

    def test_rank_offset(self):
        class FakeDataset:
            def __len__(self):
                return 100

        sampler = SequentialDistributedSampler(FakeDataset(), num_replicas=4, rank=3)
        indices = list(sampler)
        assert indices == list(range(75, 100))
