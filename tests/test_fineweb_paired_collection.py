import json

import numpy as np
import pytest

from scripts.campaigns.fineweb_paired_collection import (
    check_pair, split_for_hash, text_hash, write_npz, storage_check,
)


@pytest.fixture
def tokenizer(tmp_path):
    p = tmp_path/'tokenizer.json'
    p.write_text(json.dumps({'model': {'vocab': {'word': 0, 'other': 1}},
                             'added_tokens': [{'id': 2, 'content': '<(S>'},
                                              {'id': 3, 'content': '<S)>'},
                                              {'id': 4, 'content': '<(NP>'},
                                              {'id': 5, 'content': '<NP)>'},
                                              {'id': 6, 'content': '<|endoftext|>'}]}))
    return p


def test_exact_whitespace_duplicates_always_share_split():
    assert text_hash('caf\u00e9  is nice') == text_hash('cafe\u0301\nis nice')
    assert split_for_hash(text_hash('caf\u00e9 is nice')) == split_for_hash(text_hash('cafe\u0301\nis nice'))


def test_real_nested_pair_and_multi_document_forest(tokenizer):
    terminal = np.array([0, 1, 6, 0, 6], dtype='<u4')
    tree = np.array([2, 4, 0, 5, 1, 3, 6, 2, 0, 3, 6], dtype='<u4')
    assert check_pair(tree, terminal, tokenizer) == 2


@pytest.mark.parametrize('tree,terminal', [
    ([2, 4, 0, 3, 5, 6], [0, 6]),  # crossing typed labels
    ([2, 0, 6, 3, 6], [0, 6, 6]),  # EOS inside the tree
    ([2, 0, 3, 6], [1, 6]),        # different lexical projection
    ([2, 0, 3, 6, 1, 6], [0, 6, 1, 6]),  # second document lacks a tree
    ([2, 99, 3, 6], [99, 6]),     # token outside vocabulary
])
def test_invalid_pairs_are_rejected(tokenizer, tree, terminal):
    with pytest.raises(ValueError):
        check_pair(np.array(tree, dtype='<u4'), np.array(terminal, dtype='<u4'), tokenizer)


def test_npz_preserves_full_arrays_and_binary_hashes(tmp_path):
    path = tmp_path/'pair.npz'
    arrays = {'terminal': np.array([0, 1, 6], dtype='<u4'),
              'tree': np.array([2, 0, 1, 3, 6], dtype='<u4'),
              'records': np.array([(np.void(b'\x00'*32),)], dtype=[('sha', 'V32')])}
    write_npz(path, arrays)
    with np.load(path, allow_pickle=False) as saved:
        for key, array in arrays.items():
            assert np.array_equal(saved[key], array)


def test_storage_guard_enforces_output_cap(tmp_path):
    for split in ['train', 'validation_a', 'validation_b']:
        (tmp_path/split).mkdir()
    (tmp_path/'train'/'pair.npz').write_bytes(b'x'*100)
    with pytest.raises(OSError, match='cap'):
        storage_check(tmp_path, {'reserve_free_bytes': 0, 'max_output_bytes': 110}, 11)
