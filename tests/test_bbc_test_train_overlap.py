import hashlib

import numpy as np
import pytest

from diagnostics.audit_bbc_test_train_overlap import make_targets, scan


@pytest.mark.parametrize('chunk_tokens', [1, 2, 5, 100])
def test_overlap_counts_duplicates_and_chunk_boundaries(tmp_path, chunk_tokens):
    a = [50257, 11, 12, 50256]
    b = [50257, 13, 14, 15, 50256]
    different_whitespace = [50257, 11, 220, 12, 50256]
    absent = [50257, 16, 50256]
    test_path, train_path = tmp_path / 'test.npy', tmp_path / 'train.npy'
    np.save(test_path, np.asarray(a + a + b + absent, dtype='<u2'))
    np.save(train_path, np.asarray(different_whitespace + a + b + a, dtype='<u2'))
    result = scan(make_targets(test_path), train_path, chunk_tokens=chunk_tokens)
    assert result['scan_complete']
    assert result['train_documents'] == 4
    assert result['test_documents'] == 4
    assert result['unique_test_hashes'] == 3
    assert result['matched_test_documents'] == 3
    assert result['matched_train_documents'] == 3
    assert result['test_train_matching_pairs'] == 5
    assert result['unmatched_test_doc_ids'] == [3]
    assert result['train_copies_per_test_histogram'] == {0: 1, 1: 1, 2: 2}
    assert result['groups'][0]['train_matches_doc_and_token_offset'] == [[1, 5], [3, 14]]
    assert result['train_file_sha256'] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert result['train_documents_without_final_eos'] == 0


def test_manifest_tampering_rejected(tmp_path):
    path = tmp_path / 'test.npy'
    np.save(path, np.asarray([50257, 11, 50256], dtype='<u2'))
    target = make_targets(path)
    target['documents'][0]['ids'][1] = 12
    with pytest.raises(AssertionError):
        scan(target, path)
