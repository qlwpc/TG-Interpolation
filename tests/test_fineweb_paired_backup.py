"""Reject silently mispaired or damaged historical streams before publication."""
import hashlib
import json

import numpy as np
import pytest

from scripts.campaigns.fineweb_paired_backup import export_shard, check_tree, eos_ends, materialize_selection, sha256_file


def test_chunked_eos_and_unfinished_document():
    np.testing.assert_array_equal(eos_ends(np.array([1, 7, 2, 3, 7]), 7, 2), [2, 5])
    with pytest.raises(ValueError, match='EOS'):
        eos_ends(np.array([1, 7, 2]), 7, 2)


def test_tree_projection_and_label_validation():
    opening, closing = {3: 'S', 5: 'NP'}, {4: 'S', 6: 'NP'}
    assert check_tree(np.array([3, 5, 1, 6, 4, 7]), np.array([1, 7]), opening, closing, 7, 8) == 1
    with pytest.raises(ValueError, match='projection'):
        check_tree(np.array([3, 2, 4, 7]), np.array([1, 7]), opening, closing, 7, 8)
    with pytest.raises(ValueError, match='labels'):
        check_tree(np.array([3, 1, 6, 7]), np.array([1, 7]), opening, closing, 7, 8)


def make_pair(tmp_path, tree_tokens):
    root = tmp_path / 'source'
    for fmt in ('terminal', 'tree'):
        (root / fmt).mkdir(parents=True)
    # Select the second document; independent tree offsets are longer.
    terminal = np.array([1, 7, 2, 7], dtype='<u4')
    terminal.tofile(root / 'terminal/a.npy')
    np.array(tree_tokens, dtype='<u4').tofile(root / 'tree/a.npy')
    tok = tmp_path / 'tokenizer.json'
    tok.write_text(json.dumps({'model': {'vocab': {'a': 0, 'b': 1, 'c': 2}},
                              'added_tokens': [{'content': t, 'id': i} for i, t in enumerate(
                                  ['<(S>', '<S)>', '<(NP>', '<NP)>', '<|endoftext|>'], 3)]}))
    output = tmp_path / 'backup'
    output.mkdir()
    record = {'id': '000:2', 'metadata': {'file': str(root / 'terminal/a.npy'),
               'start_token': 2, 'end_token': 4, 'formula_decision': 'keep',
               'token_bytes_sha256': hashlib.sha256(terminal[2:].tobytes()).hexdigest()}}
    return (0, [record], str(root), str(output), str(tok)), output


def test_export_independent_offsets_and_readback(tmp_path):
    task, output = make_pair(tmp_path, [3, 1, 4, 7, 3, 5, 2, 6, 4, 7])
    receipt = export_shard(task)
    assert receipt['source_documents'] == 2
    row = json.loads((output / 'manifests/000.jsonl').read_text())
    assert row['source']['terminal']['start_token'] == 2
    assert row['source']['tree']['start_token'] == 4
    np.testing.assert_array_equal(np.load(output / 'terminal/000.npy'), [2, 7])
    np.testing.assert_array_equal(np.load(output / 'tree/000.npy'), [3, 5, 2, 6, 4, 7])


def test_equal_counts_are_insufficient_for_pairing(tmp_path):
    task, output = make_pair(tmp_path, [3, 1, 4, 7, 3, 1, 4, 7])
    with pytest.raises(ValueError, match='projection'):
        export_shard(task)
    assert not (output / 'manifests/000.jsonl').exists()


def test_eos_count_mismatch_is_rejected(tmp_path):
    task, _ = make_pair(tmp_path, [3, 1, 4, 7])
    with pytest.raises(ValueError, match='EOS count mismatch'):
        export_shard(task)


def test_same_selection_applied_to_both_streams(tmp_path):
    task, backup = make_pair(tmp_path, [3, 1, 4, 7, 3, 5, 2, 6, 4, 7])
    idx, records, root, output, tokenizer = task
    first = json.loads(json.dumps(records[0]))
    first.update(id='000:0')
    first['metadata'].update(start_token=0, end_token=2,
                             token_bytes_sha256=hashlib.sha256(np.array([1, 7], dtype='<u4').tobytes()).hexdigest())
    result = export_shard((idx, records + [first], root, output, tokenizer))
    (backup / 'receipt.json').write_text(json.dumps({'status': 'complete', 'all_exported_projections_equal': True,
               'tokenizer_sha256': sha256_file(tokenizer), 'shards': [result], 'storage_format': 'NPY uint32 EOS-only'}))
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'sample_ids': ['000:2']}))
    subset = tmp_path / 'subset'
    receipt = materialize_selection(backup, selection, subset, tokenizer)
    assert receipt['exported_documents'] == 1
    np.testing.assert_array_equal(np.load(subset / 'terminal/000.npy'), [2, 7])
    np.testing.assert_array_equal(np.load(subset / 'tree/000.npy'), [3, 5, 2, 6, 4, 7])
    selection.write_text(json.dumps({'sample_ids': ['000:999']}))
    with pytest.raises(ValueError, match='absent'):
        materialize_selection(backup, selection, tmp_path / 'bad_subset', tokenizer)
