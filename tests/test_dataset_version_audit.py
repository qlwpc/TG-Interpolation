"""Negative controls for version-audit claims; these do not certify real data."""
import json

import numpy as np

from diagnostics.audit_dataset_versions import Audit


def fixture_root(tmp_path):
    root = tmp_path / 'dataset'
    bbc = root / 'bbc-news'
    bbc.mkdir(parents=True)
    vocab = {'word': 1, 'other': 2, '<|beginoftext|>': 10,
             '<|endoftext|>': 11, '<|pad|>': 12, '<(S>': 20, '<S)>': 21}
    (bbc / 'TG_GPT2_tokenizer.json').write_text(json.dumps({
        'model': {'vocab': vocab}, 'added_tokens': []}))
    return root, Audit(root, 2)


def test_stream_rejects_equal_counts_with_bad_boundaries(tmp_path):
    root, audit = fixture_root(tmp_path)
    p = root / 'bbc-news/terminal'
    p.mkdir()
    np.save(p / 'test.npy', np.array([10, 10, 1, 11, 11], dtype=np.uint16))
    result = audit.stream('bbc-news/terminal/test.npy')
    assert result['documents'] == 2
    assert not result['framed']


def test_candidates_detect_content_and_tree_errors(tmp_path):
    root, audit = fixture_root(tmp_path)
    p = root / 'bbc-news/testppl_tree'
    p.mkdir()
    values = np.concatenate([np.tile([10, 20, 1, 21], 300),
                             np.tile([20, 1, 21, 11], 300)]).astype(np.uint16)
    np.save(p / 'tree_sent_index.npy', np.full(600, 4, dtype=np.uint16))
    np.save(p / 'tree_doc_index.npy', np.array([2], dtype=np.uint16))
    np.save(p / 'tree_300.npy', values)
    good = audit.candidates('bbc-news/testppl_tree')
    assert good['terminal_mismatches'] == good['invalid_tree_records'] == good['unnormalized_records'] == 0
    values[6] = 2  # Candidate 1 has a different terminal.
    values[11] = 20  # Candidate 2 has an unclosed tree.
    np.save(p / 'tree_300.npy', values)
    bad = audit.candidates('bbc-news/testppl_tree')
    assert bad['terminal_mismatches'] == 1
    assert bad['invalid_tree_records'] == 1


def test_run_records_truncated_npy_without_writing_dataset(tmp_path):
    root, audit = fixture_root(tmp_path)
    p = root / 'bbc-news/terminal'
    p.mkdir()
    f = p / 'test.npy'
    f.write_bytes(b'\x93NUMPY\x01\x00')
    before = f.read_bytes()
    result = audit.run()
    assert len(result['errors']) == 1
    assert result['changed_during_check'] == []
    assert f.read_bytes() == before
