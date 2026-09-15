"""Read-only, exhaustive frozen-input and merged-candidate readback audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from datatools.parse_test_docppl_data.audit_labeled_topk import audit_block
from datatools.parse_test_docppl_data.generate_native_topk import CanonicalPPLCorpus, _sha256_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT / 'dataset/bbc-news-reserved-clean-v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    start = time.time()
    base = json.loads((root / 'manifest.json').read_text())
    checked = {}
    for name, expected in base['files'].items():
        actual = _sha256_path(root / name)
        if actual != expected:
            raise ValueError(f'frozen file hash mismatch: {name}')
        checked[name] = actual
    complete = json.loads((root / 'testppl/manifest.json').read_text())
    assert complete['status'] == 'complete'
    assert _sha256_path(root / 'manifest.json') == complete['base_manifest_sha256']
    tok = ROOT / 'dataset/bbc-news/TG_GPT2_tokenizer.json'
    assert _sha256_path(tok) == base['tokenizer_sha256']
    corpus = CanonicalPPLCorpus(root / 'canonical/test', tok)
    assert len(corpus.lengths) == 129085 and len(corpus.document_counts) == 5000
    tree = root / 'testppl/tree300'
    tg = root / 'testppl/tg300'
    for name, expected in json.loads((tree / 'manifest.json').read_text())['files'].items():
        assert _sha256_path(tree / name) == expected, name
    lengths = np.load(tree / 'tree_sent_index.npy', mmap_mode='r').reshape(-1, 300)
    tokens = np.load(tree / 'tree_300.npy', mmap_mode='r')
    scores = np.load(tree / 'proposal_scores.npy', mmap_mode='r')
    counts = np.load(tree / 'valid_counts.npy', mmap_mode='r')
    tg_lengths = np.load(tg / 'tg_sent_index.npy', mmap_mode='r').reshape(-1, 300)
    tg_tokens = np.load(tg / 'tg_300.npy', mmap_mode='r')
    assert np.array_equal(counts, np.load(tg / 'valid_counts.npy'))
    assert np.array_equal(scores, np.load(tg / 'proposal_scores.npy'))
    for path in (tree / 'tree_doc_index.npy', tg / 'tg_doc_index.npy'):
        assert np.array_equal(np.load(path), corpus.document_counts), str(path)
    terminal = np.load(root / 'terminal/test.npy', mmap_mode='r')
    vocab = corpus.tokenizer.get_vocab()
    structure = np.zeros(65536, dtype=bool)
    structure[[i for s, i in vocab.items() if s.startswith('<(') or (s.startswith('<') and s.endswith(')>'))]] = True
    canonical_mask = ~structure[corpus.tree]
    assert np.array_equal(corpus.tree[canonical_mask], terminal), 'frozen terminal projection'
    terminal_lengths = np.add.reduceat(canonical_mask.astype(np.int32), corpus.sentence_offsets[:-1].astype(np.int64))
    terminal_offsets = np.r_[0, terminal_lengths.cumsum(dtype=np.int64)]
    expected_doc_offsets = terminal_offsets[np.r_[0, corpus.document_ends]]
    assert np.array_equal(expected_doc_offsets, np.load(root / 'terminal/test_doc_offsets.npy'))
    close = np.zeros(65536, dtype=bool)
    close[[i for s, i in vocab.items() if s.startswith('<') and s.endswith(')>')]] = True
    hist = Counter()
    left = tg_left = 0
    for first in range(0, len(lengths), 128):
        last = min(first + 128, len(lengths))
        right = left + int(lengths[first:last].sum(dtype=np.int64))
        block = tokens[left:right]
        hist.update(audit_block(block, lengths[first:last], scores[first:last], counts[first:last], corpus, first))
        repeats = 1 + close[block].astype(np.int8)
        converted = np.repeat(block, repeats)
        tg_right = tg_left + len(converted)
        assert np.array_equal(converted, tg_tokens[tg_left:tg_right]), f'TG conversion {first}'
        offsets = np.r_[0, lengths[first:last].reshape(-1).cumsum(dtype=np.int64)]
        expected_lengths = lengths[first:last].reshape(-1) + np.add.reduceat(close[block].astype(np.int32), offsets[:-1])
        assert np.array_equal(expected_lengths, tg_lengths[first:last].reshape(-1)), f'TG lengths {first}'
        for sent in range(first, last):
            expected = corpus.sentence_input(sent).terminal_tokens
            actual = terminal[terminal_offsets[sent]:terminal_offsets[sent+1]]
            assert np.array_equal(actual, expected), f'terminal/canonical {sent}'
        left, tg_left = right, tg_right
        if first % 4096 == 0:
            print(f'audited {last}/129085 sentences; elapsed={time.time()-start:.1f}s', flush=True)
    assert left == len(tokens) and tg_left == len(tg_tokens)
    assert terminal_offsets[-1] == len(terminal)
    derived = root / 'evaluation_terminal'
    derived.mkdir(exist_ok=True)
    target = derived / 'test.npy'
    if not target.exists():
        target.symlink_to('../terminal/test.npy')
    np.save(derived / 'test_sent_index.npy', terminal_lengths)
    np.save(derived / 'test_doc_index.npy', corpus.document_counts)
    result = dict(status='complete', claim_type='computed', documents=5000, sentences=129085,
                  terminal_tokens_including_bos=int(len(terminal)), scored_terminal_tokens=int(len(terminal)-5000),
                  frozen_files=checked, base_manifest_sha256=_sha256_path(root / 'manifest.json'),
                  tree_all_candidates_brackets_terminals_unique_padding=True,
                  tg_all_tokens_and_lengths_exact=True, all_document_boundaries_equal=True,
                  valid_counts=dict(hist), elapsed_seconds=time.time()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
