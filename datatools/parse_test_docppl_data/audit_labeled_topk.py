"""Independent readback of tree300 coverage, brackets, terminals and valid rows."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import numpy as np
from datatools.parse_test_docppl_data.generate_native_topk import (
    CanonicalPPLCorpus, _json_dump, _sha256_path,
)


def audit_block(tokens, lengths, scores, counts, corpus, first):
    vocab = corpus.tokenizer.get_vocab()
    open_to_close = {int(i): int(vocab['<' + s[2:-1] + ')>'])
                     for s, i in vocab.items() if s.startswith('<(') and s.endswith('>')}
    lookup = np.zeros(65536, dtype=np.int32)
    opening_ids = np.asarray(list(open_to_close), dtype=np.uint16)
    closing_ids = np.asarray(list(open_to_close.values()), dtype=np.uint16)
    lookup[opening_ids] = 1
    lookup[closing_ids] = -1
    changes = lookup[tokens]
    depth = np.cumsum(changes, dtype=np.int32)
    if np.any(depth < 0):
        raise ValueError('closing bracket before opening bracket')
    offsets = np.r_[0, np.cumsum(lengths.reshape(-1), dtype=np.int64)]
    if offsets[-1] != len(tokens) or np.any(lengths <= 0) or np.any(depth[offsets[1:] - 1] != 0):
        raise ValueError('record lengths or balanced brackets failed')
    roots = (changes == 1) & (depth == 1)
    if np.any(np.add.reduceat(roots, offsets[:-1]) != 1):
        raise ValueError('a candidate does not have exactly one root')
    opening = changes == 1
    closing = changes == -1
    od, cd = depth[opening], depth[closing] + 1
    pair_lookup = np.zeros(65536, dtype=np.uint16)
    pair_lookup[opening_ids] = closing_ids
    expected, actual = pair_lookup[tokens[opening]], tokens[closing]
    for level in np.unique(od):
        if not np.array_equal(expected[od == level], actual[cd == level]):
            raise ValueError('opening/closing label mismatch')
    hist = Counter()
    for local, length_row in enumerate(lengths):
        start = local * 300
        left, right = offsets[start], offsets[start + 300]
        row = corpus.sentence_input(first + local)
        projected = tokens[left:right][changes[left:right] == 0]
        if not np.array_equal(projected.reshape(300, -1),
                              np.broadcast_to(row.terminal_tokens, (300, len(row.terminal_tokens)))):
            raise ValueError(f'terminal mismatch at sentence {first + local}')
        valid = int(counts[local])
        if not 1 <= valid <= 300 or not np.isfinite(scores[local, :valid]).all():
            raise ValueError('invalid labeled valid count or score')
        if np.any(np.diff(scores[local, :valid]) > 1e-4) or not np.isneginf(scores[local, valid:]).all():
            raise ValueError('unordered scores or finite padding mass')
        rows = [tokens[offsets[start+k]:offsets[start+k+1]].tobytes() for k in range(300)]
        if len(set(rows[:valid])) != valid or any(x != rows[valid-1] for x in rows[valid:]):
            raise ValueError('duplicate valid candidates or incorrect physical padding')
        hist[valid] += 1
    return hist


def run(args):
    corpus = CanonicalPPLCorpus(args.canonical, args.tokenizer)
    hist, cursor = Counter(), 0
    started = time.time()
    paths = sorted(args.input.glob('shard-*-of-*/block-*.npz'))
    for path in paths:
        with np.load(path) as b:
            if int(b['first']) != cursor:
                raise ValueError('blocks have gaps or overlap')
            hist.update(audit_block(b['tokens'], b['lengths'], b['scores'], b['valid_counts'], corpus, cursor))
            cursor = int(b['last'])
        if cursor % 4096 == 0:
            print(f'audited={cursor} elapsed={time.time()-started:.1f}s', flush=True)
    if not args.smoke:
        if cursor != len(corpus.lengths):
            raise ValueError('labeled output is not the complete split')
        manifest = json.loads((args.input / 'manifest.json').read_text())
        if manifest['sentence_count'] != cursor or manifest['document_count'] != len(corpus.document_counts):
            raise ValueError('labeled manifest count mismatch')
        for name, digest in manifest['files'].items():
            if _sha256_path(args.input / name) != digest:
                raise ValueError('merged labeled file hash differs')
        if not np.array_equal(np.load(args.input / 'tree_doc_index.npy'), corpus.document_counts):
            raise ValueError('document index differs from frozen split')
        merged = np.load(args.input / 'tree_300.npy', mmap_mode='r')
        merged_lengths = np.load(args.input / 'tree_sent_index.npy', mmap_mode='r').reshape(-1, 300)
        merged_scores = np.load(args.input / 'proposal_scores.npy', mmap_mode='r')
        merged_counts = np.load(args.input / 'valid_counts.npy', mmap_mode='r')
        token_offset = 0
        for path in paths:
            with np.load(path) as b:
                first, last = int(b['first']), int(b['last'])
                tokens = b['tokens']
                for x, y in [(merged[token_offset:token_offset+len(tokens)], tokens),
                             (merged_lengths[first:last], b['lengths']),
                             (merged_scores[first:last], b['scores']),
                             (merged_counts[first:last], b['valid_counts'])]:
                    if not np.array_equal(x, y):
                        raise ValueError('merged file differs from independently audited block')
                token_offset += len(tokens)
        if token_offset != len(merged):
            raise ValueError('extra merged tokens')
    result = dict(complete=True, scope='smoke' if args.smoke else 'complete_test',
                  sentences=cursor, physical_slots=300, valid_count_histogram=dict(sorted(hist.items())),
                  all_candidate_terminals_match=True, all_brackets_balanced_and_labels_match=True,
                  valid_candidates_unique=True, padding_has_no_proposal_mass=True,
                  auditor_sha256=_sha256_path(Path(__file__)), elapsed_seconds=time.time()-started)
    _json_dump(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--canonical', type=Path, default=REPO / 'dataset/bbc-news-reserved-clean-v1/canonical/test')
    p.add_argument('--tokenizer', type=Path, default=REPO / 'dataset/bbc-news/TG_GPT2_tokenizer.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
