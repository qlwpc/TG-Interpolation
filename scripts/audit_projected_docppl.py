"""Exhaustive frozen clean-tree audit and post-projection candidate identities.

Run on CPU with bounded workers. Source arrays are read-only. Projection is
independently expressed with NumPy and checked against the runtime C++ extension
over every physical candidate, then deduplicated within each logical sentence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from datatools.parse_test_docppl_data.audit_labeled_topk import audit_block
from datatools.parse_test_docppl_data.generate_native_topk import CanonicalPPLCorpus
from scripts.projected_candidate_utils import stable_unique_indices
from olmo.data.tg_mask import SentencepieceVocab


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def audit_chunk(bounds):
    first, last = bounds
    lengths = LENGTHS[first:last]
    tokens = TOKENS[int(OFFSETS[first*300]):int(OFFSETS[last*300])]
    scores, counts = SCORES[first:last], COUNTS[first:last]
    hist = audit_block(tokens, lengths, scores, counts, CORPUS, first)
    local_offsets = np.r_[0, np.cumsum(lengths.reshape(-1), dtype=np.int64)]
    opening, closing = OPENING[tokens], CLOSING[tokens]
    answers = {}
    for grammar in ('tree_noont', 'tree_compress'):
        if grammar == 'tree_noont':
            keep = ~opening
            projected = tokens[keep]
            actual = RUNTIME_VOCAB.convert_treenpy_to_noont(tokens)
        else:
            keep = ~closing | ~np.r_[False, closing[:-1]]
            projected = tokens[keep].copy()
            projected[CLOSING[projected]] = MERGED_CLOSE
            actual = RUNTIME_VOCAB.convert_treenpy_to_compress(tokens)
        if not np.array_equal(projected, actual):
            raise ValueError(f'{grammar}: independent projection differs from runtime')
        del actual
        plen = np.add.reduceat(keep.astype(np.int64), local_offsets[:-1])
        poff = np.r_[0, np.cumsum(plen)]
        unique = np.zeros(last-first, dtype=np.uint16)
        firsts = np.full((last-first, 300), -1, dtype=np.int16)
        inverse = np.full_like(firsts, -1)
        for local, count in enumerate(counts):
            base = local*300
            selected, mapping = stable_unique_indices(
                projected[poff[base+k]:poff[base+k+1]] for k in range(int(count)))
            unique[local] = len(selected)
            firsts[local, :len(selected)] = selected
            inverse[local, :int(count)] = mapping
        answers[grammar] = (unique, firsts, inverse, int(plen.max()))
    return first, last, dict(hist), answers


def main():
    global CORPUS, TOKENS, LENGTHS, SCORES, COUNTS, OFFSETS, OPENING, CLOSING, MERGED_CLOSE, RUNTIME_VOCAB
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'dataset/bbc-news-reserved-clean-v1')
    p.add_argument('--tokenizer', type=Path, default=ROOT/'dataset/bbc-news/TG_GPT2_tokenizer.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('use 1..8 bounded CPU workers')
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'audit.json').exists():
        raise ValueError('completed audit exists; use a new output directory')
    data = args.data/'testppl/tree300'
    manifest = json.loads((data/'manifest.json').read_text())
    identities = {}
    for name, expected in manifest['files'].items():
        path = data/name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'frozen tree300 SHA256 mismatch: {name}')
        identities[str(path.relative_to(args.data))] = actual
    CORPUS = CanonicalPPLCorpus(args.data/'canonical/test', args.tokenizer)
    RUNTIME_VOCAB = SentencepieceVocab.from_vocab_file(str(args.tokenizer))
    for path in CORPUS.source_paths:
        identities[str(path.relative_to(args.data.resolve()))] = digest(path)
    identities['testppl/tree300/manifest.json'] = digest(data/'manifest.json')
    TOKENS = np.load(data/'tree_300.npy', mmap_mode='r')
    LENGTHS = np.load(data/'tree_sent_index.npy', mmap_mode='r').reshape(-1, 300)
    SCORES = np.load(data/'proposal_scores.npy', mmap_mode='r')
    COUNTS = np.load(data/'valid_counts.npy', mmap_mode='r')
    OFFSETS = np.r_[0, np.cumsum(LENGTHS.reshape(-1), dtype=np.int64)]
    if OFFSETS[-1] != TOKENS.size or len(COUNTS) != len(CORPUS.lengths):
        raise ValueError('physical coverage mismatch')
    doc_counts = np.load(data/'tree_doc_index.npy')
    if not np.array_equal(doc_counts, CORPUS.document_counts):
        raise ValueError('document boundaries differ from canonical corpus')
    vocab = CORPUS.tokenizer.get_vocab()
    OPENING = np.zeros(65536, dtype=bool)
    CLOSING = np.zeros(65536, dtype=bool)
    OPENING[[i for s, i in vocab.items() if s.startswith('<(') and s.endswith('>')]] = True
    CLOSING[[i for s, i in vocab.items() if s.startswith('<') and s.endswith(')>')]] = True
    MERGED_CLOSE = vocab['<X)>']
    # Independent denominator and document-boundary certification.
    terminal = np.load(args.data/'terminal/test.npy', mmap_mode='r')
    terminal_offsets = np.load(args.data/'terminal/test_doc_offsets.npy')
    stripped = CORPUS.tree[~(OPENING[CORPUS.tree] | CLOSING[CORPUS.tree])]
    if not np.array_equal(terminal, stripped):
        raise ValueError('canonical terminal stream differs from terminal/test.npy')
    canonical_offsets = CORPUS.sentence_offsets.astype(np.int64)
    sentence_tokens = np.add.reduceat(
        (~(OPENING[CORPUS.tree] | CLOSING[CORPUS.tree]) & (CORPUS.tree != CORPUS.vocab.bos)).astype(np.int64),
        canonical_offsets[:-1])
    doc_starts = np.r_[0, np.cumsum(doc_counts, dtype=np.int64)]
    doc_tokens = np.add.reduceat(sentence_tokens, doc_starts[:-1])
    if not np.array_equal(np.diff(terminal_offsets)-1, doc_tokens):
        raise ValueError('per-document denominator differs from terminal offsets')
    for left, right in zip(terminal_offsets[:-1], terminal_offsets[1:]):
        record = terminal[int(left):int(right)]
        if (record[0] != CORPUS.vocab.bos or record[-1] != CORPUS.vocab.eos
                or np.count_nonzero(record == CORPUS.vocab.bos) != 1
                or np.count_nonzero(record == CORPUS.vocab.eos) != 1):
            raise ValueError('invalid BOS/EOS document boundaries')
    if (len(doc_counts), len(COUNTS), int(doc_tokens.sum())) != (5000, 129085, 3068713):
        raise ValueError('unexpected clean-test evaluation coverage')
    for name in ('test.npy', 'test_doc_offsets.npy'):
        identities['terminal/'+name] = digest(args.data/'terminal'/name)
    arrays = {}
    for grammar in ('tree_noont', 'tree_compress'):
        folder = args.output/grammar
        folder.mkdir(exist_ok=True)
        arrays[grammar] = {
            'unique_counts': np.lib.format.open_memmap(folder/'unique_counts.npy', mode='w+', dtype=np.uint16, shape=(len(COUNTS),)),
            'first_indices': np.lib.format.open_memmap(folder/'first_indices.npy', mode='w+', dtype=np.int16, shape=(len(COUNTS), 300)),
            'inverse': np.lib.format.open_memmap(folder/'inverse.npy', mode='w+', dtype=np.int16, shape=(len(COUNTS), 300)),
        }
    histogram, maxima, seen = Counter(), Counter(), 0
    chunks = [(i, min(i+128, len(COUNTS))) for i in range(0, len(COUNTS), 128)]
    with mp.get_context('fork').Pool(args.workers) as pool:
        for first, last, hist, answers in pool.imap(audit_chunk, chunks):
            if first != seen:
                raise ValueError('audit coverage gap')
            seen = last
            histogram.update(hist)
            for grammar, (counts, firsts, inverse, maxlen) in answers.items():
                for name, array in zip(('unique_counts', 'first_indices', 'inverse'), (counts, firsts, inverse)):
                    arrays[grammar][name][first:last] = array
                maxima[grammar] = max(maxima[grammar], maxlen)
            if first % 4096 == 0:
                print(json.dumps(dict(sentences=seen, total=len(COUNTS), elapsed_seconds=time.time()-started)), flush=True)
    projections, artifacts = {}, {}
    for grammar, files in arrays.items():
        for name, array in files.items():
            array.flush()
            artifacts[f'{grammar}/{name}.npy'] = digest(args.output/grammar/(name+'.npy'))
        unique = np.asarray(files['unique_counts'])
        removed = COUNTS.astype(np.int64)-unique
        projections[grammar] = dict(sentences_with_collisions=int(np.count_nonzero(removed)),
            removed_candidates=int(removed.sum()), original_candidates=int(COUNTS.sum()),
            unique_candidates=int(unique.sum()), maximum_length=maxima[grammar],
            unique_count_histogram={str(k): v for k, v in sorted(Counter(unique.tolist()).items())})
    for name, array in (('sentence_terminal_counts', sentence_tokens), ('document_terminal_counts', doc_tokens),
                        ('document_sentence_counts', doc_counts), ('original_counts', COUNTS)):
        np.save(args.output/(name+'.npy'), array)
        artifacts[name+'.npy'] = digest(args.output/(name+'.npy'))
    source_paths = [Path(__file__), ROOT/'scripts/projected_candidate_utils.py',
                    ROOT/'datatools/parse_test_docppl_data/audit_labeled_topk.py',
                    ROOT/'datatools/parse_test_docppl_data/generate_native_topk.py',
                    ROOT/'olmo/data/tgmasking/mask.cpp', ROOT/'olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so']
    report = dict(status='complete', scope='full_clean_test', documents=len(doc_counts), sentences=seen,
        terminal_count=int(doc_tokens.sum()), physical_slots=300, original_count_histogram=dict(histogram),
        bracket_labels_and_balance=True, all_candidate_terminals_match=True, physical_padding_excluded=True,
        independent_projection_matches_runtime=True, stable_first_occurrence=True,
        data_files=identities, tokenizer_sha256=digest(args.tokenizer), artifacts=artifacts,
        source_files={str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        projections=projections, workers=args.workers, elapsed_seconds=time.time()-started)
    tmp = args.output/'audit.json.tmp'
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(args.output/'audit.json')
    print(json.dumps({k: v for k, v in report.items() if k not in ('data_files', 'artifacts', 'source_files', 'projections')}, indent=2))


if __name__ == '__main__':
    main()
