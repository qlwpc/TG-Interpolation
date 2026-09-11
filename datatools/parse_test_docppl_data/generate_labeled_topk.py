#!/usr/bin/env python3
"""Resumable tree300 generation from frozen canonical sentences, then TG export."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import numpy as np

from datatools.parse_test_docppl_data.generate_native_topk import (
    CanonicalPPLCorpus, _json_dump, _sha256_path,
)
from datatools.parse_test_docppl_data.labeled_topk import (
    LabeledKBest, label_token_pairs, serialize_candidate,
)


def decode(payload):
    row, chart, labels, pairs = payload
    ranked = LabeledKBest(chart, labels).topk(300)
    if not ranked:
        raise ValueError(f'no legal tree for sentence {row.global_sentence_id}')
    prefix = row.terminal_tokens[:row.content_start]
    suffix = row.terminal_tokens[row.content_end:]
    trees = [serialize_candidate(spans, row.word_piece_ids, pairs, prefix, suffix)
             for _score, spans in ranked]
    if len({t.tobytes() for t in trees}) != len(trees):
        raise ValueError('duplicate serialized labeled tree')
    valid = len(trees)
    # Physical shape safety only: the repeated last tree is excluded by count.
    trees += [trees[-1]] * (300 - valid)
    lengths = np.asarray([len(t) for t in trees], dtype=np.uint16)
    if max(map(len, trees)) > np.iinfo(np.uint16).max:
        raise OverflowError('tree record exceeds uint16 length')
    tokens = np.concatenate(trees)
    structure_ids = sorted({i for pair in pairs.values() for chain in pair for i in chain})
    terminals = tokens[~np.isin(tokens, structure_ids)]
    if not np.array_equal(terminals.reshape(300, -1),
                          np.broadcast_to(row.terminal_tokens, (300, len(row.terminal_tokens)))):
        raise ValueError('candidate serialization changed frozen terminals')
    scores = np.full(300, -np.inf, dtype=np.float32)
    scores[:valid] = [s for s, _ in ranked]
    return tokens, lengths, scores, valid


def sources(args, corpus):
    paths = [Path(__file__), Path(__file__).with_name('labeled_topk.py'),
             Path(__file__).with_name('generate_native_topk.py')]
    return {'inputs': corpus.fingerprints(args.tokenizer, args.test_tree),
            'code': {str(p): _sha256_path(p) for p in paths},
            'model': args.model, 'k': 300,
            'normalization': 'ADJ->ADJP; unique normalized tree ranked by max source derivation'}


def generate(args):
    import benepar
    import torch
    torch.set_num_threads(1)
    corpus = CanonicalPPLCorpus(args.ppl_dir, args.tokenizer)
    ds, de, ss, se = corpus.shard_bounds(args.shard_id, args.num_shards)
    if args.max_sentences:
        se = min(se, ss + args.max_sentences)
    out = args.output / f'shard-{args.shard_id:05d}-of-{args.num_shards:05d}'
    out.mkdir(parents=True, exist_ok=True)
    identity = sources(args, corpus)
    identity_path = out / 'sources.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('resume input/code identity changed; use a fresh output directory')
    _json_dump(identity_path, identity)
    run = dict(status='running', document_start=ds, document_end=de,
               sentence_start=ss, sentence_end=se, num_shards=args.num_shards,
               shard_id=args.shard_id, command=sys.argv, started_at=time.time(),
               device=args.device, max_sentences=args.max_sentences)
    _json_dump(out / 'run.json', run)
    parser = benepar.Parser(args.model, batch_size=args.parser_batch_size)
    parser._parser.to(torch.device(args.device))
    labels = parser._parser.decoder.label_from_index
    pairs = label_token_pairs(labels, corpus.tokenizer)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=mp.get_context('spawn')) as pool:
        for first in range(ss, se, args.chunk_sentences):
            last = min(se, first + args.chunk_sentences)
            path = out / f'block-{first:08d}-{last:08d}.npz'
            if path.exists():
                with np.load(path) as block:
                    if int(block['first']) != first or int(block['last']) != last:
                        raise ValueError('resume block boundary mismatch')
                continue
            rows = [corpus.sentence_input(i) for i in range(first, last)]
            results, pending = [], []
            cursor = 0
            while cursor < len(rows):
                batch, cost = [], 0
                while cursor < len(rows) and len(batch) < args.parser_batch_size:
                    row = rows[cursor]
                    if batch and cost + len(row.words) > args.parser_batch_tokens:
                        break
                    batch.append(row)
                    cost += len(row.words)
                    cursor += 1
                examples = [parser._with_missing_fields_filled(benepar.InputSentence(words=r.words)) for r in batch]
                charts = list(parser._parser.parse(examples, return_scores=True,
                                                  subbatch_max_tokens=args.parser_batch_tokens))
                if len(charts) != len(batch):
                    raise ValueError('parser omitted a sentence')
                for row, chart in zip(batch, charts):
                    pending.append(pool.submit(decode, (row, chart, labels, pairs)))
                while len(pending) >= 2 * args.cpu_workers:
                    results.append(pending.pop(0).result())
            results.extend(f.result() for f in pending)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.savez(handle, first=first, last=last,
                         tokens=np.concatenate([r[0] for r in results]),
                         lengths=np.stack([r[1] for r in results]),
                         scores=np.stack([r[2] for r in results]),
                         valid_counts=np.asarray([r[3] for r in results], dtype=np.uint16))
            os.replace(temporary, path)
            run.update(completed_sentences=last - ss, elapsed_seconds=time.time() - started)
            _json_dump(out / 'run.json', run)
            print(f'shard={args.shard_id} completed={last-ss}/{se-ss} elapsed={time.time()-started:.1f}s', flush=True)
    run.update(status='complete', completed_at=time.time(), elapsed_seconds=time.time() - started)
    _json_dump(out / 'run.json', run)


def finalize(args):
    corpus = CanonicalPPLCorpus(args.ppl_dir, args.tokenizer)
    shards = sorted(args.output.glob('shard-*-of-*'))
    if len(shards) != args.num_shards:
        raise ValueError('missing labeled shards')
    blocks, total, cursor = [], 0, 0
    for shard_id, shard in enumerate(shards):
        run = json.loads((shard / 'run.json').read_text())
        ds, de, ss, se = corpus.shard_bounds(shard_id, args.num_shards)
        if run['status'] != 'complete' or run['max_sentences'] or (run['sentence_start'], run['sentence_end']) != (ss, se):
            raise ValueError('incomplete or truncated labeled shard')
        if json.loads((shard / 'sources.json').read_text()) != sources(args, corpus):
            raise ValueError('labeled source fingerprints changed')
        for path in sorted(shard.glob('block-*.npz')):
            with np.load(path) as b:
                if int(b['first']) != cursor or int(b['last']) <= cursor:
                    raise ValueError('non-contiguous block coverage')
                cursor = int(b['last'])
                total += int(b['lengths'].sum(dtype=np.uint64))
                blocks.append(path)
    n = len(corpus.lengths)
    if cursor != n:
        raise ValueError('labeled candidates do not cover all frozen sentences')
    files = {'tree_300': ((total,), np.uint16), 'tree_sent_index': ((n * 300,), np.uint16),
             'proposal_scores': ((n, 300), np.float32), 'valid_counts': ((n,), np.uint16)}
    arrays = {name: np.lib.format.open_memmap(args.output / (name + '.npy.tmp'), mode='w+', dtype=dtype, shape=shape)
              for name, (shape, dtype) in files.items()}
    offset = 0
    for path in blocks:
        with np.load(path) as b:
            first, last = int(b['first']), int(b['last'])
            tokens = b['tokens']
            if len(tokens) != int(b['lengths'].sum(dtype=np.uint64)):
                raise ValueError('block token count differs from lengths')
            arrays['tree_300'][offset:offset+len(tokens)] = tokens
            arrays['tree_sent_index'][first*300:last*300] = b['lengths'].reshape(-1)
            arrays['proposal_scores'][first:last] = b['scores']
            arrays['valid_counts'][first:last] = b['valid_counts']
            offset += len(tokens)
    for name, array in arrays.items():
        array.flush()
        os.replace(args.output / (name + '.npy.tmp'), args.output / (name + '.npy'))
    np.save(args.output / 'tree_doc_index.npy', corpus.document_counts.astype(np.uint32))
    _json_dump(args.output / 'manifest.json', dict(status='complete', format='labeled-tree300-v1',
        document_count=len(corpus.document_counts), sentence_count=n, slots=300,
        valid_counts='valid_counts.npy', proposal_scores='proposal_scores.npy',
        ranking='exact canonical A/B labeled CKY; normalized aliases use maximum score',
        sources=sources(args, corpus),
        files={p.name: _sha256_path(p) for p in args.output.glob('*.npy')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['generate', 'finalize'])
    root = REPO / 'dataset/bbc-news-reserved-clean-v1'
    p.add_argument('--ppl-dir', type=Path, default=root / 'canonical/test')
    p.add_argument('--test-tree', type=Path, default=root / 'tree/test.npy')
    p.add_argument('--tokenizer', type=Path, default=REPO / 'dataset/bbc-news/TG_GPT2_tokenizer.json')
    p.add_argument('--output', type=Path, default=root / 'testppl/tree300')
    p.add_argument('--model', default='benepar_en3_large')
    p.add_argument('--num-shards', type=int, default=8)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--parser-batch-size', type=int, default=16)
    p.add_argument('--parser-batch-tokens', type=int, default=1500)
    p.add_argument('--chunk-sentences', type=int, default=128)
    p.add_argument('--max-sentences', type=int)
    args = p.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        p.error('invalid shard id')
    (generate if args.command == 'generate' else finalize)(args)


if __name__ == '__main__':
    main()
