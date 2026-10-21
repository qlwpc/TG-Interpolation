"""Locate topic candidates in processed FineWeb-Edu terminal shards.

This is lexical candidate retrieval, not the paper's DBSCAN classifier and not
an estimate of topic prevalence. Coordinates are zero-based token offsets;
end_token is exclusive. Historical raw uint32 files and EOS-only boundaries
must be selected explicitly. Sampling anchors are stratified in token space, so long
documents have a higher inclusion probability. Original arrays are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def load_tokens(path, storage_format):
    import numpy as np

    with path.open('rb') as source:
        magic = source.read(6)
    if storage_format == 'npy':
        if magic != b'\x93NUMPY':
            raise ValueError(f'missing NPY header: {path}; select raw-u32 for historical data')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
    else:
        if magic == b'\x93NUMPY' or path.stat().st_size % 4:
            raise ValueError(f'not a headerless uint32 stream: {path}')
        array = np.memmap(path, dtype='<u4', mode='r')
    if array.ndim != 1 or array.dtype != np.dtype('<u4') or not len(array):
        raise ValueError(f'expected nonempty uint32 vector: {path}')
    return array


def document_at(array, anchor, bos, eos, initial_radius=8192, max_radius=262144, boundary='bos-eos'):
    import numpy as np

    radius = initial_radius
    while True:
        left, right = max(0, anchor - radius), min(len(array), anchor + radius + 1)
        if boundary == 'eos':
            before = np.flatnonzero(array[left:anchor] == eos)
            start = left + int(before[-1]) + 1 if before.size else (0 if left == 0 else None)
        else:
            before = np.flatnonzero(array[left:anchor + 1] == bos)
            start = left + int(before[-1]) if before.size else None
        after = np.flatnonzero(array[anchor:right] == eos)
        if start is not None and after.size:
            end = anchor + int(after[0]) + 1
            # Reject ambiguous/corrupt boundaries instead of joining documents.
            if (boundary == 'eos' or np.count_nonzero(array[start:end] == bos) == 1) and np.count_nonzero(array[start:end] == eos) == 1:
                return start, end
            return None
        if radius >= max_radius:
            return None
        radius = min(radius * 2, max_radius)


def retrieve(text, topics, min_hits):
    # Most cues are single words: tokenize once rather than scanning a long
    # document independently for every cue. Keep boundary regexes for phrases.
    words = set(re.findall(r'\w+', text.lower()))
    hits = []
    for topic in topics:
        terms = [term for term, pattern in topic['_patterns']
                 if (term in words if re.fullmatch(r'\w+', term) else pattern.search(text))]
        if len(terms) >= min_hits:
            hits.append({'topic_id': topic['id'], 'distinct_terms': len(terms), 'terms': terms})
    return sorted(hits, key=lambda x: (-x['distinct_terms'], x['topic_id']))


def run_sample(args):
    import numpy as np
    from tokenizers import Tokenizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('samples.jsonl', 'summary.json')):
        raise FileExistsError('use a fresh output directory to preserve evidence')
    tok = Tokenizer.from_file(str(args.tokenizer))
    bos, eos = tok.token_to_id('<|beginoftext|>'), tok.token_to_id('<|endoftext|>')
    if bos is None or eos is None:
        raise ValueError('missing document boundary tokens')
    if tok.get_vocab_size() != 151732 or tok.token_to_id('<|SEP|>') != 151673:
        raise ValueError('unexpected Qwen3 TG tokenizer layout')
    taxonomy = json.loads(args.taxonomy.read_text())
    topics = taxonomy['topics']
    for topic in topics:
        topic['_patterns'] = [(term, re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)', re.I)) for term in topic['terms']]
    files = sorted((args.data_root / 'terminal').glob('*.npy'))
    if len(files) != args.expected_shards:
        raise ValueError(f'expected {args.expected_shards} terminal shards, got {len(files)}')
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'host': os.uname().nodename,
        'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'data_root': str(args.data_root.absolute()),
        'resolved_data_root_on_sample_node': str(args.data_root.resolve()),
        'tokenizer_path': str(args.tokenizer.resolve()),
        'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        'taxonomy_sha256': hashlib.sha256(args.taxonomy.read_bytes()).hexdigest(),
        'method': 'lexical candidate retrieval; token-stratified random anchors; no prevalence estimate',
        'seed': args.seed, 'anchors_per_shard': args.per_shard, 'min_distinct_terms': args.min_hits,
        'storage_format': args.storage_format, 'boundary': args.boundary,
        'bos_id': bos, 'eos_id': eos, 'vocab_size': tok.get_vocab_size(),
        'shards': [], 'sampled_documents': 0, 'sampled_tokens': 0,
        'skipped_boundary_anchors': 0, 'duplicate_anchors': 0,
        'full_payload_integrity_checked': False, 'original_document_metadata_available': False,
    }
    counts, best = Counter(), {t['id']: [] for t in topics}
    rng = np.random.default_rng(args.seed)
    with (output / 'samples.jsonl').open('w') as sink:
        for shard_index, path in enumerate(files):
            array = load_tokens(path, args.storage_format)
            if (args.boundary == 'bos-eos' and int(array[0]) != bos) or int(array[-1]) != eos:
                raise ValueError(f'invalid shard boundary: {path}')
            meta = {'shard_index': shard_index, 'path': str(path.absolute()), 'tokens': int(array.size),
                    'resolved_path_on_sample_node': str(path.resolve()),
                    'dtype': str(array.dtype), 'file_bytes': path.stat().st_size,
                    'mtime_ns': path.stat().st_mtime_ns, 'samples': 0}
            tree_path = args.data_root / 'tree' / path.name
            tree = load_tokens(tree_path, args.storage_format)
            meta['tree_tokens'] = int(tree.size)
            meta['tree_path'] = str(tree_path.absolute())
            del tree
            seen = set()
            for interval in range(args.per_shard):
                lo = interval * len(array) // args.per_shard
                hi = (interval + 1) * len(array) // args.per_shard
                anchor = int(rng.integers(lo, hi))
                bounds = document_at(array, anchor, bos, eos, boundary=args.boundary)
                if bounds is None:
                    summary['skipped_boundary_anchors'] += 1
                    continue
                start, end = bounds
                if start in seen:
                    summary['duplicate_anchors'] += 1
                    continue
                seen.add(start)
                ids = array[start:end]
                if int(ids.max()) >= tok.get_vocab_size():
                    raise ValueError(f'out-of-vocabulary sampled document: {path}:{start}')
                text = tok.decode(ids.tolist(), skip_special_tokens=True)
                record = {'sample_id': f'{shard_index:03d}:{start}', 'shard_index': shard_index,
                          'file': str(path.absolute()), 'resolved_file_on_sample_node': str(path.resolve()),
                          'start_token': start, 'end_token': end,
                          'tokens': end - start, 'anchor_token': anchor,
                          'token_bytes_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                          'candidates': retrieve(text, topics, args.min_hits), 'text': text}
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                summary['sampled_documents'] += 1
                summary['sampled_tokens'] += end - start
                meta['samples'] += 1
                for hit in record['candidates']:
                    topic_id = hit['topic_id']
                    counts[topic_id] += 1
                    short = {k: v for k, v in record.items() if k != 'candidates'}
                    short['retrieval'] = hit
                    best[topic_id].append(short)
                    best[topic_id].sort(key=lambda r: (-r['retrieval']['distinct_terms'], r['tokens'], r['sample_id']))
                    del best[topic_id][args.top_k:]
            summary['shards'].append(meta)
            del array
            print(f'{shard_index + 1}/{len(files)} shards; {summary["sampled_documents"]} documents', flush=True)
    summary['total_terminal_tokens'] = sum(s['tokens'] for s in summary['shards'])
    summary['total_tree_tokens'] = sum(s['tree_tokens'] for s in summary['shards'])
    summary['topic_candidate_counts'] = {t['id']: counts[t['id']] for t in topics}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output / 'topic_candidates.json').write_text(json.dumps(best, ensure_ascii=False, indent=2) + '\n')


def extract(args):
    import numpy as np
    from tokenizers import Tokenizer

    array = load_tokens(args.file, args.storage_format)
    if not (0 <= args.start < args.end <= len(array)):
        raise ValueError('invalid half-open token interval')
    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = array[args.start:args.end]
    eos = tok.token_to_id('<|endoftext|>')
    if int(ids[-1]) != eos or np.count_nonzero(ids == eos) != 1:
        raise ValueError('coordinates do not delimit a complete document')
    if args.boundary == 'bos-eos' and int(ids[0]) != tok.token_to_id('<|beginoftext|>'):
        raise ValueError('missing BOS at start')
    if args.boundary == 'eos' and args.start > 0 and int(array[args.start - 1]) != eos:
        raise ValueError('coordinates do not delimit a complete document')
    if args.sha256 and hashlib.sha256(ids.tobytes()).hexdigest() != args.sha256:
        raise ValueError('document token fingerprint changed')
    print(tok.decode(ids.tolist(), skip_special_tokens=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    s = subs.add_parser('sample')
    s.add_argument('--data-root', type=Path, required=True)
    s.add_argument('--tokenizer', type=Path, required=True)
    s.add_argument('--taxonomy', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--expected-shards', type=int, default=246)
    s.add_argument('--per-shard', type=int, default=32)
    s.add_argument('--seed', type=int, default=20261012)
    s.add_argument('--min-hits', type=int, default=3)
    s.add_argument('--top-k', type=int, default=5)
    s.add_argument('--storage-format', choices=('npy', 'raw-u32'), default='npy')
    s.add_argument('--boundary', choices=('bos-eos', 'eos'), default='bos-eos')
    s.set_defaults(func=run_sample)
    e = subs.add_parser('extract')
    e.add_argument('--file', type=Path, required=True)
    e.add_argument('--tokenizer', type=Path, required=True)
    e.add_argument('--start', type=int, required=True)
    e.add_argument('--end', type=int, required=True)
    e.add_argument('--sha256')
    e.add_argument('--storage-format', choices=('npy', 'raw-u32'), default='npy')
    e.add_argument('--boundary', choices=('bos-eos', 'eos'), default='bos-eos')
    e.set_defaults(func=extract)
    args = p.parse_args()
    if args.command == 'sample' and (args.per_shard < 1 or args.min_hits < 1 or args.top_k < 1):
        p.error('sampling and retrieval limits must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
