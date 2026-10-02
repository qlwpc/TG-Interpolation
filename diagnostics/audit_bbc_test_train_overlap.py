"""Read-only full-document terminal SHA-256 overlap audit.

targets: emit a local current-test manifest, including exact terminal IDs.
scan: consume the manifest on stdin and scan a remote/local training .npy once.
Hash equality is verified by comparing canonical uint16 bytes on every hit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path('/home/wangpch/TG-Interpolation')


def canonical(ids):
    return np.asarray(ids, dtype='<u2').tobytes()


def make_targets(path):
    arr = np.load(path)
    starts = np.flatnonzero(arr == 50257)
    assert len(starts) and starts[0] == 0
    ends = np.concatenate((starts[1:], [len(arr)]))
    records = []
    for d, (start, end) in enumerate(zip(starts, ends)):
        ids = arr[start:end]
        assert ids[-1] == 50256
        records.append({'test_doc': d, 'tokens': len(ids), 'ids': ids.tolist(),
                        'sha256': hashlib.sha256(canonical(ids)).hexdigest()})
    return {'schema': 1, 'source': str(path.resolve()),
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'definition': 'Full terminal document including whitespace, BOS and EOS; little-endian uint16.',
            'documents': records}


def scan(manifest, path, chunk_tokens=8_000_000):
    groups = {}
    for target in manifest['documents']:
        blob = canonical(target['ids'])
        key = hashlib.sha256(blob).digest()
        assert key.hex() == target['sha256'] and len(blob) == 2 * target['tokens']
        if key in groups:
            assert groups[key]['bytes'] == blob
            groups[key]['test_docs'].append(target['test_doc'])
        else:
            groups[key] = {'bytes': blob, 'test_docs': [target['test_doc']], 'matches': []}
    allowed_lengths = {t['tokens'] for t in manifest['documents']}
    arr = np.load(path, mmap_mode='r')
    assert arr.ndim == 1 and arr.dtype.str == '<u2' and arr[0] == 50257
    stat = path.stat()
    file_hash, data_hash = hashlib.sha256(), hashlib.sha256()
    with path.open('rb') as handle:
        file_hash.update(handle.read(arr.offset))
    documents = hashed = matched = invalid_eos = 0
    started = last_log = time.monotonic()

    def consume(block, offset):
        nonlocal documents, hashed, matched, invalid_eos
        d = documents
        documents += 1
        invalid_eos += int(block[-1] != 50256)
        if len(block) not in allowed_lengths:
            return
        blob = block.tobytes()
        key = hashlib.sha256(blob).digest()
        hashed += 1
        group = groups.get(key)
        if group is not None:
            assert blob == group['bytes'], 'hash collision or incorrect manifest'
            group['matches'].append([d, int(offset)])
            matched += 1

    pending = 0
    for begin in range(0, len(arr), chunk_tokens):
        block = arr[begin:begin+chunk_tokens].view(np.ndarray)
        raw = block.tobytes()
        file_hash.update(raw)
        data_hash.update(raw)
        starts = np.flatnonzero(block == 50257)
        if len(starts):
            first = begin + int(starts[0])
            if first > pending:
                consume(arr[pending:first].view(np.ndarray), pending)
            for left, right in zip(starts[:-1], starts[1:]):
                consume(block[left:right], begin + int(left))
            pending = begin + int(starts[-1])
        now = time.monotonic()
        if now - last_log >= 20:
            print(json.dumps({'tokens_read': begin+len(block), 'total_tokens': len(arr),
                'documents': documents, 'matched_train_documents': matched,
                'matched_test_documents_so_far': sum(len(g['test_docs']) for g in groups.values() if g['matches']),
                'elapsed_seconds': round(now-started, 1)}), file=sys.stderr, flush=True)
            last_log = now
    consume(arr[pending:].view(np.ndarray), pending)
    assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    output_groups = [{'sha256': key.hex(), 'test_docs': group['test_docs'],
                      'train_copies': len(group['matches']),
                      'train_matches_doc_and_token_offset': group['matches']}
                     for key, group in groups.items()]
    per_doc = {d: g['train_copies'] for g in output_groups for d in g['test_docs']}
    known_four = {1063, 1090, 1121, 1122}
    return {'claim_type': 'computed', 'scan_complete': True, 'definition': manifest['definition'],
            'target_source': manifest['source'], 'target_source_sha256': manifest['source_sha256'],
            'train_source': str(path), 'train_file_sha256': file_hash.hexdigest(),
            'train_data_sha256': data_hash.hexdigest(), 'train_tokens': len(arr),
            'train_documents': documents, 'train_documents_without_final_eos': invalid_eos,
            'length_compatible_hashed_documents': hashed,
            'test_documents': len(per_doc), 'unique_test_hashes': len(groups),
            'matched_test_documents': sum(n > 0 for n in per_doc.values()),
            'unmatched_test_documents': sum(n == 0 for n in per_doc.values()),
            'matched_unique_test_hashes': sum(g['train_copies'] > 0 for g in output_groups),
            'matched_train_documents': matched,
            'test_train_matching_pairs': sum(per_doc.values()),
            'other_test_documents': sum(d not in known_four for d in per_doc),
            'other_matched_test_documents': sum(n > 0 and d not in known_four for d, n in per_doc.items()),
            'known_four_copies': {str(d): per_doc[d] for d in sorted(known_four) if d in per_doc},
            'train_copies_per_test_histogram': dict(sorted(Counter(per_doc.values()).items())),
            'unmatched_test_doc_ids': [d for d, n in per_doc.items() if n == 0],
            'groups': output_groups, 'elapsed_seconds': round(time.monotonic()-started, 1)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('targets', 'scan'))
    parser.add_argument('--test', type=Path, default=REPO / 'dataset/bbc-news/terminal/test.npy')
    parser.add_argument('--train', type=Path, default=REPO / 'dataset/bbc-news/terminal/train.npy')
    args = parser.parse_args()
    result = make_targets(args.test) if args.mode == 'targets' else scan(json.load(sys.stdin), args.train)
    print(json.dumps(result), flush=True)
