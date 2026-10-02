"""Validate full overlap audit and export per-test-document CSV evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root, index_path, control_path=None):
    manifest = json.loads((root / 'targets.json').read_text())
    report = json.loads((root / 'scan.json').read_text())
    assert report['scan_complete']
    source_sha = hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()
    assert source_sha == manifest['source_sha256'] == report['target_source_sha256']
    targets = {t['test_doc']: t for t in manifest['documents']}
    assert sorted(targets) == list(range(report['test_documents']))
    records, seen_train, seen_test = {}, set(), set()
    for group in report['groups']:
        matches = group['train_matches_doc_and_token_offset']
        assert group['train_copies'] == len(matches)
        doc_ids = [d for d, offset in matches]
        assert doc_ids == sorted(set(doc_ids))
        assert not seen_train.intersection(doc_ids)
        seen_train.update(doc_ids)
        for d, offset in matches:
            assert 0 <= d < report['train_documents'] and 0 <= offset < report['train_tokens']
        for d in group['test_docs']:
            assert d not in seen_test
            seen_test.add(d)
            target = targets[d]
            sha = hashlib.sha256(np.asarray(target['ids'], dtype='<u2').tobytes()).hexdigest()
            assert sha == target['sha256'] == group['sha256']
            records[d] = {'test_doc': d, 'terminal_tokens': target['tokens'],
                          'full_terminal_sha256': sha, 'train_copies': len(matches),
                          'first_train_doc': matches[0][0] if matches else '',
                          'first_train_token_offset': matches[0][1] if matches else ''}
    assert seen_test == set(targets)
    assert len(seen_train) == report['matched_train_documents']
    assert sum(r['train_copies'] > 0 for r in records.values()) == report['matched_test_documents']
    assert sum(r['train_copies'] for r in records.values()) == report['test_train_matching_pairs']
    assert dict(Counter(r['train_copies'] for r in records.values())) == {
        int(k): v for k, v in report['train_copies_per_test_histogram'].items()}
    controls = None
    if control_path:
        control = json.loads(control_path.read_text())
        old_train = next(r for r in control['arrays'] if r['split'] == 'train')
        assert old_train['data_sha256'] == report['train_data_sha256']
        assert old_train['documents'] == report['train_documents']
        assert control['train_full_terminal_match_counts'] == report['known_four_copies']
        controls = {'previous_four_counts_match': True, 'same_train_data_sha256': True,
                    'control_path': str(control_path)}
    index = json.loads(index_path.read_text())
    slots = [(shard, row) for shard, rows in index.items() for row in rows]
    assert len(slots) == 5025 and len(records) == 4966
    assert all(shard == 'CC-MAIN-2013-20' for shard, row in slots[:59])
    by_shard = defaultdict(lambda: {'test_documents': 0, 'matched_test_documents': 0})
    rows = []
    for d in sorted(records):
        shard, source_row = slots[d+59]
        record = dict(records[d], index_slot_shard=shard, index_slot_source_row=source_row)
        rows.append(record)
        by_shard[shard]['test_documents'] += 1
        by_shard[shard]['matched_test_documents'] += int(record['train_copies'] > 0)
    for name, subset in [('test_overlap.csv', rows),
                         ('unmatched_test.csv', [r for r in rows if not r['train_copies']])]:
        with (root / name).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    copies = [r['train_copies'] for r in rows if r['train_copies']]
    summary = {k: v for k, v in report.items() if k not in ('groups', 'unmatched_test_doc_ids')}
    summary.update({'validation_passed': True, 'control_validation': controls,
        'matched_test_percent': 100 * report['matched_test_documents'] / len(rows),
        'other_matched_test_percent': 100 * report['other_matched_test_documents'] / report['other_test_documents'],
        'copies_among_matched_test': {'min': min(copies), 'median': float(np.median(copies)), 'max': max(copies)},
        'per_index_slot_shard': dict(by_shard),
        'index_slot_note': 'Historical test_index slot mapped by +59; the four article mismatches have different content sources.',
        'evidence_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in ('targets.json', 'scan.json', 'test_overlap.csv', 'unmatched_test.csv')}})
    with (root / 'summary.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_dir', type=Path)
    parser.add_argument('--index', type=Path, default=Path('dataset/bbc-news/test_index.json'))
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    result = summarize(args.evidence_dir, args.index, args.control)
    print(json.dumps({k: v for k, v in result.items() if k not in
        ('per_index_slot_shard', 'train_copies_per_test_histogram')}, indent=2))
