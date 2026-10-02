"""Explain BBC row-disjoint splits versus content overlap using read-only probes."""
import argparse
import bisect
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/wangpch/TG-Interpolation')


def prepare():
    test = json.loads((REPO / 'dataset/bbc-news/test_index.json').read_text())
    dev = json.loads((REPO / 'dataset/bbc-news/dev_index.json').read_text())
    counts_path = REPO / 'diagnostics/data/bbc_historical_shard_counts.json'
    counts = json.loads(counts_path.read_text())['counts']
    held = {s: sorted(set(test[s]) | set(dev[s])) for s in test}
    shards, ends, offset = list(test), [], 0
    for s in shards:
        offset += max(held[s]) + 1 - len(held[s])
        ends.append(offset)

    def decode(d):
        i = bisect.bisect_right(ends, d)
        s = shards[i]
        k = d - (ends[i-1] if i else 0)
        row = k
        while (n := k + bisect.bisect_right(held[s], row)) != row:
            row = n
        assert row not in held[s] and row < max(held[s])
        return s, row

    audit = json.loads((REPO / 'artifacts/bbc_all_test_train_overlap_20260906/scan.json').read_text())
    assert audit['train_documents'] == ends[-1]
    slots = [(s, row) for s, rows in test.items() for row in rows][59:]
    cross, same, cp, sp, probes = set(), set(), 0, 0, []
    mapped = []
    for g in audit['groups']:
        for d, token_offset in g['train_matches_doc_and_token_offset']:
            shard, row = decode(d)
            mapped.append({'train_doc': d, 'shard': shard, 'source_row': row,
                           'sha256': g['sha256'], 'test_docs': g['test_docs']})
            for t in g['test_docs']:
                if slots[t][0] == shard:
                    same.add(t); sp += 1
                else:
                    cross.add(t); cp += 1
            if row+1 in held[shard] and row+1 < max(held[shard]):
                right, excluded = row+1, []
                while right in held[shard]:
                    excluded.append({'row': right, 'split': 'test' if right in test[shard] else 'dev'})
                    right += 1
                probes.append({'shard': shard, 'left_row': row, 'right_row': right,
                               'excluded': excluded, 'train_doc': d, 'token_offset': token_offset,
                               'left_sha256': g['sha256']})
    # One nearby deletion boundary per available calendar year limits raw-prefix I/O.
    by_year = defaultdict(list)
    for p in probes:
        by_year[p['shard'].split('-')[2]].append(p)
    selected = [min(ps, key=lambda p: p['right_row']) for year, ps in sorted(by_year.items())]
    old = json.loads((REPO / 'artifacts/bbc_terminal_hash_sources_20260906/verification.json').read_text())
    lookup = {(m['shard'], m['source_row'], m['sha256']) for m in mapped}
    assert all((h['shard'], h['source_row'], h['full_sha256']) in lookup for h in old['verified_raw_hits'])
    witnesses = []
    for t in (0, 1):
        g = next(g for g in audit['groups'] if t in g['test_docs'])
        d, token_offset = g['train_matches_doc_and_token_offset'][0]
        shard, row = decode(d)
        witnesses.append({'test_doc': t, 'test_index_slot': slots[t], 'shard': shard,
                          'source_row': row, 'train_doc': d, 'token_offset': token_offset,
                          'sha256': g['sha256']})
    return {'claim_type': 'computed', 'mapping': 'Keep rows below final held-out row, excluding all dev/test rows; concatenate in test-index shard order.',
            'raw_documents_89_shards': sum(counts[s] for s in test),
            'held_out_rows': sum(map(len, held.values())),
            'proper_complement_documents': sum(counts[s]-len(held[s]) for s in test),
            'legacy_truncated_documents': ends[-1], 'observed_train_documents': audit['train_documents'],
            'extra_tail_rows_dropped': sum(counts[s]-max(held[s])-1 for s in test),
            'prior_raw_hits_mapped_correctly': len(old['verified_raw_hits']),
            'cross_shard_test_documents': len(cross), 'same_shard_test_documents': len(same),
            'same_shard_only_test_documents': len(same-cross), 'cross_shard_matching_pairs': cp,
            'same_shard_matching_pairs': sp, 'available_boundary_probes': len(probes),
            'probes': selected, 'witnesses': witnesses,
            'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                REPO / 'dataset/bbc-news/test_index.json', REPO / 'dataset/bbc-news/dev_index.json',
                counts_path]}}


def remote(request):
    needed = defaultdict(set)
    for p in request['probes']:
        needed[p['shard']].update([p['left_row'], p['right_row']] + [e['row'] for e in p['excluded']])
    for w in request['witnesses']:
        needed[w['shard']].add(w['source_row'])
    sources = {}
    for shard, rows in needed.items():
        path = REPO / f'dataset/bbc-news-parsed-raw/{shard}.txt'
        stat = path.stat()
        records = {}
        with path.open('rb') as handle:
            for i, raw in enumerate(handle):
                if i in rows:
                    records[str(i)] = {'parsed': raw.decode(), 'raw_sha256': hashlib.sha256(raw).hexdigest()}
                if i >= max(rows):
                    break
        assert len(records) == len(rows)
        assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
        sources[shard] = records
        print(f'read {shard} through row {i}', file=sys.stderr, flush=True)
    path = REPO / 'dataset/bbc-news/terminal/train.npy'
    arr = np.load(path, mmap_mode='r')
    def following(offset, count):
        size = 65536
        while True:
            block = arr[offset:offset+size]
            starts = np.flatnonzero(block == 50257)
            assert starts[0] == 0
            if len(starts) > count:
                return [block[starts[i]:starts[i+1]].tolist() for i in range(count)]
            if offset+size >= len(arr):
                raise ValueError('not enough following documents')
            size *= 2
    return {'sources': sources,
            'probe_documents': [following(p['token_offset'], 2) for p in request['probes']],
            'witness_documents': [following(w['token_offset'], 1)[0] for w in request['witnesses']]}


def verify(root):
    from tokenizers import Tokenizer
    sys.path.insert(0, str(REPO))
    from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_format, legacy_tokenizer
    request = json.loads((root / 'request.json').read_text())
    result = json.loads((root / 'remote.json').read_text())
    tok = legacy_tokenizer(Tokenizer.from_file(str(REPO / 'dataset/bbc-news/TG_GPT2_tokenizer.json')))
    def encode(shard, row):
        record = result['sources'][shard][str(row)]
        assert hashlib.sha256(record['parsed'].encode()).hexdigest() == record['raw_sha256']
        ids = tok.encode(legacy_format(record['parsed'], tok.get_vocab())).ids
        return [50257] + [x for x in ids if not 50268 <= x <= 50319] + [50256]
    def sha(ids):
        return hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
    probes = []
    for p, (left, right) in zip(request['probes'], result['probe_documents']):
        assert encode(p['shard'], p['left_row']) == left
        assert encode(p['shard'], p['right_row']) == right
        assert sha(left) == p['left_sha256']
        for e in p['excluded']:
            assert encode(p['shard'], e['row']) not in (left, right)
        probes.append(dict(p, left_and_right_exact=True, held_out_rows_skipped=True))
    targets = json.loads((REPO / 'artifacts/bbc_all_test_train_overlap_20260906/targets.json').read_text())
    witnesses = []
    for w, ids in zip(request['witnesses'], result['witness_documents']):
        assert encode(w['shard'], w['source_row']) == ids == targets['documents'][w['test_doc']]['ids']
        assert sha(ids) == w['sha256']
        witnesses.append(dict(w, exact_terminal_equal=True, preview=tok.decode(ids[1:45])))
    request.update({'validation_passed': True, 'probes': probes, 'witnesses': witnesses,
                    'scope_note': 'All match positions mapped under the historical rule; rule checked against counts, 17 prior raw sources and sampled deletion boundaries, not a full raw-to-train rebuild.'})
    with (root / 'verification.json').open('x') as handle:
        json.dump(request, handle, indent=2)
        handle.write('\n')
    return request


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'remote', 'verify'))
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    result = prepare() if args.mode == 'prepare' else remote(json.load(sys.stdin)) if args.mode == 'remote' else verify(args.evidence_dir)
    print(json.dumps(result), flush=True)
