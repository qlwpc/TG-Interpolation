"""Independently re-encode raw hash hits with the established legacy builder."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_format, legacy_tokenizer, split_tree_stream
from diagnostics.audit_bbc_structure_provenance import sentence_trees
from diagnostics.find_bbc_terminal_sources import digest


def verify(root):
    manifest = json.loads((root / 'targets.json').read_text())
    path = Path(manifest['source'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['source_sha256']
    current = split_tree_stream(path)
    targets = {t['current_doc']: t for t in manifest['targets']}
    tok = legacy_tokenizer(Tokenizer.from_str(manifest['tokenizer_json']))
    arrays = [json.loads(s) for s in (root / 'array_scan.jsonl').read_text().splitlines()]
    assert arrays[-1]['type'] == 'complete'
    scans = [json.loads(s) for s in (root / 'parsed_scan.jsonl').read_text().splitlines()]
    completed_shards = [s for s in scans if s['type'] == 'shard']
    verified = []
    for shard in completed_shards:
        for hit in shard['hits']:
            target = targets[hit['current_doc']]
            assert hashlib.sha256(hit['parsed'].encode()).hexdigest() == hit['raw_sha256']
            tree = np.asarray([50257, *tok.encode(legacy_format(hit['parsed'], tok.get_vocab())).ids,
                               50256], dtype=np.uint16)
            full = tree[(tree < 50268) | (tree > 50319)].tolist()
            sentences = sentence_trees(tree)
            core = [x for s in sentences for x in s['leaves']]
            assert digest(core) == target['core_sha256'] and core == target['core_ids']
            assert digest(full) == target['full_sha256'] and full == target['full_ids']
            record = {k: v for k, v in hit.items() if k != 'parsed'}
            record['independently_verified_core_and_full'] = True
            record['tree_exact'] = bool(np.array_equal(tree, current[hit['current_doc']]))
            verified.append(record)
    assert set(targets) == {h['current_doc'] for h in verified}
    return {'claim_type': 'computed',
            'array_scan_complete': True,
            'arrays': [{k: v for k, v in s.items() if k != 'prefix_matches'} for s in arrays if s['type'] == 'array'],
            'train_full_terminal_match_counts': dict(Counter(h['current_doc'] for s in arrays
                 if s.get('split') == 'train' for h in s['hits'])),
            'parsed_scan_complete': scans[-1]['type'] == 'complete',
            'parsed_stop_reason': 'Intentionally stopped after all four targets had exact raw parsed sources; '
                                 'only completed-shard records are used as verified provenance.',
            'completed_parsed_shards': [{k: v for k, v in s.items() if k not in
                ('hits', 'prefix_matches_with_core_difference')} for s in completed_shards],
            'completed_parsed_rows': sum(s['rows_scanned'] for s in completed_shards),
            'verified_raw_hits': verified,
            'inputs': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ('targets.json', 'array_scan.jsonl', 'parsed_scan.jsonl')}}


if __name__ == '__main__':
    root = Path(sys.argv[1])
    result = verify(root)
    with (root / 'verification.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('train_full_terminal_match_counts',
        'completed_parsed_rows', 'parsed_scan_complete')}, indent=2))
    print('verified raw hits:', len(result['verified_raw_hits']))
