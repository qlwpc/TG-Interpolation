"""Compare the read-only remote source extraction against pinned local trees."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_bbc_structure_provenance import sentence_trees, sha256_file, split_tree_stream


def verify(root):
    report_path = root / 'structure_differences.json'
    remote_path = root / 'remote_sources.json'
    report = json.loads(report_path.read_text())
    remote = json.loads(remote_path.read_text())
    for name in ('old', 'current'):
        record = report['inputs'][name]
        assert sha256_file(Path(record['path'])) == record['sha256'], name
    old = split_tree_stream(Path(report['inputs']['old']['path']))
    current = split_tree_stream(Path(report['inputs']['current']['path']))
    counts, documents, rank_summary = Counter(), [], Counter()
    for doc in report['documents']:
        key = str(doc['current_doc'])
        before = [x['tokens'] for x in sentence_trees(old[doc['old_doc']])]
        after = [x['tokens'] for x in sentence_trees(current[doc['current_doc']])]
        shard = remote['shards'][doc['shard']]
        group = ('structure' if doc['terminals_equal'] else
                 'adj' if doc['current_doc'] in (1550, 2626) else 'article')
        comparisons = {}
        for stage, trees in [('deprecated', remote['deprecated'][key]),
                             ('shard_npy', shard['npy_candidate_zero'][key]),
                             ('source_txt', shard['txt_candidate_zero'][key])]:
            match = 'current' if trees == after else 'old' if trees == before else 'neither'
            comparisons[stage] = match
            counts[f'{group}/{stage}/{match}'] += 1
            assert match == ('old' if group == 'adj' else 'current'), (key, stage, match)
        documents.append({'current_doc': doc['current_doc'], 'old_doc': doc['old_doc'],
                          'group': group, 'matches': comparisons})
    per_shard = {}
    for name, shard in remote['shards'].items():
        ranks = shard['old_tree_ranks']
        matched = [x for x in ranks if x['old_tree_candidate_ids']]
        histogram = Counter(min(x['old_tree_candidate_ids']) for x in matched)
        rank_summary.update(histogram)
        per_shard[name] = {'changed_sentences': len(ranks), 'old_tree_found': len(matched),
                           'old_tree_absent': len(ranks) - len(matched)}
    assert sum(x['changed_sentences'] for x in per_shard.values()) == 403
    return {'claim_type': 'computed',
            'inputs': {str(p): sha256_file(p) for p in (report_path, remote_path)},
            'scope': 'Complete top-level tree tokens for all sentences in the 97 changed documents; '
                     'not a comparison of outside-tree whitespace or all candidates globally.',
            'stage_matches': dict(counts), 'per_shard': per_shard,
            'old_tree_minimum_candidate_id_histogram': dict(sorted(rank_summary.items())),
            'old_tree_found_sentences': sum(rank_summary.values()),
            'old_tree_absent_sentences': 403 - sum(rank_summary.values()),
            'documents': documents}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_dir', type=Path)
    args = parser.parse_args()
    result = verify(args.evidence_dir)
    output = args.evidence_dir / 'source_verification.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'documents'}, indent=2))
