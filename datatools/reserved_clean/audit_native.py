"""Read back native candidates against a frozen base split and canonical input."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from datatools.reserved_clean.common import atomic_json, records, sha_file
from datatools.parse_test_docppl_data.generate_native_topk import CanonicalPPLCorpus
from datatools.parse_test_docppl_data.native_topk import capped_catalan, capped_little_schroeder
from olmo.eval.native_model_topk_corpus import NativeModelTopKCorpus


def run(args):
    manifest_hash = sha_file(args.dataset / 'manifest.json')
    validation = json.loads((args.dataset / 'audit/validation.json').read_text())
    if not validation['complete'] or validation['manifest_sha256'] != manifest_hash:
        raise ValueError('base dataset has not passed its frozen validation')
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    selected = list(records(args.dataset / (args.split + '_selected.jsonl.gz')))
    selection = json.loads(args.selection.read_text()) if args.selection else None
    if selection and selection['candidate_receipt_sha256'] != manifest['candidate_receipt_sha256']:
        raise ValueError('selection belongs to another candidate pool')
    ids = selection['candidate_ids'] if selection else [r['candidate_id'] for r in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate selection')
    lookup = {r['candidate_id']: i for i, r in enumerate(selected)}
    if not set(ids) <= lookup.keys():
        raise ValueError('selection includes documents outside its declared split')
    tree = np.load(args.dataset / 'tree' / (args.split + '.npy'), mmap_mode='r')
    offsets = np.load(args.dataset / 'tree' / (args.split + '_doc_offsets.npy'))
    expected_tree = np.concatenate([tree[offsets[lookup[i]]:offsets[lookup[i] + 1]] for i in ids])
    canonical = CanonicalPPLCorpus(args.canonical, args.tokenizer)
    if not np.array_equal(expected_tree, canonical.tree):
        raise ValueError('canonical input differs from frozen base documents')
    audit = json.loads((args.native / 'alignment_audit.json').read_text())
    audit_tokenizer = args.equivalent_tokenizer or args.tokenizer
    if json.loads(audit_tokenizer.read_text()) != json.loads(args.tokenizer.read_text()):
        raise ValueError('candidate tokenizer config differs from frozen scoring tokenizer')
    expected_sources = {sha_file(path) for path in canonical.source_paths}
    expected_sources.add(sha_file(audit_tokenizer))
    if set(audit['source_sha256'].values()) != expected_sources or audit['exceptions'] or audit['document_offset']:
        raise ValueError('native alignment receipt does not certify these inputs')
    native = NativeModelTopKCorpus(args.native)
    if len(native) != len(canonical.lengths) or native.manifest['document_count'] != len(ids):
        raise ValueError('native output does not cover the complete subset')
    hist = {'gpst': Counter(), 'pushdown': Counter()}
    for i in range(len(native)):
        a, b = native.sentence(i), canonical.sentence_input(i)
        if a.global_sentence_id != i or a.document_id != b.document_id:
            raise ValueError('native sentence/document identity mismatch')
        if not np.array_equal(a.tokens, b.terminal_tokens) or a.content_bounds != (b.content_start, b.content_end):
            raise ValueError('native scoring tokens or bounds changed')
        starts = np.r_[0, np.cumsum([len(x) for x in b.word_piece_ids])[:-1]]
        if not np.array_equal(a.word_starts, starts):
            raise ValueError('native word boundaries changed')
        for component in hist:
            count = getattr(a, component + '_valid_count')
            scores = getattr(a, component + '_proposal_scores')
            expected_count = (capped_catalan if component == 'gpst' else capped_little_schroeder)(len(b.words), 300)
            if count != expected_count:
                raise ValueError(f'{component} does not contain the complete requested top-300 support')
            if not 1 <= count <= 300 or len(scores) != count or not np.isfinite(scores).all():
                raise ValueError('invalid logical candidate count or proposal score')
            if np.any(scores[1:] > scores[:-1] + 1e-4):
                raise ValueError('proposal ranking is not descending')
            hist[component][count] += 1
        width = max(b.content_end - b.content_start - 1, 0)
        if a.gpst_merge_orders.shape != (a.gpst_valid_count, width):
            raise ValueError('GPST merge width differs from scoring text')
        if not np.all(np.sort(a.gpst_merge_orders, axis=1) == np.arange(width)):
            raise ValueError('GPST candidate does not merge every BPE gap exactly once')
        if len({r.tobytes() for r in a.gpst_merge_orders}) != a.gpst_valid_count:
            raise ValueError('duplicate logical GPST candidate')
        spans, counts = a.pushdown_spans, a.pushdown_span_counts
        if np.any(counts > spans.shape[1]):
            raise ValueError('Pushdown span count exceeds storage')
        if len(b.words) > 1 and (np.any(counts < 1) or not np.all(
                spans[np.arange(len(counts)), counts.astype(int) - 1] ==
                [b.content_start, b.content_end - 1, b.content_end - 1])):
            raise ValueError('Pushdown candidate is missing its sentence root')
        valid = spans[np.arange(spans.shape[1])[None, :] < counts[:, None]]
        if (np.any(valid[:, 0] < b.content_start) or np.any(valid[:, 2] >= b.content_end)
                or np.any(valid[:, 0] >= valid[:, 2]) or np.any(valid[:, 1] != valid[:, 2])):
            raise ValueError('Pushdown candidate has invalid real constituent bounds')
        if not np.all(np.isin(valid[:, 0], starts + b.content_start)):
            raise ValueError('Pushdown constituent starts inside a word')
        ends = np.r_[starts[1:] - 1, b.content_end - b.content_start - 1] + b.content_start
        if not np.all(np.isin(valid[:, 2], ends)):
            raise ValueError('Pushdown constituent ends inside a word')
        keys = {row[:int(count)].tobytes() for row, count in zip(spans, counts)}
        if len(keys) != a.pushdown_valid_count:
            raise ValueError('duplicate logical Pushdown candidate')
    for shard in native.shards:
        for component in hist:
            scores = getattr(shard, component + '_proposal_scores')
            counts = getattr(shard, component + '_valid_counts')
            padding = np.arange(300)[None, :] >= counts[:, None]
            if not np.isneginf(scores[padding]).all():
                raise ValueError('padding candidates have finite proposal mass')
    atomic_json(args.output, {'complete': True, 'dataset_manifest_sha256': manifest_hash,
        'selection_sha256': sha_file(args.selection) if args.selection else None,
        'scope': 'selected_subset' if args.selection else 'complete_split',
        'split': args.split, 'documents': len(ids),
        'sentences': len(native), 'all_native_tokens_and_word_boundaries_match': True,
        'logical_candidates_unique': True, 'padding_has_no_proposal_mass': True,
        'valid_counts_match_catalan_and_schroeder_limits': True,
        'valid_count_histograms': {k: dict(sorted(v.items())) for k, v in hist.items()},
        'auditor_sha256': sha_file(__file__),
        'native_files': {str(p.relative_to(args.native)): sha_file(p)
                         for p in args.native.rglob('*') if p.is_file()}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'canonical', 'native', 'output', 'tokenizer'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--selection', type=Path, help='Omit to audit the entire frozen split in original order')
    p.add_argument('--split', choices=('dev', 'test'), required=True)
    p.add_argument('--equivalent-tokenizer', type=Path)
    run(p.parse_args())
