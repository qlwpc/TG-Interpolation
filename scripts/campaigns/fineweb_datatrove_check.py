"""Run pinned DataTrove diagnostics without modifying source training data.

Quality filters are evaluated independently and in FineWeb's official order.
C4 text rewrites are diagnostic only: a paired terminal/tree selection must use
the same document IDs and preserve original text, or rebuild both representations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import struct
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def records(folder):
    for p in sorted(Path(folder).glob('*.jsonl')):
        with p.open() as f:
            for line in f:
                yield json.loads(line)


def prepare(args):
    root = args.output
    if root.exists():
        raise FileExistsError('use a fresh check output directory')
    root.mkdir(parents=True)
    (root / 'input').mkdir()
    (root / 'filtered_text_backup').mkdir()
    rows, hashes = [], {}
    for decision in ('keep', 'drop', 'review'):
        source = args.source / f'{decision}.jsonl'
        import shutil
        copied = root / 'filtered_text_backup' / source.name
        shutil.copyfile(source, copied)
        hashes[source.name] = sha256_file(source)
        if sha256_file(copied) != hashes[source.name]:
            raise ValueError('text backup hash mismatch')
        for line in copied.open():
            r = json.loads(line)
            if r['formula_filter']['decision'] != decision:
                raise ValueError('formula decision mismatch')
            meta = {k: r[k] for k in ('shard_index', 'file', 'start_token', 'end_token', 'tokens', 'token_bytes_sha256')}
            meta['formula_decision'] = decision
            meta['source_text_sha256'] = hashlib.sha256(r.get('source_text', r['text']).encode()).hexdigest()
            meta['native_text_sha256'] = hashlib.sha256(r['text'].encode()).hexdigest()
            rows.append({'id': r['sample_id'], 'text': r['text'], 'metadata': meta})
    rows.sort(key=lambda r: (r['metadata']['shard_index'], r['metadata']['start_token']))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate input IDs')
    handles = [(root / 'input' / f'{i:05d}.jsonl').open('w') for i in range(4)]
    try:
        for i, r in enumerate(rows):
            handles[i % 4].write(json.dumps(r, ensure_ascii=False) + '\n')
    finally:
        for handle in handles:
            handle.close()
    write_json(root / 'prepare_receipt.json', {'status': 'complete', 'records': len(rows),
               'formula_counts': dict(Counter(r['metadata']['formula_decision'] for r in rows)),
               'original_text_backups_sha256': hashes, 'input_sha256': {p.name: sha256_file(p) for p in sorted((root / 'input').glob('*.jsonl'))},
               'original_data_modified': False, 'input_text': 'native bracket normalization from existing formula trial',
               'scope': '7872 sampled EOS spans; token-stratified, not uniform document sampling'})


def as_result(result):
    return {'pass': bool(result[0]), 'reason': result[1]} if isinstance(result, tuple) else {'pass': bool(result), 'reason': None}


def quality_file(task):
    input_path, output = map(Path, task)
    from datatrove.data import Document
    from datatrove.pipeline.filters import GopherRepetitionFilter, GopherQualityFilter, C4QualityFilter, FineWebQualityFilter
    filters = {'gopher_repetition': GopherRepetitionFilter(), 'gopher_quality': GopherQualityFilter(),
               'c4': C4QualityFilter(filter_no_terminal_punct=False), 'fineweb': FineWebQualityFilter()}
    def evaluate(name, text, sample_id):
        doc = Document(text=text, id=sample_id)
        result = as_result(filters[name].filter(doc))
        if name != 'c4' and doc.text != text:
            raise ValueError('unexpected text mutation')
        return result, doc.text
    with input_path.open() as f, (output / input_path.name).open('w') as out, (output / f'c4_preview_{input_path.name}').open('w') as previews:
        for idx, line in enumerate(f):
            r = json.loads(line)
            text = r['text']
            res = {name: evaluate(name, text, r['id'])[0] for name in ('gopher_repetition', 'gopher_quality', 'fineweb')}
            res['c4'], c4_text = evaluate('c4', text, r['id'])
            res['fineweb_rstrip'], _ = evaluate('fineweb', '\n'.join(line.rstrip() for line in text.splitlines()), r['id'])
            if res['c4']['pass']:
                res['fineweb_after_c4'], _ = evaluate('fineweb', c4_text, r['id'])
            reason = next((name + ':' + str(res[name]['reason']) for name in ('gopher_repetition', 'gopher_quality', 'c4', 'fineweb_after_c4') if name in res and not res[name]['pass']), None)
            changes = {'text_changed': c4_text != text, 'characters_before': len(text), 'characters_after': len(c4_text),
                       'non_whitespace_changed': ''.join(text.split()) != ''.join(c4_text.split())}
            row = {'sample_id': r['id'], 'metadata': r['metadata'], 'independent': res,
                   'official_text_chain_pass': reason is None, 'official_text_chain_first_failure': reason,
                   'c4_changes': changes}
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            if res['c4']['pass'] and changes['text_changed']:
                previews.write(json.dumps({'id': r['id'], 'text': c4_text, 'metadata': r['metadata'], 'diagnostic_only': True}, ensure_ascii=False) + '\n')
            if idx % 100 == 0:
                print(f'quality {input_path.name}: {idx + 1} records', flush=True)
    return str(output / input_path.name)


def quality(args):
    output = args.output / 'quality'
    output.mkdir()
    start = time.monotonic()
    inputs = sorted((args.output / 'input').glob('*.jsonl'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(quality_file, [(str(p), str(output)) for p in inputs]))
    independent, chain, mutations, formula_keeps = Counter(), Counter(), Counter(), Counter()
    n = 0
    for path in sorted(output.glob('[0-9]*.jsonl')):
        for line in path.open():
            r = json.loads(line)
            n += 1
            for name, res in r['independent'].items():
                if not res['pass']:
                    independent[name + ':' + str(res['reason'])] += 1
            chain[r['official_text_chain_first_failure'] or 'pass'] += 1
            for name in ('text_changed', 'non_whitespace_changed'):
                if r['c4_changes'][name]:
                    mutations[name] += 1
            if r['metadata']['formula_decision'] == 'keep':
                formula_keeps[r['official_text_chain_first_failure'] or 'pass'] += 1
    prepare_receipt = json.loads((args.output / 'prepare_receipt.json').read_text())
    if n != prepare_receipt['records']:
        raise ValueError('quality coverage mismatch')
    write_json(output / 'summary.json', {'status': 'complete', 'records': n, 'elapsed_seconds': time.monotonic() - start,
               'independent_failures': dict(independent), 'official_chain_results': dict(chain),
               'formula_keep_chain_results': dict(formula_keeps), 'c4_mutations': dict(mutations),
               'order': ['gopher_repetition', 'gopher_quality', 'c4', 'fineweb_after_c4'],
               'c4_config': {'filter_no_terminal_punct': False, 'other_parameters': 'pinned defaults'},
               'url_language_extraction_run': False, 'active_candidates_modified': False,
               'quality_hard_filters_adopted': False, 'rewritten_text_has_matching_old_tree': False})


def minhash(args):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.text import TextNormConfig
    root = args.output / 'minhash'
    root.mkdir()
    config = MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1,
                          norm_config=TextNormConfig(norm_numbers=False), hash_config=HashConfig(hash_fc='sha1', precision=64))
    def reader():
        return JsonlReader(str(args.output / 'input'), glob_pattern='*.jsonl', shuffle_files=False)
    stages = [([reader(), MinhashDedupSignature(str(root / 'signatures'), config=config)], 4),
              ([MinhashDedupBuckets(str(root / 'signatures'), str(root / 'buckets'), config=config)], 14),
              ([MinhashDedupCluster(str(root / 'buckets'), str(root / 'remove_ids'), config=config,
                                   save_cluster_id=True, save_cluster_size=True)], 1),
              ([reader(), MinhashDedupFilter(str(root / 'remove_ids'),
                    exclusion_writer=JsonlWriter(str(root / 'removed'), compression=None)),
                    JsonlWriter(str(root / 'kept'), compression=None)], 4)]
    durations = []
    for i, (pipeline, tasks) in enumerate(stages, 1):
        start = time.monotonic()
        exe = LocalPipelineExecutor(pipeline=pipeline, tasks=tasks, workers=min(args.workers, tasks),
                                    logging_dir=str(root / f'logs_stage{i}'), skip_completed=False)
        exe.run()
        if len(exe.get_incomplete_ranks(skip_completed=True)):
            raise ValueError(f'incomplete MinHash stage {i}')
        durations.append(time.monotonic() - start)
        print(f'MinHash stage {i} complete', flush=True)
    source = {r['id']: r for r in records(args.output / 'input')}
    kept, removed = {}, {}
    for dest, folder in ((kept, root / 'kept'), (removed, root / 'removed')):
        for r in records(folder):
            if r['id'] in dest or r['id'] not in source or r['text'] != source[r['id']]['text']:
                raise ValueError('MinHash identity/text mismatch')
            dest[r['id']] = r
    if set(kept) & set(removed) or set(kept) | set(removed) != set(source):
        raise ValueError('MinHash coverage mismatch')
    rank_rows = [list(records_one(p)) for p in sorted((args.output / 'input').glob('*.jsonl'))]
    clusters = defaultdict(list)
    for path in sorted((root / 'remove_ids').glob('*.clusters')):
        rank = int(path.stem)
        for idx, cid in struct.iter_unpack('<II', path.read_bytes()):
            clusters[cid].append(rank_rows[rank][idx]['id'])
    cluster_rows = []
    for cid, ids in sorted(clusters.items()):
        representatives = sorted(set(ids) & set(kept))
        if len(representatives) != 1:
            raise ValueError('cluster does not have exactly one representative')
        cluster_rows.append({'cluster_id': cid, 'sample_ids': sorted(ids), 'kept': representatives[0],
                             'removed': sorted(set(ids) & set(removed))})
    edge_set = set()
    for path in sorted((root / 'buckets').glob('*.dups')):
        for f1, d1, f2, d2 in struct.iter_unpack('<IIII', path.read_bytes()):
            edge_set.add(tuple(sorted((rank_rows[f1][d1]['id'], rank_rows[f2][d2]['id']))))
    sig = MinhashDedupSignature(str(root / 'unused_validation_signatures'), config=config)
    shingles = {}
    pairs = []
    for a, b in sorted(edge_set):
        for sid in (a, b):
            if sid not in shingles:
                shingles[sid] = set(sig.get_shingles(source[sid]['text']).ravel().tolist())
        sa, sb = shingles[a], shingles[b]
        pairs.append({'a': a, 'b': b, 'hashed_5gram_jaccard': len(sa & sb) / len(sa | sb),
                      'exact_native_text_equal': source[a]['text'] == source[b]['text'],
                      'same_source_token_hash': source[a]['metadata']['token_bytes_sha256'] == source[b]['metadata']['token_bytes_sha256']})
    known = {'065:270803770', '179:92001306'}
    if known <= set(source) and not any(known <= set(c['sample_ids']) for c in cluster_rows):
        raise ValueError('known exact duplicate not found by MinHash')
    write_json(root / 'clusters.json', cluster_rows)
    write_json(root / 'matched_pair_overlap.json', pairs)
    # Only exact-source-token duplicates alter this conservative selection.
    groups = defaultdict(list)
    for r in source.values():
        if r['metadata']['formula_decision'] == 'keep':
            groups[r['metadata']['token_bytes_sha256']].append(r['id'])
    selection, exact_removed = [], []
    for ids in groups.values():
        ids.sort()
        selection.append(ids[0])
        exact_removed.extend({'sample_id': sid, 'duplicate_of': ids[0]} for sid in ids[1:])
    selection.sort()
    write_json(args.output / 'paired_keep_selection.json', {'policy': 'existing formula keep plus exact source-token dedup; quality/near-dup diagnostic only',
               'sample_ids': selection, 'exact_duplicate_removals': exact_removed,
               'known_formula_false_negative': '197:290371261', 'certified_formula_free': False})
    write_json(root / 'summary.json', {'status': 'complete', 'input_records': len(source), 'kept': len(kept), 'removed': len(removed),
               'clusters': len(cluster_rows), 'unique_matched_pairs': len(pairs), 'stage_seconds': durations,
               'known_exact_pair_detected': True, 'all_ids_accounted_for': True, 'all_output_text_unchanged': True,
               'config': {'n_grams': 5, 'num_buckets': 14, 'hashes_per_bucket': 8, 'seed': 1,
                          'hash': 'sha1', 'precision': 64, 'norm_numbers': False, 'other_text_normalization': 'pinned defaults'},
               'near_duplicate_policy_adopted': False, 'paired_keep_records': len(selection),
               'exact_source_duplicate_removals': exact_removed})


def records_one(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def selection(args):
    """Keep the old rule receipt intact while excluding a verified pilot miss."""
    parent = json.loads((args.output / 'paired_keep_selection.json').read_text())
    source = {r['id']: r for r in records(args.output / 'input')}
    sid = '197:290371261'
    r = source[sid]
    evidence = 'u(x ) = ln(x )'
    if evidence not in r['text'] or sid not in parent['sample_ids']:
        raise ValueError('known formula miss identity/evidence changed')
    reviewed = dict(parent)
    reviewed['policy'] = 'existing formula keep plus exact source-token dedup and one manually verified formula miss exclusion; quality/near-dup diagnostic only'
    reviewed['sample_ids'] = [s for s in parent['sample_ids'] if s != sid]
    reviewed['reviewed_math_exclusions'] = [{'sample_id': sid, 'evidence': evidence,
        'source_token_sha256': r['metadata']['token_bytes_sha256'],
        'native_text_sha256': r['metadata']['native_text_sha256'],
        'reason': 'verified mathematical function equation missed by formula-syntax-v5'}]
    reviewed['known_formula_false_negative_excluded'] = True
    reviewed['parent_selection_sha256'] = sha256_file(args.output / 'paired_keep_selection.json')
    write_json(args.output / 'paired_keep_selection_reviewed.json', reviewed)


def environment(args):
    from datatrove.pipeline.filters import GopherRepetitionFilter, GopherQualityFilter, C4QualityFilter, FineWebQualityFilter
    from datatrove.pipeline.dedup.minhash import MinhashDedupSignature
    from datatrove.executor import LocalPipelineExecutor
    classes = (GopherRepetitionFilter, GopherQualityFilter, C4QualityFilter, FineWebQualityFilter, MinhashDedupSignature, LocalPipelineExecutor)
    write_json(args.output / 'environment.json', {'python': os.sys.version, 'executable': os.sys.executable,
               'packages': {p: importlib.metadata.version(p) for p in ('datatrove', 'spacy', 'numpy', 'multiprocess', 'tokenizers', 'huggingface_hub')},
               'actual_modules': {c.__name__: {'file': inspect.getfile(c), 'sha256': sha256_file(inspect.getfile(c))} for c in classes},
               'driver_sha256': sha256_file(__file__), 'scope': 'CPU text diagnostics; no GPST model or training',
               'isolation': 'temporary venv with system-site-packages; shared environment unmodified'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'quality', 'minhash', 'environment', 'selection'))
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
