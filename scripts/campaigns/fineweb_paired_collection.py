"""Collect whole, filtered, deduplicated FineWeb pairs as lossless NPZ shards.

Run ``index`` and ``export`` inside Slurm on SIST. The terminal budget excludes
document-ending EOS; tree structure tokens are counted separately. Source arrays
are read-only. Each NPZ stores full terminal/tree uint32 arrays and coordinates,
so it is independent of the source and needs no custom tree reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.campaigns.fineweb_collection_filter import VERSION, classify
from scripts.campaigns.fineweb_paired_backup import raw_array, eos_ends, structural_tables

PROTOCOL = 'fineweb-filtered-paired-npz-v1'
RECORD = np.dtype([('ordinal', '<u4'), ('start', '<u8'), ('end', '<u8'),
                   ('token_sha', 'V32'), ('text_sha', 'V32'), ('split', 'u1')])
PACK_RECORD = np.dtype(RECORD.descr + [('tree_start', '<u8'), ('tree_end', '<u8'),
                                      ('terminal_offset', '<u8'), ('tree_offset', '<u8')])
SPLITS = ('train', 'validation_a', 'validation_b')
WORK = None


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def identity(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}


def split_for_hash(digest):
    bucket = int.from_bytes(digest[:8], 'big') % 2048
    return 1 if bucket == 0 else (2 if bucket == 1 else 0)


def text_hash(text):
    canonical = ' '.join(unicodedata.normalize('NFC', text).split())
    return hashlib.sha256(canonical.encode('utf-8')).digest()


def source_code_identity():
    repo = Path(__file__).resolve().parents[2]
    return {name: sha_file(repo/name) for name in [
        'scripts/campaigns/fineweb_paired_collection.py',
        'scripts/campaigns/fineweb_collection_filter.py',
        'scripts/campaigns/fineweb_formula_filter.py',
        'scripts/campaigns/fineweb_paired_backup.py', 'datatools/ptb_brackets.py']}


def load_protocol(root):
    p = json.loads((root/'protocol.json').read_text())
    if p['protocol'] != PROTOCOL or p['filter'] != VERSION or p['code_sha256'] != source_code_identity():
        raise ValueError('protocol or executed source differs from inventory')
    if sha_file(root/'tokenizer.json') != p['tokenizer_sha256']:
        raise ValueError('tokenizer identity changed')
    return p


def init_scan(path, tokenizer, workspace):
    global WORK
    from tokenizers import Tokenizer
    WORK = (raw_array(Path(path)), Tokenizer.from_file(str(tokenizer)), Path(workspace))


def scan_block(task):
    block, ordinal, start, ends = task
    a, tok, work = WORK
    out = work / f'{block:05d}.npy'
    rows, counts, token_counts, examples = [], Counter(), Counter(), {}
    for j, end in enumerate(ends):
        end = int(end)
        size = end - start - 1
        if size < 32 or size > 262144:
            reasons = ['short_document' if size < 32 else 'oversized_document']
            text = ''
        else:
            ids = a[start:end-1]
            text = tok.decode(ids.tolist(), skip_special_tokens=False)
            reasons = classify(text)['reasons']
        if reasons:
            counts['drop'] += 1
            token_counts['drop'] += size
            counts.update(reasons)
            for reason in reasons:
                if reason not in examples:
                    examples[reason] = {'ordinal': ordinal+j, 'start': start, 'end': end,
                                        'text_head': text[:1000]}
        else:
            digest = text_hash(text)
            token_digest = hashlib.sha256(a[start:end].tobytes()).digest()
            rows.append((ordinal+j, start, end, np.void(token_digest), np.void(digest),
                         split_for_hash(digest)))
            counts['keep'] += 1
            token_counts['keep'] += size
        start = end
    np.save(out, np.array(rows, dtype=RECORD), allow_pickle=False)
    return {'block': block, 'file': str(out), 'counts': dict(counts),
            'lexical_tokens': dict(token_counts), 'examples': examples}


def scan_tasks(ends, block_tokens=1 << 20):
    ordinal, start, block = 0, 0, 0
    while ordinal < len(ends):
        limit = max(ordinal+1, int(np.searchsorted(ends, start+block_tokens, side='right')))
        stop = min(limit, len(ends))
        yield block, ordinal, start, ends[ordinal:stop]
        start, ordinal, block = int(ends[stop-1]), stop, block+1


def create_inventory(args):
    root, source = args.output, args.source
    if (root/'protocol.json').exists():
        raise FileExistsError('inventory already exists; resume index/export instead')
    root.mkdir(parents=True, exist_ok=True)
    for d in ['selections', 'scan_receipts', 'scratch', 'export_receipts', *SPLITS]:
        (root/d).mkdir(exist_ok=True)
    tokenizer = root/'tokenizer.json'
    if not tokenizer.exists():
        shutil.copyfile(args.tokenizer, tokenizer)
    if sha_file(tokenizer) != sha_file(args.tokenizer):
        raise ValueError('tokenizer mismatch')
    acceptance = json.loads((args.repair/'acceptance_final.json').read_text())
    if acceptance['status'] != 'passed' or not acceptance['full_repair_complete']:
        raise ValueError('source repair not accepted')
    shards = []
    for index, terminal in enumerate(sorted((source/'terminal').glob('*.npy'))):
        journal = args.repair/'backups'/f'{index:03d}'/'journal.json'
        j = json.loads(journal.read_text())
        tree = source/'tree'/terminal.name
        if j['status'] != 'committed' or j['name'] != terminal.name:
            raise ValueError('repair identity mismatch')
        shards.append({'index': index, 'name': terminal.name,
                       'paths': {'terminal': str(terminal), 'tree': str(tree)},
                       'identity': {fmt: identity(p) for fmt, p in [('terminal', terminal), ('tree', tree)]},
                       'sha256': j['new_sha256'], 'journal_sha256': sha_file(journal)})
    if len(shards) != 246:
        raise ValueError('expected 246 repaired source pairs')
    order = np.random.default_rng(20261013).permutation(len(shards)).tolist()
    protocol = {'protocol': PROTOCOL, 'filter': VERSION, 'source': str(source),
                'source_modified': False, 'tokenizer_sha256': sha_file(tokenizer),
                'source_acceptance_sha256': sha_file(args.repair/'acceptance_final.json'),
                'train_lexical_target': args.train_tokens, 'validation_lexical_target_each': args.validation_tokens,
                'counting': 'terminal lexical tokens excluding ending EOS; tree tokens counted separately',
                'boundary': 'whole documents, historical EOS-only, no initial BOS',
                'splits': 'SHA256(NFC + collapsed whitespace text), buckets 0/1 of 2048 for validation_a/b',
                'dedup': 'global exact normalized-text SHA256 among selected records; no MinHash claim',
                'selection': 'seeded source shard permutation; stable source document order within each shard',
                'source_order': order, 'source_shards': shards, 'storage': 'standard lossless ZIP_DEFLATED NPZ, full terminal/tree uint32',
                'max_output_bytes': args.max_output_bytes, 'reserve_free_bytes': args.reserve_free_bytes,
                'minimum_document_lexical_tokens': 32, 'maximum_document_lexical_tokens': 262144,
                'formula_review_policy': 'exclude uncertain as well as confirmed formula candidates',
                'whole_document_web_boilerplate_policy': True, 'created': time.time()}
    protocol.update(code_sha256=source_code_identity(), pilot_scan_blocks=args.pilot_scan_blocks)
    atomic_json(root/'protocol.json', protocol)
    print(json.dumps({'source_pairs': len(shards), 'output': str(root)}), flush=True)


def verify_source_identity(shard):
    for fmt, path in shard['paths'].items():
        if identity(path) != shard['identity'][fmt]:
            raise ValueError(f'source identity changed: {fmt}, {shard["index"]}')


def collect_index(args):
    root = args.output
    p = load_protocol(root)
    targets = [p['train_lexical_target'], p['validation_lexical_target_each'], p['validation_lexical_target_each']]
    totals, docs, seen, completed = [0]*3, [0]*3, set(), []
    for index in p['source_order']:
        receipt = root/'scan_receipts'/f'{index:03d}.json'
        if not receipt.exists():
            continue
        r = json.loads(receipt.read_text())
        selection = root/'selections'/f'{index:03d}.npy'
        if r['status'] != 'complete' or sha_file(selection) != r['selection_sha256']:
            raise ValueError('invalid previous selection')
        verify_source_identity(p['source_shards'][index])
        for row in np.load(selection, allow_pickle=False):
            digest = row['text_sha'].tobytes()
            if digest in seen:
                raise ValueError('duplicate previous selected text')
            seen.add(digest)
            split = int(row['split'])
            if split != split_for_hash(digest):
                raise ValueError('incorrect previous split')
            totals[split] += int(row['end']-row['start'])-1
            docs[split] += 1
        completed.append(index)
    for index in p['source_order']:
        if all(a >= b for a, b in zip(totals, targets)):
            break
        if index in completed:
            continue
        shard = p['source_shards'][index]
        verify_source_identity(shard)
        path = Path(shard['paths']['terminal'])
        start_time = time.time()
        # Authenticate the scanned source against the independently accepted SHA.
        if sha_file(path) != shard['sha256']['terminal']:
            raise ValueError('terminal source SHA changed')
        a = raw_array(path)
        ends = eos_ends(a, 151643)
        work = root/'scratch'/f'{index:03d}'
        work.mkdir(exist_ok=True)
        results, selected, counts, token_counts, examples = [], [], Counter(), Counter(), {}
        duplicate_docs = 0
        with ProcessPoolExecutor(args.workers, initializer=init_scan,
                                 initargs=(path, root/'tokenizer.json', work)) as pool:
            tasks = scan_tasks(ends)
            if p['pilot_scan_blocks']:
                tasks = itertools.islice(tasks, p['pilot_scan_blocks'])
            for result in pool.map(scan_block, tasks, chunksize=1):
                counts.update(result['counts'])
                token_counts.update(result['lexical_tokens'])
                for reason, example in result['examples'].items():
                    examples.setdefault(reason, example)
                for row in np.load(result['file'], allow_pickle=False):
                    split = int(row['split'])
                    digest = row['text_sha'].tobytes()
                    if digest in seen:
                        duplicate_docs += 1
                        continue
                    if totals[split] >= targets[split]:
                        continue
                    seen.add(digest)
                    selected.append(row.copy())
                    totals[split] += int(row['end']-row['start'])-1
                    docs[split] += 1
                Path(result['file']).unlink()
                results.append(result['block'])
        verify_source_identity(shard)
        selection = root/'selections'/f'{index:03d}.npy'
        np.save(selection, np.array(selected, dtype=RECORD), allow_pickle=False)
        receipt = {'status': 'complete', 'shard': index, 'source': shard,
                   'source_documents': len(ends), 'scanned_documents': counts['keep']+counts['drop'],
                   'pilot_scan_blocks': p['pilot_scan_blocks'],
                   'counts': dict(counts), 'lexical_tokens': dict(token_counts),
                   'duplicate_selected_texts_skipped': duplicate_docs,
                   'selected_documents': len(selected), 'selection_sha256': sha_file(selection),
                   'cumulative_lexical_tokens': dict(zip(SPLITS, totals)),
                   'cumulative_documents': dict(zip(SPLITS, docs)), 'examples': examples,
                   'elapsed': time.time()-start_time, 'job_id': os.environ.get('SLURM_JOB_ID')}
        atomic_json(root/'scan_receipts'/f'{index:03d}.json', receipt)
        completed.append(index)
        progress = {'status': 'indexing', 'completed_source_shards': completed,
                    'lexical_tokens': dict(zip(SPLITS, totals)), 'documents': dict(zip(SPLITS, docs)),
                    'target_lexical_tokens': dict(zip(SPLITS, targets)), 'updated': time.time()}
        atomic_json(root/'progress.json', progress)
        print(json.dumps({'shard': index, 'elapsed': receipt['elapsed'], 'tokens': totals, 'documents': docs}), flush=True)
    if not all(a >= b for a, b in zip(totals, targets)):
        raise ValueError(f'candidate source exhausted before targets: {totals}')
    # Resume may already have met every target without entering the loop.
    atomic_json(root/'progress.json', {'status': 'index_complete', 'completed_source_shards': completed,
                                     'lexical_tokens': dict(zip(SPLITS, totals)),
                                     'documents': dict(zip(SPLITS, docs)), 'updated': time.time()})


@lru_cache(maxsize=4)
def pair_tables(tokenizer):
    vocab, opening, closing = structural_tables(tokenizer)
    size = max(vocab.values())+1
    nt = np.zeros(size, dtype=bool)
    delta = np.zeros(size, dtype=np.int8)
    label = np.zeros(size, dtype=np.int16)
    labels = sorted(set(opening.values()))
    for i, name in enumerate(labels, 1):
        op, cl = vocab[f'<({name}>'], vocab[f'<{name})>']
        nt[[op, cl]] = True
        delta[op], delta[cl] = 1, -1
        label[[op, cl]] = i
    return vocab, size, nt, delta, label


def check_pair(tree, terminal, tokenizer):
    vocab, size, nt, delta, label = pair_tables(str(tokenizer))
    if not len(terminal) or not len(tree) or max(int(tree.max()), int(terminal.max())) >= size:
        raise ValueError('empty or out-of-vocabulary pair')
    mask = nt[tree]
    if np.any(nt[terminal]) or not np.array_equal(tree[~mask], terminal):
        raise ValueError('tree projection differs from terminal')
    depth = np.cumsum(delta[tree], dtype=np.int32)
    eos = vocab['<|endoftext|>']
    if depth.min() < 0 or depth[-1] != 0 or np.any(depth[tree == eos] != 0):
        raise ValueError('unbalanced forest or interior EOS')
    root_counts = np.cumsum((delta[tree] == 1) & (depth == 1))
    if np.any(np.diff(np.r_[0, root_counts[tree == eos]]) == 0):
        raise ValueError('document without a tree root')
    events = tree[mask]
    if not len(events):
        raise ValueError('missing tree structure')
    kinds = delta[events]
    d = np.cumsum(kinds, dtype=np.int32)-(kinds == 1)
    order = np.argsort(d, kind='stable')
    d, k, tags = d[order], kinds[order], label[events[order]]
    starts = np.r_[0, np.flatnonzero(np.diff(d))+1]
    sizes = np.diff(np.r_[starts, len(d)])
    if np.any(sizes % 2) or np.any(k[::2] != 1) or np.any(k[1::2] != -1) or np.any(tags[::2] != tags[1::2]):
        raise ValueError('mismatched tree labels')
    if terminal[-1] != eos:
        raise ValueError('missing ending EOS')
    return int(np.count_nonzero(terminal == eos))


def write_npz(path, arrays):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name, a in arrays.items():
            with z.open(name+'.npy', 'w') as f:
                np.lib.format.write_array(f, a, allow_pickle=False)


def storage_check(root, protocol, incoming):
    # Serialize capacity checks and publication among export workers.
    free = shutil.disk_usage(root).free
    if free < protocol['reserve_free_bytes'] + incoming:
        raise OSError('storage guard: insufficient filesystem free bytes')
    current = sum(p.stat().st_size for split in SPLITS for p in (root/split).glob('*.npz'))
    if current+incoming > protocol['max_output_bytes']:
        raise OSError('storage guard: collection byte cap exceeded')


def export_source(task):
    root, index, block_tokens = task
    root = Path(root)
    protocol = load_protocol(root)
    shard = protocol['source_shards'][index]
    verify_source_identity(shard)
    selection_path = root/'selections'/f'{index:03d}.npy'
    scan_receipt = json.loads((root/'scan_receipts'/f'{index:03d}.json').read_text())
    if sha_file(selection_path) != scan_receipt['selection_sha256']:
        raise ValueError('selection identity changed')
    receipt_path = root/'export_receipts'/f'{index:03d}.json'
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        if r['status'] != 'complete' or r['selection_sha256'] != scan_receipt['selection_sha256']:
            raise ValueError('incompatible export receipt')
        for f in r['files']:
            if sha_file(root/f['file']) != f['sha256']:
                raise ValueError('previous export SHA mismatch')
        return r
    start_time = time.time()
    arrays, ends = {}, {}
    for fmt, path in shard['paths'].items():
        if sha_file(path) != shard['sha256'][fmt]:
            raise ValueError('source SHA mismatch during export')
        arrays[fmt] = raw_array(Path(path))
        ends[fmt] = eos_ends(arrays[fmt], 151643)
    if len(ends['tree']) != len(ends['terminal']):
        raise ValueError('source EOS count mismatch')
    rows = np.load(selection_path, allow_pickle=False)
    files = []
    for split_index, split in enumerate(SPLITS):
        selected = rows[rows['split'] == split_index]
        blocks, bucket, tokens = [], [], 0
        for row in selected:
            if bucket and tokens >= block_tokens:
                blocks.append(bucket)
                bucket, tokens = [], 0
            bucket.append(row)
            tokens += int(row['end']-row['start'])
        if bucket:
            blocks.append(bucket)
        for number, bucket in enumerate(blocks):
            payloads = {'terminal': [], 'tree': []}
            offsets = {'terminal': 0, 'tree': 0}
            packed = np.zeros(len(bucket), dtype=PACK_RECORD)
            for pos, row in enumerate(bucket):
                for field in RECORD.names:
                    packed[pos][field] = row[field]
                ordinal, start, end = int(row['ordinal']), int(row['start']), int(row['end'])
                if end != ends['terminal'][ordinal] or start != (int(ends['terminal'][ordinal-1]) if ordinal else 0):
                    raise ValueError('selection is not a whole document')
                ts = int(ends['tree'][ordinal-1]) if ordinal else 0
                te = int(ends['tree'][ordinal])
                lexical = arrays['terminal'][start:end]
                if hashlib.sha256(lexical.tobytes()).digest() != row['token_sha'].tobytes():
                    raise ValueError('selected source token SHA mismatch')
                packed[pos]['tree_start'], packed[pos]['tree_end'] = ts, te
                for fmt, part in [('terminal', lexical), ('tree', arrays['tree'][ts:te])]:
                    packed[pos][fmt+'_offset'] = offsets[fmt]
                    offsets[fmt] += len(part)
                    payloads[fmt].append(part)
            combined = {fmt: np.concatenate(parts).astype('<u4', copy=False) for fmt, parts in payloads.items()}
            if check_pair(combined['tree'], combined['terminal'], root/'tokenizer.json') != len(bucket):
                raise ValueError('export document count differs')
            relative = Path(split)/f'{index:03d}-{number:04d}.npz'
            path = root/relative
            temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
            with (root/'.storage.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                storage_check(root, protocol, sum(a.nbytes for a in combined.values())+packed.nbytes)
            try:
                write_npz(temp, {**combined, 'records': packed})
                with np.load(temp, allow_pickle=False) as saved:
                    for fmt, a in combined.items():
                        if not np.array_equal(saved[fmt], a):
                            raise ValueError('NPZ readback differs')
                    if not np.array_equal(saved['records'], packed):
                        raise ValueError('NPZ coordinate readback differs')
                with (root/'.storage.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    storage_check(root, protocol, temp.stat().st_size)
                    os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
            files.append({'file': str(relative), 'bytes': path.stat().st_size, 'sha256': sha_file(path),
                          'terminal_tokens': offsets['terminal'], 'tree_tokens': offsets['tree'],
                          'lexical_tokens': offsets['terminal']-len(bucket), 'documents': len(bucket),
                          'terminal_token_bytes_sha256': hashlib.sha256(combined['terminal'].tobytes()).hexdigest(),
                          'tree_token_bytes_sha256': hashlib.sha256(combined['tree'].tobytes()).hexdigest()})
    verify_source_identity(shard)
    result = {'status': 'complete', 'shard': index, 'selection_sha256': scan_receipt['selection_sha256'],
              'files': files, 'source_full_sha_verified': True, 'all_projections_equal': True,
              'all_typed_labels_balanced': True, 'all_outputs_read_back': True,
              'elapsed': time.time()-start_time, 'job_id': os.environ.get('SLURM_JOB_ID')}
    atomic_json(receipt_path, result)
    print(json.dumps({'exported_shard': index, 'files': len(files), 'elapsed': result['elapsed']}), flush=True)
    return result


def export_all(args):
    root = args.output
    progress = json.loads((root/'progress.json').read_text())
    if progress['status'] != 'index_complete':
        raise ValueError('index must meet all budgets before exporting')
    shards = progress['completed_source_shards']
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(export_source, [(str(root), i, args.block_tokens) for i in shards]):
            results.append(r)
    totals = {split: Counter() for split in SPLITS}
    for r in results:
        for f in r['files']:
            totals[Path(f['file']).parts[0]].update({k: f[k] for k in ['lexical_tokens', 'terminal_tokens', 'tree_tokens', 'documents', 'bytes']})
    for split in SPLITS:
        for key in ['lexical_tokens', 'documents']:
            if totals[split][key] != progress[key][split]:
                raise ValueError('index/export totals differ')
    receipt = {'status': 'complete', 'protocol': PROTOCOL, 'filter': VERSION,
               'totals': {k: dict(v) for k, v in totals.items()}, 'source_modified': False,
               'all_selected_pairs_project_exactly': True, 'all_selected_typed_trees_balanced': True,
               'all_compressed_arrays_read_back_exactly': True, 'exact_text_dedup_and_split_disjointness_verified': True,
               'protocol_sha256': sha_file(root/'protocol.json'), 'tokenizer_sha256': sha_file(root/'tokenizer.json'),
               'source_shards': shards, 'job_id': os.environ.get('SLURM_JOB_ID'), 'completed': time.time()}
    atomic_json(root/'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inventory', 'index', 'export'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--repair', type=Path)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--train-tokens', type=int, default=20_000_000_000)
    p.add_argument('--validation-tokens', type=int, default=10_000_000)
    p.add_argument('--max-output-bytes', type=int, default=95 << 30)
    p.add_argument('--reserve-free-bytes', type=int, default=15 << 30)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--block-tokens', type=int, default=4 << 20)
    p.add_argument('--pilot-scan-blocks', type=int, default=0,
                   help='inventory only: bounded pilot blocks per source, zero means full shards')
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    if args.action != 'inventory' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('batch computation must run inside sbatch')
    if args.action == 'inventory':
        if not all([args.source, args.repair, args.tokenizer]):
            p.error('inventory requires --source --repair --tokenizer')
        create_inventory(args)
    elif args.action == 'index':
        collect_index(args)
    else:
        export_all(args)


if __name__ == '__main__':
    main()
