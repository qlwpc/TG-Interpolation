"""Back up EOS-delimited historical FineWeb terminal/tree records together.

Both streams are indexed independently. Equal document ordinals are candidates,
not proof: every exported tree must project to the exact source terminal bytes.
This preserves historical tokens (including PTB escapes); it does not apply text
rewrites, add BOS, or certify these arrays for a new training protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def raw_array(path):
    with path.open('rb') as f:
        if f.read(6) == b'\x93NUMPY':
            raise ValueError(f'expected historical raw uint32: {path}')
    if not path.stat().st_size or path.stat().st_size % 4:
        raise ValueError(f'invalid raw uint32 size: {path}')
    return np.memmap(path, mode='r', dtype='<u4')


def eos_ends(array, eos, chunk_tokens=4 << 20):
    chunks = []
    for start in range(0, len(array), chunk_tokens):
        loc = np.flatnonzero(array[start:start + chunk_tokens] == eos)
        if len(loc):
            chunks.append(loc.astype(np.int64) + start + 1)
    ends = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    if not len(ends) or ends[-1] != len(array):
        raise ValueError('source does not end at EOS')
    return ends


def structural_tables(tokenizer):
    spec = json.loads(Path(tokenizer).read_text())
    vocab = dict(spec['model']['vocab'])
    vocab.update({t['content']: t['id'] for t in spec['added_tokens']})
    opening = {i: s[2:-1] for s, i in vocab.items() if re.fullmatch(r'<\([A-Za-z0-9]+>', s)}
    closing = {i: s[1:-2] for s, i in vocab.items() if re.fullmatch(r'<[A-Za-z0-9]+\)>', s)}
    if not opening or set(opening.values()) != set(closing.values()):
        raise ValueError('unrecognized structural vocabulary')
    return vocab, opening, closing


def check_tree(tree, terminal, opening, closing, eos, vocab_size):
    if int(tree.max()) >= vocab_size or int(terminal.max()) >= vocab_size:
        raise ValueError('out-of-vocabulary tokens')
    nt = np.array(sorted(opening.keys() | closing.keys()), dtype='<u4')
    if not np.array_equal(tree[~np.isin(tree, nt)], terminal):
        raise ValueError('tree projection differs from terminal')
    if int(tree[-1]) != eos or np.count_nonzero(tree == eos) != 1:
        raise ValueError('invalid tree EOS boundary')
    stack, roots = [], 0
    for token in tree[np.isin(tree, nt)].tolist():
        if token in opening:
            if not stack:
                roots += 1
            stack.append(opening[token])
        elif not stack or stack.pop() != closing[token]:
            raise ValueError('unbalanced or mismatched tree labels')
    if stack or not roots:
        raise ValueError('unfinished or empty tree forest')
    return roots


def export_shard(task):
    shard_index, records, data_root, output, tokenizer = task
    vocab, opening, closing = structural_tables(tokenizer)
    eos = vocab['<|endoftext|>']
    name = Path(records[0]['metadata']['file']).name
    paths = {fmt: Path(data_root) / fmt / name for fmt in ('terminal', 'tree')}
    before = {fmt: (p.stat().st_size, p.stat().st_mtime_ns) for fmt, p in paths.items()}
    arrays = {fmt: raw_array(p) for fmt, p in paths.items()}
    ends = {fmt: eos_ends(a, eos) for fmt, a in arrays.items()}
    if len(ends['terminal']) != len(ends['tree']):
        raise ValueError(f'EOS count mismatch: shard {shard_index}')
    rows, payloads = [], {fmt: [] for fmt in arrays}
    offsets = {fmt: 0 for fmt in arrays}
    for record in sorted(records, key=lambda r: r['metadata']['start_token']):
        meta = record['metadata']
        start, end = meta['start_token'], meta['end_token']
        ordinal = int(np.searchsorted(ends['terminal'], end))
        if ordinal >= len(ends['terminal']) or ends['terminal'][ordinal] != end:
            raise ValueError('sample end is not an EOS boundary')
        expected_start = int(ends['terminal'][ordinal - 1]) if ordinal else 0
        if start != expected_start:
            raise ValueError('sample start is not an EOS boundary')
        tree_start = int(ends['tree'][ordinal - 1]) if ordinal else 0
        tree_end = int(ends['tree'][ordinal])
        slices = {'terminal': np.array(arrays['terminal'][start:end]),
                  'tree': np.array(arrays['tree'][tree_start:tree_end])}
        source_hash = hashlib.sha256(slices['terminal'].tobytes()).hexdigest()
        if source_hash != meta['token_bytes_sha256']:
            raise ValueError('source terminal fingerprint changed')
        roots = check_tree(slices['tree'], slices['terminal'], opening, closing, eos, len(vocab))
        row = {'sample_id': record['id'], 'shard_index': shard_index,
               'source_document_ordinal': ordinal, 'formula_decision': meta['formula_decision'],
               'tree_roots': roots, 'projection_equal': True, 'balanced_tree_labels': True,
               'text_representation': 'original historical tokens; cleaned JSONL is separate',
               'source': {}, 'backup': {}}
        for fmt, data in slices.items():
            source_start, source_end = (start, end) if fmt == 'terminal' else (tree_start, tree_end)
            row['source'][fmt] = {'file': str(paths[fmt]), 'start_token': source_start,
                                  'end_token': source_end, 'token_bytes_sha256': hashlib.sha256(data.tobytes()).hexdigest()}
            row['backup'][fmt] = {'file': f'{fmt}/{shard_index:03d}.npy',
                                  'start_token': offsets[fmt], 'end_token': offsets[fmt] + len(data)}
            offsets[fmt] += len(data)
            payloads[fmt].append(data)
        rows.append(row)
    result = {'shard_index': shard_index, 'source_files': {}, 'source_documents': len(ends['terminal']),
              'exported_documents': len(rows), 'outputs': {}}
    for fmt, data in payloads.items():
        path = Path(output) / fmt / f'{shard_index:03d}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.concatenate(data).astype('<u4', copy=False), allow_pickle=False)
        saved = np.load(path, mmap_mode='r', allow_pickle=False)
        for row, expected in zip(rows, data):
            bounds = row['backup'][fmt]
            if not np.array_equal(saved[bounds['start_token']:bounds['end_token']], expected):
                raise ValueError('backup readback differs')
        result['outputs'][fmt] = {'file': str(path), 'tokens': int(saved.size),
                                  'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        after = (paths[fmt].stat().st_size, paths[fmt].stat().st_mtime_ns)
        if before[fmt] != after:
            raise ValueError('source file changed during export')
        result['source_files'][fmt] = {'file': str(paths[fmt]), 'bytes': after[0], 'mtime_ns': after[1],
                                       'full_file_sha256_computed': False}
    manifest = Path(output) / 'manifests' / f'{shard_index:03d}.jsonl'
    manifest.parent.mkdir(exist_ok=True)
    with manifest.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result['manifest_sha256'] = sha256_file(manifest)
    print(f'shard {shard_index:03d}: {len(rows)} pairs verified; {len(ends["terminal"])} source documents', flush=True)
    return result


def materialize_selection(backup, selection, output, tokenizer):
    """Apply one ID list to both immutable backed-up representations."""
    backup, output = Path(backup), Path(output)
    requested = json.loads(Path(selection).read_text())
    ids = requested['sample_ids']
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('selection must contain unique IDs')
    wanted = set(ids)
    parent = json.loads((backup / 'receipt.json').read_text())
    if parent['status'] != 'complete' or not parent['all_exported_projections_equal']:
        raise ValueError('parent backup incomplete or unverified')
    if sha256_file(tokenizer) != parent['tokenizer_sha256']:
        raise ValueError('tokenizer fingerprint mismatch')
    if output.exists():
        raise FileExistsError('use a fresh subset directory')
    vocab, opening, closing = structural_tables(tokenizer)
    rows_by_shard, seen = {}, set()
    for shard in parent['shards']:
        idx = shard['shard_index']
        manifest = backup / 'manifests' / f'{idx:03d}.jsonl'
        if sha256_file(manifest) != shard['manifest_sha256']:
            raise ValueError('backup manifest fingerprint mismatch')
        for line in manifest.open():
            row = json.loads(line)
            if row['sample_id'] in seen:
                raise ValueError('duplicate parent sample ID')
            seen.add(row['sample_id'])
            if row['sample_id'] in wanted:
                rows_by_shard.setdefault(idx, []).append(row)
    if not wanted <= seen:
        raise ValueError('selection IDs absent from backup')
    output.mkdir(parents=True)
    for fmt in ('terminal', 'tree', 'manifests'):
        (output / fmt).mkdir()
    results = []
    for shard in parent['shards']:
        idx = shard['shard_index']
        rows = rows_by_shard.get(idx, [])
        if not rows:
            continue
        arrays = {}
        for fmt in ('terminal', 'tree'):
            path = backup / fmt / f'{idx:03d}.npy'
            if sha256_file(path) != shard['outputs'][fmt]['sha256']:
                raise ValueError('parent token file fingerprint mismatch')
            arrays[fmt] = np.load(path, mmap_mode='r', allow_pickle=False)
        payloads, offsets, new_rows = {'terminal': [], 'tree': []}, {'terminal': 0, 'tree': 0}, []
        for original in rows:
            row = json.loads(json.dumps(original))
            slices = {}
            for fmt, array in arrays.items():
                bounds = original['backup'][fmt]
                slices[fmt] = np.array(array[bounds['start_token']:bounds['end_token']])
                if hashlib.sha256(slices[fmt].tobytes()).hexdigest() != original['source'][fmt]['token_bytes_sha256']:
                    raise ValueError('backed-up document fingerprint mismatch')
                row['backup'][fmt] = {'file': f'{fmt}/{idx:03d}.npy', 'start_token': offsets[fmt],
                                      'end_token': offsets[fmt] + len(slices[fmt])}
                offsets[fmt] += len(slices[fmt])
                payloads[fmt].append(slices[fmt])
            check_tree(slices['tree'], slices['terminal'], opening, closing, vocab['<|endoftext|>'], len(vocab))
            new_rows.append(row)
        result = {'shard_index': idx, 'exported_documents': len(rows), 'outputs': {}}
        for fmt in arrays:
            path = output / fmt / f'{idx:03d}.npy'
            expected = np.concatenate(payloads[fmt]).astype('<u4', copy=False)
            np.save(path, expected, allow_pickle=False)
            if not np.array_equal(np.load(path, mmap_mode='r', allow_pickle=False), expected):
                raise ValueError('subset readback mismatch')
            result['outputs'][fmt] = {'file': str(path), 'tokens': len(expected), 'bytes': path.stat().st_size,
                                      'sha256': sha256_file(path)}
        manifest = output / 'manifests' / f'{idx:03d}.jsonl'
        with manifest.open('w') as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        result['manifest_sha256'] = sha256_file(manifest)
        results.append(result)
    receipt = {'status': 'complete', 'scope': 'paired sample subset; historical source tokens preserved',
               'exported_documents': sum(r['exported_documents'] for r in results),
               'selection': requested, 'selection_sha256': sha256_file(selection),
               'parent_backup_receipt_sha256': sha256_file(backup / 'receipt.json'),
               'tokenizer_sha256': parent['tokenizer_sha256'], 'script_sha256': sha256_file(__file__),
               'all_exported_projections_equal': True, 'all_exported_tree_labels_balanced': True,
               'original_source_modified': False, 'outputs_read_back': True,
               'storage_format': parent['storage_format'], 'shards': results}
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='stable DataTrove JSONL shards')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--from-backup', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--only-shard', type=int)
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    if args.from_backup:
        if not args.selection:
            p.error('--selection is required with --from-backup')
        materialize_selection(args.from_backup, args.selection, args.output, args.tokenizer)
        return
    if not args.input or not args.data_root:
        p.error('--input and --data-root are required for source backup')
    if args.output.exists():
        raise FileExistsError('use a fresh backup directory')
    groups, seen = {}, set()
    files = sorted(args.input.glob('*.jsonl'))
    for file in files:
        for line in file.open():
            record = json.loads(line)
            if record['id'] in seen:
                raise ValueError('duplicate sample ID')
            seen.add(record['id'])
            idx = record['metadata']['shard_index']
            if args.only_shard is None or idx == args.only_shard:
                groups.setdefault(idx, []).append(record)
    if not groups:
        raise ValueError('empty selection')
    args.output.mkdir(parents=True)
    tasks = [(idx, records, str(args.data_root), str(args.output), str(args.tokenizer)) for idx, records in sorted(groups.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(export_shard, tasks))
    receipt = {'status': 'complete', 'scope': 'paired sample backup, not full 100BT filtering',
               'host': os.uname().nodename, 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
               'exported_documents': sum(r['exported_documents'] for r in results),
               'source_documents_in_scanned_shards': sum(r['source_documents'] for r in results),
               'tokenizer_sha256': sha256_file(args.tokenizer),
               'script_sha256': sha256_file(__file__),
               'input_sha256': {f.name: sha256_file(f) for f in files},
               'storage_format': 'standard NPY uint32; historical EOS-only documents; no BOS inserted',
               'original_source_modified': False, 'all_exported_projections_equal': True,
               'all_exported_tree_labels_balanced': True, 'outputs_read_back': True,
               'shards': results}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
