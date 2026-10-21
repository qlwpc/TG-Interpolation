"""Unpack complete paired NPZ collection into headerless training uint32 files.

Use sbatch on SIST. A full terminal+tree materialization needs about 270 GB;
compressed collection files remain untouched. Capacity is checked up front.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def unpack(collection, output, split='train', formats=('terminal', 'tree'), reserve=15 << 30):
    collection, output = Path(collection), Path(output)
    receipt = json.loads((collection/'receipt.json').read_text())
    if receipt['status'] != 'complete':
        raise ValueError('collection is incomplete')
    if output.exists():
        raise FileExistsError('use a fresh unpack destination')
    files = []
    for index in receipt['source_shards']:
        r = json.loads((collection/'export_receipts'/f'{index:03d}.json').read_text())
        files.extend(f for f in r['files'] if Path(f['file']).parts[0] == split)
    required = sum(f[fmt+'_tokens']*4 for f in files for fmt in formats)
    if shutil.disk_usage(output.parent).free < required+reserve:
        raise OSError(f'insufficient space for {required} raw bytes plus {reserve} reserve')
    output.mkdir()
    for fmt in formats:
        (output/fmt).mkdir()
    outputs = []
    for entry in files:
        path = collection/entry['file']
        if sha_file(path) != entry['sha256']:
            raise ValueError('compressed archive identity mismatch')
        with np.load(path, allow_pickle=False) as z:
            for fmt in formats:
                a = z[fmt]
                if a.dtype != np.dtype('<u4') or hashlib.sha256(a.tobytes()).hexdigest() != entry[fmt+'_token_bytes_sha256']:
                    raise ValueError('uncompressed token identity mismatch')
                dest = output/fmt/(path.stem+'.npy')
                a.tofile(dest)
                if sha_file(dest) != entry[fmt+'_token_bytes_sha256']:
                    raise ValueError('raw output readback identity mismatch')
                outputs.append({'file': str(dest.relative_to(output)), 'tokens': len(a),
                                'sha256': entry[fmt+'_token_bytes_sha256']})
    result = {'status': 'complete', 'source_receipt_sha256': sha_file(collection/'receipt.json'),
              'split': split, 'formats': list(formats), 'storage': 'headerless little-endian uint32; .npy name is historical convention',
              'source_modified': False, 'raw_bytes': required, 'outputs': outputs}
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation_a', 'validation_b'], default='train')
    p.add_argument('--format', choices=['terminal', 'tree', 'both'], default='both')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('SIST materialization must run through sbatch')
    formats = ('terminal', 'tree') if args.format == 'both' else (args.format,)
    print(json.dumps(unpack(args.collection, args.output, args.split, formats)), flush=True)


if __name__ == '__main__':
    main()
