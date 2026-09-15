"""Pin real checkpoint contents and the evaluation code snapshot once per campaign."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    campaign = root.parent
    tasks = json.loads((campaign / 'tasks.json').read_text())
    records = {}
    for task in tasks:
        model = task['model']
        if model in records:
            continue
        argv = task['argv']
        cp = Path(argv[argv.index('--checkpoint') + 1]).resolve()
        records[model] = {}
        for name in ('config.yaml', 'model.pt'):
            path = cp / name
            before = path.stat()
            value = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'checkpoint changed while hashing: {path}')
            records[model][name] = dict(path=str(path), resolved=str(path.resolve()),
                                        sha256=value, bytes=after.st_size, mtime_ns=after.st_mtime_ns)
        print(f'hashed {model}', flush=True)
    files = [p for folder in ('olmo', 'scripts', 'olmo_data') for p in (root/folder).rglob('*')
             if p.suffix in ('.py', '.so', '.sbatch')]
    output = dict(status='complete', claim_type='computed', models=records,
                  code_sha256={str(p.relative_to(root)):sha(p) for p in sorted(files)},
                  python=sys.version, platform=platform.platform(), completed_at=time.time())
    tmp = campaign / 'provenance.json.tmp'
    tmp.write_text(json.dumps(output, indent=2)+'\n')
    tmp.replace(campaign / 'provenance.json')


if __name__ == '__main__':
    main()
