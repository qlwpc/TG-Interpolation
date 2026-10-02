#!/usr/bin/env python3
"""Check public source membership, documentation links and frozen inputs.

With --export DIR, copy the validated working-tree candidate (including
uncommitted changes) into a new directory without .git or local work records.
This is a source/export check, not a clean-environment training validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(['git', '-C', str(ROOT), *args], input=data, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode())
    return result.stdout


def candidates() -> tuple[set[str], set[str]]:
    if (ROOT / '.git').exists():
        tracked = set(git('ls-files', '-z').decode().strip('\0').split('\0')) - {''}
        others = set(git('ls-files', '--others', '--exclude-standard', '-z').decode().strip('\0').split('\0')) - {''}
        names = tracked | others
        # Git also reports matching negation rules; these paths are public.
        matches = git('check-ignore', '--no-index', '-v', '-z', '--stdin',
                      data=('\0'.join(sorted(names)) + '\0').encode()).decode().split('\0')
        ignored = {matches[i + 3] for i in range(0, len(matches) - 1, 4)
                   if not matches[i + 2].startswith('!')}
        return {p for p in names - ignored if (ROOT / p).is_file()}, tracked & ignored
    exported_manifest = ROOT / '_public_source_manifest.json'
    if exported_manifest.is_file():
        return {row['path'] for row in json.loads(exported_manifest.read_text())}, set()
    return {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
            and p.name != '_public_source_manifest.json'}, set()


def anchors(text: str) -> set[str]:
    result = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        heading = re.sub(r'<[^>]+>', '', heading).replace('`', '').lower()
        slug = re.sub(r'[^\w\- ]', '', heading).replace(' ', '-')
        count = counts.get(slug, 0)
        result.add(slug + (f'-{count}' if count else ''))
        counts[slug] = count + 1
    return result


def check(names: set[str]) -> list[str]:
    errors = []
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            errors.append(f'{name}: missing public source file')
            continue
        if path.is_symlink():
            errors.append(f'{name}: publish a regular file, not a local symlink')
        if path.suffix != '.md':
            continue
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            target = target.strip('<>')
            if re.match(r'^[a-zA-Z][\w+.-]*:', target):
                continue
            relative, _, fragment = unquote(target).partition('#')
            dest = (path.parent / relative).resolve() if relative else path.resolve()
            try:
                rel = dest.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append(f'{name}: link outside public tree: {target}')
                continue
            if rel not in names and not any(p.startswith(rel.rstrip('/') + '/') for p in names):
                errors.append(f'{name}: target absent from public tree: {target}')
            elif fragment and dest.suffix == '.md' and rel in names:
                if fragment not in anchors(dest.read_text()):
                    errors.append(f'{name}: missing anchor: {target}')
    configs = json.loads((ROOT / 'train_configs/paper_pretraining_manifest.json').read_text())
    for row in configs['runs']:
        name = row['source_config']
        if name not in names:
            errors.append(f"{row['id']}: config excluded: {name}")
        elif hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != row['source_config_sha256']:
            errors.append(f"{row['id']}: config hash mismatch")
    for manifest, key in [('reproducibility/evidence/manifest.json', 'files'),
                          ('reproducibility/configs/dedup500m/manifest.json', 'models')]:
        for row in json.loads((ROOT / manifest).read_text())[key]:
            name = row['path']
            if name not in names or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != row['sha256']:
                errors.append(f'{manifest}: missing or changed {name}')
    if 'reproducibility/tokenizer-reference.json' not in names:
        errors.append('Missing public tokenizer reference')
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--allow-tracked-private', action='store_true',
                        help='Audit a cleanup in progress before removing private records from the index')
    args = parser.parse_args()
    names, tracked_private = candidates()
    errors = check(names)
    if tracked_private and not args.allow_tracked_private:
        errors.append(f'{len(tracked_private)} ignored files are still tracked; ignore rules alone do not untrack them')
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    if args.export:
        if tracked_private:
            raise SystemExit('Refusing export until tracked private records are removed from the index')
        dest = args.export.resolve()
        if dest == ROOT or ROOT in dest.parents:
            raise SystemExit('Export must be outside the working tree')
        dest.mkdir(parents=True, exist_ok=False)
        rows = []
        for name in sorted(names):
            source = ROOT / name
            output = dest / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            rows.append({'path': name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        (dest / '_public_source_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'public_files': len(names), 'tracked_private': len(tracked_private),
                      'links_and_frozen_inputs': 'passed', 'export': str(args.export) if args.export else None}))


if __name__ == '__main__':
    main()
