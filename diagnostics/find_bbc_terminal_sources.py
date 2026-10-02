"""Read-only terminal-hash search; remote modes accept a target manifest on stdin.

make-targets runs locally. arrays scans all archived train/dev/test documents.
parsed scans every raw parsed row, using necessary literal anchors only to avoid
expensive tree parsing of obvious nonmatches; final matches require full hashes
and exact token lists. All hashes serialize IDs as little-endian uint16.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import threading

import numpy as np

REPO = Path('/home/wangpch/TG-Interpolation')
DOCS = (1063, 1090, 1121, 1122)
ANCHORS = {1063: 'Hamas', 1090: 'Merkinch', 1121: 'Stockholm', 1122: 'Victorian'}
OUTPUT_LOCK = threading.Lock()


def digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()


def emit(value):
    with OUTPUT_LOCK:
        print(json.dumps(value, ensure_ascii=False), flush=True)


def make_targets():
    sys.path.insert(0, str(REPO))
    from diagnostics.audit_bbc_structure_provenance import sentence_trees, split_tree_stream
    from tokenizers import Tokenizer
    path = REPO / 'dataset/bbc-news/tree/test.npy'
    docs = split_tree_stream(path)
    tok = Tokenizer.from_file(str(REPO / 'dataset/bbc-news/TG_GPT2_tokenizer.json'))
    targets = []
    for d in DOCS:
        trees = sentence_trees(docs[d])
        core = [t for tree in trees for t in tree['leaves']]
        full = [int(t) for t in docs[d] if not 50268 <= t <= 50319]
        targets.append({'current_doc': d, 'anchor': ANCHORS[d], 'full_ids': full,
                        'core_ids': core, 'full_sha256': digest(full),
                        'core_sha256': digest(core), 'sentences': len(trees),
                        'preview': tok.decode(core[:60])})
    emit({'schema': 1, 'hash_encoding': 'little-endian uint16', 'source': str(path),
          'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'tokenizer_json': tok.to_str(), 'targets': targets})


def arrays(manifest):
    targets = manifest['targets']
    for split in ('test', 'dev', 'train'):
        path = REPO / f'dataset/bbc-news/terminal/{split}.npy'
        stat = path.stat()
        arr = np.load(path, mmap_mode='r')
        starts = []
        data_hash = hashlib.sha256()
        for begin in range(0, len(arr), 8_000_000):
            block = arr[begin:begin+8_000_000]
            data_hash.update(block.tobytes())
            starts.append(np.flatnonzero(block == 50257) + begin)
        starts = np.concatenate(starts)
        assert starts[0] == 0
        ends = np.concatenate((starts[1:], [len(arr)]))
        lengths = ends - starts
        hits, near, hashed = [], [], 0
        for target in targets:
            expected = target['full_ids']
            candidates = np.flatnonzero(lengths == len(expected))
            for d in candidates:
                ids = arr[starts[d]:ends[d]]
                hashed += 1
                if digest(ids) == target['full_sha256']:
                    assert ids.tolist() == expected
                    hits.append({'current_doc': target['current_doc'], 'split_doc': int(d),
                                 'token_offset': int(starts[d]), 'tokens': len(ids),
                                 'full_sha256': digest(ids), 'tokenwise_equal': True})
            # Separately expose prefix matches even if whitespace/content differs.
            possible = np.flatnonzero(lengths >= 12)
            for offset, token in enumerate(expected[:12]):
                possible = possible[arr[starts[possible]+offset] == token]
            for d in possible:
                ids = arr[starts[d]:ends[d]]
                near.append({'current_doc': target['current_doc'], 'split_doc': int(d),
                             'tokens': len(ids), 'full_sha256': digest(ids),
                             'full_ids': ids.tolist()})
        assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
        emit({'type': 'array', 'path': str(path), 'split': split, 'documents': len(starts),
              'tokens': len(arr), 'data_sha256': data_hash.hexdigest(),
              'length_compatible_hash_comparisons': hashed, 'hits': hits, 'prefix_matches': near})
        print(f'{split}: {len(starts)} docs, {len(hits)} exact full-terminal hits', file=sys.stderr, flush=True)
    emit({'type': 'complete'})


def parsed(manifest, workers, shards=None):
    from nltk import Tree
    from tokenizers import Tokenizer
    config = json.loads(manifest['tokenizer_json'])
    for token in config['added_tokens']:
        if 50262 <= token['id'] <= 50267:
            token.update(single_word=True, lstrip=True, normalized=True)
    tok = Tokenizer.from_str(json.dumps(config))
    vocab = tok.get_vocab()
    targets = manifest['targets']
    anchors = [t['anchor'].encode() for t in targets]
    root = REPO / 'dataset/bbc-news-parsed-raw'
    test = json.loads((REPO / 'dataset/bbc-news/test_index.json').read_text())
    dev = json.loads((REPO / 'dataset/bbc-news/dev_index.json').read_text())
    files = sorted(root.glob('*.txt'), key=lambda p: (p.stem != 'CC-MAIN-2015-06', p.name))
    if shards:
        wanted = set(shards.split(','))
        assert wanted <= {p.stem for p in files}, 'unknown requested shard'
        files = [p for p in files if p.stem in wanted]
    emit({'type': 'header', 'scope': [str(p) for p in files],
          'bytes': sum(p.stat().st_size for p in files), 'targets': targets,
          'prefilter': 'Necessary literal word anchor in full parsed row; no prefix truncation. '
                       'Full core and full terminal hashes computed after parsing candidate rows.'})

    def scan(path):
        stat = path.stat()
        count, parsed_count, hits, near = 0, 0, [], []
        raw_hash = hashlib.sha256()
        for_test, for_dev = set(test.get(path.stem, [])), set(dev.get(path.stem, []))
        with path.open('rb') as handle:
            for row, raw in enumerate(handle):
                count += 1
                raw_hash.update(raw)
                candidates = [t for t, anchor in zip(targets, anchors) if anchor in raw]
                if not candidates:
                    continue
                # Fast lossless leaf extraction rejects unrelated articles before NLTK.
                text = raw.decode('utf-8')
                leaves = re.findall(r'\([^()\s]+\s+([^()\s]+)\)', text)
                first = tok.encode(''.join(' ' + x for x in leaves[:25] if x != 'Ċ')).ids
                candidates = [t for t in candidates if first[:12] == t['core_ids'][:12]]
                if not candidates:
                    continue
                parsed_count += 1
                doc = Tree.fromstring('(DOCROOT ' + text.strip() + ')')

                def render(node):
                    if isinstance(node[0], str):
                        assert len(node) == 1
                        return ' ' + ('\n' if node[0] == 'Ċ' else node[0])
                    content = ''.join(render(child) for child in node)
                    label = node.label()
                    if label == 'DOCROOT':
                        return content
                    if f'<({label}>' in vocab and f'<{label})>' in vocab:
                        return f'<({label}>{content}<{label})>'
                    return f' ({label}{content} {label})'

                tree_ids = tok.encode(render(doc)).ids
                core_ids, depth, sentence_count = [], 0, 0
                for token in tree_ids:
                    if 50268 <= token <= 50293:
                        sentence_count += int(depth == 0)
                        depth += 1
                    elif 50294 <= token <= 50319:
                        depth -= 1
                        assert depth >= 0
                    elif depth:
                        core_ids.append(token)
                assert depth == 0
                full_ids = [50257] + [x for x in tree_ids if not 50268 <= x <= 50319] + [50256]
                item = {'shard': path.stem, 'source_row': row,
                        'split_by_released_index': 'test' if row in for_test else 'dev' if row in for_dev else 'train',
                        'raw_sha256': hashlib.sha256(raw).hexdigest(), 'sentences': sentence_count,
                        'core_tokens': len(core_ids), 'full_tokens': len(full_ids),
                        'core_sha256': digest(core_ids), 'full_sha256': digest(full_ids)}
                for target in candidates:
                    match = dict(item, current_doc=target['current_doc'])
                    match['core_equal'] = item['core_sha256'] == target['core_sha256']
                    match['full_equal'] = item['full_sha256'] == target['full_sha256']
                    if match['core_equal']:
                        assert core_ids == target['core_ids']
                        match['tokenwise_equal'] = True
                        match['parsed'] = text
                        hits.append(match)
                        print('EXACT ' + json.dumps({k: v for k, v in match.items() if k != 'parsed'}), file=sys.stderr, flush=True)
                    else:
                        match['core_ids'] = core_ids
                        near.append(match)
        assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
        result = {'type': 'shard', 'path': str(path), 'rows_scanned': count,
                  'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                  'raw_file_sha256': raw_hash.hexdigest(), 'fully_parsed_candidates': parsed_count,
                  'hits': hits, 'prefix_matches_with_core_difference': near}
        emit(result)
        print(f'scanned {path.name}: {count} rows, {len(hits)} core hash hits', file=sys.stderr, flush=True)
        return count

    with ThreadPoolExecutor(max_workers=workers) as pool:
        total = sum(pool.map(scan, files))
    emit({'type': 'complete', 'shards': len(files), 'rows_scanned': total})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('make-targets', 'arrays', 'parsed'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--shards', help='comma-separated parsed shard names; default scans all')
    args = parser.parse_args()
    if args.mode == 'make-targets':
        make_targets()
    else:
        manifest = json.load(sys.stdin)
        (arrays(manifest) if args.mode == 'arrays' else parsed(manifest, args.workers, args.shards))
