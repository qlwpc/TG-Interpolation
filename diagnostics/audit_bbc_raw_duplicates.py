"""Full raw BBC duplicate census, with exact legacy terminal serialization.

Remote run is read-only. Exact raw lines are cached with byte verification;
distinct parses are encoded in worker processes, and terminal hash collisions
are checked by full token bytes. Reports are written to stdout only.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sys
import time

REPO = Path('/home/wangpch/TG-Interpolation')
TOKENIZER = None
VOCAB = None
TREE_EVENTS = re.compile(r'\(([^()\s]+)\s+([^()\s]+)\)|\(([^()\s]+)|\)')


def legacy_format(parsed, vocab):
    from nltk import Tree
    tree = Tree.fromstring('(qlwpcRegen ' + parsed.strip() + ')')
    def render(node):
        if not isinstance(node, Tree):
            raise ValueError('non-tree parsed node')
        # Historical pformat_flat returns immediately at its first string child.
        if any(isinstance(child, str) for child in node):
            leaf = next(child for child in node if isinstance(child, str))
            return ' ' + ('\n' if leaf == 'Ċ' else leaf)
        content = ''.join(render(child) for child in node)
        label = node.label()
        if label == 'qlwpcRegen':
            return content
        if f'<({label}>' in vocab and f'<{label})>' in vocab:
            return f'<({label}>{content}<{label})>'
        return f' ({label}{content if node else " "} {label})'
    return render(tree)


def fast_legacy_format(parsed, vocab):
    """Same serialization as legacy_format without constructing NLTK node objects.

The finite event parser accepts ordinary one-leaf preterminals and balanced
constituents. Unusual inputs fall back to the established strict serializer.
"""
    output, stack = [], []
    cursor = 0
    for match in TREE_EVENTS.finditer(parsed):
        if parsed[cursor:match.start()].strip():
            return legacy_format(parsed, vocab)
        cursor = match.end()
        preterminal, leaf, label = match.groups()
        if preterminal is not None:
            output.append(' ' + ('\n' if leaf == 'Ċ' else leaf))
        elif label is not None:
            opening, closing = f'<({label}>', f'<{label})>'
            if opening in vocab and closing in vocab:
                output.append(opening)
                stack.append((closing, len(output), False))
            else:
                output.append(' ('+label)
                stack.append((' '+label+')', len(output), True))
        else:
            if not stack:
                return legacy_format(parsed, vocab)
            closing, after_open, unknown = stack.pop()
            if unknown and len(output) == after_open:
                output.append(' ')
            output.append(closing)
    if stack or parsed[cursor:].strip():
        return legacy_format(parsed, vocab)
    return ''.join(output)


def init_worker(config):
    global TOKENIZER, VOCAB
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    for token in config['added_tokens']:
        if 50262 <= token['id'] <= 50267:
            token.update(single_word=True, lstrip=True, normalized=True)
    TOKENIZER = Tokenizer.from_str(json.dumps(config))
    VOCAB = TOKENIZER.get_vocab()


def encode_batch(raws):
    import numpy as np
    output = []
    for raw in raws:
        try:
            ids = TOKENIZER.encode(fast_legacy_format(raw.decode('utf-8'), VOCAB)).ids
            terminal = [50257] + [x for x in ids if not 50268 <= x <= 50319] + [50256]
            output.append(np.asarray(terminal, dtype='<u2').tobytes())
        except Exception as exc:
            raise ValueError(f'encoding failed for raw sha256={hashlib.sha256(raw).hexdigest()}: {exc}') from exc
    return output


def make_manifest():
    folder = REPO / 'datatools/parse_pretrain_data'
    counts = json.loads((REPO / 'diagnostics/data/bbc_historical_shard_counts.json').read_text())['counts']
    order = [s.strip() for s in (folder / 'bbc_configs.txt').read_text().splitlines()
             if s.strip() and not s.startswith('#')]
    return {'order': order, 'expected_rows': counts,
            'dev': json.loads((REPO / 'dataset/bbc-news/dev_index.json').read_text()),
            'test': json.loads((REPO / 'dataset/bbc-news/test_index.json').read_text()),
            'tokenizer': json.loads((REPO / 'dataset/bbc-news/TG_GPT2_tokenizer.json').read_text()),
            'train_control': json.loads((REPO / 'artifacts/bbc_all_test_train_overlap_20260906/summary.json').read_text())}


def summarize_groups(groups, total, index, mask_filter=None):
    # group layout: token bytes, all count, historic count, reserved count,
    # train count, dev count, test count, omitted-tail count, shard bitset, first location.
    hist = Counter()
    cross_groups = cross_docs = 0
    for group in groups.values():
        count = group[index]
        if not count:
            continue
        hist[count] += 1
        mask = group[8] if mask_filter is None else group[8] & mask_filter
        if mask.bit_count() > 1:
            cross_groups += 1
            cross_docs += count
    unique = sum(hist.values())
    assert sum(k*v for k, v in hist.items()) == total
    return {'documents': total, 'unique_terminal_documents': unique,
            'extra_duplicate_copies': total-unique,
            'extra_duplicate_percent': 100*(total-unique)/total if total else 0,
            'duplicate_groups': unique-hist[1], 'singleton_documents': hist[1],
            'documents_in_duplicate_groups': total-hist[1],
            'documents_in_duplicate_groups_percent': 100*(total-hist[1])/total if total else 0,
            'groups_present_in_multiple_shards': cross_groups,
            'documents_in_cross_shard_groups': cross_docs,
            'copies_per_unique_document_histogram': dict(sorted(hist.items())),
            'maximum_copies': max(hist, default=0)}


def run(manifest, workers, batch_rows, pilot_rows):
    import numpy as np
    order = manifest['order']
    assert len(order) == len(set(order)) == 94
    dev, test = manifest['dev'], manifest['test']
    assert list(test) == [s for s in order if s in test] and set(dev) == set(test)
    paths = [REPO / f'dataset/bbc-news-parsed-raw/{s}.txt' for s in order]
    assert all(p.is_file() for p in paths)
    # Cache retains exact raw bytes to verify raw-hash hits without probabilistic equality.
    cache, groups, held_tokens, sources = {}, {}, {}, []
    totals = [0]*8
    raw_unique_bytes = token_unique_bytes = 0
    train_hash = hashlib.sha256()
    train_tokens = 0
    historical_mask = sum(1 << i for i,s in enumerate(order) if s in test)
    reserved_mask = ((1 << len(order))-1) ^ historical_mask
    started = last_progress = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                             initializer=init_worker, initargs=(manifest['tokenizer'],)) as pool:
        for shard_id, path in enumerate(paths):
            shard = path.stem
            stat = path.stat()
            held_dev, held_test = set(dev.get(shard, [])), set(test.get(shard, []))
            held = held_dev | held_test
            assert not held_dev.intersection(held_test)
            final_held = max(held, default=-1)
            rows = unique_in_shard = cached_rows = 0
            raw_sha = hashlib.sha256()
            bit = 1 << shard_id
            with path.open('rb') as handle:
                while True:
                    batch = []
                    for _ in range(batch_rows):
                        if pilot_rows and rows+len(batch) >= pilot_rows:
                            break
                        raw = handle.readline()
                        if not raw:
                            break
                        assert raw.endswith(b'\n'), (shard, rows+len(batch), 'unterminated row')
                        raw_sha.update(raw)
                        batch.append(raw)
                    if not batch:
                        break
                    keys, pending = [], {}
                    for raw in batch:
                        key = hashlib.sha256(raw).digest()
                        keys.append(key)
                        if key in cache:
                            assert cache[key][0] == raw, 'raw SHA collision'
                            cached_rows += 1
                        elif key in pending:
                            assert pending[key] == raw, 'batch raw SHA collision'
                            cached_rows += 1
                        else:
                            pending[key] = raw
                    pending_items = list(pending.items())
                    jobs = [[raw for _,raw in pending_items[i:i+32]] for i in range(0,len(pending_items),32)]
                    encoded = (blob for result in pool.map(encode_batch, jobs) for blob in result)
                    for (key, raw), blob in zip(pending_items, encoded):
                        terminal_key = hashlib.sha256(blob).digest()
                        if terminal_key in groups:
                            assert groups[terminal_key][0] == blob, 'terminal SHA collision'
                        else:
                            groups[terminal_key] = [blob, 0, 0, 0, 0, 0, 0, 0, 0, None]
                            token_unique_bytes += len(blob)
                        cache[key] = (raw, terminal_key)
                        raw_unique_bytes += len(raw)
                    for local, key in enumerate(keys):
                        row = rows+local
                        terminal_key = cache[key][1]
                        group = groups[terminal_key]
                        if not group[8] & bit:
                            unique_in_shard += 1
                            group[8] |= bit
                        if group[9] is None:
                            group[9] = [shard, row]
                        group[1] += 1; totals[1] += 1
                        if shard not in test:
                            category = 3
                        else:
                            group[2] += 1; totals[2] += 1
                            if row in held_dev:
                                category = 5
                            elif row in held_test:
                                category = 6
                            elif row < final_held:
                                category = 4
                            else:
                                category = 7
                        group[category] += 1; totals[category] += 1
                        if category == 4:
                            train_hash.update(group[0]); train_tokens += len(group[0])//2
                        elif category in (5,6):
                            held_tokens[shard, row] = group[0]
                    rows += len(batch)
                    now = time.monotonic()
                    if now-last_progress >= 30:
                        print(json.dumps({'type': 'progress', 'shard': shard, 'shard_rows': rows,
                            'total_rows': totals[1], 'unique_raw_parses': len(cache),
                            'unique_terminal_documents': len(groups),
                            'cache_raw_GiB': round(raw_unique_bytes/2**30,2),
                            'elapsed_seconds': round(now-started,1)}), file=sys.stderr, flush=True)
                        last_progress = now
            assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
            if not pilot_rows:
                assert rows == manifest['expected_rows'][shard], (shard, rows)
            sources.append({'shard': shard, 'rows': rows, 'unique_within_shard': unique_in_shard,
                            'within_shard_extra_copies': rows-unique_in_shard, 'exact_raw_cache_hits': cached_rows,
                            'file_sha256': raw_sha.hexdigest(), 'file_size': stat.st_size,
                            'mtime_ns': stat.st_mtime_ns, 'complete_file': not bool(pilot_rows)})
            print(json.dumps({'type': 'shard_complete', **sources[-1]}), file=sys.stderr, flush=True)
    validation = {'raw_and_terminal_hash_hits_byte_verified': True}
    if not pilot_rows:
        control = manifest['train_control']
        assert totals[4] == control['train_documents']
        validation['full_raw_to_historical_train_data_hash_equal'] = train_hash.hexdigest() == control['train_data_sha256']
        validation['full_raw_to_historical_train_token_count_equal'] = train_tokens == control['train_tokens']
        for name, index in [('dev',dev), ('test',test)]:
            rebuilt = hashlib.sha256()
            for shard, selected in index.items():
                for row in selected:
                    rebuilt.update(held_tokens[shard,row])
            arr = np.load(REPO / f'dataset/bbc-news/terminal/{name}.npy')
            actual = hashlib.sha256(arr.tobytes()).hexdigest()
            validation[f'full_raw_to_historical_{name}_data_hash_equal'] = actual == rebuilt.hexdigest()
    profiles = {name: summarize_groups(groups, totals[i], i, mask) for name,i,mask in [
        ('all_94_shards',1,None), ('historical_89_shard_pool',2,historical_mask),
        ('reserved_5_shards',3,reserved_mask)]}
    # Train counts by group are exact, but its cross-shard mask is not tracked separately.
    train_hist = Counter(g[4] for g in groups.values() if g[4])
    reserved_overlap = {
        'reserved_document_instances_with_historical_pool_match': sum(g[3] for g in groups.values() if g[2]),
        'reserved_unique_contents_with_historical_pool_match': sum(g[3]>0 and g[2]>0 for g in groups.values()),
        'reserved_document_instances_with_actual_train_match': sum(g[3] for g in groups.values() if g[4]),
        'reserved_unique_contents_with_actual_train_match': sum(g[3]>0 and g[4]>0 for g in groups.values()),
        'reserved_document_instances_without_historical_pool_match': sum(g[3] for g in groups.values() if not g[2]),
        'reserved_unique_contents_without_historical_pool_match': sum(g[3]>0 and not g[2] for g in groups.values())}
    top = sorted(groups.items(), key=lambda item:item[1][1], reverse=True)[:20]
    return {'claim_type': 'computed', 'complete': not bool(pilot_rows), 'pilot_rows_per_shard': pilot_rows,
            'definition': 'Exact full legacy terminal IDs, including whitespace, BOS/EOS; SHA-256 little-endian uint16.',
            'profiles': profiles, 'reserved_overlap': reserved_overlap,
            'actual_train_unique_contents': sum(train_hist.values()),
            'actual_train_extra_copies': totals[4]-sum(train_hist.values()),
            'actual_train_copies_histogram': dict(sorted(train_hist.items())),
            'historical_membership_counts': dict(zip(['train','dev','test','omitted_tail'],totals[4:8])),
            'reconstructed_train_data_sha256': train_hash.hexdigest(),
            'unique_raw_parse_lines': len(cache), 'raw_cache_bytes': raw_unique_bytes,
            'unique_terminal_bytes': token_unique_bytes, 'validation': validation,
            'sources': sources, 'elapsed_seconds': round(time.monotonic()-started,1),
            'top_repeated_contents': [{'sha256': key.hex(), 'copies': g[1], 'tokens': len(g[0])//2,
                'shards': [s for i,s in enumerate(order) if g[8] & (1<<i)], 'first_source': g[9],
                'preview_token_ids': np.frombuffer(g[0],dtype='<u2')[:80].tolist()} for key,g in top]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('manifest','run'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-rows', type=int, default=2048)
    parser.add_argument('--pilot-rows', type=int, default=0)
    args = parser.parse_args()
    result = make_manifest() if args.mode == 'manifest' else run(json.load(sys.stdin),args.workers,args.batch_rows,args.pilot_rows)
    print(json.dumps(result), flush=True)
