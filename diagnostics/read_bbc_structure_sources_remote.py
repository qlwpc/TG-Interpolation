"""Read-only RTX3090 evidence extraction; request JSON is supplied on stdin."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

repo = Path('/home/wangpch/TG-Interpolation')
requests = json.load(sys.stdin)
tokenizer = Tokenizer.from_file(str(repo / 'datatools/TG_GPT2_tokenizer.json'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def core(block):
    opening = np.flatnonzero((block >= 50268) & (block <= 50293))
    closing = np.flatnonzero((block >= 50294) & (block <= 50319))
    return block[int(opening[0]):int(closing[-1])+1].tolist() if len(opening) and len(closing) else block.tolist()


def corpus(data_path, sent_path, doc_path):
    data = np.load(data_path, mmap_mode='r')
    lengths = np.load(sent_path, mmap_mode='r').reshape(-1, 300)
    docs = np.load(doc_path)
    offsets = np.concatenate(([0], np.cumsum(lengths.sum(axis=1, dtype=np.int64))))
    boundaries = np.concatenate(([0], np.cumsum(docs, dtype=np.int64)))
    assert offsets[-1] == len(data) and boundaries[-1] == len(lengths)
    return data, lengths, offsets, boundaries


deprecated_root = repo / 'dataset/testppl_tree_deprecated'
data, lens, offsets, docs = corpus(deprecated_root / 'tree_300.npy',
    deprecated_root / 'tree_sent_index.npy', deprecated_root / 'tree_doc_index.npy')
output = {'deprecated': {}, 'shards': {}, 'cached_raw': []}
for request in requests:
    d = request['current_doc']
    output['deprecated'][str(d)] = [core(data[int(offsets[s]):int(offsets[s])+int(lens[s,0])])
                                    for s in range(int(docs[d]), int(docs[d+1]))]
output['deprecated_data_sha256'] = sha(deprecated_root / 'tree_300.npy')
for shard in sorted({r['shard'] for r in requests}):
    root = repo / 'dataset/bbc-news/test300'
    data_path = root / f'tree_300_{shard}.npy'
    sent_path, doc_path = root / f'sent_index_{shard}.npy', root / f'doc_index_{shard}.npy'
    data, lens, offsets, docs = corpus(data_path, sent_path, doc_path)
    subset = [r for r in requests if r['shard'] == shard]
    wanted_lines, records, ranks = {}, {}, []
    for request in subset:
        d, doc_id = request['shard_doc'], request['current_doc']
        records[str(doc_id)] = []
        for local_s, s in enumerate(range(int(docs[d]), int(docs[d+1]))):
            start = int(offsets[s])
            records[str(doc_id)].append(core(data[start:start+int(lens[s,0])]))
            wanted_lines[s*300] = (doc_id, local_s)
        for changed in request.get('changes', []):
            s = int(docs[d]) + changed['sentence_in_doc']
            start = int(offsets[s]); matches = []
            for candidate in range(300):
                end = start + int(lens[s,candidate])
                if core(data[start:end]) == changed['old_token_ids']:
                    matches.append(candidate)
                start = end
            ranks.append({'current_doc': doc_id, 'sentence_in_doc': changed['sentence_in_doc'],
                          'old_tree_candidate_ids': matches})
    text_records = {str(r['current_doc']): [None] * len(records[str(r['current_doc'])]) for r in subset}
    source_path = root / f'{shard}.txt'
    with source_path.open() as handle:
        for line_no, line in enumerate(handle):
            if line_no in wanted_lines:
                d, s = wanted_lines[line_no]
                ids = np.asarray(tokenizer.encode(line.strip()).ids, dtype=np.uint16)
                ids[ids == 50261] = 198
                text_records[str(d)][s] = core(ids)
    assert all(all(x is not None for x in rows) for rows in text_records.values())
    output['shards'][shard] = {'npy_candidate_zero': records, 'txt_candidate_zero': text_records,
        'old_tree_ranks': ranks, 'source_txt': str(source_path), 'source_txt_sha256': sha(source_path),
        'source_npy_sha256': sha(data_path), 'source_doc_counts': np.diff(docs).tolist()}
    print(f'read original source {shard}', file=sys.stderr, flush=True)

# Read cached raw rows for the four article mismatches, without a Hub request.
cache = Path('/home/wangpch/.cache/huggingface/datasets/permutans___fineweb-bbc-news/CC-MAIN-2015-06')
output['cache_arrow_files'] = [str(p) for p in cache.glob('**/*.arrow')]
try:
    from datasets import Dataset, concatenate_datasets
    for directory in sorted({p.parent for p in cache.glob('**/*train*.arrow')}):
        paths = sorted(directory.glob('*train*.arrow'))
        dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
        entries = []
        for request in requests:
            if request['current_doc'] not in [1063,1090,1121,1122]:
                continue
            i = request['source_row']
            row = dataset[i]
            entries.append({'current_doc': request['current_doc'], 'source_row': i,
                            'text': row['text'], 'id': row.get('id'), 'url': row.get('url')})
        output['cached_raw'].append({'directory': str(directory), 'rows': len(dataset), 'entries': entries})
except Exception as exc:
    output['cache_read_error'] = repr(exc)
print(json.dumps(output))
