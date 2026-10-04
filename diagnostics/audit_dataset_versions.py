#!/usr/bin/env python3
"""Read-only, bounded BBC version diagnostics; stdout is the JSON receipt.

Requires only NumPy. Small dev/test streams are checked in full; candidate
indexes are summed in full but candidate bodies and training caches are sampled.
This is not a full training/content/contamination audit or deletion authority.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1. No dataset files are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


def digest(a):
    """Hash canonical int64 values (not NPY file bytes), across integer dtypes."""
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()


class Audit:
    def __init__(self, root, samples):
        self.root, self.samples, self.identities = root.resolve(), samples, {}
        tok = json.loads(self.read(root / 'bbc-news/TG_GPT2_tokenizer.json'))
        vocab = {**tok['model']['vocab'], **{t['content']: t['id'] for t in tok['added_tokens']}}
        self.opens = {i: s[2:-1] for s, i in vocab.items() if re.fullmatch(r'<\([A-Za-z0-9]+>', s)}
        self.closes = {i: s[1:-2] for s, i in vocab.items() if re.fullmatch(r'<[A-Za-z0-9]+\)>', s)}
        self.open_ids, self.close_ids = list(self.opens), list(self.closes)
        self.structure = self.open_ids + self.close_ids
        self.bos, self.eos, self.pad = vocab['<|beginoftext|>'], vocab['<|endoftext|>'], vocab['<|pad|>']
        self.valid_ids = list(set(vocab.values()))

    @staticmethod
    def signature(p):
        s = p.stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

    def track(self, p):
        self.identities.setdefault(str(p), self.signature(p))

    def read(self, p):
        self.track(p)
        return p.read_bytes()

    def array(self, p):
        self.track(p)
        return np.load(p, mmap_mode='r', allow_pickle=False)

    def terminal(self, a):
        return a[~np.isin(a, self.structure)]

    def tg(self, a):
        return np.repeat(a, 1 + np.isin(a, self.close_ids))

    def balanced(self, a):
        stack = []
        roots = 0
        for t in a[np.isin(a, self.structure)]:
            t = int(t)
            if t in self.opens:
                roots += not stack
                stack.append(self.opens[t])
            elif not stack or stack.pop() != self.closes[t]:
                return False, roots
        return not stack, roots

    def stream(self, rel):
        a = self.array(self.root / rel)
        starts = np.flatnonzero(a == self.bos)
        ends = np.flatnonzero(a == self.eos)
        framed = bool(len(starts) and len(starts) == len(ends) and starts[0] == 0
                      and ends[-1] == len(a)-1 and np.all(ends >= starts)
                      and np.array_equal(starts[1:], ends[:-1]+1))
        result = dict(path=rel, tokens=a.size, dtype=str(a.dtype), documents=len(starts),
                      framed=framed, pad=int(np.count_nonzero(a == self.pad)),
                      invalid_ids=int(np.count_nonzero(~np.isin(a, self.valid_ids))),
                      values_sha256=digest(a), terminal_values_sha256=digest(self.terminal(a)))
        if '/tree/' in rel or rel.startswith('tree/'):
            ok, roots = self.balanced(a)
            result.update(label_balanced=ok, roots=roots, tg_values_sha256=digest(self.tg(a)))
        return result

    def candidates(self, directory, suffix=''):
        p = self.root / directory
        tokens = self.array(p / f'tree_300{suffix}.npy')
        lengths = self.array(p / f'tree_sent_index{suffix}.npy').reshape(-1, 300)
        docs = self.array(p / 'tree_doc_index.npy')
        row_sizes = lengths.sum(axis=1, dtype=np.int64)
        offsets = np.r_[0, row_sizes.cumsum()]
        doc_ends = docs.cumsum(dtype=np.int64)
        doc_starts = set(np.r_[0, doc_ends[:-1]].tolist())
        last_sentences = set((doc_ends-1).tolist())
        indices = sorted(set(range(min(12, len(lengths)))) |
                         set(np.linspace(0, len(lengths)-1, self.samples, dtype=int).tolist()))
        out = dict(path=directory, suffix=suffix, documents=len(docs), sentences=len(lengths),
                   indexes_cover_tokens=int(offsets[-1]) == tokens.size,
                   indexes_cover_sentences=int(doc_ends[-1]) == len(lengths),
                   nonpositive_lengths=int(np.count_nonzero(lengths <= 0)),
                   sampled_sentence_ids=indices, sampled_candidates=0,
                   terminal_mismatches=0, invalid_tree_records=0, invalid_id_records=0,
                   unnormalized_records=0, examples=[], candidate0_digest=None)
        h = hashlib.sha256()
        for s in indices:
            off = int(offsets[s])
            row = tokens[off:int(offsets[s+1])]
            ends = np.r_[0, lengths[s].cumsum(dtype=np.int64)]
            ref = self.terminal(row[:int(ends[1])])
            for k in range(300):
                rec = row[int(ends[k]):int(ends[k+1])]
                term = self.terminal(rec)
                balanced, roots = self.balanced(rec)
                checks = dict(terminal_mismatches=not np.array_equal(term, ref),
                              invalid_tree_records=not balanced or roots != 1,
                              invalid_id_records=not np.all(np.isin(rec, self.valid_ids)),
                              unnormalized_records=(int(np.count_nonzero(rec == self.bos)) != int(s in doc_starts)
                                or int(np.count_nonzero(rec == self.eos)) != int(s in last_sentences)
                                or bool(np.any(rec == self.pad))
                                or (s in doc_starts and rec[0] != self.bos)
                                or (s in last_sentences and rec[-1] != self.eos)))
                for key, bad in checks.items():
                    out[key] += int(bad)
                if any(checks.values()) and len(out['examples']) < 8:
                    out['examples'].append(dict(sentence=s, candidate=k,
                                                failures=[x for x, bad in checks.items() if bad]))
                if k == 0:
                    h.update(np.asarray(rec, dtype='<i8').tobytes())
                out['sampled_candidates'] += 1
        out['candidate0_digest'] = h.hexdigest()
        return out

    def precomputed(self, directory):
        files = {p.stem: self.array(p) for p in directory.glob('*.npy')}
        ids, spans, counts = (files[k] for k in ['input_ids', 'spans', 'span_counts'])
        idx = sorted(set(np.linspace(0, len(ids)-1, 16, dtype=int).tolist()))
        out = dict(path=str(directory.relative_to(self.root)), chunks=len(ids),
                   shapes={k: list(v.shape) for k,v in files.items()},
                   dtypes={k: str(v.dtype) for k,v in files.items()},
                   same_row_counts=all(len(v) == len(ids) for v in files.values()),
                   sampled_chunk_ids=idx, bad_span_rows=0, bad_token_rows=0,
                   invalid_counts=0, input_values_sha256=digest(ids[idx]),
                   span_values_sha256=digest(spans[idx]))
        for i in idx:
            count = int(counts[i])
            if count < 0 or count > spans.shape[1]:
                out['invalid_counts'] += 1
                continue
            a = spans[i, :count]
            n = int(np.count_nonzero(ids[i] != self.pad))
            if len(a) and not np.all((a[:,0] >= 0) & (a[:,0] <= a[:,1]) &
                                     (a[:,1] <= a[:,2]) & (a[:,2] < n)):
                out['bad_span_rows'] += 1
            out['bad_token_rows'] += int(not np.all(np.isin(ids[i], self.valid_ids)))
        return out

    def run(self):
        results = dict(streams=[], candidates=[], precomputed=[], errors=[])
        def check(section, name, fn):
            try:
                results[section].append(fn())
            except Exception as e:
                results['errors'].append(dict(section=section, path=str(name), error=f'{type(e).__name__}: {e}'))
        for corpus in ['bbc-news', 'bbc-news-reserved-clean-v1', 'bbc-news-parsed-dedup']:
            for fmt in ['terminal', 'tree', 'tg', 'tree_noont', 'tree_compress', 'tree_triplecnt']:
                for split in ['dev', 'test', 'test.pre_testppl_alignment', 'test.updated_20260822']:
                    rel = f'{corpus}/{fmt}/{split}.npy'
                    p = self.root / rel
                    if p.is_file() and 0 < p.stat().st_size <= 64*2**20:
                        check('streams', rel, lambda: self.stream(rel))
        for directory in ['bbc-news/testppl_tree', 'bbc-news/testppl_tree_updated_20260822',
                          'testppl_tree_deprecated', 'bbc-news-reserved-clean-v1/testppl/tree300']:
            for suffix in ['', '.pre_bos_eos_normalization']:
                if (self.root/directory/f'tree_300{suffix}.npy').is_file():
                    check('candidates', directory, lambda: self.candidates(directory, suffix))
        for directory in sorted((self.root/'bbc-news/parse_aligned').glob('*')):
            if (directory/'input_ids.npy').is_file():
                check('precomputed', directory, lambda: self.precomputed(directory))
        # Only this small unresolved cross-machine file gets a new full hash.
        p = self.root/'hellaswag/hellaswag_train.txt'
        if p.exists():
            self.track(p)
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda: f.read(1024*1024), b''):
                    h.update(b)
            results['hellaswag_train_sha256'] = h.hexdigest()
        results['read_file_signatures'] = self.identities
        results['changed_during_check'] = [p for p,s in self.identities.items() if self.signature(Path(p)) != s]
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    start = time.time()
    result = Audit(args.root, args.samples).run()
    result.update(host=args.host, elapsed_seconds=time.time()-start,
                  scope='full small streams/index totals; sampled candidate bodies/cache rows',
                  dataset_mutations=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
