"""Exact lazy K-best for the historical tree300 canonical A/B grammar.

A split joins an A forest on the left and a B tree on the right. Non-leaf B
roots and the sentence root must be labeled; A may have a null root. Leaves
may carry a collapsed unary label. Thus erasing null nodes is injective.
Unlike the old autograd KMax implementation, only demanded ranks are expanded.
"""
from __future__ import annotations

import heapq

import numpy as np


class LabeledKBest:
    def __init__(self, scores, label_names=None):
        self.scores = np.asarray(scores, dtype=np.float64)
        n, m, labels = self.scores.shape
        if not n or n != m or labels < 2:
            raise ValueError('expected a nonempty square labeled score chart')
        if not np.isfinite(self.scores[np.triu_indices(n)]).all():
            raise ValueError('nonfinite labeled scores')
        self.n = n
        self.cells = {}
        self.label_cache = {}
        # ADJ->ADJP is the established tokenizer normalization. Multiple source
        # labels with the same emitted chain use their highest scoring source.
        # This ranks unique normalized trees by their best original derivation.
        self.label_names = label_names
        self.groups = {}
        for label in range(labels):
            name = label if label_names is None else '::'.join(
                'ADJP' if s == 'ADJ' else s for s in label_names[label].split('::'))
            self.groups.setdefault(name, []).append(label)

    def labels(self, left, right, real):
        key = left, right, real
        if key in self.label_cache:
            return self.label_cache[key]
        values = []
        for group in self.groups.values():
            allowed = [x for x in group if x != 0 or not real]
            if allowed:
                label = max(allowed, key=lambda x: (self.scores[left, right, x], -x))
                values.append((float(self.scores[left, right, label]), label))
        self.label_cache[key] = sorted(values, key=lambda x: (-x[0], x[1]))
        return self.label_cache[key]

    def get(self, key, rank):
        state, left, right = key
        if key not in self.cells:
            self.cells[key] = ([], [], set())
            if state == 'F':
                for split in range(left, right):
                    self.push(key, (split, 0, 0))
            else:
                self.push(key, (0, 0, 0))
        out, heap, _seen = self.cells[key]
        while len(out) <= rank:
            # Delay successors until a further rank is requested.
            if out:
                _score, point = out[-1]
                a, b, c = point
                if state == 'F':
                    self.push(key, (a, b + 1, c))
                    self.push(key, (a, b, c + 1))
                else:
                    self.push(key, (a + 1, b, c))
                    if left != right:
                        self.push(key, (a, b + 1, c))
            if not heap:
                return None
            neg, point = heapq.heappop(heap)
            out.append((-neg, point))
        return out[rank]

    def push(self, key, point):
        out, heap, seen = self.cells[key]
        if point in seen:
            return
        seen.add(point)
        state, left, right = key
        a, b, c = point
        if state == 'F':
            x = self.get(('A', left, a), b)
            y = self.get(('B', a + 1, right), c)
            if x is None or y is None:
                return
            score = x[0] + y[0]
        else:
            real = state == 'R' or (state == 'B' and left != right)
            labels = self.labels(left, right, real)
            if a >= len(labels):
                return
            score = labels[a][0]
            if left != right:
                forest = self.get(('F', left, right), b)
                if forest is None:
                    return
                score += forest[0]
        heapq.heappush(heap, (-score, point))

    def spans(self, rank):
        result = []

        def visit(key, r):
            state, left, right = key
            _score, (a, b, c) = self.get(key, r)
            if state == 'F':
                visit(('A', left, a), b)
                visit(('B', a + 1, right), c)
            else:
                if left != right:
                    visit(('F', left, right), b)
                real = state == 'R' or (state == 'B' and left != right)
                label = self.labels(left, right, real)[a][1]
                if label:
                    result.append((left, right, label))

        visit(('R', 0, self.n - 1), rank)
        return tuple(result)

    def topk(self, k=300):
        result = []
        for rank in range(k):
            root = self.get(('R', 0, self.n - 1), rank)
            if root is None:
                break
            result.append((root[0], self.spans(rank)))
        return result


def label_token_pairs(label_names, tokenizer):
    pairs = {}
    for index, chain in label_names.items():
        names = [('ADJP' if x == 'ADJ' else x) for x in chain.split('::') if x]
        opens = tuple(tokenizer.token_to_id('<(' + x + '>') for x in names)
        closes = tuple(tokenizer.token_to_id('<' + x + ')>') for x in reversed(names))
        if any(x is None for x in opens + closes):
            raise ValueError(f'unsupported label chain {chain!r}')
        pairs[index] = opens, closes
    return pairs


def serialize_candidate(spans, word_piece_ids, pairs, prefix=(), suffix=()):
    """Insert structure IDs around original BPE leaves, without re-tokenizing."""
    starts, ends = {}, {}
    for left, right, label in spans:
        starts.setdefault(left, []).append((right, label))
        ends.setdefault(right, []).append((left, label))
    output = list(prefix)
    for i, pieces in enumerate(word_piece_ids):
        for _right, label in sorted(starts.get(i, ()), reverse=True):
            output.extend(pairs[label][0])
        output.extend(pieces)
        for _left, label in sorted(ends.get(i, ()), reverse=True):
            output.extend(pairs[label][1])
    output.extend(suffix)
    return np.asarray(output, dtype=np.uint16)
