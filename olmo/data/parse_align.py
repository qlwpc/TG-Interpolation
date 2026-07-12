"""Parse-tree alignment utilities for the no-extra-token syntactic-LM baselines
(Pushdown Layers, Murty et al. 2023; TreeReg, Nandi et al. 2025).

Both baselines train on the *terminal* token sequence plus an aligned
*constituency parse*. The repo stores parses as tokenized "tree" streams
(``dataset/bbc-news/tree/*.npy``) where non-terminal brackets are interleaved as
special tokens::

    [BOS, <(S>, <(NP>, ĠMexico, ', s, <NP)>, ..., <S)>, EOS, BOS, ...]

Each ``[BOS ... EOS]`` block is one complete parse tree (a document). The block's
*leaves* (non-NT, non-BOS tokens, keeping EOS) are exactly the corresponding
``terminal/*.npy`` block — so building from the tree stream is self-contained and
guarantees alignment.

This module is pure-python/numpy (no torch) so it runs in data-loader workers and
in unit tests. The algorithms:

* :class:`TreeVocab` — non-terminal detection by token-id range (HF json format
  ``<(LABEL>`` / ``<LABEL)>``), mirroring the C++ ``SentencepieceVocab`` in
  ``olmo/data/tgmasking/mask.cpp``.
* :func:`parse_tree_block` — stack-parse a ``[BOS, <(S>...<S)>, EOS]`` block into a
  nested ``(label, children)`` tree; BOS/EOS are returned as surrounding leaves.
* :func:`binarize_tree` — left/right binarize a non-binary tree (preserves leaves).
* :func:`tree_spans` — enumerate a binarized tree's constituent spans
  ``(i, split, j)`` (leaf indices, 0-based within the tree's content leaves) and its
  content-leaf token ids.
* :func:`compute_depth_matrix` — the Pushdown "stale stack tape"
  ``S[k,j] = #{constituents (l,r): l<=j<=r and r<=k}`` (int8, lower-triangular),
  computed in O(n^2) via a difference array + cumulative sum.
* :func:`chunk_units` — greedily pack whole-tree units into chunks ``<= max_len``
  terminals (whole-sentence integrity; a unit whose content exceeds ``max_len`` is
  split at its top-level child trees).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

import numpy as np

# A parse tree node: (label, children). Leaves are plain ints (token ids).
TreeNode = Tuple[str, List[Any]]
Leaf = int

# (left_leaf, split_leaf, right_leaf): a constituent spanning leaves [left..right]
# that bifurcates at ``split`` into [left..split] / [split+1..right] (0-based).
Span = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Vocab / non-terminal detection
# --------------------------------------------------------------------------- #
_OPEN_RE = re.compile(r"^<\([A-Za-z0-9]+>$")
_CLOSE_RE = re.compile(r"^<[A-Za-z0-9]+\)>$")


@dataclass
class TreeVocab:
    """Detects non-terminal bracket tokens in the HF tokenizer format.

    Opening NTs are contiguous id range ``[op_lo, op_hi]`` (tokens ``<(LABEL>``);
    closing NTs are ``[cl_lo, cl_hi]`` (tokens ``<LABEL)>``). The two ranges are
    paired label-for-label (opening label X <-> closing label X), matching the
    GPT-2-style ``TG_GPT2_tokenizer.json`` used by this repo.
    """

    op_lo: int
    op_hi: int
    cl_lo: int
    cl_hi: int
    bos: int = 50257
    eos: int = 50256
    pad: int = 50258
    id2tok: Dict[int, str] = field(default_factory=dict)
    _tokenizer_path: Optional[str] = None  # set by from_tokenizer_file; used to
    # rebuild a light vocab in multiprocessing workers without pickling id2tok.

    @classmethod
    def from_tokenizer_file(cls, path: str) -> "TreeVocab":
        with open(path) as f:
            tok = json.load(f)
        added = {a["content"]: a["id"] for a in tok.get("added_tokens", [])}
        vocab = tok.get("model", {}).get("vocab", {})
        allt: Dict[str, int] = {**vocab, **added}
        op = [v for k, v in allt.items() if _OPEN_RE.match(k)]
        cl = [v for k, v in allt.items() if _CLOSE_RE.match(k)]
        if not op or not cl:
            raise ValueError(
                f"No <(LABEL>/<LABEL)> non-terminal tokens found in {path}; "
                "is this a TG tokenizer?"
            )
        id2tok = {v: k for k, v in allt.items()}
        return cls(
            op_lo=min(op), op_hi=max(op), cl_lo=min(cl), cl_hi=max(cl),
            bos=allt.get("<|beginoftext|>", 50257),
            eos=allt.get("</s>", 50256),
            pad=allt.get("<|pad|>", allt.get("<pad>", 50258)),
            id2tok=id2tok,
            _tokenizer_path=path,
        )

    def is_opening(self, i: int) -> bool:
        return self.op_lo <= i <= self.op_hi

    def is_closing(self, i: int) -> bool:
        return self.cl_lo <= i <= self.cl_hi

    def is_non_terminal(self, i: int) -> bool:
        return self.op_lo <= i <= self.cl_hi

    def label_of_opening(self, i: int) -> str:
        tok = self.id2tok.get(int(i), "")
        return tok[2:-1] if _OPEN_RE.match(tok) else ""


# --------------------------------------------------------------------------- #
# Tree parsing
# --------------------------------------------------------------------------- #
# A segment of a block: either a run of plain leaf tokens (BOS/EOS/whitespace) or
# one complete top-level parse tree (its content leaves + constituent spans).
# A block = [BOS, tree1, ws, tree2, ws, ..., treeK, EOS] -> a list of segments.
Segment = Tuple[str, Any]  # ('leaves', List[int]) | ('tree', (List[int], List[Span]))


def _parse_one_tree(block: Sequence[int], start: int, vocab: TreeVocab) -> Tuple[Optional[TreeNode], int]:
    """Stack-parse the single tree beginning at ``block[start]`` (an opening NT).

    Returns ``(tree, end_plus_one)`` where ``end_plus_one`` is the index just past
    the tree's root-closing token. Returns ``(None, start)`` if ``block[start]`` is
    not an opening NT.
    """
    if not vocab.is_opening(int(block[start])):
        return None, start
    stack: List[List[Any]] = [[vocab.label_of_opening(int(block[start])), []]]
    i = start + 1
    n = len(block)
    while i < n:
        tok = int(block[i])
        if vocab.is_opening(tok):
            stack.append([vocab.label_of_opening(tok), []])
        elif vocab.is_closing(tok):
            node = tuple(stack.pop())
            if not stack:
                return node, i + 1
            stack[-1][1].append(node)
        else:
            stack[-1][1].append(tok)
        i += 1
    # Unterminated root (malformed); keep what we have.
    return tuple(stack[0]) if stack else None, n


def parse_block_segments(block: Sequence[int], vocab: TreeVocab) -> List[Segment]:
    """Split a ``[BOS, tree1, ws, tree2, ..., treeK, EOS]`` block into segments.

    Returns a list of ``('leaves', ids)`` (plain tokens: BOS/EOS/whitespace) and
    ``('tree', (content_leaves, spans))`` (a complete top-level parse tree, with
    spans over 0-based content-leaf indices). Captures ALL top-level trees in the
    block (a document is typically a forest of sentence trees).
    """
    segments: List[Segment] = []
    leaf_run: List[int] = []
    i = 0
    n = len(block)
    while i < n:
        tok = int(block[i])
        if vocab.is_opening(tok):
            if leaf_run:
                segments.append(("leaves", leaf_run))
                leaf_run = []
            tree, end = _parse_one_tree(block, i, vocab)
            if tree is not None:
                from_seg = ("tree", tree)
                segments.append(from_seg)
            i = end
        else:
            leaf_run.append(tok)
            i += 1
    if leaf_run:
        segments.append(("leaves", leaf_run))
    return segments


def parse_tree_block(block: Sequence[int], vocab: TreeVocab) -> Tuple[List[int], Optional[TreeNode], List[int]]:
    """Convenience: parse the FIRST tree in a block (legacy single-tree API).

    Returns ``(prefix_leaves, tree_or_None, suffix_leaves)``. For multi-sentence
    blocks only the first top-level tree is returned; prefer
    :func:`parse_block_segments` to capture the whole forest.
    """
    segments = parse_block_segments(block, vocab)
    prefix: List[int] = []
    tree: Optional[TreeNode] = None
    suffix: List[int] = []
    seen_tree = False
    for seg in segments:
        kind, data = seg
        if kind == "leaves":
            (suffix if seen_tree else prefix).extend(data)
        else:  # tree
            if not seen_tree:
                tree = data
                seen_tree = True
            else:
                suffix.extend(_collect_leaves(data))
    return prefix, tree, suffix


def _collect_leaves(node: Union[TreeNode, Leaf]) -> List[int]:
    if isinstance(node, int):
        return [node]
    _, children = node
    out: List[int] = []
    for c in children:
        out.extend(_collect_leaves(c))
    return out


def count_leaves(node: Union[TreeNode, Leaf]) -> int:
    if isinstance(node, int):
        return 1
    return sum(count_leaves(c) for c in node[1])


# --------------------------------------------------------------------------- #
# Binarization
# --------------------------------------------------------------------------- #
def binarize_tree(node: Union[TreeNode, Leaf], direction: str = "left") -> Union[TreeNode, Leaf]:
    """Left/right-binarize a non-binary constituency tree.

    Leaves (ints) are unchanged. For an internal node with children
    ``[c1, c2, ..., ck]`` (k>2):

    * ``direction="left"``: ``(X c1 (X|< c2 (X|< c3 ...)))`` — right-recursive tail
      (left-branching). The introduced nodes reuse the parent label suffixed with
      ``"|<"`` so they are distinguishable but still phrasal.
    * ``direction="right"``: ``(X (X|> ... (X|> c2 c3) c4) ck)`` — left-recursive tail
      (right-branching).

    Leaves are preserved in order, so the terminal sequence is identical pre/post
    binarization.
    """
    if isinstance(node, int):
        return node
    # Iterative post-order binarization (explicit stack) — the train split has
    # deeply-nested trees that overflow Python's recursion limit. Each internal
    # node is expanded (children pushed), then processed bottom-up once its
    # children are binarized. Results cached by id(node).
    result: Dict[int, Union[TreeNode, Leaf]] = {}
    stack: List[Tuple] = [("node", node)]
    while stack:
        entry = stack.pop()
        node_i = entry[1]
        if entry[0] == "node":
            if isinstance(node_i, int):
                result[id(node_i)] = node_i
                continue
            label, children = node_i
            # Re-push as 'ready' (binarizes after children), then push children.
            stack.append(("ready", node_i))
            for c in reversed(children):
                stack.append(("node", c))
        else:  # 'ready'
            label, children = node_i
            children = [result[id(c)] for c in children]
            if len(children) <= 2:
                result[id(node_i)] = (label, children)
                continue
            if direction == "left":
                tail = children[1]
                for c in children[2:]:
                    tail = (f"{label}|<", [tail, c])
                result[id(node_i)] = (label, [children[0], tail])
            elif direction == "right":
                acc = (f"{label}|>", [children[0], children[1]])
                for c in children[2:-1]:
                    acc = (f"{label}|>", [acc, c])
                result[id(node_i)] = (label, [acc, children[-1]])
            else:
                raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
    return result[id(node)]


# --------------------------------------------------------------------------- #
# Spans + depth matrix
# --------------------------------------------------------------------------- #
def tree_spans(tree: TreeNode) -> Tuple[List[int], List[Span]]:
    """Enumerate a binarized tree's content leaves and constituent spans.

    Returns ``(leaves, spans)`` where ``leaves`` is the in-order list of leaf token
    ids (the terminal content), and ``spans`` is a list of ``(left, split, right)``
    for each *internal* node: the node spans leaves ``[left..right]`` and splits at
    ``split`` (left child = ``[left..split]``, right child = ``[split+1..right]``).
    A unary internal node (single child) has ``split == right`` (degenerate split);
    callers may skip such spans for the TreeReg CE.
    """
    leaves: List[int] = []
    spans: List[Span] = []

    # Iterative post-order traversal (an explicit stack) instead of recursion:
    # the train split has deeply-nested parse trees that exceed Python's default
    # 1000-frame recursion limit (RecursionError at ~97% of a prior run). Each
    # stack entry is either ('node', node) — to be expanded — or ('ready', node,
    # left) — children processed, ready to emit the span. A node's leaf-range
    # (left, right) is stored in `ranges` keyed by id(node) so its parent can
    # read it without recursion.
    ranges: Dict[int, Tuple[int, int]] = {}
    stack: List[Tuple] = [("node", tree)]
    while stack:
        entry = stack.pop()
        if entry[0] == "node":
            node = entry[1]
            if isinstance(node, int):
                idx = len(leaves)
                leaves.append(node)
                ranges[id(node)] = (idx, idx)
                continue
            label, children = node
            left = len(leaves)
            # Re-push as 'ready' (emits span after children), then push children
            # in reverse so they're processed left-to-right.
            stack.append(("ready", node, left))
            for c in reversed(children):
                stack.append(("node", c))
        else:  # 'ready'
            _, node, left = entry
            label, children = node
            child_ranges = [ranges[id(c)] for c in children]
            right = child_ranges[-1][1]
            if len(children) == 2:
                split = child_ranges[0][1]
                spans.append((left, split, right))
            elif len(children) == 1:
                spans.append((left, right, right))  # unary; degenerate split
            elif len(children) > 2:
                # Should not happen after binarize_tree, but stay robust: pick
                # the split after the first child.
                split = child_ranges[0][1]
                spans.append((left, split, right))
            ranges[id(node)] = (left, right)
    return leaves, spans


def compute_depth_matrix(spans: Sequence[Span], n_leaves: int, max_depth: int = 127) -> np.ndarray:
    """Pushdown "stale stack tape" ``S[k,j]`` from constituent spans.

    ``S[k,j] = #{constituents (l,r): l<=j<=r and r<=k}`` for ``j<=k`` (else 0),
    i.e. the number of constituents that contain leaf ``j`` and have already closed
    by prefix ``k``. This is exactly the depth of token ``j`` in the incremental
    parse as of prefix ``k`` (the number of reduce operations ``j`` has participated
    in), matching Murty et al. 2023 §3.1 (the depth of each prefix token tracked on
    the stack tape, which grows stale as later reduces fire).

    A constituent ``(l,_,r)`` contributes ``+1`` to the rectangle ``[r:n, l:r+1]`` of
    ``S`` (rows ``k>=r``, cols ``j in [l,r]``). This is a 2D range update, done in
    O(#spans) via two :func:`np.add.at` difference markers per span (``+1`` at
    ``(r,l)``, ``-1`` at ``(r, r+1)``) followed by two cumulative sums — no Python
    loop over spans. Output is ``int8`` ``(n_leaves, n_leaves)``, lower-triangular,
    clipped to ``max_depth``.
    """
    n = n_leaves
    if n == 0:
        return np.zeros((0, 0), dtype=np.int8)
    spans = np.asarray(spans, dtype=np.int64)
    if spans.size == 0:
        return np.zeros((n, n), dtype=np.int8)
    l = spans[:, 0]
    r = spans[:, 2]
    # Clamp to valid range (defensive).
    r = np.clip(r, 0, n - 1)
    l = np.clip(l, 0, r)
    D = np.zeros((n, n), dtype=np.int32)
    # +1 at (r, l): opens the column range [l, ...] starting at row r.
    np.add.at(D, (r, l), 1)
    # -1 at (r, r+1): closes the column range after r (skip when r+1 == n).
    r1 = r + 1
    valid = r1 < n
    if np.any(valid):
        np.add.at(D, (r[valid], r1[valid]), -1)
    # 2D prefix sum: propagate row starts downward, then column ranges rightward.
    S = np.cumsum(np.cumsum(D, axis=0), axis=1)
    S = np.minimum(S, max_depth)
    # Lower-triangular: a query at k can only attend to keys j<=k. (Spans already
    # satisfy l<=r<=k for active cells, so this only zeroes the strict upper triangle.)
    S = np.tril(S)
    return S.astype(np.int8)


# --------------------------------------------------------------------------- #
# Chunking (whole-tree units)
# --------------------------------------------------------------------------- #
@dataclass
class TreeUnit:
    """One whole parse-tree unit = a document block (or a sub-forest when a block
    exceeds ``max_len``). Stores the segment list so terminal/spans can be rebuilt
    with correct chunk-local offsets. Each ``('tree', ...)`` segment carries its own
    binarized content leaves + spans (0-based over those content leaves)."""

    segments: List[Segment]

    @property
    def n_terminals(self) -> int:
        total = 0
        for kind, data in self.segments:
            if kind == "leaves":
                total += len(data)
            else:  # tree -> (leaves, spans)
                total += len(data[0])
        return total

    @property
    def n_trees(self) -> int:
        return sum(1 for k, _ in self.segments if k == "tree")


def _binarize_segments(segments: List[Segment], direction: str) -> List[Segment]:
    out: List[Segment] = []
    for kind, data in segments:
        if kind == "leaves":
            out.append((kind, data))
        else:
            btree = binarize_tree(data, direction)
            leaves, spans = tree_spans(btree)
            out.append(("tree", (leaves, spans)))
    return out


def _unit_from_segments(segments: List[Segment]) -> TreeUnit:
    return TreeUnit(segments=segments)


def _split_oversized(segments: List[Segment], max_len: int) -> List[TreeUnit]:
    """Split a block's segments into sub-units each ``<= max_len`` terminals, at
    tree boundaries. Leaf-runs (BOS/whitespace) attach to the following tree;
    trailing leaves attach to the last sub-unit. An atomic single-tree sub-unit
    that still exceeds ``max_len`` is emitted unchanged (never cut mid-tree)."""
    # Greedily group segments into sub-units by terminal budget.
    units: List[TreeUnit] = []
    cur: List[Segment] = []
    cur_len = 0
    for seg in segments:
        seg_len = len(seg[1]) if seg[0] == "leaves" else len(seg[1][0])
        if cur and cur_len + seg_len > max_len and any(k == "tree" for k, _ in cur):
            units.append(_unit_from_segments(cur))
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += seg_len
    if cur:
        units.append(_unit_from_segments(cur))
    return units if units else [_unit_from_segments(segments)]


def chunk_units(
    blocks: Sequence[Sequence[int]],
    vocab: TreeVocab,
    direction: str,
    max_len: int = 2048,
) -> List[List[TreeUnit]]:
    """Greedily pack whole-tree units into chunks of ``<= max_len`` terminals.

    Each block becomes a :class:`TreeUnit` (a forest of its top-level sentence
    trees); blocks longer than ``max_len`` are split at tree boundaries into sub-
    units. Units are then greedily packed into chunks. No parse tree is ever split
    across a chunk boundary (an atomic tree exceeding ``max_len`` is emitted as its
    own oversize chunk rather than cut).
    """
    chunks: List[List[TreeUnit]] = []
    cur: List[TreeUnit] = []
    cur_len = 0
    for block in blocks:
        segments = _binarize_segments(parse_block_segments(block, vocab), direction)
        if not segments:
            continue
        unit = _unit_from_segments(segments)
        units = [unit] if unit.n_terminals <= max_len else _split_oversized(segments, max_len)
        for u in units:
            ulen = u.n_terminals
            if cur and cur_len + ulen > max_len:
                chunks.append(cur)
                cur, cur_len = [], 0
            cur.append(u)
            cur_len += ulen
    if cur:
        chunks.append(cur)
    return chunks


def chunk_to_tensors(chunk: Sequence[TreeUnit]) -> Dict[str, Any]:
    """Flatten a chunk (list of units) into terminal ids, the depth matrix S, and
    the constituent spans (chunk-local terminal indices).

    * ``terminals``: ``(T,)`` int64 — the model input (BOS...EOS per block, concatenated).
    * ``depth_matrix``: ``(T, T)`` int8 — Pushdown stale tape. Leaves not in any
      constituent (BOS/EOS/whitespace) stay at depth 0.
    * ``spans``: ``(M, 3)`` int32 — TreeReg ``(left, split, right)`` in terminal indices.
    """
    terminals: List[int] = []
    spans: List[Span] = []
    for u in chunk:
        for kind, data in u.segments:
            if kind == "leaves":
                terminals.extend(int(x) for x in data)
            else:  # tree -> (content_leaves, tree_spans)
                leaves, tspans = data
                t0 = len(terminals)
                terminals.extend(int(x) for x in leaves)
                for (l, sp, r) in tspans:
                    spans.append((t0 + l, t0 + sp, t0 + r))
    T = len(terminals)
    S = compute_depth_matrix(spans, T) if T > 0 else np.zeros((0, 0), dtype=np.int8)
    return {
        "terminals": np.asarray(terminals, dtype=np.int64),
        "depth_matrix": S,
        "spans": np.asarray(spans, dtype=np.int32).reshape(-1, 3),
    }


# --------------------------------------------------------------------------- #
# Tree-stream chunk iterator (for preprocessing / dataset indexing)
# --------------------------------------------------------------------------- #
def _block_ranges(n_blocks: int, workers: int) -> List[Tuple[int, int]]:
    """Split ``[0, n_blocks)`` into ``workers`` contiguous (lo, hi) ranges."""
    base = n_blocks // workers
    rem = n_blocks % workers
    ranges: List[Tuple[int, int]] = []
    lo = 0
    for w in range(workers):
        sz = base + (1 if w < rem else 0)
        if sz <= 0:
            break
        ranges.append((lo, lo + sz))
        lo += sz
    return ranges


def _warm_tree_cache(tree_npy: str, block: int = 16 * 1024 * 1024) -> None:
    """Read the whole tree file sequentially into the OS page cache.

    A single sequential pass maximises mechanical-disk throughput (no seeking)
    and, when RAM can hold the file (~49 GB train.npy << 251 GB RAM), leaves it
    cached so the subsequent parallel scan AND parse read from RAM instead of
    re-hitting the disk. Without this, 16 workers cold-reading disjoint block
    ranges of a 49 GB file cause a seek storm (~0 throughput on a spinning disk).
    Uses os.read (real read syscalls, unlike mmap fault-in) so read_bytes is
    visible and the kernel prefetcher engages. Also issues POSIX_FADV_SEQUENTIAL.
    """
    import os
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        POSIX_FADV_SEQUENTIAL = 2
        fd = os.open(tree_npy, os.O_RDONLY)
        libc.posix_fadvise(fd, ctypes.c_long(0), ctypes.c_long(0),
                           ctypes.c_int(POSIX_FADV_SEQUENTIAL))
    except Exception:
        fd = os.open(tree_npy, os.O_RDONLY)
    try:
        while True:
            buf = os.read(fd, block)
            if not buf:
                break
    finally:
        os.close(fd)


def _block_totals_inline(tree_arr, vocab, BLOCK, block_starts, bi_lo, bi_hi):
    """Phase A: per-block (delta_total, leaf_total) only — O(1) output per block.

    Each block's delta_total = #op - #cl (the bracket-depth carry it contributes);
    leaf_total = #leaves. These let the caller compute the true depth/leaf carry
    at every block's start in a serial O(#blocks) prefix scan, WITHOUT recording
    any per-close data. This is the memory-bounded alternative to recording every
    closing NT (the train stream has ~2.4B closes → ~58 GB just for the close
    arrays, which OOM'd/thrashed).
    """
    delta_totals = np.empty(bi_hi - bi_lo, dtype=np.int64)
    leaf_totals = np.empty(bi_hi - bi_lo, dtype=np.int64)
    for k, bi in enumerate(range(bi_lo, bi_hi)):
        b0 = int(block_starts[bi])
        seg = np.asarray(tree_arr[b0 : b0 + BLOCK])
        is_op = (seg >= vocab.op_lo) & (seg <= vocab.op_hi)
        is_cl = (seg >= vocab.cl_lo) & (seg <= vocab.cl_hi)
        n_op = int(is_op.sum()); n_cl = int(is_cl.sum())
        delta_totals[k] = n_op - n_cl
        leaf_totals[k] = int(seg.shape[0]) - n_op - n_cl
    return delta_totals, leaf_totals


def _block_totals_worker(args):
    tree_npy, tokenizer_path, BLOCK, block_starts, rng = args
    import numpy as np
    tree = np.load(tree_npy, mmap_mode="r")
    _madvise_sequential(tree)
    vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
    return _block_totals_inline(tree, vocab, BLOCK, block_starts, rng[0], rng[1])


def _toplevel_closes_inline(tree_arr, vocab, BLOCK, block_starts, depth_carry,
                            leaf_carry, bi_lo, bi_hi):
    """Phase B: record ONLY top-level closes (depth==0) for blocks [bi_lo, bi_hi),
    given the true per-block depth_carry / leaf_carry arrays (from phase A's prefix
    scan). Output is O(#top-level-trees) (~5M for train), NOT O(#closes) (~2.4B).

    Returns (close_pos, close_leaf) where close_leaf is the GLOBAL cumulative leaf
    count up to and including each close (ready for the greedy packer).
    """
    close_pos: List[np.ndarray] = []
    close_leaf: List[np.ndarray] = []
    for k, bi in enumerate(range(bi_lo, bi_hi)):
        b0 = int(block_starts[bi])
        seg = np.asarray(tree_arr[b0 : b0 + BLOCK])
        is_op = (seg >= vocab.op_lo) & (seg <= vocab.op_hi)
        is_cl = (seg >= vocab.cl_lo) & (seg <= vocab.cl_hi)
        is_leaf = (~is_op) & (~is_cl)
        d0 = int(depth_carry[bi]); l0 = int(leaf_carry[bi])
        leaf_cum = np.cumsum(is_leaf.astype(np.int64)) + l0  # global cum leaves
        delta = is_op.astype(np.int32) - is_cl.astype(np.int32)
        d = np.cumsum(delta) + d0  # true depth after each token
        closes = np.nonzero(is_cl & (d == 0))[0]
        if closes.size:
            close_pos.append((b0 + closes).astype(np.int64))
            close_leaf.append(leaf_cum[closes].astype(np.int64))
    if not close_pos:
        return None, None
    return np.concatenate(close_pos), np.concatenate(close_leaf)


def _toplevel_closes_worker(args):
    """Phase B worker: re-mmap the tree file, scan a block range with the known
    per-block carries, return only top-level closes."""
    (tree_npy, tokenizer_path, BLOCK, block_starts, depth_carry, leaf_carry, rng) = args
    import numpy as np
    tree = np.load(tree_npy, mmap_mode="r")
    _madvise_sequential(tree)
    vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
    return _toplevel_closes_inline(tree, vocab, BLOCK, block_starts, depth_carry,
                                   leaf_carry, rng[0], rng[1])


def iter_tree_chunks(tree_arr: "np.ndarray", vocab: TreeVocab, direction: str,
                     max_len: int = 2048, workers: int = 1,
                     tree_npy: Optional[str] = None) -> List[Tuple[int, int]]:
    """Scan the tree-token stream and return chunk spans ``(tree_start, tree_len)``
    with whole-tree integrity.

    A chunk is a maximal run of complete top-level parse trees (plus intervening
    leaf tokens like BOS/EOS/whitespace) whose total **terminal-leaf** count is
    ``<= max_len``. Top-level trees are found with a bracket-depth scan (a tree
    opens at depth 0 and closes when depth returns to 0). An atomic single tree
    whose leaves exceed ``max_len`` is emitted as its own (oversize) chunk.

    Vectorized with numpy (scans the stream in fixed-size blocks, computing
    per-token NT deltas and leaf counts), so it scales to the 24.7B-token train
    stream in minutes rather than the ~hours a Python per-token scan would take.

    The data loader mmaps ``tree_arr`` and slices ``[start:start+len]`` per chunk;
    parsing that slice yields both the spans and the terminal leaves (which are
    bit-identical to the corresponding ``terminal/*.npy`` tokens).
    """
    n = int(tree_arr.shape[0])
    if n == 0:
        return []
    # Pass 1 (single-pass parallel block scan): find every top-level tree's CLOSE
    # position (a closing NT where the running bracket depth returns to 0). For
    # well-formed input, tree k spans [prev_close, this_close). For each top-level
    # tree close we record (global close position, cumulative leaves up to and
    # including that close). Both arrays are O(#trees) — NOT O(#stream) — so the
    # 24.7B-token train stream (~5M trees) needs only ~120 MB.
    #
    # Two-phase, memory-bounded, parallel (the tree file is read twice but stays
    # in the page cache, so the second read is from RAM):
    #   Phase A: each worker computes per-block (delta_total, leaf_total) only —
    #     O(1) output per block (~3K blocks), no per-close data. A serial O(#blocks)
    #     prefix scan then gives the true depth/leaf carry at every block's start.
    #   Phase B: each worker re-scans its blocks WITH the known carries and records
    #     ONLY top-level closes (depth==0) — O(#trees) output, not O(#closes).
    # (An earlier single-pass design recorded EVERY closing NT + its local depth —
    # ~2.4B closes on train → ~58 GB of arrays → 182 GB RSS + thrashing. This
    # two-phase design keeps peak output at O(#trees)≈5M.)
    BLOCK = 8_000_000  # tokens per numpy block (~16 MB int16)
    block_starts_arr = np.arange(0, n, BLOCK)
    n_blocks = int(block_starts_arr.shape[0])
    ranges = _block_ranges(n_blocks, max(1, workers))
    tokenizer_path = vocab._tokenizer_path

    # --- Phase A: per-block delta/leaf totals (parallel), then prefix-scan carries. ---
    if workers and workers > 1 and n_blocks >= 2 and tree_npy:
        from multiprocessing import Pool
        argsA = [(tree_npy, tokenizer_path, BLOCK, block_starts_arr, r) for r in ranges]
        with Pool(workers) as pool:
            tot_results = pool.map(_block_totals_worker, argsA)
        delta_all = np.concatenate([t[0] for t in tot_results])
        leaf_all = np.concatenate([t[1] for t in tot_results])
    else:
        delta_all, leaf_all = _block_totals_inline(tree_arr, vocab, BLOCK,
                                                   block_starts_arr, 0, n_blocks)
    # depth_carry[bi] = true bracket depth at the START of block bi.
    depth_carry = np.concatenate(([0], np.cumsum(delta_all)[:-1]))
    leaf_carry = np.concatenate(([0], np.cumsum(leaf_all)[:-1]))
    total_leaves = int(leaf_all.sum()) if leaf_all.size else 0

    # --- Phase B: record only top-level closes (parallel), with known carries. ---
    close_pos_parts: List[np.ndarray] = []
    close_leaf_parts: List[np.ndarray] = []
    if workers and workers > 1 and n_blocks >= 2 and tree_npy:
        argsB = [(tree_npy, tokenizer_path, BLOCK, block_starts_arr,
                  depth_carry, leaf_carry, r) for r in ranges]
        with Pool(workers) as pool:
            cl_results = pool.map(_toplevel_closes_worker, argsB)
        for cp, cl in cl_results:
            if cp is not None and cp.size:
                close_pos_parts.append(cp)
                close_leaf_parts.append(cl)
    else:
        cp, cl = _toplevel_closes_inline(tree_arr, vocab, BLOCK, block_starts_arr,
                                         depth_carry, leaf_carry, 0, n_blocks)
        if cp is not None and cp.size:
            close_pos_parts.append(cp)
            close_leaf_parts.append(cl)

    # Tree k spans [close_idxs[k-1]+1, close_idxs[k]+1) (close is inclusive).
    closes_arr = (np.concatenate(close_pos_parts) if close_pos_parts
                  else np.zeros(0, np.int64))
    T = len(closes_arr)
    if T == 0:
        return [(0, n)] if n > 0 else []
    # Tree starts: 0 for the first, then prev_close+1.
    tree_starts = np.empty(T, dtype=np.int64)
    tree_starts[0] = 0
    tree_starts[1:] = closes_arr[:-1] + 1
    tree_ends = closes_arr + 1  # exclusive end
    # Leaves in tree k = leaves-at-close[k] - leaves-at-close[k-1] (0 for k=0).
    clc = (np.concatenate(close_leaf_parts) if close_leaf_parts
           else np.zeros(0, np.int64))
    tree_leaves = clc.copy()
    tree_leaves[1:] = clc[1:] - clc[:-1]

    # Pass 2 (vectorized greedy packing): pack trees into chunks of <= max_len
    # leaves. A tree whose leaves exceed max_len is its own (oversize) chunk.
    #
    # For every candidate start tree k, the furthest tree e with
    # sum(leaves[k..e]) <= max_len is found in ONE vectorized searchsorted on the
    # prefix-sum array: e[k] = searchsorted(csum, csum[k]-1 + max_len, 'right')-1,
    # clamped to [k, T-1] (an oversize tree maps to e==k, i.e. its own chunk).
    # The chunks are then the pointer chain 0 -> e[0]+1 -> e[e[0]+1]+1 -> ...,
    # which advances by a full chunk per step — O(#chunks) iterations (~5M for
    # train), NOT O(#trees) (~250M). The old per-tree Python loop was the
    # bottleneck that stalled the 24.7B-token train scan for >15 minutes.
    leaves = tree_leaves.astype(np.int64)
    csum = np.concatenate(([0], np.cumsum(leaves)))  # csum[k] = sum leaves[0..k-1]
    # For start k: target prefix = (leaves before k) + max_len = csum[k] + max_len.
    # e[k] = largest index with csum[e[k]+1] - csum[k] <= max_len, i.e. csum[e[k]+1] <= target.
    # searchsorted(csum, target, 'right')-1 gives that index in the csum array,
    # which is e (tree index). Clamp so e >= k (oversize -> single-tree chunk).
    targets = csum[:T] + max_len                 # one per start tree
    e_arr = np.searchsorted(csum, targets, side="right") - 1
    e_arr = np.clip(e_arr, np.arange(T), T - 1)  # oversize trees -> e == k
    e_plus1 = e_arr + 1                          # next start tree after a chunk
    # Walk the pointer chain from k=0; collect chunk start trees.
    start_trees: List[int] = []
    k = 0
    while k < T:
        start_trees.append(k)
        k = int(e_plus1[k])
    st = np.asarray(start_trees, dtype=np.int64)
    et = e_plus1[st]                             # exclusive end-tree per chunk
    chunk_start_pos = tree_starts[st]
    chunk_end_pos = tree_ends[et - 1]            # exclusive stream end
    chunks = list(zip(chunk_start_pos.tolist(),
                      (chunk_end_pos - chunk_start_pos).tolist()))
    # Trailing leaves after the last tree close (e.g. a final EOS): append as a
    # tail chunk, merged into the last chunk if it fits.
    last_end = int(tree_ends[-1])
    if last_end < n:
        # total_leaves is the count across the whole stream; leaves at the last
        # close is clc[-1]. The tail = everything after.
        tail_leaves = total_leaves - int(clc[-1])
        if chunks and tail_leaves <= max_len:
            s0, l0 = chunks[-1]
            chunks[-1] = (s0, last_end + (n - last_end) - s0)
        else:
            chunks.append((last_end, n - last_end))
    return chunks


def _count_leaves_in_span(tree_arr, start, end, vocab) -> int:
    """Number of terminal leaves (non-NT tokens) in ``tree_arr[start:end]``."""
    sub = np.asarray(tree_arr[start:end])
    is_nt = (sub >= vocab.op_lo) & (sub <= vocab.cl_hi)
    return int((~is_nt).sum())


def parse_chunk_slice(tree_slice: Sequence[int], vocab: TreeVocab, direction: str) -> Dict[str, Any]:
    """Parse one chunk's tree-token slice -> terminal leaves (input_ids) + spans.

    This is what the data loader calls per chunk. Returns:
    * ``input_ids``: ``(T,)`` int64 — terminal leaves (== terminal.npy tokens).
    * ``spans``: ``(M, 3)`` int32 — constituent ``(left, split, right)`` over input_ids.
    """
    segments = _binarize_segments(parse_block_segments(tree_slice, vocab), direction)
    terminals: List[int] = []
    spans: List[Span] = []
    for kind, data in segments:
        if kind == "leaves":
            terminals.extend(int(x) for x in data)
        else:
            leaves, tspans = data
            t0 = len(terminals)
            terminals.extend(int(x) for x in leaves)
            for (l, sp, r) in tspans:
                spans.append((t0 + l, t0 + sp, t0 + r))
    return {
        "input_ids": np.asarray(terminals, dtype=np.int64),
        "spans": np.asarray(spans, dtype=np.int32).reshape(-1, 3),
    }


# --------------------------------------------------------------------------- #
# Dataset (plugs into the OLMo train/eval loaders)
# --------------------------------------------------------------------------- #
class ParseAlignedDataset(torch.utils.data.Dataset):
    """Dataset for the Pushdown/TreeReg baselines.

    Serves whole-tree chunks (from a precomputed ``chunk_index.npy`` over
    ``tree/*.npy``) as ``{input_ids, tree_spans, tree_span_mask, attention_mask}``.
    The terminal ``input_ids`` are the chunk's tree leaves (bit-identical to
    ``terminal/*.npy``); ``tree_spans`` are the binarized constituent spans over
    those leaves. The Pushdown depth matrix is computed on the GPU in the model
    forward (not here). Duck-types :class:`MemMapDataset` (``__len__``/``__getitem__``
    + an ``offsets`` property) so the existing :class:`IterableDataset` wrapper and
    samplers work unchanged.
    """

    def __init__(
        self,
        tree_npy: str,
        chunk_index_npy: str,
        tokenizer: str,
        direction: str = "left",
        max_len: int = 2048,
        pad_token_id: int = 50258,
        generate_attention_mask: bool = True,
    ):
        import os
        self.tree_npy = tree_npy
        self._chunk_index = np.load(chunk_index_npy)  # (n_chunks, 2): tree_start, tree_len
        self._tree = np.load(tree_npy, mmap_mode="r")
        # Random chunk-index access into tree.npy -> disable readahead (cgroup OOM
        # guard; see PrecomputedParseDataset / _madvise_random rationale).
        _madvise_random(self._tree)
        self.vocab = TreeVocab.from_tokenizer_file(tokenizer)
        self.direction = direction
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.generate_attention_mask = generate_attention_mask
        self.transformer_grammar_type = ""  # set by caller if needed
        # MemMap-compat attributes.
        self._num_instances = int(len(self._chunk_index))
        self._mmap_offsets = [(0, self._num_instances)]

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        return self._mmap_offsets

    def __len__(self) -> int:
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        s, l = self._chunk_index[index]
        s, l = int(s), int(l)
        out = parse_chunk_slice(np.asarray(self._tree[s : s + l]), self.vocab, self.direction)
        input_ids = out["input_ids"]
        spans = out["spans"]
        # Truncate to max_len (atomic trees slightly over max_len).
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            spans = spans[(spans[:, 2] < self.max_len) & (spans[:, 0] < self.max_len)] if len(spans) else spans
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        item: Dict[str, Any] = {"input_ids": input_ids}
        if len(spans):
            tspans = torch.tensor(spans, dtype=torch.long)
            item["tree_spans"] = tspans
            item["tree_span_mask"] = torch.ones(len(tspans), dtype=torch.bool)
        else:
            item["tree_spans"] = torch.zeros((0, 3), dtype=torch.long)
            item["tree_span_mask"] = torch.zeros((0,), dtype=torch.bool)
        if self.generate_attention_mask:
            item["attention_mask"] = torch.ones(len(input_ids), dtype=torch.bool)
        return item


class PrecomputedParseDataset(torch.utils.data.Dataset):
    """Dataset that reads precomputed parse-aligned chunks from a directory produced
    by :func:`preprocess_split`.

    Loads (memory-mapped) the saved ``input_ids.npy``, ``spans.npy``,
    ``span_counts.npy`` (and optionally ``depth_matrix.npy``) and serves
    ``{input_ids, tree_spans, tree_span_mask, attention_mask}`` (+ optional
    ``tree_depth_matrix``). No ``tree.npy`` parsing at load time — faster than
    :class:`ParseAlignedDataset`. The Pushdown depth matrix is computed on the GPU
    in the model forward from the spans (faster than loading the ~6.8 GB int8
    ``depth_matrix`` from disk per batch), so ``depth_matrix.npy`` is loaded only if
    ``load_depth=True`` (e.g. for analysis / a depth-loading variant).
    """

    def __init__(self, data_dir: str, pad_token_id: int = 50258,
                 load_depth: bool = False):
        import os
        self.data_dir = data_dir
        self.pad_token_id = pad_token_id
        self.input_ids = np.load(os.path.join(data_dir, "input_ids.npy"), mmap_mode="r")
        # Prefer the int16 spans file if present (spans are terminal indices < 2048,
        # so int16 is lossless and halves the mmap from 149 GB to 75 GB for the train
        # split — directly cutting cgroup page-cache pressure under slurm, which was
        # the cause of the num_workers>0 OOM). Fall back to the int32 spans.npy.
        # __getitem__ builds tensors with dtype=torch.long, which upcasts int16 -> int64.
        spans_int16 = os.path.join(data_dir, "spans_int16.npy")
        spans_path = spans_int16 if os.path.exists(spans_int16) else os.path.join(data_dir, "spans.npy")
        self.spans = np.load(spans_path, mmap_mode="r")
        self.span_counts = np.load(os.path.join(data_dir, "span_counts.npy"))
        # These mmaps are read by RANDOM chunk index across 4.88M chunks. Hint the
        # kernel MADV_RANDOM so it stops readaheading (the default readahead hoards
        # a large fraction of the 189 GB file into page cache -> cgroup OOM under
        # slurm). See _madvise_random for the full rationale.
        _madvise_random(self.input_ids)
        _madvise_random(self.spans)
        self.n_chunks = int(self.input_ids.shape[0])
        self.max_len = int(self.input_ids.shape[1])
        self.load_depth = load_depth
        self.depth = None
        if load_depth and os.path.exists(os.path.join(data_dir, "depth_matrix.npy")):
            self.depth = np.load(os.path.join(data_dir, "depth_matrix.npy"), mmap_mode="r")
            _madvise_random(self.depth)
        # MemMapDataset-compat.
        self._num_instances = self.n_chunks
        self._mmap_offsets = [(0, self.n_chunks)]
        self.transformer_grammar_type = ""

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        return self._mmap_offsets

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        input_ids = torch.tensor(np.asarray(self.input_ids[index]), dtype=torch.long)
        attention_mask = input_ids != self.pad_token_id
        m = int(self.span_counts[index])
        if m > 0:
            tspans = torch.tensor(np.asarray(self.spans[index, :m]), dtype=torch.long)
            span_mask = torch.ones(m, dtype=torch.bool)
        else:
            tspans = torch.zeros((0, 3), dtype=torch.long)
            span_mask = torch.zeros((0,), dtype=torch.bool)
        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tree_spans": tspans,
            "tree_span_mask": span_mask,
        }
        if self.depth is not None:
            item["tree_depth_matrix"] = torch.tensor(
                np.asarray(self.depth[index]), dtype=torch.int8)
        return item


# --------------------------------------------------------------------------- #
# Offline preprocessing: binarize trees + generate the stale depth matrix
# --------------------------------------------------------------------------- #
def preprocess_split(
    tree_npy: str,
    tokenizer_path: str,
    direction: str,
    out_dir: str,
    max_len: int = 2048,
    pad_token_id: int = 50258,
    save_depth_matrix: bool = True,
    workers: int = 1,
    scan_workers: int = 1,
    warm_cache: bool = False,
    max_spans_override: Optional[int] = None,
    load_tree_to_ram: bool = False,
) -> Dict[str, str]:
    """Process a ``tree/*.npy`` parse stream into binarized chunks + the stale
    Pushdown depth matrix, saved to ``out_dir``.

    For each whole-tree chunk (``<= max_len`` terminal leaves, packed by
    :func:`iter_tree_chunks`): binarize (left/right), extract terminal leaves
    (``input_ids`` — bit-identical to ``terminal/*.npy``) and constituent spans,
    and (optionally) compute the stale depth matrix ``S[k,j]`` (int8). Saves:

    * ``chunk_index.npy`` ``(n_chunks, 2)`` int64 — tree-stream slice per chunk.
    * ``input_ids.npy`` ``(n_chunks, max_len)`` int32 — terminal leaves, padded.
    * ``spans.npy`` ``(n_chunks, max_spans, 3)`` int32 — constituent spans, pad -1.
    * ``span_counts.npy`` ``(n_chunks,)`` int32 — valid spans per chunk.
    * ``depth_matrix.npy`` ``(n_chunks, max_len, max_len)`` int8 — the stale tape
      ``S[k,j] = #{constituents (l,r): l<=j<=r<=k}``, padded. Only when
      ``save_depth_matrix`` (a full depth matrix is ~4.2 MB/chunk; the train split
      is ~20 TB and must be computed on the GPU at train time instead).

    Streams to preallocated memory-maps (one chunk in RAM at a time) and parses in
    parallel across ``workers`` processes — so it scales to the 4.94M-chunk train
    split (~40 min/direction on 16 workers) without OOM.

    ``scan_workers`` controls the chunk-scan parallelism separately from the parse
    ``workers``. The scan reads the tree stream in BLOCKS; with ``scan_workers=1``
    (default) it reads sequentially — maximising mechanical-disk throughput and
    warming the page cache for the parse phase. With ``scan_workers>1`` the scan
    parallelises (fast when the file is already cached, e.g. a warm second
    direction) but on a COLD mechanical disk many workers reading disjoint block
    ranges cause a seek storm (~0 throughput). Use ``warm_cache`` to read the
    whole tree file into the page cache (single sequential pass) before scanning
    when RAM (>=~file size) can hold it — then both scan and parse hit cache.

    Returns the dict of saved file paths.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
    if warm_cache:
        _warm_tree_cache(tree_npy)
    tree_mmap = np.load(tree_npy, mmap_mode="r")
    chunks = iter_tree_chunks(tree_mmap, vocab, direction=direction, max_len=max_len,
                              workers=scan_workers, tree_npy=tree_npy)
    n_chunks = len(chunks)
    print(f"[{out_dir}] {n_chunks} chunks (direction={direction}); "
          f"scan_workers={scan_workers} parse_workers={workers}")

    # Optionally load the whole tree into RAM (49 GB < 251 GB RAM). Parse workers
    # then inherit it via fork (copy-on-write, read-only → no duplication) and read
    # from RAM with ZERO disk input I/O — the fix for the seek storm that wedges
    # 16 mmap-based workers in D state on a contended mechanical disk. The mmap is
    # kept for the scan (already done) and dropped before the RAM load to save RSS.
    tree_ram = None
    if load_tree_to_ram:
        print(f"  loading tree.npy into RAM ({tree_mmap.nbytes/1e9:.1f} GB) for fork-shared parse...")
        tree_ram = np.array(tree_mmap, dtype=tree_mmap.dtype)  # real ndarray copy
        del tree_mmap
        import gc; gc.collect()

    # Estimate max_spans from a sample (avoid a full two-pass parse). Cap generously.
    # The sample is read in STREAM ORDER (sorted by start) so its tree.npy reads
    # are sequential/cache-friendly, not a seek storm of random small reads.
    _tree_for_sample = tree_ram if tree_ram is not None else tree_mmap
    if max_spans_override is not None:
        max_spans = max_spans_override
    else:
        sample = chunks[:: max(1, n_chunks // 2000)][:2000]
        sample = sorted(sample, key=lambda c: c[0])  # stream order -> cache-friendly
        max_spans = 1
        for (s, l) in sample:
            out = parse_chunk_slice(np.asarray(_tree_for_sample[s : s + l]), vocab, direction)
            if len(out["spans"]) > max_spans:
                max_spans = len(out["spans"])
        max_spans = max(max_spans + 8, 1)  # headroom
    print(f"  max_spans={max_spans}")

    # Preallocate memory-maps (streaming; one chunk in RAM at a time).
    p_chunk_index = os.path.join(out_dir, "chunk_index.npy")
    p_input_ids = os.path.join(out_dir, "input_ids.npy")
    p_spans = os.path.join(out_dir, "spans.npy")
    p_span_counts = os.path.join(out_dir, "span_counts.npy")
    np.save(p_chunk_index, np.asarray(chunks, dtype=np.int64))
    input_ids_m = np.lib.format.open_memmap(p_input_ids, mode="w+",
        dtype=np.int32, shape=(n_chunks, max_len))
    spans_m = np.lib.format.open_memmap(p_spans, mode="w+",
        dtype=np.int32, shape=(n_chunks, max_spans, 3))
    span_counts_m = np.lib.format.open_memmap(p_span_counts, mode="w+",
        dtype=np.int32, shape=(n_chunks,))
    input_ids_m[:] = pad_token_id
    spans_m[:] = -1

    # Parse + write (parallel). With load_tree_to_ram, the fork-inherited RAM
    # array is the tree source (zero disk input I/O); otherwise workers re-mmap
    # tree.npy. Workers write to disjoint rows of the output memmaps. The RAM
    # array is stashed in a module global so fork-inherited workers read it
    # copy-on-write WITHOUT it being pickled per-task (which would copy 49 GB).
    global _TREE_RAM
    _TREE_RAM = tree_ram  # None when not loading to RAM
    tree_src = tree_npy  # path passed to workers; they use _TREE_RAM if set
    if workers > 1:
        from multiprocessing import Pool
        import functools
        ranges = [(r[0], min(r[0] + 2000, r[1])) for r in
                  [(i, min(i + 2000, n_chunks)) for i in range(0, n_chunks, 2000)]]
        fn = functools.partial(_parse_range_to_memmap, tree_src, tokenizer_path,
                               direction, max_len, pad_token_id,
                               p_input_ids, p_spans, p_span_counts, max_spans)
        with Pool(workers) as pool:
            for done in pool.imap_unordered(fn, ranges):
                pass
    else:
        _parse_range_to_memmap(tree_src, tokenizer_path, direction, max_len,
                               pad_token_id, p_input_ids, p_spans, p_span_counts,
                               max_spans, (0, n_chunks))
    _TREE_RAM = None
    input_ids_m.flush(); spans_m.flush(); span_counts_m.flush()

    saved = {"chunk_index": p_chunk_index, "input_ids": p_input_ids,
             "spans": p_spans, "span_counts": p_span_counts}

    if save_depth_matrix:
        p_depth = os.path.join(out_dir, "depth_matrix.npy")
        depth = np.lib.format.open_memmap(p_depth, mode="w+",
            dtype=np.int8, shape=(n_chunks, max_len, max_len))
        try:
            for i in range(n_chunks):
                m = int(span_counts_m[i])
                n = int((input_ids_m[i] != pad_token_id).sum())
                if n > 0 and m > 0:
                    depth[i, :n, :n] = compute_depth_matrix(spans_m[i, :m], n)
                if (i + 1) % 2000 == 0:
                    print(f"  depth matrix {i+1}/{n_chunks}")
        finally:
            depth.flush()
        saved["depth_matrix"] = p_depth
        print(f"  depth_matrix: {os.path.getsize(p_depth)/1e9:.2f} GB")

    print(f"[{out_dir}] done. input_ids ({n_chunks},{max_len}), spans ({n_chunks},{max_spans},3), "
          f"depth {'saved' if save_depth_matrix else 'skipped (compute on GPU at train time)'}")
    return saved


def _madvise_sequential(mmap_arr) -> None:
    """Hint the kernel to prefetch + drop-behind for the tree mmap.

    Without this, 16 parse workers each faulting pages across a 49 GB mmap
    accumulate ~220 GB of resident file pages, evicting the warm page cache and
    forcing a re-read from disk (seek storm). MADV_SEQUENTIAL makes the kernel
    drop pages behind the read cursor, keeping per-worker RSS small and the cache
    stable. No-op if ctypes/madvise is unavailable.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        addr = ctypes.c_void_p(mmap_arr.ctypes.data)
        size = ctypes.c_size_t(mmap_arr.nbytes)
        # MADV_SEQUENTIAL = 2
        libc.madvise(addr, size, ctypes.c_int(2))
    except Exception:
        pass


def _madvise_random(mmap_arr) -> None:
    """Hint the kernel to DISABLE readahead for a randomly-accessed mmap.

    ``PrecomputedParseDataset`` mmaps ``spans.npy`` (149 GB) + ``input_ids.npy``
    (40 GB) and reads them by random chunk index. With the default hint the kernel
    readaheads ~128 KB per fault, so 4 dataloader workers faulting across 4.88M
    chunks hoard a large fraction of the 189 GB file into page cache. Under a slurm
    cgroup memory limit, page cache counts toward the job -> the cgroup OOM killer
    SIGKILLs the process (this is NOT a CUDA OOM; it hits pushdown and treereg
    identically because both load the same spans). MADV_RANDOM (=1) makes each
    fault bring in only the needed page, capping resident pages at the working set.
    No-op if ctypes/madvise is unavailable.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        addr = ctypes.c_void_p(mmap_arr.ctypes.data)
        size = ctypes.c_size_t(mmap_arr.nbytes)
        # MADV_RANDOM = 1
        libc.madvise(addr, size, ctypes.c_int(1))
    except Exception:
        pass


# Module global for the fork-inherited in-RAM tree array (set by preprocess_split
# when load_tree_to_ram=True). Workers read it copy-on-write WITHOUT it being
# pickled per-task (which would copy the 49 GB array per dispatch).
_TREE_RAM: Optional["np.ndarray"] = None


def _parse_range_to_memmap(tree_src, tokenizer_path, direction, max_len, pad_token_id,
                           p_input_ids, p_spans, p_span_counts, max_spans, rng):
    """Worker: parse chunks [rng[0], rng[1]) and write to the shared memmaps.

    ``tree_src`` is the tree.npy path. If the module global ``_TREE_RAM`` is set
    (fork-inherited in-RAM array, copy-on-write), workers read from it with ZERO
    disk input I/O — the fix for the seek storm that wedges 16 mmap workers in D
    state on a contended mechanical disk. Otherwise workers re-mmap tree.npy.
    """
    import numpy as np
    vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
    if _TREE_RAM is not None:
        tree = _TREE_RAM  # fork-inherited RAM array (read-only -> no copy)
    else:
        tree = np.load(tree_src, mmap_mode="r")
        _madvise_sequential(tree)
    chunk_index = np.load(p_input_ids.rsplit("/", 1)[0] + "/chunk_index.npy", mmap_mode="r")
    input_ids_m = np.lib.format.open_memmap(p_input_ids, mode="r+")
    spans_m = np.lib.format.open_memmap(p_spans, mode="r+")
    span_counts_m = np.lib.format.open_memmap(p_span_counts, mode="r+")
    lo, hi = rng
    for i in range(lo, hi):
        s, l = int(chunk_index[i, 0]), int(chunk_index[i, 1])
        out = parse_chunk_slice(np.asarray(tree[s : s + l]), vocab, direction)
        ids = out["input_ids"]
        sp = out["spans"]
        if len(ids) > max_len:
            ids = ids[:max_len]
            sp = sp[sp[:, 2] < max_len] if len(sp) else sp
        input_ids_m[i, : len(ids)] = ids
        m = min(len(sp), max_spans)
        if m > 0:
            spans_m[i, :m] = sp[:m]
        span_counts_m[i] = m
    # NOTE: no per-range flush. Calling flush() (msync) here forces synchronous
    # disk writes that, with 16 parallel workers on a contended mechanical disk,
    # wedge every worker in D state (rq_qos_wait) at ~0 throughput. The memmap
    # pages stay in the page cache and the OS writes them back asynchronously;
    # the single end-of-run flush in preprocess_split persists everything.
    return hi


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Binarize tree/*.npy + generate the stale depth matrix.")
    ap.add_argument("--tree_npy", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--direction", default="both", choices=["left", "right", "both"])
    ap.add_argument("--out_dir", required=True, help="output dir (a <direction> subdir is created)")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--pad_token_id", type=int, default=50258)
    ap.add_argument("--no_depth", action="store_true", help="skip the (large) depth matrix")
    ap.add_argument("--workers", type=int, default=1, help="parallel parse workers")
    ap.add_argument("--scan_workers", type=int, default=1,
                    help="chunk-scan parallelism (1=sequential; >1 only helps if tree.npy is cached)")
    ap.add_argument("--warm_cache", action="store_true",
                    help="read tree.npy into the page cache (single sequential pass) before "
                         "scanning/parsing — avoids a seek storm when 16 workers cold-read a "
                         "large file on a mechanical disk. Use when RAM >= file size.")
    ap.add_argument("--max_spans", type=int, default=None,
                    help="override the max_spans-per-chunk column count (skip the sampling "
                         "estimate). Train left/right is ~2546. Use to avoid a slow random-read "
                         "sample on a cold mechanical disk.")
    ap.add_argument("--load_tree_to_ram", action="store_true",
                    help="load tree.npy into RAM once (49 GB < 251 GB); parse workers inherit it "
                         "via fork (copy-on-write) and read from RAM with ZERO disk input I/O. "
                         "The fix for the seek storm that wedges 16 mmap workers in D state on a "
                         "contended mechanical disk. Requires RAM >= tree.npy size.")
    args = ap.parse_args()
    dirs = ["left", "right"] if args.direction == "both" else [args.direction]
    for d in dirs:
        preprocess_split(
            args.tree_npy, args.tokenizer, d, f"{args.out_dir}_{d}",
            max_len=args.max_len, pad_token_id=args.pad_token_id,
            save_depth_matrix=not args.no_depth, workers=args.workers,
            scan_workers=args.scan_workers, warm_cache=args.warm_cache,
            max_spans_override=args.max_spans, load_tree_to_ram=args.load_tree_to_ram,
        )
