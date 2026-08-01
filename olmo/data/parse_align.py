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
* :func:`collapse_unary_tree` — collapse unary chains, matching the paper's
  BLLIP preprocessing.
* :func:`binarize_tree` — binarize a non-binary tree into a left- or
  right-recursive spine (preserves leaves); the ``direction`` name follows
  NLTK ``chomsky_normal_form(factor=...)``.
* :func:`tree_spans` — enumerate a tree's constituent spans ``(i, split, j)``
  (leaf indices, 0-based within the tree's content leaves) and its content-leaf
  token ids. Works on the RAW tree (one span per real internal node; n-ary/unary
  nodes get a degenerate ``split==j``) so the Pushdown stack tape counts only true
  constituents. :func:`binarize_tree` first only if a real binary split is needed
  (TreeReg CE).
* :func:`compute_depth_matrix` — the Pushdown "stale stack tape"
  ``S[k,j] = #{constituents (l,r): l<=j<=r and r<=k}`` (int8, lower-triangular),
  computed in O(n^2) via a difference array + cumulative sum.
* :func:`iter_tree_chunks` — scan a tree-token stream and return whole-tree chunk
  spans ``(tree_start, tree_len)`` packed to ``<= max_len`` terminal leaves
  (vectorized two-phase bracket-depth scan; an atomic tree exceeding ``max_len``
  is emitted as its own oversize chunk). Writes ``chunk_index.npy`` for
  :class:`ParseAlignedDataset`.
* :func:`parse_chunk_slice` — parse one chunk's tree-token slice into terminal
  leaves (``input_ids``, bit-identical to ``terminal/*.npy``) and constituent
  spans. Called per chunk by :class:`ParseAlignedDataset`.
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
Segment = Tuple[str, Any]
# Before ``_binarize_segments``:
#   ('leaves', List[int]) | ('tree', TreeNode)
# Afterwards:
#   ('leaves', List[int]) |
#   ('tree', (List[int], List[Span], List[bool]))
# The boolean list marks the first BPE token of every parser preterminal.  It is
# deliberately derived from tree structure instead of GPT-2's ``Ġ`` spelling so
# punctuation/newline tokenization cannot silently change the TreeReg candidate
# split set.


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


def tree_leaves_and_word_boundaries(
    tree: Union[TreeNode, Leaf],
) -> Tuple[List[int], List[bool]]:
    """Return terminal leaves and exact parser-word starts for ``tree``.

    A constituency parser preterminal contains the one or more tokenizer pieces
    belonging to a single source word.  Its first terminal is therefore a word
    start and the remaining terminals are continuations.  Bare terminals under a
    malformed/mixed node are conservatively treated as individual word starts.

    The traversal is iterative because the training tree stream contains trees
    deeper than Python's recursion limit.
    """
    leaves: List[int] = []
    boundaries: List[bool] = []
    stack: List[Union[TreeNode, Leaf]] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, int):
            leaves.append(int(node))
            boundaries.append(True)
            continue
        _, children = node
        if children and all(isinstance(child, int) for child in children):
            for child_idx, child in enumerate(children):
                leaves.append(int(child))
                boundaries.append(child_idx == 0)
            continue
        for child in reversed(children):
            stack.append(child)
    return leaves, boundaries


# --------------------------------------------------------------------------- #
# Unary collapse + binarization
# --------------------------------------------------------------------------- #
def collapse_unary_tree(
    node: Union[TreeNode, Leaf], join_char: str = "+"
) -> Union[TreeNode, Leaf]:
    """Collapse every internal unary chain while preserving terminal leaves.

    ``(A (B (C x)))`` becomes ``(A+B+C x)``. A single preterminal
    ``(C x)`` remains an internal node, matching NLTK's
    ``collapse_unary(collapsePOS=True, collapseRoot=True)`` behavior used by
    the Pushdown Layers preprocessing: labels are merged, but the final
    preterminal-to-token edge is not removed.

    The traversal is iterative so malformed/deep parser output cannot overflow
    Python's recursion limit.
    """
    if isinstance(node, int):
        return node
    result: Dict[int, Union[TreeNode, Leaf]] = {}
    stack: List[Tuple[str, Union[TreeNode, Leaf]]] = [("node", node)]
    while stack:
        phase, current = stack.pop()
        if phase == "node":
            if isinstance(current, int):
                result[id(current)] = current
                continue
            stack.append(("ready", current))
            for child in reversed(current[1]):
                stack.append(("node", child))
            continue

        label, original_children = current
        children = [result[id(child)] for child in original_children]
        while len(children) == 1 and not isinstance(children[0], int):
            child_label, child_children = children[0]
            label = f"{label}{join_char}{child_label}"
            children = child_children
        result[id(current)] = (label, children)
    return result[id(node)]


def binarize_tree(node: Union[TreeNode, Leaf], direction: str = "right") -> Union[TreeNode, Leaf]:
    """Binarize a non-binary constituency tree (left/right spine direction).

    The ``direction`` parameter is named to match NLTK's
    :meth:`~nltk.tree.Tree.chomsky_normal_form` ``factor`` argument, so the
    name denotes the **recursion direction of the introduced spine** (not which
    child is kept): ``"right"`` = right-recursive spine, ``"left"`` =
    left-recursive spine.

    Leaves (ints) are unchanged. For an internal node with children
    ``[c1, c2, ..., ck]`` (k>2):

    * ``direction="right"`` (default): ``(X c1 (X|< c2 (X|< c3 c4)))`` — a
      **right-recursive** spine (NLTK ``factor="right"``). The leftmost child
      ``c1`` stays the parent's direct left child; the remaining ``[c2..ck]``
      fold right-to-left into a right-recursive chain (innermost pair
      ``(ck-1, ck)``) hung on the right. This reproduces the official TreeReg
      ``binarize_tree`` (``ananjan-nandi-9/tree_regularization``,
      ``src/data/data_utils.py``), which returns ``(c1, (c2, (c3, ...)))``, and
      NLTK ``chomsky_normal_form()``'s default (``factor="right"``). The
      introduced nodes reuse the parent label suffixed with ``"|<"`` so they are
      distinguishable but still phrasal.
    * ``direction="left"``: ``(X (X|> (X|> c1 c2) c3) c4)`` — a
      **left-recursive** spine (NLTK ``factor="left"``), the mirror of
      ``"right"``. The rightmost child ``ck`` stays the parent's direct right
      child; the remaining ``[c1..ck-1]`` fold left-to-right into a left-recursive
      chain (innermost pair ``(c1, c2)``) hung on the left. This is a repo-local
      mirror direction with no counterpart in official TreeReg (which binarizes
      in one direction only); it matches the "Binary" variant of the
      compositional_SLMs corpus (e.g. ``1100_dev_doc.txt``, left-recursive).

    Leaves are preserved in order, so the terminal sequence is identical pre/post
    binarization. This function leaves unary nodes unchanged; the TreeReg
    preprocessing pipeline calls :func:`collapse_unary_tree` first, matching the
    official target extractor's drill-through behavior.
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
            if direction == "right":
                # Right-recursive spine (NLTK factor="right"), matching official
                # TreeReg ``binarize_tree`` (ananjan-nandi-9/tree_regularization,
                # src/data/data_utils.py), which returns ``(c1, (c2, (c3, ...)))``
                # for children [c1..ck]: c1 stays the parent's left child; [c2..ck]
                # fold into a right-recursive chain hung on the right, innermost
                # pair (ck-1, ck). Fold [c2..ck] RIGHT-TO-LEFT to get that.
                # (An earlier left-to-right fold produced ``(c1, ((c2 c3) c4)...)``
                # — a left-recursive spine that is NOT what TreeReg uses.)
                tail = children[-1]
                for c in reversed(children[1:-1]):
                    tail = (f"{label}|<", [c, tail])
                result[id(node_i)] = (label, [children[0], tail])
            elif direction == "left":
                # Left-recursive spine (NLTK factor="left"), the mirror of "right".
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
    """Enumerate a tree's content leaves and constituent spans (no binarization).

    Works on a RAW (non-binarized) tree: emits exactly one ``(left, split, right)``
    span per real internal node, so the span set is the tree's true constituents
    (no artificial ``X|<``/``X|>`` nodes). This is the form the Pushdown stack tape
    needs — :func:`compute_depth_matrix` uses only ``(left, right)`` and must NOT
    count binarization-induced artificial constituents (they would inflate depth).

    Returns ``(leaves, spans)`` where ``leaves`` is the in-order list of leaf token
    ids (the terminal content), and ``spans`` is a list of ``(left, split, right)``
    for each *internal* node: the node spans leaves ``[left..right]``. ``split`` is:

    * the real bifurcation point for a binary node (left child = ``[left..split]``,
      right child = ``[split+1..right]``) — what TreeReg's CE loss consumes;
    * **degenerate** (``split == right``) for unary (1 child) and n-ary (>2 children)
      nodes, signalling "no binary bifurcation imposed". Callers that need a real
      binary split (e.g. TreeReg CE) should :func:`binarize_tree` first; callers that
      only use ``(left, right)`` (e.g. the Pushdown stack tape) may skip binarization
      and ignore ``split``.
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
                # Non-binarized n-ary node: one real constituent over [left..right]
                # with NO imposed binary bifurcation -> degenerate split (split==right),
                # matching the unary convention. compute_depth_matrix (the Pushdown
                # stack tape) uses only (left, right) so this is correct for it;
                # callers needing a real binary split (TreeReg CE) must binarize_tree
                # first (which replaces this node with nested binary children).
                spans.append((left, right, right))
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
# Tree-stream chunk iterator (for preprocessing / dataset indexing)
# --------------------------------------------------------------------------- #
def _binarize_segments(
    segments: List[Segment],
    direction: str,
    binarize: bool = True,
    collapse_unary: bool = False,
) -> List[Segment]:
    """Convert each tree segment into ``(leaves, spans, word_boundaries)``.

    When ``binarize=True`` (the paper-faithful TreeReg/Pushdown setting),
    :func:`binarize_tree` the tree first so every internal node is binary and
    ``split`` is a real bifurcation point. ``binarize=False`` is retained for
    legacy raw-tree ablations: one span is emitted per original internal node and
    n-ary nodes receive a degenerate split.
    """
    out: List[Segment] = []
    for kind, data in segments:
        if kind == "leaves":
            out.append((kind, data))
        else:
            original_leaves, word_boundaries = tree_leaves_and_word_boundaries(data)
            tree = collapse_unary_tree(data) if collapse_unary else data
            btree = binarize_tree(tree, direction) if binarize else tree
            leaves, spans = tree_spans(btree)
            if leaves != original_leaves:
                raise ValueError("unary collapse/binarization changed terminal leaves")
            out.append(("tree", (leaves, spans, word_boundaries)))
    return out
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
    """Phase B: record top-level opens/closes for blocks [bi_lo, bi_hi),
    given the true per-block depth_carry / leaf_carry arrays (from phase A's prefix
    scan). Output is O(#top-level-trees) (~5M for train), NOT O(#NTs) (~2.4B).

    Returns ``(open_pos, open_leaf, close_pos, close_leaf)``. The cumulative leaf
    values are measured immediately before the opening and at the closing token,
    respectively, so their difference is the number of terminal leaves inside a
    top-level tree. Recording openings is required when preprocessing synthesizes
    per-tree ROOT/EOS boundaries and must discard document-level leaves outside
    the parses.
    """
    open_pos: List[np.ndarray] = []
    open_leaf: List[np.ndarray] = []
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
        opens = np.nonzero(is_op & (d == 1))[0]
        closes = np.nonzero(is_cl & (d == 0))[0]
        if opens.size:
            open_pos.append((b0 + opens).astype(np.int64))
            open_leaf.append(leaf_cum[opens].astype(np.int64))
        if closes.size:
            close_pos.append((b0 + closes).astype(np.int64))
            close_leaf.append(leaf_cum[closes].astype(np.int64))
    if not open_pos and not close_pos:
        return None, None, None, None
    return (
        np.concatenate(open_pos) if open_pos else None,
        np.concatenate(open_leaf) if open_leaf else None,
        np.concatenate(close_pos) if close_pos else None,
        np.concatenate(close_leaf) if close_leaf else None,
    )


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
                     tree_npy: Optional[str] = None,
                     wrap_toplevel_trees: bool = False) -> List[Tuple[int, int]]:
    """Scan the tree-token stream and return chunk spans ``(tree_start, tree_len)``
    with whole-tree integrity.

    A chunk is a maximal run of complete top-level parse trees (plus intervening
    leaf tokens like BOS/EOS/whitespace) whose total **terminal-leaf** count is
    ``<= max_len``. With ``wrap_toplevel_trees=True``, outside leaves are discarded
    and two synthesized tokens (ROOT and EOS) are counted for every tree. Top-level
    trees are found with a bracket-depth scan (a tree
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
    open_pos_parts: List[np.ndarray] = []
    open_leaf_parts: List[np.ndarray] = []
    close_pos_parts: List[np.ndarray] = []
    close_leaf_parts: List[np.ndarray] = []
    if workers and workers > 1 and n_blocks >= 2 and tree_npy:
        argsB = [(tree_npy, tokenizer_path, BLOCK, block_starts_arr,
                  depth_carry, leaf_carry, r) for r in ranges]
        with Pool(workers) as pool:
            cl_results = pool.map(_toplevel_closes_worker, argsB)
        for op, ol, cp, cl in cl_results:
            if op is not None and op.size:
                open_pos_parts.append(op)
                open_leaf_parts.append(ol)
            if cp is not None and cp.size:
                close_pos_parts.append(cp)
                close_leaf_parts.append(cl)
    else:
        op, ol, cp, cl = _toplevel_closes_inline(
            tree_arr, vocab, BLOCK, block_starts_arr,
            depth_carry, leaf_carry, 0, n_blocks
        )
        if op is not None and op.size:
            open_pos_parts.append(op)
            open_leaf_parts.append(ol)
        if cp is not None and cp.size:
            close_pos_parts.append(cp)
            close_leaf_parts.append(cl)

    # Tree k spans [close_idxs[k-1]+1, close_idxs[k]+1) (close is inclusive).
    closes_arr = (np.concatenate(close_pos_parts) if close_pos_parts
                  else np.zeros(0, np.int64))
    T = len(closes_arr)
    if T == 0:
        return [] if wrap_toplevel_trees else ([(0, n)] if n > 0 else [])
    tree_ends = closes_arr + 1  # exclusive end
    clc = (np.concatenate(close_leaf_parts) if close_leaf_parts
           else np.zeros(0, np.int64))
    if wrap_toplevel_trees:
        opens_arr = (np.concatenate(open_pos_parts) if open_pos_parts
                     else np.zeros(0, np.int64))
        olc = (np.concatenate(open_leaf_parts) if open_leaf_parts
               else np.zeros(0, np.int64))
        if len(opens_arr) != T or len(olc) != T:
            raise ValueError(
                "malformed tree stream: top-level opening/closing counts differ "
                f"({len(opens_arr)} opens, {T} closes)"
            )
        if np.any(opens_arr >= closes_arr):
            raise ValueError("malformed tree stream: top-level close precedes opening")
        tree_starts = opens_arr
        # The two synthetic leaves are [ROOT] and [EOS].
        tree_leaves = (clc - olc) + 2
    else:
        # Legacy terminal-preserving mode includes leaves outside/between parses.
        tree_starts = np.empty(T, dtype=np.int64)
        tree_starts[0] = 0
        tree_starts[1:] = closes_arr[:-1] + 1
        tree_leaves = clc.copy()
        tree_leaves[1:] = clc[1:] - clc[:-1]

    # Pass 2 (vectorized greedy packing): pack trees into chunks of <= max_len
    # leaves. A tree whose leaves exceed max_len is its own (oversize) chunk.
    #
    # For every candidate start tree k, the furthest tree e with
    # sum(leaves[k..e]) <= max_len is found in ONE vectorized searchsorted on the
    # prefix-sum array, clamped to [k, T-1] (an oversize tree maps to e==k, i.e.
    # its own chunk).
    # The chunks are then the pointer chain 0 -> e[0]+1 -> e[e[0]+1]+1 -> ...,
    # which advances by a full chunk per step — O(#chunks) iterations (~5M for
    # train), NOT O(#trees) (~250M). The old per-tree Python loop was the
    # bottleneck that stalled the 24.7B-token train scan for >15 minutes.
    leaves = tree_leaves.astype(np.int64)
    csum = np.concatenate(([0], np.cumsum(leaves)))  # csum[k] = sum leaves[0..k-1]
    # For start k: target prefix = (leaves before k) + max_len = csum[k] + max_len.
    # We want the largest tree index e with sum(leaves[k..e]) <= max_len, i.e.
    # csum[e+1] - csum[k] <= max_len, i.e. csum[e+1] <= target.
    # searchsorted(csum, target, 'right') returns the first position p with
    # csum[p] > target, so csum[p-1] <= target and the largest valid e+1 is p-1,
    # i.e. e = p - 2. (Using p-1 here was an off-by-one: it gave csum[e+1] > target,
    # so chunks packed max_len+overflow leaves and ParseAlignedDataset truncated
    # the tail — dropping ~0.7% of dev tokens.) Clamp so e >= k: an atomic tree
    # whose own leaves exceed max_len has p = k+1, e = k-1 -> clamp to k (own chunk).
    targets = csum[:T] + max_len                 # one per start tree
    e_arr = np.searchsorted(csum, targets, side="right") - 2
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
    if not wrap_toplevel_trees and last_end < n:
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


def parse_chunk_slice(
    tree_slice: Sequence[int],
    vocab: TreeVocab,
    direction: str,
    binarize: bool = True,
    collapse_unary: bool = False,
    add_boundary_root: bool = False,
    wrap_toplevel_trees: bool = False,
    root_token_id: Optional[int] = None,
    sentence_eos_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse one chunk's tree-token slice -> terminal leaves (input_ids) + spans.

    This is what the data loader calls per chunk. Returns:
    * ``input_ids``: ``(T,)`` int64 — terminal leaves (== terminal.npy tokens).
    * ``spans``: ``(M, 3)`` int32 — constituent ``(left, split, right)`` over input_ids.
    * ``word_boundaries``: ``(T,)`` bool — first BPE token of each parser word.
    * ``sentence_ids``: ``(T,)`` int32 — top-level parse-tree id, or ``-1`` for
      BOS/EOS/whitespace outside a parsed tree.

    ``binarize=True`` binarizes the tree so every span's ``split`` is a real
    bifurcation (required by TreeReg and used by the official Pushdown pipeline).
    ``binarize=False`` spans the raw tree for legacy ablations.

    ``wrap_toplevel_trees=True`` implements the Pushdown paper's sentence
    convention exactly: discard document-level leaves outside parses, turn each
    top-level tree into ``[ROOT] tree-leaves [EOS]``, and add the EOS-to-ROOT
    attachment span. ``root_token_id`` and ``sentence_eos_token_id`` are then
    required. This mode is mutually exclusive with the legacy
    ``add_boundary_root`` option, which only spans boundary tokens already in the
    source stream.
    """
    if add_boundary_root and wrap_toplevel_trees:
        raise ValueError("add_boundary_root and wrap_toplevel_trees are mutually exclusive")
    if wrap_toplevel_trees and (
        root_token_id is None or sentence_eos_token_id is None
    ):
        raise ValueError(
            "wrap_toplevel_trees requires root_token_id and sentence_eos_token_id"
        )
    segments = _binarize_segments(
        parse_block_segments(tree_slice, vocab),
        direction,
        binarize,
        collapse_unary,
    )
    terminals: List[int] = []
    spans: List[Span] = []
    word_boundaries: List[bool] = []
    sentence_ids: List[int] = []
    sentence_id = 0
    for kind, data in segments:
        if kind == "leaves":
            if wrap_toplevel_trees:
                continue
            plain = [int(x) for x in data]
            terminals.extend(plain)
            word_boundaries.extend([False] * len(plain))
            sentence_ids.extend([-1] * len(plain))
        else:
            leaves, tspans, tree_word_boundaries = data
            t0 = len(terminals)
            if wrap_toplevel_trees:
                terminals.append(int(root_token_id))
                word_boundaries.append(False)
                sentence_ids.append(-1)
            content_start = len(terminals)
            terminals.extend(int(x) for x in leaves)
            word_boundaries.extend(bool(x) for x in tree_word_boundaries)
            sentence_ids.extend([sentence_id] * len(leaves))
            for (l, sp, r) in tspans:
                spans.append((content_start + l, content_start + sp, content_start + r))
            if wrap_toplevel_trees:
                eos_position = len(terminals)
                terminals.append(int(sentence_eos_token_id))
                word_boundaries.append(False)
                sentence_ids.append(-1)
                spans.append((t0, eos_position, eos_position))
            sentence_id += 1
    if add_boundary_root:
        # The paper initializes every sentence with a ROOT token and forces EOS
        # to attach to it. Here we can add that span only for explicit
        # [BOS ... EOS] blocks already present in the terminal stream. A corpus
        # with one BOS/EOS pair per document rather than per top-level parse is a
        # terminal-preserving adaptation, not the paper's per-sentence boundary
        # convention.
        root_start: Optional[int] = None
        for position, token in enumerate(terminals):
            if vocab.bos == vocab.eos:
                # GPT-2 uses the same id for BOS and EOS. Boundary occurrences
                # therefore alternate ROOT, EOS, ROOT, EOS (including adjacent
                # EOS/ROOT pairs between packed sentences).
                if token == vocab.bos:
                    if root_start is None:
                        root_start = position
                    else:
                        spans.append((root_start, position, position))
                        root_start = None
            else:
                if token == vocab.bos:
                    root_start = position
                elif token == vocab.eos and root_start is not None:
                    spans.append((root_start, position, position))
                    root_start = None
    return {
        "input_ids": np.asarray(terminals, dtype=np.int64),
        "spans": np.asarray(spans, dtype=np.int32).reshape(-1, 3),
        "word_boundaries": np.asarray(word_boundaries, dtype=np.bool_),
        "sentence_ids": np.asarray(sentence_ids, dtype=np.int32),
    }


# --------------------------------------------------------------------------- #
# Dataset (plugs into the OLMo train/eval loaders)
# --------------------------------------------------------------------------- #
class ParseAlignedDataset(torch.utils.data.Dataset):
    """Dataset for the Pushdown/TreeReg baselines.

    Serves whole-tree chunks (from a precomputed ``chunk_index.npy`` over
    ``tree/*.npy``) as ``{input_ids, tree_spans, tree_span_mask, attention_mask}``.
    The terminal ``input_ids`` are the chunk's tree leaves (bit-identical to
    ``terminal/*.npy``); ``tree_spans`` are the constituent spans over those leaves
    (binarized after unary collapse for paper-faithful TreeReg and Pushdown data).
    The Pushdown depth matrix is computed on the GPU in the model forward (not here).
    Duck-types :class:`MemMapDataset` (``__len__``/``__getitem__`` + an ``offsets``
    property) so the existing :class:`IterableDataset` wrapper and samplers work
    unchanged.
    """

    def __init__(
        self,
        tree_npy: str,
        chunk_index_npy: str,
        tokenizer: str,
        direction: str = "right",
        max_len: int = 2048,
        pad_token_id: int = 50258,
        eos_token_id: int = 50256,
        generate_attention_mask: bool = True,
        include_instance_metadata: bool = True,
        binarize: bool = True,
        collapse_unary: bool = False,
        add_boundary_root: bool = False,
        wrap_toplevel_trees: bool = False,
        root_token_id: Optional[int] = None,
        treereg_metadata: bool = False,
        generate_doc_lengths: bool = False,
    ):
        import os
        self.tree_npy = tree_npy
        self._include_instance_metadata = include_instance_metadata
        self._metadata = {"path": str(tree_npy)}
        self._chunk_index = np.load(chunk_index_npy)  # (n_chunks, 2): tree_start, tree_len
        self._tree = np.load(tree_npy, mmap_mode="r")
        # Random chunk-index access into tree.npy -> disable readahead (cgroup OOM
        # guard; see PrecomputedParseDataset / _madvise_random rationale).
        _madvise_random(self._tree)
        self.vocab = TreeVocab.from_tokenizer_file(tokenizer)
        self.direction = direction
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.generate_attention_mask = generate_attention_mask
        self.generate_doc_lengths = generate_doc_lengths
        self.binarize = binarize
        self.collapse_unary = collapse_unary
        self.add_boundary_root = add_boundary_root
        self.wrap_toplevel_trees = wrap_toplevel_trees
        self.root_token_id = root_token_id
        self.treereg_metadata = treereg_metadata
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
        out = parse_chunk_slice(
            np.asarray(self._tree[s : s + l]),
            self.vocab,
            self.direction,
            self.binarize,
            collapse_unary=self.collapse_unary,
            add_boundary_root=self.add_boundary_root,
            wrap_toplevel_trees=self.wrap_toplevel_trees,
            root_token_id=self.root_token_id,
            sentence_eos_token_id=self.eos_token_id,
        )
        input_ids = out["input_ids"]
        spans = out["spans"]
        word_boundaries = out["word_boundaries"]
        sentence_ids = out["sentence_ids"]
        # Truncate to max_len (atomic trees slightly over max_len).
        if len(input_ids) > self.max_len:
            full_sentence_ids = sentence_ids
            input_ids = input_ids[: self.max_len]
            spans = spans[(spans[:, 2] < self.max_len) & (spans[:, 0] < self.max_len)] if len(spans) else spans
            word_boundaries = word_boundaries[: self.max_len]
            sentence_ids = sentence_ids[: self.max_len]
            # Never regularize a prefix of an overlength top-level tree.  At most
            # one contiguous top-level tree can cross the truncation boundary.
            crossing_id = int(sentence_ids[-1]) if len(sentence_ids) else -1
            if crossing_id >= 0 and np.any(full_sentence_ids[self.max_len :] == crossing_id):
                crossing = sentence_ids == crossing_id
                sentence_ids = sentence_ids.copy()
                word_boundaries = word_boundaries.copy()
                sentence_ids[crossing] = -1
                word_boundaries[crossing] = False
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        item: Dict[str, Any] = {"input_ids": input_ids}
        if self.wrap_toplevel_trees:
            # ROOT is an input-only SOS token for each independent sentence; it
            # must never be an LM prediction target when sentences are packed.
            item["label_mask"] = input_ids != int(self.root_token_id)
        if len(spans):
            tspans = torch.tensor(spans, dtype=torch.long)
            item["tree_spans"] = tspans
            item["tree_span_mask"] = torch.ones(len(tspans), dtype=torch.bool)
        else:
            item["tree_spans"] = torch.zeros((0, 3), dtype=torch.long)
            item["tree_span_mask"] = torch.zeros((0,), dtype=torch.bool)
        if self.generate_attention_mask:
            item["attention_mask"] = torch.ones(len(input_ids), dtype=torch.bool)
        if self.treereg_metadata:
            item["treereg_word_boundaries"] = torch.tensor(word_boundaries, dtype=torch.bool)
            item["treereg_sentence_ids"] = torch.tensor(sentence_ids, dtype=torch.int32)
        # Document lengths (eval-only): split the chunk by EOS so the model can
        # apply intra-document attention masking for faithful PPL. The chunk packs
        # multiple complete top-level trees, so it contains multiple documents.
        if self.generate_doc_lengths:
            from .util import get_document_lengths
            item["doc_lens"] = get_document_lengths(input_ids, self.eos_token_id)
        # Instance metadata for the `lm` evaluator (see PrecomputedParseDataset).
        if self._include_instance_metadata:
            from copy import deepcopy
            item["metadata"] = deepcopy(self._metadata)
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
                 eos_token_id: int = 50256,
                 load_depth: bool = False, include_instance_metadata: bool = True,
                 generate_doc_lengths: bool = False,
                 require_treereg_metadata: bool = False,
                 require_pushdown_root_token_id: Optional[int] = None,
                 expected_binarize_direction: Optional[str] = None):
        import os
        self.data_dir = data_dir
        if os.path.exists(os.path.join(data_dir, "PREPROCESSING_INCOMPLETE")):
            raise RuntimeError(
                f"{data_dir} is an interrupted preprocessing output and must "
                "not be loaded; remove it and preprocess again"
            )
        if require_pushdown_root_token_id is not None:
            import json
            manifest_path = os.path.join(data_dir, "preprocessing.json")
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    f"{data_dir} has no preprocessing.json and cannot be verified "
                    "as per-tree ROOT/EOS Pushdown data; regenerate it with "
                    "scripts/precompute_pushdown_unary.py"
                )
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            required_contract = {
                "binarize": True,
                "collapse_unary": True,
                "wrap_toplevel_trees": True,
                "root_token_id": int(require_pushdown_root_token_id),
                "sentence_eos_token_id": int(eos_token_id),
            }
            if expected_binarize_direction is not None:
                required_contract["direction"] = expected_binarize_direction
            mismatches = {
                key: (manifest.get(key), expected)
                for key, expected in required_contract.items()
                if manifest.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"{data_dir} does not match the configured Pushdown "
                    f"preprocessing contract: {mismatches}"
                )
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.pushdown_root_token_id = require_pushdown_root_token_id
        self.generate_doc_lengths = generate_doc_lengths
        self._include_instance_metadata = include_instance_metadata
        self._metadata = {"path": str(data_dir)}
        self.input_ids = np.load(os.path.join(data_dir, "input_ids.npy"), mmap_mode="r")
        # New preprocessing writes the chosen dtype directly to canonical
        # ``spans.npy`` (TreeReg uses int16). Retain a fallback for legacy outputs
        # that stored compact spans under ``spans_int16.npy``. __getitem__ builds
        # torch.long tensors, so either on-disk integer dtype is upcast to int64.
        spans_path = os.path.join(data_dir, "spans.npy")
        if not os.path.exists(spans_path):
            spans_path = os.path.join(data_dir, "spans_int16.npy")
        self.spans = np.load(spans_path, mmap_mode="r")
        self.span_counts = np.load(os.path.join(data_dir, "span_counts.npy"))
        word_boundaries_path = os.path.join(data_dir, "treereg_word_boundaries.npy")
        sentence_ids_path = os.path.join(data_dir, "treereg_sentence_ids.npy")
        have_treereg_metadata = (
            os.path.exists(word_boundaries_path) and os.path.exists(sentence_ids_path)
        )
        if require_treereg_metadata and not have_treereg_metadata:
            raise FileNotFoundError(
                f"TreeReg metadata is missing from {data_dir}: expected "
                "treereg_word_boundaries.npy and treereg_sentence_ids.npy. "
                "Regenerate this split with scripts/precompute_treereg.py; the "
                "legacy *_right data is not paper-faithful."
            )
        self.word_boundaries = (
            np.load(word_boundaries_path, mmap_mode="r") if have_treereg_metadata else None
        )
        self.sentence_ids = (
            np.load(sentence_ids_path, mmap_mode="r") if have_treereg_metadata else None
        )
        # These mmaps are read by RANDOM chunk index across 4.88M chunks. Hint the
        # kernel MADV_RANDOM so it stops readaheading (the default readahead hoards
        # a large fraction of the 189 GB file into page cache -> cgroup OOM under
        # slurm). See _madvise_random for the full rationale.
        _madvise_random(self.input_ids)
        _madvise_random(self.spans)
        if self.word_boundaries is not None:
            _madvise_random(self.word_boundaries)
        if self.sentence_ids is not None:
            _madvise_random(self.sentence_ids)
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
        if self.pushdown_root_token_id is not None:
            item["label_mask"] = attention_mask & (
                input_ids != int(self.pushdown_root_token_id)
            )
        if self.word_boundaries is not None and self.sentence_ids is not None:
            item["treereg_word_boundaries"] = torch.tensor(
                np.asarray(self.word_boundaries[index]), dtype=torch.bool
            )
            item["treereg_sentence_ids"] = torch.tensor(
                np.asarray(self.sentence_ids[index]), dtype=torch.int32
            )
        if self.depth is not None:
            item["tree_depth_matrix"] = torch.tensor(
                np.asarray(self.depth[index]), dtype=torch.int8)
        # Document lengths (eval-only): split by EOS so the model applies
        # intra-document attention masking for faithful PPL. Computed on the FULL
        # (padded) input_ids, matching MemMapDataset: trailing pads (pad != EOS)
        # fall into a final trailing "doc" so cu_doc_lens covers every seq_len
        # position (required by flash_attn_varlen_func on the treereg path); pads
        # are masked out of the CE by get_labels via attention_mask, and out of
        # pushdown attention by am[b, kv_idx].
        if self.generate_doc_lengths:
            from .util import get_document_lengths
            item["doc_lens"] = get_document_lengths(input_ids, self.eos_token_id)
        # Instance metadata: the `lm` evaluator (evaluator.py EvaluatorType.lm) zips
        # `batch["metadata"]` with per-instance CE loss, so every eval item must emit
        # one. MemMapDataset emits {"path": ...}; mirror that here (single data dir,
        # so the dict is constant across instances). Built in __init__ + deep-copied
        # per item (matching MemMapDataset's deepcopy at memmap_dataset.py:263).
        if self._include_instance_metadata:
            from copy import deepcopy
            item["metadata"] = deepcopy(self._metadata)
        return item


# --------------------------------------------------------------------------- #
# Offline preprocessing: binarize trees + generate the stale depth matrix
# --------------------------------------------------------------------------- #
def _available_memory_bytes() -> int:
    """Read Linux MemAvailable without adding a psutil dependency."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


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
    binarize: bool = True,
    collapse_unary: bool = False,
    add_boundary_root: bool = False,
    wrap_toplevel_trees: bool = False,
    root_token_id: Optional[int] = None,
    sentence_eos_token_id: Optional[int] = None,
    save_treereg_metadata: bool = False,
    input_dtype: Any = np.int32,
    span_dtype: Any = np.int32,
    pin_workers: bool = False,
    worker_task_multiplier: int = 32,
    min_free_disk_bytes: int = 0,
    min_free_memory_bytes: int = 0,
) -> Dict[str, str]:
    """Process a ``tree/*.npy`` parse stream into chunks + the stale
    Pushdown depth matrix, saved to ``out_dir``.

    For each whole-tree chunk (``<= max_len`` terminal leaves, packed by
    :func:`iter_tree_chunks`): extract terminal leaves (``input_ids`` — bit-identical
    to ``terminal/*.npy``) and constituent spans, and (optionally) compute the stale
    depth matrix ``S[k,j]`` (int8). ``binarize=True`` makes every span binary;
    ``binarize=False`` is a legacy raw-tree ablation. ``collapse_unary=True``
    removes unary-chain duplicates before span extraction; together with
    ``binarize=True`` this reproduces the paper/reference preprocessing.
    ``add_boundary_root=True`` adds a span across each explicit BOS/ROOT-to-EOS
    block so its final attachment is supervised to ROOT. It does not synthesize
    missing per-sentence boundary tokens. ``wrap_toplevel_trees=True`` instead
    emits ``[ROOT] tree-leaves [EOS]`` for every top-level tree and counts those
    added tokens during chunk packing. Saves:

    * ``chunk_index.npy`` ``(n_chunks, 2)`` int64 — tree-stream slice per chunk.
    * ``input_ids.npy`` ``(n_chunks, max_len)`` — terminal leaves, padded.
      ``input_dtype`` defaults to int32; uint16 is lossless for this tokenizer.
    * ``spans.npy`` ``(n_chunks, max_spans, 3)`` — constituent spans, padded
      with -1. ``span_dtype`` defaults to int32; int16 is lossless for sequences
      shorter than 32768 and is written directly without a conversion pass.
    * ``span_counts.npy`` ``(n_chunks,)`` int32 — valid spans per chunk.
    * ``treereg_word_boundaries.npy`` ``(n_chunks, max_len)`` bool — exact
      parser-word starts. Only when ``save_treereg_metadata``.
    * ``treereg_sentence_ids.npy`` ``(n_chunks, max_len)`` int16 — complete
      top-level tree id per token, ``-1`` outside parsed trees. Only when
      ``save_treereg_metadata``.
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
    incomplete_marker = os.path.join(out_dir, "PREPROCESSING_INCOMPLETE")
    complete_marker = os.path.join(out_dir, "PREPROCESSING_COMPLETE")

    input_dtype = np.dtype(input_dtype)
    span_dtype = np.dtype(span_dtype)
    if input_dtype.kind not in ("u", "i"):
        raise ValueError(f"input_dtype must be an integer dtype, got {input_dtype}")
    if span_dtype.kind != "i":
        raise ValueError(
            f"span_dtype must be signed so padding -1 is representable, got {span_dtype}"
        )
    input_info = np.iinfo(input_dtype)
    if not input_info.min <= pad_token_id <= input_info.max:
        raise ValueError(
            f"pad_token_id={pad_token_id} does not fit input_dtype={input_dtype}"
        )
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if worker_task_multiplier < 1:
        raise ValueError("worker_task_multiplier must be at least 1")
    if add_boundary_root and wrap_toplevel_trees:
        raise ValueError("add_boundary_root and wrap_toplevel_trees are mutually exclusive")
    if wrap_toplevel_trees and (
        root_token_id is None or sentence_eos_token_id is None
    ):
        raise ValueError(
            "wrap_toplevel_trees requires root_token_id and sentence_eos_token_id"
        )

    vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
    if warm_cache:
        _warm_tree_cache(tree_npy)
    tree_mmap = np.load(tree_npy, mmap_mode="r")
    chunks = iter_tree_chunks(tree_mmap, vocab, direction=direction, max_len=max_len,
                              workers=scan_workers, tree_npy=tree_npy,
                              wrap_toplevel_trees=wrap_toplevel_trees)
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
        available_memory = _available_memory_bytes()
        required_memory = int(tree_mmap.nbytes) + int(min_free_memory_bytes)
        if available_memory < required_memory:
            raise RuntimeError(
                "insufficient memory for --load-tree-to-ram: "
                f"available={available_memory / 2**30:.1f} GiB, "
                f"tree={tree_mmap.nbytes / 2**30:.1f} GiB, "
                f"required reserve={min_free_memory_bytes / 2**30:.1f} GiB"
            )
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
            out = parse_chunk_slice(np.asarray(_tree_for_sample[s : s + l]), vocab,
                                    direction, binarize, collapse_unary, add_boundary_root,
                                    wrap_toplevel_trees, root_token_id,
                                    sentence_eos_token_id)
            if len(out["spans"]) > max_spans:
                max_spans = len(out["spans"])
        max_spans = max(max_spans + 8, 1)  # headroom
    print(f"  max_spans={max_spans}")

    estimated_output_bytes = (
        n_chunks * max_len * input_dtype.itemsize
        + n_chunks * max_spans * 3 * span_dtype.itemsize
        + n_chunks * np.dtype(np.int32).itemsize
        + n_chunks * 2 * np.dtype(np.int64).itemsize
    )
    if save_treereg_metadata:
        estimated_output_bytes += n_chunks * max_len * (
            np.dtype(np.bool_).itemsize + np.dtype(np.int16).itemsize
        )
    if save_depth_matrix:
        estimated_output_bytes += n_chunks * max_len * max_len
    disk = __import__("shutil").disk_usage(out_dir)
    print(
        f"  estimated output={estimated_output_bytes / 2**30:.1f} GiB; "
        f"disk free={disk.free / 2**30:.1f} GiB; "
        f"required reserve={min_free_disk_bytes / 2**30:.1f} GiB",
        flush=True,
    )
    if disk.free - estimated_output_bytes < min_free_disk_bytes:
        raise RuntimeError(
            "insufficient disk headroom: estimated output would leave "
            f"{(disk.free - estimated_output_bytes) / 2**30:.1f} GiB, below "
            f"the required {min_free_disk_bytes / 2**30:.1f} GiB reserve"
        )

    with open(incomplete_marker, "w", encoding="utf-8") as marker:
        marker.write("Output is incomplete until PREPROCESSING_COMPLETE exists.\n")
    if os.path.exists(complete_marker):
        os.remove(complete_marker)

    # Preallocate memory-maps (streaming; one chunk in RAM at a time).
    p_chunk_index = os.path.join(out_dir, "chunk_index.npy")
    p_input_ids = os.path.join(out_dir, "input_ids.npy")
    p_spans = os.path.join(out_dir, "spans.npy")
    p_span_counts = os.path.join(out_dir, "span_counts.npy")
    p_word_boundaries = os.path.join(out_dir, "treereg_word_boundaries.npy")
    p_sentence_ids = os.path.join(out_dir, "treereg_sentence_ids.npy")
    np.save(p_chunk_index, np.asarray(chunks, dtype=np.int64))
    input_ids_m = np.lib.format.open_memmap(p_input_ids, mode="w+",
        dtype=input_dtype, shape=(n_chunks, max_len))
    spans_m = np.lib.format.open_memmap(p_spans, mode="w+",
        dtype=span_dtype, shape=(n_chunks, max_spans, 3))
    span_counts_m = np.lib.format.open_memmap(p_span_counts, mode="w+",
        dtype=np.int32, shape=(n_chunks,))
    word_boundaries_m = None
    sentence_ids_m = None
    if save_treereg_metadata:
        word_boundaries_m = np.lib.format.open_memmap(
            p_word_boundaries, mode="w+", dtype=np.bool_, shape=(n_chunks, max_len)
        )
        sentence_ids_m = np.lib.format.open_memmap(
            p_sentence_ids, mode="w+", dtype=np.int16, shape=(n_chunks, max_len)
        )
    # Do not fill these potentially hundreds-of-GB arrays in the parent. That
    # serialized setup previously occupied one CPU for hours before Pool workers
    # even existed. Each worker initializes only its own contiguous rows below.

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
        ranges = _block_ranges(
            n_chunks, min(n_chunks, workers * worker_task_multiplier)
        )
        fn = functools.partial(_parse_range_to_memmap, tree_src, tokenizer_path,
                               direction, max_len, pad_token_id, binarize, collapse_unary,
                               p_input_ids, p_spans, p_span_counts, max_spans,
                               add_boundary_root=add_boundary_root,
                               wrap_toplevel_trees=wrap_toplevel_trees,
                               root_token_id=root_token_id,
                               sentence_eos_token_id=sentence_eos_token_id,
                               p_word_boundaries=(
                                   p_word_boundaries if save_treereg_metadata else None
                               ),
                               p_sentence_ids=(
                                   p_sentence_ids if save_treereg_metadata else None
                               ))
        cpu_ids = _worker_cpu_candidates(workers) if pin_workers else []
        completed = 0
        # Emit roughly one-percent milestones. With 32 tasks per worker, large
        # train runs now return a completed task every few minutes rather than
        # appearing frozen for nearly an hour between synchronized block waves.
        report_every = max(n_chunks // 100, 1)
        next_report = report_every
        with Pool(
            workers,
            initializer=_parse_worker_initializer,
            initargs=(cpu_ids,),
        ) as pool:
            for done in pool.imap_unordered(fn, ranges, chunksize=1):
                completed += done
                if completed >= next_report or completed == n_chunks:
                    print(f"  parsed {completed}/{n_chunks} chunks", flush=True)
                    next_report += report_every
    else:
        _parse_range_to_memmap(tree_src, tokenizer_path, direction, max_len,
                               pad_token_id, binarize, collapse_unary, p_input_ids, p_spans,
                               p_span_counts, max_spans, (0, n_chunks),
                               add_boundary_root=add_boundary_root,
                               wrap_toplevel_trees=wrap_toplevel_trees,
                               root_token_id=root_token_id,
                               sentence_eos_token_id=sentence_eos_token_id,
                               p_word_boundaries=(
                                   p_word_boundaries if save_treereg_metadata else None
                               ),
                               p_sentence_ids=(
                                   p_sentence_ids if save_treereg_metadata else None
                               ))
    _TREE_RAM = None
    input_ids_m.flush(); spans_m.flush(); span_counts_m.flush()
    if word_boundaries_m is not None:
        word_boundaries_m.flush()
    if sentence_ids_m is not None:
        sentence_ids_m.flush()

    saved = {"chunk_index": p_chunk_index, "input_ids": p_input_ids,
             "spans": p_spans, "span_counts": p_span_counts}
    if save_treereg_metadata:
        saved["treereg_word_boundaries"] = p_word_boundaries
        saved["treereg_sentence_ids"] = p_sentence_ids

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

    # Persist the preprocessing contract alongside the arrays. Without this,
    # terminal-preserving legacy Pushdown data and per-tree ROOT/EOS data have
    # identical filenames and can be accidentally interchanged.
    import json
    manifest_path = os.path.join(out_dir, "preprocessing.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "format_version": 2,
                "tree_npy": os.path.abspath(tree_npy),
                "tokenizer_path": os.path.abspath(tokenizer_path),
                "direction": direction,
                "max_len": max_len,
                "pad_token_id": pad_token_id,
                "binarize": binarize,
                "collapse_unary": collapse_unary,
                "add_boundary_root": add_boundary_root,
                "wrap_toplevel_trees": wrap_toplevel_trees,
                "root_token_id": root_token_id,
                "sentence_eos_token_id": sentence_eos_token_id,
                "input_dtype": input_dtype.name,
                "span_dtype": span_dtype.name,
                "n_chunks": n_chunks,
                "max_spans": max_spans,
            },
            manifest_file,
            indent=2,
            sort_keys=True,
        )
        manifest_file.write("\n")
    saved["manifest"] = manifest_path

    print(f"[{out_dir}] done. input_ids ({n_chunks},{max_len}), spans ({n_chunks},{max_spans},3), "
          f"depth {'saved' if save_depth_matrix else 'skipped (compute on GPU at train time)'}")
    # Publish completion before removing the rejection marker. A crash between
    # these operations therefore remains fail-closed rather than exposing a
    # partially published dataset as a legacy marker-less output.
    with open(complete_marker, "w", encoding="utf-8") as marker:
        marker.write("ok\n")
    os.remove(incomplete_marker)
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


def _worker_cpu_candidates(workers: int) -> List[int]:
    """Choose distinct allowed CPUs, preferring one SMT thread per physical core."""
    import os

    allowed = sorted(os.sched_getaffinity(0))
    representatives: List[int] = []
    seen_cores = set()
    for cpu in allowed:
        try:
            with open(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id",
                encoding="utf-8",
            ) as f:
                package = int(f.read())
            with open(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id",
                encoding="utf-8",
            ) as f:
                core = int(f.read())
            key = (package, core)
        except (OSError, ValueError):
            key = (0, cpu)
        if key not in seen_cores:
            seen_cores.add(key)
            representatives.append(cpu)
    candidates = representatives or allowed
    count = min(workers, len(candidates))
    if count == 0:
        return []
    # Spread workers across the allowed physical-core list while leaving all
    # unselected cores free for other users and services.
    return [
        candidates[(idx * len(candidates)) // count]
        for idx in range(count)
    ]


def _parse_worker_initializer(cpu_ids: Sequence[int]) -> None:
    """Pin each parser process to a distinct allowed CPU when requested."""
    if not cpu_ids:
        return
    import multiprocessing
    import os

    identity = multiprocessing.current_process()._identity
    worker_index = (identity[-1] - 1) if identity else 0
    cpu = int(cpu_ids[worker_index % len(cpu_ids)])
    os.sched_setaffinity(0, {cpu})


def _parse_range_to_memmap(tree_src, tokenizer_path, direction, max_len, pad_token_id,
                           binarize, collapse_unary, p_input_ids, p_spans,
                           p_span_counts, max_spans, rng, add_boundary_root=False,
                           wrap_toplevel_trees=False, root_token_id=None,
                           sentence_eos_token_id=None,
                           p_word_boundaries=None, p_sentence_ids=None):
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
    word_boundaries_m = (
        np.lib.format.open_memmap(p_word_boundaries, mode="r+")
        if p_word_boundaries is not None
        else None
    )
    sentence_ids_m = (
        np.lib.format.open_memmap(p_sentence_ids, mode="r+")
        if p_sentence_ids is not None
        else None
    )
    lo, hi = rng
    for i in range(lo, hi):
        s, l = int(chunk_index[i, 0]), int(chunk_index[i, 1])
        out = parse_chunk_slice(
            np.asarray(tree[s : s + l]), vocab, direction, binarize,
            collapse_unary, add_boundary_root, wrap_toplevel_trees,
            root_token_id, sentence_eos_token_id
        )
        ids = out["input_ids"]
        sp = out["spans"]
        word_boundaries = out["word_boundaries"]
        sentence_ids = out["sentence_ids"]
        if len(ids) > max_len:
            full_sentence_ids = sentence_ids
            ids = ids[:max_len]
            sp = sp[sp[:, 2] < max_len] if len(sp) else sp
            word_boundaries = word_boundaries[:max_len]
            sentence_ids = sentence_ids[:max_len]
            crossing_id = int(sentence_ids[-1]) if len(sentence_ids) else -1
            if crossing_id >= 0 and np.any(full_sentence_ids[max_len:] == crossing_id):
                crossing = sentence_ids == crossing_id
                sentence_ids = sentence_ids.copy()
                word_boundaries = word_boundaries.copy()
                sentence_ids[crossing] = -1
                word_boundaries[crossing] = False
        if len(sp) > max_spans:
            raise ValueError(
                f"chunk {i} has {len(sp)} spans but output capacity is {max_spans}; "
                "rerun with --max_spans at least this large"
            )
        input_info = np.iinfo(input_ids_m.dtype)
        if len(ids) and (
            int(ids.min()) < input_info.min or int(ids.max()) > input_info.max
        ):
            raise ValueError(
                f"chunk {i} token range [{int(ids.min())}, {int(ids.max())}] "
                f"does not fit {input_ids_m.dtype}"
            )
        span_info = np.iinfo(spans_m.dtype)
        if len(sp) and (
            int(sp.min()) < span_info.min or int(sp.max()) > span_info.max
        ):
            raise ValueError(
                f"chunk {i} span range [{int(sp.min())}, {int(sp.max())}] "
                f"does not fit {spans_m.dtype}"
            )

        # Initialize this row in the worker so padding is correct without a
        # serial full-file pass in the parent process.
        input_ids_m[i, :] = pad_token_id
        spans_m[i, :, :] = -1
        if word_boundaries_m is not None:
            word_boundaries_m[i, :] = False
        if sentence_ids_m is not None:
            sentence_ids_m[i, :] = -1

        input_ids_m[i, : len(ids)] = ids
        if word_boundaries_m is not None:
            word_boundaries_m[i, : len(ids)] = word_boundaries
        if sentence_ids_m is not None:
            if len(sentence_ids) and int(sentence_ids.max()) > np.iinfo(np.int16).max:
                raise ValueError(f"chunk {i} has too many top-level trees for int16 ids")
            sentence_ids_m[i, : len(ids)] = sentence_ids
        m = len(sp)
        if m > 0:
            spans_m[i, :m] = sp[:m]
        span_counts_m[i] = m
    # NOTE: no per-range flush. Calling flush() (msync) here forces synchronous
    # disk writes that, with 16 parallel workers on a contended mechanical disk,
    # wedge every worker in D state (rq_qos_wait) at ~0 throughput. The memmap
    # pages stay in the page cache and the OS writes them back asynchronously;
    # the single end-of-run flush in preprocess_split persists everything.
    return hi - lo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Parse tree/*.npy into constituent spans + the stale Pushdown depth matrix. "
                    "The default binarizes for TreeReg/Pushdown; --no_binarize is a legacy ablation."
    )
    ap.add_argument("--tree_npy", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--direction",
        default="both",
        choices=["left", "right", "both"],
        help="binarization spine direction, named per NLTK chomsky_normal_form(factor=...): "
             "'right' = right-recursive spine (default TreeReg, NLTK factor='right'); "
             "'left' = left-recursive spine (NLTK factor='left'). 'both' runs each. "
             "NOTE: the <direction> output subdir suffix follows this name, so after the "
             "left/right rename a re-run produces _left=left-recursive / _right=right-recursive; "
             "pre-existing on-disk _left/_right dirs were generated under the OLD naming "
             "(_left was right-recursive) and are NOT auto-renamed — regenerate to refresh.",
    )
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
    ap.add_argument("--no_binarize", action="store_true",
                    help="span the RAW tree (one span per real constituent) instead of binarizing. "
                         "Legacy ablation only; official TreeReg and Pushdown preprocessing binarize.")
    ap.add_argument(
        "--collapse_unary",
        action="store_true",
        help="collapse internal unary chains before span extraction (paper-faithful "
             "Pushdown preprocessing; normally used with binarization enabled)",
    )
    ap.add_argument(
        "--add_boundary_root",
        action="store_true",
        help="add spans across explicit BOS/ROOT-to-EOS blocks; does not insert "
             "missing per-sentence boundary tokens",
    )
    ap.add_argument(
        "--treereg_metadata",
        action="store_true",
        help="save exact top-level-tree ids and preterminal word-start masks for TreeReg",
    )
    args = ap.parse_args()
    dirs = ["left", "right"] if args.direction == "both" else [args.direction]
    for d in dirs:
        preprocess_split(
            args.tree_npy, args.tokenizer, d, f"{args.out_dir}_{d}",
            max_len=args.max_len, pad_token_id=args.pad_token_id,
            save_depth_matrix=not args.no_depth, workers=args.workers,
            scan_workers=args.scan_workers, warm_cache=args.warm_cache,
            max_spans_override=args.max_spans, load_tree_to_ram=args.load_tree_to_ram,
            binarize=not args.no_binarize, collapse_unary=args.collapse_unary,
            add_boundary_root=args.add_boundary_root,
            save_treereg_metadata=args.treereg_metadata,
        )
