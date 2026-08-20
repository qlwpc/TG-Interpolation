"""Supervised (gold-tree) dataset + collator for GPST.

Reads the repo's BBC tree-stream corpus
(``dataset/bbc-news/tree/{train,dev,test}.npy`` — uint16, BOS/NT-bracket
interleaved with GPT-2 token leaves) and converts each sentence's gold
constituency parse into:

* the terminal token-id sequence (the sentence's leaves); and
* a **merge-order** array — the per-sentence ``(L-1,)`` sequence of gap indices
  consumed by ``CPPChartTableManager`` (see Phase 0). For a binarized gold tree,
  each internal node bifurcating at leaf ``split`` contributes gap ``split``;
  nodes are emitted post-order (bottom-up), matching
  ``get_tree_from_merge_trajectory``'s reconstruction.

This is the supervised counterpart to the unsupervised ``GPT2Dataset``
(where merge orders are *predicted* by the top-down parser). The trainer feeds
these gold merge orders directly into the composition model, bypassing the
parser's induction (the parser is still trained, supervised, to predict the
gold split order).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from olmo.data.parse_align import (
    TreeVocab,
    binarize_tree,
    collapse_unary_tree,
    parse_block_segments,
    tree_spans,
)


def tree_to_merge_orders(tree, direction: str = "right") -> Tuple[List[int], List[int]]:
    """Convert a (possibly n-ary) constituency tree to a merge-order sequence.

    Returns ``(leaves, merge_orders)`` where ``leaves`` is the in-order list of
    leaf token ids and ``merge_orders`` is the ``(L-1,)`` gap-index sequence
    (bottom-up post-order) consumed by ``CPPChartTableManager``.

    The tree is first binarized (``direction`` per NLTK
    ``chomsky_normal_form(factor=...)``: ``'right'`` = right-recursive spine,
    default) so every internal node has a real binary bifurcation. For a binary
    node spanning leaves ``[left..right]`` that splits at ``split`` (left child
    ``[left..split]``, right child ``[split+1..right]``), the gap index at merge
    time is ``split`` (0-based, between leaves ``split`` and ``split+1``).
    Emitted post-order (children before parents) so merges happen bottom-up.
    """
    # The released GPST preprocessing collapses unary chains before
    # binarization.  Leaving unary nodes in place creates more internal nodes
    # than gaps and therefore cannot be represented by an L-1 merge trajectory.
    bin_tree = binarize_tree(collapse_unary_tree(tree), direction=direction)
    leaves, spans = tree_spans(bin_tree)
    # spans are already post-order (tree_spans emits children before parents).
    merge_orders = [split for (_l, split, _r) in spans if _l != _r]
    expected = list(range(max(len(leaves) - 1, 0)))
    if sorted(merge_orders) != expected:
        raise ValueError(
            "gold merge orders are not a gap permutation: "
            f"got={merge_orders}, expected={expected}"
        )
    return leaves, merge_orders


def _build_tree_vocab(tokenizer_path: str) -> TreeVocab:
    """Build a TreeVocab from the repo's GPT-2 tokenizer json (NT-bracket range)."""
    return TreeVocab.from_tokenizer_file(tokenizer_path)


class GoldTreeDataset(torch.utils.data.Dataset):
    """Dataset over sentences with gold merge orders.

    Each ``__getitem__`` returns ``{"text": np.ndarray(leaves),
    "sentence_splits": [...], "merge_orders": np.ndarray}``.

    The tree-stream corpus packs whole documents between BOS/EOS and may contain
    many top-level trees.  Each top-level tree is indexed as one supervised
    sample; ``sentence_splits == [len(leaves)]``.
    """

    def __init__(self, tree_npy: str, tokenizer_path: str,
                 max_seq_len: int = 1024, num_samples: Optional[int] = None,
                 direction: str = "right"):
        self.tree_arr = np.load(tree_npy, mmap_mode="r")
        self.vocab = _build_tree_vocab(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.direction = direction
        self._tree_index = self._index_trees()
        self.num_samples = min(num_samples, len(self._tree_index)) \
            if num_samples is not None else len(self._tree_index)

    def _index_trees(self) -> List[Tuple[int, int]]:
        """Find every complete top-level parse tree in the stream.

        A BOS/EOS block is a document and normally contains many top-level
        sentence trees.  The old implementation indexed documents and then
        accidentally returned the prefix before the first tree (usually BOS).
        Supervised GPST examples are sentence trees, so index roots directly.
        """
        arr = self.tree_arr
        trees: List[Tuple[int, int]] = []
        i = 0
        n = len(arr)
        while i < n:
            if self.vocab.is_opening(int(arr[i])):
                start = i
                depth = 1
                j = i + 1
                while j < n and depth:
                    tok = int(arr[j])
                    if self.vocab.is_opening(tok):
                        depth += 1
                    elif self.vocab.is_closing(tok):
                        depth -= 1
                    j += 1
                if depth == 0:
                    trees.append((start, j))
                i = j
            else:
                i += 1
        return trees

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % len(self._tree_index)
        start, end = self._tree_index[idx]
        block = self.tree_arr[start:end].astype(np.int64).tolist()
        segments = parse_block_segments(block, self.vocab)
        trees = [data for kind, data in segments if kind == "tree"]
        if len(trees) != 1:
            raise ValueError(f"expected one tree at [{start}:{end}], found {len(trees)}")
        leaves_ids, merge_orders = tree_to_merge_orders(trees[0], direction=self.direction)
        if not leaves_ids:
            raise ValueError(f"gold tree at [{start}:{end}] has no terminal leaves")
        if len(leaves_ids) > self.max_seq_len:
            raise ValueError(
                f"gold tree has {len(leaves_ids)} leaves, exceeding max_seq_len="
                f"{self.max_seq_len}; truncating would invalidate its parse"
            )
        return {
            "text": np.array(leaves_ids, dtype=np.int32),
            "sentence_splits": [len(leaves_ids)],
            "merge_orders": np.array(merge_orders, dtype=np.int32),
        }


class GoldTreeCollator:
    """Collator for supervised GPST: like DefaultCollator but injects gold
    ``merge_orders`` (padded to ``(N, max_seq_len-1)``) into the batch."""

    def __init__(self, external_vocab_path: str = None):
        from olmo.gpst.reader.data_collator import DefaultCollator
        self._base = DefaultCollator(enable_group=True, external_vocab_path=external_vocab_path)

    def __call__(self, input_list):
        # In the ungrouped case TableManager consumes one local row per segment.
        # In the grouped case its C++ API instead consumes one GLOBAL gap
        # permutation per item and maps those gaps back to the flattened
        # segments.  Supplying segment rows in grouped mode causes an unchecked
        # C++ out-of-bounds read, so construct and validate that representation
        # here.
        item_orders_list = []
        segment_counts = []
        for item in input_list:
            item_orders = item["merge_orders"]
            if isinstance(item_orders, np.ndarray) and item_orders.ndim == 1:
                item_orders = [item_orders]
            expected_segments = 0
            previous = 0
            for split in list(item["sentence_splits"]) + [len(item["text"])]:
                if split > previous:
                    expected_segments += 1
                    previous = split
            if len(item_orders) != expected_segments:
                raise ValueError(
                    f"got {len(item_orders)} merge-order rows for "
                    f"{expected_segments} non-empty segments"
                )
            item_orders_list.append(item_orders)
            segment_counts.append(expected_segments)
        # strip merge_orders before delegating (base collator ignores unknown keys)
        # Copy sentence_splits because DefaultCollator appends the final length.
        base_items = [{"text": item["text"], "sentence_splits": list(item["sentence_splits"])}
                      for item in input_list]
        batch = self._base.generative_r2d2_collate_fn_ext(base_items)
        grouped = any(count > 1 for count in segment_counts)
        if grouped:
            text_lengths = [len(item["text"]) for item in input_list]
            if len(set(text_lengths)) != 1:
                raise ValueError(
                    "grouped gold-tree batches require equal total token lengths; "
                    "batch candidates from one sentence together"
                )
            width = text_lengths[0] - 1
            padded = np.empty((len(input_list), width), dtype=np.int64)
            for item_idx, (item, segment_orders) in enumerate(zip(input_list, item_orders_list)):
                global_orders = []
                boundaries = []
                offset = 0
                splits = list(item["sentence_splits"])
                if not splits or splits[-1] != len(item["text"]):
                    splits.append(len(item["text"]))
                for segment_idx, (end, local_orders) in enumerate(zip(splits, segment_orders)):
                    segment_len = end - offset
                    if len(local_orders) != max(segment_len - 1, 0):
                        raise ValueError(
                            f"segment length {segment_len} requires {segment_len - 1} merges, "
                            f"got {len(local_orders)}"
                        )
                    global_orders.extend(offset + int(order) for order in local_orders)
                    if segment_idx + 1 < len(splits):
                        boundaries.append(end - 1)
                    offset = end
                global_orders.extend(boundaries)
                if sorted(global_orders) != list(range(width)):
                    raise ValueError("grouped merge orders are not a global gap permutation")
                padded[item_idx] = global_orders
        else:
            merge_orders = [orders[0] for orders in item_orders_list]
            max_len = max((len(mo) for mo in merge_orders), default=0)
            padded = np.full((len(merge_orders), max_len), -1, dtype=np.int64)
            for i, mo in enumerate(merge_orders):
                padded[i, :len(mo)] = mo
        batch["merge_orders"] = torch.tensor(padded, dtype=torch.long)
        return batch
