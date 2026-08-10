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
    parse_tree_block,
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
    bin_tree = binarize_tree(tree, direction=direction)
    leaves, spans = tree_spans(bin_tree)
    # spans are already post-order (tree_spans emits children before parents).
    merge_orders = [split for (_l, split, _r) in spans if _l != _r]
    assert len(merge_orders) == max(len(leaves) - 1, 0), \
        f"merge_orders {len(merge_orders)} != leaves-1 {len(leaves)-1}"
    return leaves, merge_orders


def _build_tree_vocab(tokenizer_path: str) -> TreeVocab:
    """Build a TreeVocab from the repo's GPT-2 tokenizer json (NT-bracket range)."""
    return TreeVocab.from_tokenizer_file(tokenizer_path)


class GoldTreeDataset(torch.utils.data.Dataset):
    """Dataset over sentences with gold merge orders.

    Each ``__getitem__`` returns ``{"text": np.ndarray(leaves),
    "sentence_splits": [...], "merge_orders": np.ndarray}``.

    The tree-stream corpus packs sentences separated by BOS/EOS; each
    ``[BOS ... EOS]`` block is one sample. ``text`` is the block's terminal
    leaves; ``sentence_splits`` marks the end of each sentence within the block
    (one sentence per block here, so ``sentence_splits == [len(leaves)]``).
    """

    def __init__(self, tree_npy: str, tokenizer_path: str,
                 max_seq_len: int = 1024, num_samples: Optional[int] = None,
                 direction: str = "right"):
        self.tree_arr = np.load(tree_npy, mmap_mode="r")
        self.vocab = _build_tree_vocab(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.direction = direction
        self._block_index = self._index_blocks()
        self.num_samples = num_samples or len(self._block_index)

    def _index_blocks(self) -> List[Tuple[int, int]]:
        """Find (start, end) of each [BOS ... EOS] block in the tree stream."""
        bos = self.vocab.bos
        eos = self.vocab.eos
        arr = self.tree_arr
        blocks = []
        i = 0
        n = len(arr)
        while i < n:
            if int(arr[i]) == bos:
                start = i
                j = i + 1
                while j < n and int(arr[j]) != eos:
                    j += 1
                end = j
                blocks.append((start, end + 1))
                i = end + 1
            else:
                i += 1
        return blocks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % len(self._block_index)
        start, end = self._block_index[idx]
        block = self.tree_arr[start:end].astype(np.int64).tolist()
        leaves_ids, tree, _ = parse_tree_block(block, self.vocab)
        if tree is None or len(leaves_ids) < 2:
            leaves_ids = leaves_ids or [0]
            merge_orders = [0]
            leaves_ids = leaves_ids + leaves_ids[:1]
        else:
            _, merge_orders = tree_to_merge_orders(tree, direction=self.direction)
        if len(leaves_ids) > self.max_seq_len:
            leaves_ids = leaves_ids[:self.max_seq_len]
            merge_orders = merge_orders[:self.max_seq_len - 1]
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
        merge_orders = [item["merge_orders"] for item in input_list]
        # strip merge_orders before delegating (base collator ignores unknown keys)
        base_items = [{"text": item["text"], "sentence_splits": item["sentence_splits"]}
                      for item in input_list]
        batch = self._base.generative_r2d2_collate_fn_ext(base_items)
        max_len = max(len(mo) for mo in merge_orders)
        N = len(merge_orders)
        padded = np.full((N, max_len), -1, dtype=np.int64)
        for i, mo in enumerate(merge_orders):
            padded[i, :len(mo)] = mo
        batch["merge_orders"] = torch.tensor(padded, dtype=torch.long)
        return batch
