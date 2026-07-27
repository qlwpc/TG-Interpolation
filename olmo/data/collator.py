from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
import torch.nn.functional as F

from ..config import PaddingDirection, TrainConfig
from .tg_mask import SentencepieceVocab
from .util import get_document_lengths
__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    pad_direction: PaddingDirection
    pad_token_id: int
    generate_attention_mask: bool
    shuffle_tree: str

    @classmethod
    def from_train_config(cls, config: TrainConfig) -> DataCollator:
        obj = cls(pad_direction=config.data.pad_direction, pad_token_id=config.model.pad_token_id, 
                   generate_attention_mask=config.data.generate_attention_mask, shuffle_tree=config.model.transformer_grammar_type)
        if obj.shuffle_tree:
            obj.vocab = SentencepieceVocab.from_vocab_file(config.tokenizer.vocabulary)
        return obj

    def __call__(self, items: Union[List[Dict[str, Any]], List[torch.Tensor]]) -> Dict[str, Any]:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        all_label_mask = []
        all_indices = []
        all_metadata = []
        all_instance_mask = []
        all_doc_lens = []
        all_max_doc_lens = []
        all_gold_summary = []
        all_tree_spans = []
        all_tree_span_mask = []
        all_treereg_word_boundaries = []
        all_treereg_sentence_ids = []
        has_treereg_metadata = any(
            isinstance(x, dict)
            and (
                "treereg_word_boundaries" in x
                or "treereg_sentence_ids" in x
            )
            for x in items
        )
        max_docs = max((len(x["doc_lens"]) if isinstance(x, dict) and "doc_lens" in x else 0 for x in items))
        max_spans = max((len(x["tree_spans"]) if isinstance(x, dict) and "tree_spans" in x else 0 for x in items))

        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if self.shuffle_tree[:12] == "tree_shuffle":
                if not isinstance(input_ids, np.ndarray):
                    input_ids = input_ids.numpy()
                input_ids = self.vocab.random_shuffle_tree(input_ids)

            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # Pad input IDs.
            all_input_ids.append(
                F.pad(
                    input_ids.to(dtype=torch.long),
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None or self.generate_attention_mask:
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    attention_mask.masked_fill_(input_ids == self.pad_token_id, False)
                elif not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                pad_value = False if attention_mask.dtype == torch.bool else 0.0
                all_attention_mask.append(
                    F.pad(
                        attention_mask,
                        pad_shape,
                        value=pad_value,
                    )
                )

            # Pad attention bias.
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

            # Pad label mask.
            label_mask = x.get("label_mask") if isinstance(x, dict) else None
            if self.shuffle_tree == "tree_shuffle_mask":
                cur_label_mask = self.vocab.get_non_terminal_mask(input_ids)
                if label_mask is not None:
                    label_mask = torch.bitwise_and(label_mask, torch.tensor(cur_label_mask))
                else:
                    label_mask = cur_label_mask
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(
                    F.pad(
                        label_mask.to(dtype=torch.bool),
                        pad_shape,
                        value=False,
                    )
                )

            # Indices.
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            # Parse-aligned spans (Pushdown/TreeReg baselines): pad (M, 3) with -1
            # and a validity mask, to (max_spans_in_batch, 3).
            tree_spans = x.get("tree_spans") if isinstance(x, dict) else None
            if tree_spans is not None:
                if not isinstance(tree_spans, torch.Tensor):
                    tree_spans = torch.tensor(tree_spans)
                tree_spans = tree_spans.to(dtype=torch.long)
                if tree_spans.dim() == 1:
                    tree_spans = tree_spans.view(-1, 3)
                if self.pad_direction == PaddingDirection.left and pad_shape[0] > 0:
                    tree_spans = tree_spans.clone()
                    valid_span = tree_spans[:, 0] >= 0
                    tree_spans[valid_span] += pad_shape[0]
                m = tree_spans.shape[0]
                span_pad = (0, 0, 0, max(0, max_spans - m))
                all_tree_spans.append(F.pad(tree_spans, span_pad, value=-1))
                span_mask = torch.ones(m, dtype=torch.bool)
                span_mask = F.pad(span_mask, (0, max(0, max_spans - m)), value=False)
                all_tree_span_mask.append(span_mask)

            # TreeReg sentence-local metadata. ``sentence_ids`` identifies each
            # complete top-level parse tree; -1 denotes BOS/EOS/whitespace/pad.
            # Word starts are exact preterminal boundaries from the tree stream.
            if has_treereg_metadata:
                if not isinstance(x, dict):
                    raise ValueError("TreeReg metadata requires dictionary dataset items")
                word_boundaries = x.get("treereg_word_boundaries")
                sentence_ids = x.get("treereg_sentence_ids")
                if word_boundaries is None or sentence_ids is None:
                    raise ValueError(
                        "TreeReg batch mixes items with and without sentence metadata"
                    )
                if not isinstance(word_boundaries, torch.Tensor):
                    word_boundaries = torch.tensor(word_boundaries)
                if not isinstance(sentence_ids, torch.Tensor):
                    sentence_ids = torch.tensor(sentence_ids)
                if len(word_boundaries) != len(input_ids) or len(sentence_ids) != len(input_ids):
                    raise ValueError(
                        "TreeReg word boundaries/sentence ids must align with input_ids"
                    )
                all_treereg_word_boundaries.append(
                    F.pad(
                        word_boundaries.to(dtype=torch.bool),
                        pad_shape,
                        value=False,
                    )
                )
                all_treereg_sentence_ids.append(
                    F.pad(
                        sentence_ids.to(dtype=torch.int32),
                        pad_shape,
                        value=-1,
                    )
                )

            # Instance mask.
            instance_mask = x.get("instance_mask") if isinstance(x, dict) else None
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # Document lengths. Computed by the dataset (split by EOS). After
            # right-padding input_ids to max_len, ensure doc_lens covers EVERY
            # padded position: flash_attn_varlen_func (treereg doc-mask path)
            # needs cu_doc_lens to sum to seq_len per instance. If the dataset
            # emitted doc_lens over the unpadded prefix (ParseAlignedDataset
            # fallback), extend the trailing doc with the pad tail so the sum
            # equals the padded length; pads are still ignored in the CE
            # (get_labels masks via attention_mask) and in pushdown (am[b,kv]).
            doc_lens = x.get("doc_lens") if isinstance(x, dict) else None
            if doc_lens is not None:
                if not isinstance(doc_lens, torch.Tensor):
                    doc_lens = torch.tensor(doc_lens)
                doc_lens = doc_lens.to(dtype=torch.long)
                # Pad length for this instance == len(padded input_ids).
                inst_len = max_len
                covered = int(doc_lens.sum())
                if covered < inst_len and covered > 0:
                    doc_lens = doc_lens.clone()
                    doc_lens[-1] = doc_lens[-1] + (inst_len - covered)
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()) if len(doc_lens) else 0)

            # Metadata.
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

            gold_summary = x.get("gold_summary") if isinstance(x, dict) else None
            if gold_summary is not None:
                all_gold_summary.append(gold_summary)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_instance_mask:
            out["instance_mask"] = torch.stack(all_instance_mask)
        if all_doc_lens:
            out["doc_lens"] = torch.stack(all_doc_lens)
        if all_max_doc_lens:
            out["max_doc_lens"] = all_max_doc_lens
        if all_tree_spans:
            out["tree_spans"] = torch.stack(all_tree_spans)
            out["tree_span_mask"] = torch.stack(all_tree_span_mask)
        if all_treereg_word_boundaries:
            out["treereg_word_boundaries"] = torch.stack(all_treereg_word_boundaries)
            out["treereg_sentence_ids"] = torch.stack(all_treereg_sentence_ids)
        if all_metadata:
            out["metadata"] = all_metadata
        if all_gold_summary:
            out["gold_summary"] = all_gold_summary

        return out
