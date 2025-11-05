from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
import torch.nn.functional as F

from ..config import PaddingDirection, TrainConfig
from .tg_mask import SentencepieceVocab
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
        max_docs = max((len(x["doc_lens"]) if isinstance(x, dict) and "doc_lens" in x else 0 for x in items))

        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if self.shuffle_tree[:12] == "tree_shuffle":
                if not isinstance(input_ids, np.ndarray):
                    input_ids = input_ids.numpy()
                input_ids = self.vocab.random_shuffle_tree(input_ids)
            elif self.shuffle_tree == "pause1/2":
                paused_input = torch.zeros(2 * len(input_ids), dtype=input_ids.dtype, device=input_ids.device)
                paused_input[::2] = input_ids
                paused_input[1::2] = 50260  # Special token
                input_ids = paused_input

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
            if self.shuffle_tree[-4:] == "mask":
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

            # Instance mask.
            instance_mask = x.get("instance_mask") if isinstance(x, dict) else None
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # Document lengths.
            doc_lens = x.get("doc_lens") if isinstance(x, dict) else None
            if doc_lens is not None:
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()))

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
        if all_metadata:
            out["metadata"] = all_metadata
        if all_gold_summary:
            out["gold_summary"] = all_gold_summary

        return out
