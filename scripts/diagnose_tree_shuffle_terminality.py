#!/usr/bin/env python3
"""Compare Tree-Shuffle next-token distributions on terminal and tree inputs.

This diagnostic intentionally uses candidate 0 from the same first test document
for both streams.  It reports whether the model assigns meaningful probability
to gold non-terminal targets when it is given the tree-format context.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from olmo.data.tg_mask import SentencepieceVocab
from olmo.model import OLMo


DEFAULT_CHECKPOINT = ROOT / "saved_models/Tree_shuffle_pretrain/step49440-unsharded"
VOCAB_PATH = ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json"
TREE_DIR = ROOT / "dataset/bbc-news/testppl_tree"
TERMINAL_DIR = ROOT / "dataset/bbc-news/terminal"


def _records(data_path: Path, lengths_path: Path, count: int, stride: int = 1):
    values = np.load(data_path, mmap_mode="r")
    lengths = np.load(lengths_path, mmap_mode="r")
    offsets = np.concatenate(([0], np.cumsum(lengths[: count * stride], dtype=np.int64)))
    return [np.asarray(values[offsets[i * stride] : offsets[i * stride + 1]]) for i in range(count)]


def _summarize(logits: torch.Tensor, targets: torch.Tensor, vocab: SentencepieceVocab):
    probs = torch.softmax(logits.float(), dim=-1)
    prediction = logits.argmax(dim=-1)
    ids = torch.arange(logits.shape[-1], device=logits.device)
    nt = torch.tensor([vocab.is_non_terminal(int(i)) for i in ids.tolist()], device=logits.device)
    terminal = torch.tensor(
        [vocab.is_terminal(int(i)) or int(i) == vocab.eos for i in ids.tolist()],
        device=logits.device,
    )
    gold_nt = nt[targets]
    gold_terminal = terminal[targets]
    correct_probability = probs.gather(-1, targets[:, None]).squeeze(-1)

    def mean(value: torch.Tensor) -> float | None:
        return float(value.float().mean()) if value.numel() else None

    return {
        "predicted_positions": int(targets.numel()),
        "gold_nonterminal_positions": int(gold_nt.sum()),
        "gold_terminal_or_eos_positions": int(gold_terminal.sum()),
        "mean_probability_mass_nonterminal": float(probs[:, nt].sum(dim=-1).mean()),
        "top1_nonterminal_rate_all": mean(nt[prediction]),
        "top1_nonterminal_rate_at_gold_nonterminal": mean(nt[prediction[gold_nt]]),
        "mean_correct_probability_at_gold_nonterminal": mean(correct_probability[gold_nt]),
        "mean_correct_probability_at_gold_terminal_or_eos": mean(
            correct_probability[gold_terminal]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--document-sentences", type=int, default=37)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    vocab = SentencepieceVocab.from_vocab_file(str(VOCAB_PATH))
    terminal = _records(
        TERMINAL_DIR / "test.npy", TERMINAL_DIR / "test_sent_index.npy", args.document_sentences
    )
    # tree_sent_index has 300 parse candidates per sentence; candidate 0 is
    # every 300th record and is the exact path used by a K=1 tree comparison.
    tree = _records(
        TREE_DIR / "tree_300.npy", TREE_DIR / "tree_sent_index.npy", args.document_sentences, 300
    )
    streams = {
        "terminal_doc": np.concatenate(terminal),
        "tree_doc_candidate0": np.concatenate(tree),
        # This is the transform applied by DataCollator during Tree-Shuffle
        # training. Keep it as a distinct stream so an unshuffled test tree
        # cannot be mistaken for the actual training-input contract.
        "tree_doc_candidate0_shuffled": np.concatenate(
            [np.asarray(vocab.random_shuffle_tree(sentence)) for sentence in tree]
        ),
    }
    model = OLMo.from_checkpoint(args.checkpoint, device=args.device).eval()
    result = {"checkpoint": str(args.checkpoint), "device": args.device, "sentences": args.document_sentences}
    with torch.no_grad():
        for name, stream in streams.items():
            input_ids = torch.tensor(stream, dtype=torch.long, device=args.device)[None, :]
            output = model(input_ids)
            result[name] = {
                "input_tokens": int(input_ids.shape[1]),
                **_summarize(output.logits[0, :-1], input_ids[0, 1:], vocab),
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
