#!/usr/bin/env python
"""Preprocess a corpus into the GPST lazy memmap format (one big int32 file +
.len.pkl), for the unsupervised dataset.

Supports two modes:
- ``wikitext103``: a single wiki.train.txt (sentences per line)
- ``raw``: a directory of .txt files (e.g. OpenWebText extracts), one doc per line

Usage:
  python scripts/gpst/preprocess_corpus.py --mode raw \
    --raw_corpus_path PATH --tokenizer_path data/gpt2-small/vocab.json \
    --output_path corpus/mycorpus.lazy
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

PYTHONPATH = os.environ.get("PYTHONPATH", "")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from transformers import AutoTokenizer  # noqa: E402


def iter_docs(raw_path, mode):
    if mode == "wikitext103":
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.rstrip("\n")
    else:
        for root, _dirs, files in os.walk(raw_path):
            for fn in sorted(files):
                if not fn.endswith(".txt"):
                    continue
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip():
                            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wikitext103", "raw"], default="raw")
    ap.add_argument("--raw_corpus_path", required=True)
    ap.add_argument("--tokenizer_path", required=True,
                    help="HF tokenizer dir or vocab file (GPT-2 style)")
    ap.add_argument("--output_path", required=True,
                    help="output dir ending in .lazy")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    os.makedirs(args.output_path, exist_ok=True)
    data_file = os.path.join(args.output_path, "data")
    len_file = os.path.join(args.output_path, "data.len.pkl")
    import pickle
    lengths = []
    with open(data_file, "wb") as out:
        for doc in tqdm(iter_docs(args.raw_corpus_path, args.mode)):
            ids = tok.encode(doc, add_special_tokens=False)[:args.max_seq_len]
            if len(ids) < 2:
                continue
            out.write(np.array(ids, dtype=np.int32).tobytes(order="C"))
            lengths.append(len(ids))
            out.write(np.array([], dtype=np.int32).tobytes(order="C"))  # doc separator
            lengths.append(0)
    with open(len_file, "wb") as f:
        pickle.dump(lengths, f)
    print(f"wrote {len(lengths)//2} docs to {data_file}")


if __name__ == "__main__":
    main()
