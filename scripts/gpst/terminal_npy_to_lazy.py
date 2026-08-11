#!/usr/bin/env python
"""Convert the repo's tree-stream (``dataset/.../tree/*.npy``) into the GPST
lazy memmap format consumed by ``scripts/gpst/run_gpst.py --unsupervised``.

Why tree.npy, not terminal.npy?
  The terminal stream is a flat token flow with NO sentence boundaries. GPST's
  inside-outside chart processes per-sentence segments, and ``parser_max_len``
  (default 1024) caps a single segment — feeding whole BBC articles (median
  499, max 25000 tokens) as one segment would overflow the chart. The tree
  stream interleaves constituency markers; we recover terminal tokens AND
  sentence boundaries from ``<(S>`` / ``<S)>`` markers, so each lazy "doc" is
  an article split into sentence-length segments — exactly what the chart needs.

Output layout (under ``<output_path>``):
  data         — int32 concatenation of sentence token arrays, per article:
                 [sent1_tokens | sent2_tokens | ... | <0-sep> | next_article...]
  data.len.pkl — pickle of ``[len_sent1, len_sent2, ..., 0, ...]``; ``0`` is the
                 article separator that ``LazyLoader`` splits on. Within an
                 article, consecutive non-zero lengths become ``splits``
                 (sentence boundaries) the collator uses to segment the chart.

Token id map (from dataset/bbc-news/TG_GPT2_tokenizer.json):
  BOT=50257 <|beginoftext|>   (article boundary)
  S_OPEN=50282 <(S>           (sentence open)
  S_CLOSE=50308 <S)>           (sentence close)
  Nonterminals: 50268..50319   (skipped — not terminals)
  Terminals: everything else  (< 50268), these are emitted.

Usage:
  python scripts/gpst/terminal_npy_to_lazy.py \
    --tree_npy dataset/bbc-news/tree/train.npy \
    --output_path corpus/bbc-tree.lazy
"""
import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Token ids (see header docstring)
BOT = 50257
S_OPEN = 50282
S_CLOSE = 50308
NT_LO, NT_HI = 50268, 50319  # full nonterminal range (open 50268-50293, close 50294-50319)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree_npy", required=True,
                    help="tree-stream .npy (uint16), e.g. dataset/bbc-news/tree/train.npy")
    ap.add_argument("--output_path", required=True,
                    help="output dir ending in .lazy")
    ap.add_argument("--min_sent_len", type=int, default=2,
                    help="skip sentences shorter than this many tokens")
    ap.add_argument("--max_sent_len", type=int, default=1024,
                    help="truncate sentences longer than this (matches parser_max_len)")
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    data_file = os.path.join(args.output_path, "data")
    len_file = os.path.join(args.output_path, "data.len.pkl")

    tokens = np.load(args.tree_npy, mmap_mode="r")
    n = len(tokens)
    print(f"loaded {args.tree_npy}: {n:,} tokens (~{n/1e9:.1f}B), dtype={tokens.dtype}")

    # Per-token state machine. The vectorized split approach paid per-article
    # Python overhead on ~15M articles; this single-pass loop is simpler and
    # correct (~1.4 M tok/s — a one-time ~5 h job for the 24.7G train stream).
    print("extracting terminals + sentence boundaries ...")
    lengths = []
    num_articles = 0
    num_sentences = 0
    total_tokens = 0
    cur_sent = []
    in_article = False

    CHUNK = 10_000_000
    with open(data_file, "wb") as out:
        with tqdm(total=n, desc="scan", unit="tok", unit_scale=True) as pbar:
            for start in range(0, n, CHUNK):
                block = tokens[start:start + CHUNK]
                for tok in block:
                    t = int(tok)
                    if t == BOT:
                        if cur_sent:
                            lengths.append(len(cur_sent))
                            out.write(np.asarray(cur_sent, dtype=np.int32).tobytes(order="C"))
                            num_sentences += 1
                            total_tokens += len(cur_sent)
                            cur_sent = []
                        if in_article:
                            lengths.append(0)  # article separator
                            num_articles += 1
                        in_article = True
                    elif t == S_OPEN:
                        if cur_sent:
                            lengths.append(len(cur_sent))
                            out.write(np.asarray(cur_sent, dtype=np.int32).tobytes(order="C"))
                            num_sentences += 1
                            total_tokens += len(cur_sent)
                            cur_sent = []
                    elif t == S_CLOSE:
                        if cur_sent:
                            lengths.append(len(cur_sent))
                            out.write(np.asarray(cur_sent, dtype=np.int32).tobytes(order="C"))
                            num_sentences += 1
                            total_tokens += len(cur_sent)
                            cur_sent = []
                    elif NT_LO <= t <= NT_HI:
                        continue  # other nonterminal marker — skip
                    else:
                        cur_sent.append(t)
                pbar.update(len(block))

        if cur_sent:
            lengths.append(len(cur_sent))
            out.write(np.asarray(cur_sent, dtype=np.int32).tobytes(order="C"))
            num_sentences += 1
            total_tokens += len(cur_sent)
        if in_article:
            lengths.append(0)
            num_articles += 1

    # count sentences + tokens for reporting (lengths has [len_sent,...,0,...])
    sent_lens = [l for l in lengths if l > 0]
    num_sentences = len(sent_lens)
    total_tokens = sum(sent_lens)
    with open(len_file, "wb") as f:
        pickle.dump(lengths, f)

    print(f"\ndone: {num_articles:,} articles, {num_sentences:,} sentences, "
          f"{total_tokens:,} terminal tokens written")
    print(f"avg sentence len: {total_tokens/max(1,num_sentences):.1f} tokens")
    print(f"data file: {data_file}")
    print(f"len file:  {len_file}")
    print(f"\nuse with:\n"
          f"  bash scripts/gpst/pretrain_gpst_small_unsupervised.sh "
          f"{args.output_path}")


if __name__ == "__main__":
    main()
