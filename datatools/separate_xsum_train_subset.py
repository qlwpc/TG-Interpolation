#!/usr/bin/env python3
"""Materialize the XSum train subset actually consumed by finetuning.

``XsumDataset`` (olmo/eval/downstream.py) filters the full 204,017-example train
split at load time: an example is kept only when its gold summary ``id`` appears
in ``save_ids.json`` (the decontaminated train-id list produced by
``datatools/filter_xsum_train_id.py`` from lighteval/summarization). This script
applies the identical filter offline and writes the surviving triples as
standalone files, so the real finetune corpus can be released without the
unfiltered source.

Outputs (under --out_dir, default <xsum_dir>/train_filtered):
    xsum_train.txt                  passages as constituency trees
    xsum_train_summary.txt          tree-format summaries (SFT targets)
    gold_train_summary.jsonl        {"summary": ..., "id": ...} per line
    MANIFEST.json                   counts + sha256 of every input/output

Usage:
    python datatools/separate_xsum_train_subset.py \
        [--xsum_dir dataset/Xsum] [--out_dir dataset/Xsum/train_filtered]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip("\n") for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xsum_dir", default="./dataset/Xsum")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    xsum_dir = os.path.abspath(args.xsum_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(xsum_dir, "train_filtered"))
    os.makedirs(out_dir, exist_ok=True)

    passages_path = os.path.join(xsum_dir, "xsum_train.txt")
    summaries_path = os.path.join(xsum_dir, "xsum_train_summary.txt")
    golds_path = os.path.join(xsum_dir, "gold_train_summary.jsonl")
    ids_path = os.path.join(xsum_dir, "save_ids.json")

    for path in (passages_path, summaries_path, golds_path, ids_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(ids_path, "r", encoding="utf-8") as f:
        keep_ids = set(json.load(f))

    passages = read_lines(passages_path)
    summaries = read_lines(summaries_path)
    golds = [json.loads(line) for line in read_lines(golds_path)]

    lengths = {len(passages), len(summaries), len(golds)}
    if len(lengths) != 1:
        raise ValueError(
            f"parallel files misaligned: passages={len(passages)} "
            f"summaries={len(summaries)} golds={len(golds)}"
        )

    kept = [
        (p, s, g)
        for p, s, g in zip(passages, summaries, golds)
        if g["id"] in keep_ids
    ]
    if not kept:
        raise RuntimeError("filter kept 0 examples; check save_ids.json contents")

    out_passages = os.path.join(out_dir, "xsum_train.txt")
    out_summaries = os.path.join(out_dir, "xsum_train_summary.txt")
    out_golds = os.path.join(out_dir, "gold_train_summary.jsonl")

    with open(out_passages, "w", encoding="utf-8") as fp, \
         open(out_summaries, "w", encoding="utf-8") as fs, \
         open(out_golds, "w", encoding="utf-8") as fg:
        for passage, summary, gold in kept:
            fp.write(passage + "\n")
            fs.write(summary + "\n")
            fg.write(json.dumps({"summary": gold["summary"], "id": gold["id"]}) + "\n")

    manifest = {
        "source_dir": xsum_dir,
        "inputs": {
            name: {"lines": n, "sha256": sha256(path)}
            for name, path, n in (
                ("xsum_train.txt", passages_path, len(passages)),
                ("xsum_train_summary.txt", summaries_path, len(summaries)),
                ("gold_train_summary.jsonl", golds_path, len(golds)),
                ("save_ids.json", ids_path, len(keep_ids)),
            )
        },
        "kept_examples": len(kept),
        "outputs": {
            name: {"lines": len(kept), "sha256": sha256(path)}
            for name, path in (
                ("xsum_train.txt", out_passages),
                ("xsum_train_summary.txt", out_summaries),
                ("gold_train_summary.jsonl", out_golds),
            )
        },
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"kept {len(kept)} / {len(passages)} examples -> {out_dir}")


if __name__ == "__main__":
    main()
