"""Read-only structural audit of BBC archived single parses versus DocPPL.

Writes per-sentence constituent-span differences and a source request manifest.
All spans use half-open BPE terminal positions within one sentence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from datatools.parse_test_docppl_data.reproduce_bbc_test import selected_rows, sha256_file, split_tree_stream


def sentence_trees(document):
    roots, stack, spans, leaves = [], [], [], []
    start = None
    for pos, raw in enumerate(document):
        token = int(raw)
        if 50268 <= token <= 50293:
            if not stack:
                start, spans, leaves = pos, [], []
            stack.append((token - 50268, len(leaves)))
        elif 50294 <= token <= 50319:
            label, left = stack.pop()
            if label != token - 50294:
                raise ValueError("mismatched constituent labels")
            spans.append((label, left, len(leaves)))
            if not stack:
                roots.append({"tokens": document[start:pos+1].tolist(), "leaves": leaves,
                              "spans": spans})
        elif stack:
            leaves.append(token)
    if stack:
        raise ValueError("unclosed constituent")
    return roots


def analyze(args):
    old = split_tree_stream(args.old)
    current = split_tree_stream(args.current)
    source = list(selected_rows(args.selected))
    assert len(old) == len(source) == 5025 and len(current) == 4966
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    labels = {i: tokenizer.id_to_token(i + 50268)[2:-1] for i in range(26)}
    report = {"claim_type": "computed", "inputs": {}, "offset": 59, "documents": [],
              "structure_docs_by_shard": {}, "sentence_change_types": {}, "changed_sentences": 0}
    for name in ("old", "current", "selected", "tokenizer"):
        path = getattr(args, name)
        report["inputs"][name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    source_positions = {}
    for row in source:
        shard = row["shard"]
        source_positions.setdefault(shard, []).append(row["source_row"])
    requests, by_shard, types, lengths = [], Counter(), Counter(), []
    for j, target in enumerate(current):
        original = old[j + 59]
        if np.array_equal(original, target):
            continue
        row = source[j + 59]
        item = {"old_doc": j+59, "current_doc": j, "shard": row["shard"],
                "source_row": row["source_row"],
                "shard_doc": source_positions[row["shard"]].index(row["source_row"])}
        before, after = sentence_trees(original), sentence_trees(target)
        item["old_sentences"], item["current_sentences"] = len(before), len(after)
        terminal_equal = (len(before) == len(after) and
                          all(a["leaves"] == b["leaves"] for a, b in zip(before, after)))
        item["terminals_equal"] = terminal_equal
        requests.append(dict(item))
        changes = []
        if terminal_equal:
            by_shard[row["shard"]] += 1
            for s, (a, b) in enumerate(zip(before, after)):
                if a["tokens"] == b["tokens"]:
                    continue
                la, lb = Counter(map(tuple, a["spans"])), Counter(map(tuple, b["spans"]))
                ua, ub = Counter((l, r) for _, l, r in a["spans"]), Counter((l, r) for _, l, r in b["spans"])
                kind = ("serialization_only" if la == lb else "label_only" if ua == ub
                        else "same_spans_different_unary_multiplicity" if set(ua) == set(ub)
                        else "changed_constituent_spans")
                def describe(counter):
                    return [{"label": labels[label], "span": [l, r],
                             "text": tokenizer.decode(a["leaves"][l:r]), "count": count}
                            for (label, l, r), count in counter.items()]
                changes.append({"sentence_in_doc": s, "kind": kind,
                    "terminals": len(a["leaves"]), "text": tokenizer.decode(a["leaves"]),
                    "old_tree": tokenizer.decode(a["tokens"], skip_special_tokens=False),
                    "current_tree": tokenizer.decode(b["tokens"], skip_special_tokens=False),
                    "old_token_ids": a["tokens"], "current_token_ids": b["tokens"],
                    "removed_constituents": describe(la-lb), "added_constituents": describe(lb-la)})
                types[kind] += 1
                lengths.append(len(a["leaves"]))
            item["changes"] = changes
        report["documents"].append(item)
    report["structure_docs_by_shard"] = dict(by_shard)
    report["sentence_change_types"] = dict(types)
    report["changed_sentences"] = sum(types.values())
    report["changed_sentence_terminal_lengths"] = {"min": min(lengths), "max": max(lengths),
        "median": float(np.median(lengths)), "under_10": sum(x < 10 for x in lengths),
        "over_200": sum(x > 200 for x in lengths)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "structure_differences.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    (args.output / "source_requests.json").write_text(json.dumps(requests, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("documents", "inputs")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=REPO / "dataset/bbc-news/tree/test.pre_testppl_alignment.npy")
    parser.add_argument("--current", type=Path, default=REPO / "dataset/bbc-news/tree/test.npy")
    parser.add_argument("--selected", type=Path, default=REPO / "artifacts/bbc_test_reproduction_20260906/selected.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=REPO / "dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
