#!/usr/bin/env python
"""Count ``(ADJ ... ADJ)`` label-leak annotations and other leaked constituency
labels in the tree-stream corpora, with multiprocessing for the large files.

Background: ``dataset/bbc-news/testppl_tree/tree_300.npy`` packs 300 parse-tree
candidates per sentence.  Some candidates were produced by benepar models whose
label set (EWT) contains the constituent label ``ADJ``, but the repo's
``TG_GPT2_tokenizer.json`` has **no** ``<(ADJ>`` / ``<ADJ)>`` NT-bracket token.
When those trees are serialized, the ``ADJ`` label cannot map to an NT token and
is instead written as *literal surface tokens* ``(``, ``AD``, ``J``, ``)`` ->
the `` (ADJ ... ADJ)`` text seen leaking into the terminal stream (job 45195).

This script:
  1. Builds the set of *missing* labels: top-level constituent labels present in
     the benepar EWT/WSJ label vocabs but absent from the tokenizer's NT-bracket
     added-tokens.  (Empirically: ``ADJ`` from EWT, ``NX`` from WSJ.)
  2. For each missing label, tokenizes its leak forms (" (LABEL" open, " LABEL)"
     close) into id sequences and searches every .npy for them, in parallel.
  3. Runs a *generic* detector: scan for any ``(`` opener token followed by a
     run of all-uppercase-letter tokens (catches ADJ, NX, or anything else) on a
     sampled prefix of the huge files.
  4. Reports per-file counts, ratios (per token, per sentence), and open/close
     balance.

Targets:
  /home/wangpch/TG-Interpolation/dataset/bbc-news/tree/{train,dev,test}.npy
  /home/wangpch/TG-Interpolation/dataset/testppl_tree/tree_300.npy

Run:
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/count_adj_leaks.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TOK_PATH = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"

TREE_DIR = f"{REPO}/dataset/bbc-news/tree"
TESTPPL_DIR = f"{REPO}/dataset/testppl_tree"

# Files to scan: (label, path, sentences_or_None)
# sentence counts: train/dev/test sentence counts approximated from sent_index
# where available; None -> skip per-sentence ratio.
TARGETS: List[Tuple[str, str, int | None]] = [
    ("tree/dev.npy",        f"{TREE_DIR}/dev.npy",        None),
    ("tree/test.npy",       f"{TREE_DIR}/test.npy",       None),
    ("tree/train.npy",      f"{TREE_DIR}/train.npy",      None),
    ("testppl/tree_300.npy", f"{TESTPPL_DIR}/tree_300.npy", 148836),
]

# Union of top-level constituent labels used by the three benepar models
# (EWT: en3/en3_large contain ADJ; WSJ: en3_wsj contains NX).  The tokenizer's
# NT-bracket set is derived at runtime from added_tokens.
CANDIDATE_LABELS = [
    "ADJ", "ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST", "NAC", "NML", "NP", "NX",
    "PP", "PRN", "PRT", "QP", "RRC", "S", "SBAR", "SBARQ", "SINV", "SQ",
    "UCP", "VP", "WHADJP", "WHADVP", "WHNP", "WHPP", "X",
]

CHUNK_TOKENS = 64_000_000      # tokens per chunk
OVERLAP = 8                    # overlap tokens to avoid splitting patterns
GENERIC_SAMPLE_PREFIX = 300_000_000  # tokens scanned by generic detector on huge files


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def get_nt_labels(tok: Tokenizer) -> set:
    """Top-level labels that DO have NT-bracket tokens in the tokenizer."""
    import json
    with open(TOK_PATH) as f:
        tj = json.load(f)
    labels = set()
    for a in tj.get("added_tokens", []):
        c = a["content"]
        if c.startswith("<(") and c.endswith(">"):
            labels.add(c[2:-1])           # <(ADJP> -> ADJP
        elif c.startswith("<") and c.endswith(")>"):
            labels.add(c[1:-2])           # <ADJP)> -> ADJP
    return labels


def decode_id(tok: Tokenizer, i: int) -> str:
    try:
        return tok.decode([i])
    except Exception:  # noqa: BLE001
        return ""


def build_pattern_db(tok: Tokenizer, missing_labels: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """For each missing label, build open/close id-sequence patterns.

    Leak serialization writes the label string through the GPT-2 tokenizer, so:
      open  " (LABEL"  ->  token ids of "(LABEL" with leading space
      close " LABEL)"  ->  token ids of "LABEL)" with leading space
    We probe a few space variants and keep whichever tokenizes to a stable
    prefix/suffix.  In practice ADJ open=[357,2885,41], close=[5984,41,8].
    """
    db: Dict[str, Dict[str, np.ndarray]] = {}
    for lab in missing_labels:
        opens, closes = [], []
        for space in (" ", ""):
            o = tok.encode(f"{space}({lab}").ids
            c = tok.encode(f"{space}{lab})").ids
            opens.append(o)
            closes.append(c)
        # also no-space label-only closers
        closes.append(tok.encode(f"{lab})").ids)
        db[lab] = {
            "opens": [np.array(p, dtype=np.int32) for p in opens],
            "closes": [np.array(p, dtype=np.int32) for p in closes],
        }
    return db


def classify_upper_ids(tok: Tokenizer, vocab_size: int) -> Tuple[set, set]:
    """Return (opener_ids, upper_letter_ids).

    opener_ids: token ids whose decoded form (stripped) == "("  -> the literal
                parenthesis that starts a leaked bracket.
    upper_letter_ids: token ids whose decoded form, stripped of leading space,
                      is non-empty and all uppercase A-Z (e.g. "AD", "J", "NX").
    """
    opener_ids, upper_ids = set(), set()
    for i in range(vocab_size):
        s = decode_id(tok, i)
        st = s.strip()
        if st == "(":
            opener_ids.add(i)
        if st and all(c.isupper() and c.isalpha() for c in st):
            upper_ids.add(i)
    return opener_ids, upper_ids


# --------------------------------------------------------------------------- #
# Worker: count fixed patterns in one chunk
# --------------------------------------------------------------------------- #
_PATTERN_DB: Dict = None
_OPENER_IDS: set = None
_UPPER_IDS: set = None
_TOK: Tokenizer = None


def _init_worker(pattern_db, opener_ids, upper_ids, tok_path):
    global _PATTERN_DB, _OPENER_IDS, _UPPER_IDS, _TOK
    _PATTERN_DB = pattern_db
    _OPENER_IDS = opener_ids
    _UPPER_IDS = upper_ids
    _TOK = Tokenizer.from_file(tok_path)


def _count_chunk(args: Tuple[str, int, int, bool]) -> Dict:
    path, start, end, generic_on = args
    arr = np.load(path, mmap_mode="r")
    n = len(arr)
    lo = max(0, start - OVERLAP)
    hi = min(n, end + OVERLAP)
    seg = arr[lo:hi].astype(np.int32, copy=True)
    seg_len = len(seg)
    result: Dict = {"opens": Counter(), "closes": Counter(), "generic": Counter()}

    # ---- fixed label patterns ----
    for lab, pat in _PATTERN_DB.items():
        for key in ("opens", "closes"):
            best = 0
            for p in pat[key]:
                L = len(p)
                if seg_len < L:
                    continue
                # vectorized rolling match
                m = np.ones(seg_len - L + 1, dtype=bool)
                for k in range(L):
                    m &= (seg[k:seg_len - L + 1 + k] == p[k])
                best = max(best, int(m.sum()))
            result[key][lab] = best

    # ---- generic detector: '(' followed by all-uppercase token run ----
    if generic_on:
        opener_positions = np.where(np.isin(seg, np.fromiter(_OPENER_IDS, dtype=np.int32)))[0]
        upper_arr = np.fromiter(_UPPER_IDS, dtype=np.int32)
        upper_set = set(int(x) for x in upper_arr)
        for p in opener_positions:
            q = p + 1
            label_ids: List[int] = []
            while q < seg_len and len(label_ids) < 6 and int(seg[q]) in upper_set:
                label_ids.append(int(seg[q]))
                q += 1
            if 1 <= len(label_ids) <= 6:
                label = _TOK.decode(label_ids).strip()
                # must be a plausible ALL-CAPS label (>=2 chars), not normal text
                if len(label) >= 2 and label.isupper() and label.isalpha():
                    result["generic"][label] += 1
    return result


def merge_results(results: List[Dict]) -> Dict:
    opens, closes, generic = Counter(), Counter(), Counter()
    for r in results:
        opens.update(r["opens"])
        closes.update(r["closes"])
        generic.update(r["generic"])
    return {"opens": opens, "closes": closes, "generic": generic}


def make_chunks(n: int) -> List[Tuple[int, int]]:
    starts = list(range(0, n, CHUNK_TOKENS))
    return [(s, min(s + CHUNK_TOKENS, n)) for s in starts]


def scan_file(path: str, label: str, n_sentences: int | None,
              pool, pattern_db, opener_ids, upper_ids, tok_path) -> Dict:
    arr = np.load(path, mmap_mode="r")
    n = len(arr)
    chunks = make_chunks(n)
    # generic detector only on a prefix to bound cost on huge files
    generic_limit = GENERIC_SAMPLE_PREFIX if n > 2_000_000_000 else n

    tasks = []
    for s, e in chunks:
        generic_on = s < generic_limit
        tasks.append((path, s, e, generic_on))

    t0 = time.time()
    print(f"\n[{label}] {n:,} tokens, {len(tasks)} chunks "
          f"(generic scan first {min(generic_limit,n):,} tokens)...", flush=True)
    results = pool.map(_count_chunk, tasks)
    merged = merge_results(results)
    elapsed = time.time() - t0

    # report
    print(f"[{label}] done in {elapsed:.1f}s", flush=True)
    print(f"  total tokens            : {n:,}")
    if n_sentences:
        print(f"  sentences               : {n_sentences:,}")
    print(f"  fixed-pattern opens     : {dict(merged['opens'])}")
    print(f"  fixed-pattern closes    : {dict(merged['closes'])}")
    print(f"  generic detector (sample): {dict(merged['generic'])}")

    # ratios for the dominant leaked label
    for lab in merged["opens"]:
        o = merged["opens"][lab]
        if o == 0:
            continue
        per_tok = o / n
        line = (f"  {lab}: opens={o:,} closes={merged['closes'][lab]:,} "
                f"per-token={per_tok:.6f} ({o/n*1e6:.1f} per 1M tokens)")
        if n_sentences:
            line += f"  per-sentence={o/n_sentences:.4f}"
        print(line)
    return {"path": path, "label": label, "n_tokens": n, "n_sentences": n_sentences,
            **merged}


def main() -> None:
    tok = Tokenizer.from_file(TOK_PATH)
    vocab_size = tok.get_vocab_size()

    nt_labels = get_nt_labels(tok)
    missing = sorted(set(CANDIDATE_LABELS) - nt_labels)

    hr("0. Setup")
    print(f"tokenizer vocab size   : {vocab_size}")
    print(f"tokenizer NT labels    : {len(nt_labels)}  {sorted(nt_labels)}")
    print(f"CANDIDATE labels       : {len(CANDIDATE_LABELS)}")
    print(f"MISSING labels (leak)  : {missing}")
    if not missing:
        print("No missing labels found; nothing to count.")
        return

    pattern_db = build_pattern_db(tok, missing)
    print("\nleak patterns (token id sequences):")
    for lab, pat in pattern_db.items():
        print(f"  {lab}:")
        print(f"    opens  = {[p.tolist() for p in pat['opens']]}")
        print(f"    closes = {[p.tolist() for p in pat['closes']]}")

    opener_ids, upper_ids = classify_upper_ids(tok, vocab_size)
    print(f"\nopener '(' ids          : {sorted(opener_ids)}")
    print(f"uppercase-letter ids   : {len(upper_ids)} (e.g. "
          f"{sorted(upper_ids)[:12]} ...)")

    import multiprocessing as mp
    n_workers = min(mp.cpu_count(), 32)
    print(f"\nworkers                 : {n_workers}")
    pool = mp.Pool(n_workers, initializer=_init_worker,
                   initargs=(pattern_db, opener_ids, upper_ids, TOK_PATH))

    hr("1. Scanning files")
    summaries = []
    for label, path, n_sent in TARGETS:
        if not os.path.exists(path):
            print(f"\n[{label}] MISSING: {path}")
            continue
        summaries.append(scan_file(path, label, n_sent, pool,
                                   pattern_db, opener_ids, upper_ids, TOK_PATH))
    pool.close()
    pool.join()

    hr("2. Summary table")
    print(f"{'file':<24} {'tokens':>14} {'ADJ opens':>11} {'ADJ closes':>12} "
          f"{'NX opens':>10} {'per 1M tok':>11}")
    print("-" * 86)
    for s in summaries:
        adj_o = s["opens"].get("ADJ", 0)
        adj_c = s["closes"].get("ADJ", 0)
        nx_o = s["opens"].get("NX", 0)
        per_1m = adj_o / s["n_tokens"] * 1e6 if s["n_tokens"] else 0
        print(f"{s['label']:<24} {s['n_tokens']:>14,} {adj_o:>11,} {adj_c:>12,} "
              f"{nx_o:>10,} {per_1m:>11.1f}")

    hr("3. Verdict")
    tree_sum = sum(s["opens"].get("ADJ", 0) for s in summaries
                   if s["label"].startswith("tree/"))
    tp_sum = sum(s["opens"].get("ADJ", 0) for s in summaries
                 if s["label"].startswith("testppl/"))
    print(f"literal '(ADJ' opens in tree/* (train/dev/test): {tree_sum:,}")
    print(f"literal '(ADJ' opens in testppl/tree_300.npy   : {tp_sum:,}")
    if tree_sum == 0 and tp_sum > 0:
        print("-> ADJ leak is ABSENT from the raw tree/ corpus but PRESENT in")
        print("   testppl_tree/tree_300.npy => injected during 300-candidate")
        print("   generation (EWT-label parse variant serialized as literal text).")
    elif tree_sum > 0:
        print("-> ADJ leak ALSO present in raw tree/ corpus (upstream defect).")


if __name__ == "__main__":
    main()
