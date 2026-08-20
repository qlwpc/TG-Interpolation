#!/usr/bin/env python
"""Verify the repaired tree_300_repaired.npy in parallel: NT balance + terminal
consistency. Rebuilds per-record offsets from the repaired record lengths using
multiprocessing (the slow part of the original repair script's sequential
verification).
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/wangpch/TG-Interpolation")
import numpy as np
import multiprocessing as mp
from olmo.data.parse_align import TreeVocab
from diagnostics.repair_adj_leak import repair_block

TREE = "/home/wangpch/TG-Interpolation/dataset/testppl_tree/tree_300.npy"
SENT_IDX = "/home/wangpch/TG-Interpolation/dataset/testppl_tree/tree_sent_index.npy"
REPAIRED = "/home/wangpch/TG-Interpolation/dataset/testppl_tree/tree_300_repaired.npy"
SPS = 300

vocab = TreeVocab.from_tokenizer_file("/home/wangpch/TG-Interpolation/dataset/bbc-news/TG_GPT2_tokenizer.json")
OPLO, OPHI, CLLO, CLHI = vocab.op_lo, vocab.op_hi, vocab.cl_lo, vocab.cl_hi

lengths = np.load(SENT_IDX, mmap_mode="r")
num_records = len(lengths)
num_sentences = num_records // SPS
offs = np.empty(num_records + 1, dtype=np.uint64); offs[0] = 0
np.cumsum(lengths, dtype=np.uint64, out=offs[1:])
arr = np.load(TREE, mmap_mode="r")

def len_worker(rng):
    s0, s1 = rng
    out = []
    for s in range(s0, s1):
        f = s * SPS
        for c in range(SPS):
            a = int(offs[f + c]); b = int(offs[f + c + 1])
            blk = arr[a:b].astype(np.int64).tolist()
            rb, _, _, _ = repair_block(blk)
            out.append(len(rb))
    return out

t0 = time.time()
per = max(1, num_sentences // 128)
rngs = [(i, min(i + per, num_sentences)) for i in range(0, num_sentences, per)]
with mp.Pool(32) as p:
    nl = p.map(len_worker, rngs)
new_lengths = np.array([x for r in nl for x in r], dtype=np.uint64)
new_offs = np.empty(num_records + 1, dtype=np.uint64); new_offs[0] = 0
np.cumsum(new_lengths, dtype=np.uint64, out=new_offs[1:])
print(f"offsets rebuilt in {time.time()-t0:.0f}s  sum={int(new_lengths.sum())}", flush=True)

flat = np.load(REPAIRED, mmap_mode="r")
print("saved flat size:", flat.size, " match:", int(new_lengths.sum()) == flat.size, flush=True)

def bal_worker(rng):
    s0, s1 = rng
    unbal = under = 0
    for i in range(s0, s1):
        a = int(new_offs[i]); b = int(new_offs[i + 1])
        blk = flat[a:b].astype(np.int64).tolist()
        depth = 0; u = False
        for t in blk:
            if OPLO <= t <= OPHI: depth += 1
            elif CLLO <= t <= CLHI:
                depth -= 1
                if depth < 0: u = True
        if u: under += 1
        elif depth != 0: unbal += 1
    return under, unbal

t0 = time.time()
rrngs = [(i, min(i + 200000, num_records)) for i in range(0, num_records, 200000)]
with mp.Pool(32) as p:
    res = p.map(bal_worker, rrngs)
under = sum(r[0] for r in res); unbal = sum(r[1] for r in res)
print(f"NT balance: checked {num_records:,}  under={under}  unbal={unbal}  ({time.time()-t0:.0f}s)", flush=True)

def term_worker(rng):
    s0, s1 = rng
    broken = 0; badc = 0
    for s in range(s0, s1):
        f = s * SPS
        a0 = int(new_offs[f]); b0 = int(new_offs[f + 1])
        ref = [int(x) for x in flat[a0:b0] if not (OPLO <= int(x) <= CLHI)]
        nbad = 0
        for c in range(1, SPS):
            ac = int(new_offs[f + c]); bc = int(new_offs[f + c + 1])
            cand = [int(x) for x in flat[ac:bc] if not (OPLO <= int(x) <= CLHI)]
            if cand != ref: nbad += 1
        if nbad: broken += 1; badc += nbad
    return broken, badc

t0 = time.time()
srngs = [(i, min(i + 4000, num_sentences)) for i in range(0, num_sentences, 4000)]
with mp.Pool(32) as p:
    res = p.map(term_worker, srngs)
broken = sum(r[0] for r in res); badc = sum(r[1] for r in res)
print(f"Terminal consistency: broken={broken:,}/{num_sentences:,} ({100*broken/num_sentences:.2f}%)  "
      f"avg bad/broken={badc/max(broken,1):.1f}  ({time.time()-t0:.0f}s)", flush=True)

print()
print("=== VERDICT ===")
print(f"NT balance: {num_records-under-unbal:,}/{num_records:,} balanced")
print(f"Terminals : {num_sentences-broken:,}/{num_sentences:,} clean")
if broken == 0 and under == 0 and unbal == 0:
    print("PERFECT: repaired tree_300 fully balanced AND terminal-consistent.")
