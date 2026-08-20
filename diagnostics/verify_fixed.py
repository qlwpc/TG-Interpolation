#!/usr/bin/env python
"""Verify tree_300_fixed.npy: NT balance + terminal consistency (parallel).

Uses tree_sent_index_fixed.npy (the repaired per-record lengths) for offsets.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/wangpch/TG-Interpolation")
import numpy as np
import multiprocessing as mp
from olmo.data.parse_align import TreeVocab

REPAIRED = "/home/wangpch/TG-Interpolation/dataset/testppl_tree/tree_300_fixed.npy"
SENT_IDX = "/home/wangpch/TG-Interpolation/dataset/testppl_tree/tree_sent_index_fixed.npy"
SPS = 300

vocab = TreeVocab.from_tokenizer_file("/home/wangpch/TG-Interpolation/dataset/bbc-news/TG_GPT2_tokenizer.json")
OPLO, OPHI, CLLO, CLHI = vocab.op_lo, vocab.op_hi, vocab.cl_lo, vocab.cl_hi

lengths = np.load(SENT_IDX, mmap_mode="r")
num_records = len(lengths)
num_sentences = num_records // SPS
offs = np.empty(num_records + 1, dtype=np.uint64); offs[0] = 0
np.cumsum(lengths, dtype=np.uint64, out=offs[1:])
flat = np.load(REPAIRED, mmap_mode="r")
print(f"records={num_records:,} sentences={num_sentences:,} tokens={flat.size:,}", flush=True)
print(f"sum(lengths)==flat.size? {int(np.sum(lengths))==flat.size}", flush=True)

def bal_worker(rng):
    s0, s1 = rng
    under = unbal = 0
    for i in range(s0, s1):
        blk = flat[int(offs[i]):int(offs[i+1])].astype(np.int64).tolist()
        d = 0; u = False
        for t in blk:
            if OPLO <= t <= OPHI: d += 1
            elif CLLO <= t <= CLHI:
                d -= 1
                if d < 0: u = True
        if u: under += 1
        elif d != 0: unbal += 1
    return under, unbal

t0 = time.time()
rrngs = [(i, min(i+200000, num_records)) for i in range(0, num_records, 200000)]
with mp.Pool(32) as p:
    res = p.map(bal_worker, rrngs)
under = sum(r[0] for r in res); unbal = sum(r[1] for r in res)
print(f"NT balance: checked {num_records:,}  under={under} unbal={unbal}  ({time.time()-t0:.0f}s)", flush=True)

def term_worker(rng):
    s0, s1 = rng
    broken = 0; badc = 0
    for s in range(s0, s1):
        f = s * SPS
        ref = [int(x) for x in flat[int(offs[f]):int(offs[f+1])] if not (OPLO<=int(x)<=CLHI)]
        nbad = 0
        for c in range(1, SPS):
            cand = [int(x) for x in flat[int(offs[f+c]):int(offs[f+c+1])] if not (OPLO<=int(x)<=CLHI)]
            if cand != ref: nbad += 1
        if nbad: broken += 1; badc += nbad
    return broken, badc

t0 = time.time()
srngs = [(i, min(i+4000, num_sentences)) for i in range(0, num_sentences, 4000)]
with mp.Pool(32) as p:
    res = p.map(term_worker, srngs)
broken = sum(r[0] for r in res); badc = sum(r[1] for r in res)
print(f"Terminal consistency: broken={broken:,}/{num_sentences:,} ({100*broken/num_sentences:.4f}%)  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== VERDICT ===")
print(f"NT balance: {num_records-under-unbal:,}/{num_records:,} balanced")
print(f"Terminals : {num_sentences-broken:,}/{num_sentences:,} clean")
if broken == 0 and under == 0 and unbal == 0:
    print("PERFECT: tree_300_fixed fully balanced AND terminal-consistent.")
