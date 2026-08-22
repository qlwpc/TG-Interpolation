# Shared native n-ary top-300 corpus

## Semantic contract

Each sentence first receives one shared set of up to 300 unique, unlabeled,
unary-free n-ary trees over Benepar parser words. Benepar non-null label scores
are marginalized with `logsumexp`; an exact canonical n-ary K-best DP ranks the
resulting structures.

- Pushdown stores only real n-ary constituent spans mapped to BPE intervals.
- GPST deterministically right-binarizes the same tree after BPE expansion.
- If multiple n-ary trees map to the same GPST merge trajectory, the trajectory
  is evaluated once and its proposal mass is the `logsumexp` of the source
  candidate scores. Multiplicity is also retained for a uniform-mixture ablation.
- Physical candidate width is always 300. `valid_counts` masks the Catalan/
  Schröder-limited short-sentence tail.

The canonical PPL boundary is `testppl_tree/tree_doc_index.npy`: 4,966 documents
and 148,836 sentences. `alignment_audit.json` records the `test.npy` document
offset of 59 and the six explicit terminal exceptions.

## Shard layout

All large arrays are ordinary `.npy` files and are opened with `mmap_mode="r"`.
Tokens are stored once per sentence. Candidate arrays are contiguous within a
sentence, so runtime loading requires no tree parsing or per-candidate terminal
comparison.

Important arrays include:

```text
terminal_tokens.npy / terminal_offsets.npy
content_bounds.npy
valid_counts.npy / proposal_scores.npy

pushdown_spans.npy / pushdown_offsets.npy
pushdown_span_counts.npy

gpst_merge_orders.npy / gpst_offsets.npy
candidate_to_gpst.npy
gpst_unique_counts.npy
gpst_source_slots.npy
gpst_multiplicities.npy
gpst_log_masses.npy
```

`olmo.eval.native_nary_corpus.NativeNaryShard` exposes sentence-local views, and
`NativeNaryCorpus` traverses finalized shards without concatenating multi-GB
files.

## Production commands

```bash
python scripts/generate_native_nary_test.py audit
sbatch scripts/slurm/generate_native_nary_300_rtx3090.sbatch
python scripts/generate_native_nary_test.py finalize --num-shards 28
```

Generation is resumable at sentence granularity through `completed.npy`. Each
GPU task batches Benepar chart scoring and uses eight spawn-based CPU workers for
the exact K-best DP and model adapters.
