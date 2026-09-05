# Native document-PPL recovery and RTX3090 integration record

Updated 2026-09-05. This record describes the durable parts recovered from
`RTX3090:/home/wangpch/TG-Interpolation` after comparing its dirty worktree with
GitHub `main` at `1c9e2a8`. Untracked files were outside the audit. The remote
checkout had no unique commits, was 12 commits behind `main`, and contained 15
modified tracked files at local HEAD `f5f8da6`. The original read-only snapshot and patch are retained
locally under `/tmp/rtx3090-tracked-audit/` for this integration session.

## Integrated behavior

- Native Pushdown scoring can reuse candidate-0 transformer K/V, final hidden
  states, input IDs, and sentence IDs. When complete-sentence context truncation
  slides the window, the evaluator discards the stale cache and rebuilds it from
  the retained suffix. `--no-kv-cache` remains the full-prefix correctness
  reference.
- The attachment head computes embeddings and its MLP only for the requested
  query range. Cached and full-prefix paths retain the selected versioned
  attachment normalization: v1 `stack_legal` or v2 `sentence_causal`.
- GPST and Pushdown bound candidate batches by both their linear token/action
  budget and `batch × context_length²`. Pushdown retries the same candidate
  group at half the batch size after CUDA OOM, down to one candidate.
- Both evaluators can atomically commit one JSON result per completed document.
  `--resume-document-results` skips only documents already stored under an
  identical run fingerprint. The fingerprint binds checkpoint, corpus,
  tokenizer, model type, and scoring settings. `--max-sentences` is rejected for
  resumable output because it can stop inside a document.
- The shard launcher reads document and candidate counts from the finalized
  corpus manifest, waits for every worker, refuses to publish an aggregate after
  any worker failure, and supports explicit complete `DOCUMENT_BOUNDS`.
- The merger validates finite likelihoods, positive counts, candidate counts,
  protocol agreement, duplicate IDs, filename/ID agreement, exact document
  coverage, and the expected candidates per sentence. It writes the aggregate
  atomically only after all checks pass.

The corpus default follows the current evaluation entry:
`dataset/bbc-news/testppl/native_model_topk_300_v2`. A host that keeps the
finalized corpus at `dataset/bbc-news/native_model_topk_300_v2` can pass that
path explicitly or set `NATIVE_DATA` in the Slurm wrapper.

## Full resumable run

```bash
PYTHON_BIN=/home/wangpch/.conda/envs/LLM/bin/python \
PUSHDOWN_CHECKPOINT=saved_models/pushdown/step34354-unsharded \
bash scripts/run_native_document_ppl_shards.sh pushdown 8 \
  dataset/bbc-news/testppl/native_model_topk_300_v2 \
  docppl_runs/pushdown_native_full
```

Re-running the same command resumes completed documents. Changing the
checkpoint, corpus metadata, tokenizer, normalization, context policy, or cache
mode requires a new result directory. The launcher publishes
`aggregate.json` only when every corpus document is present exactly once.

For an individual process, the equivalent controls are:

```bash
PYTHONPATH=. python scripts/evaluate_pushdown_document_ppl.py \
  --checkpoint <checkpoint> \
  --native-data <native_model_topk_300_v2> \
  --start-document 0 --end-document 621 \
  --attachment-normalization stack_legal \
  --document-result-dir <run>/documents \
  --resume-document-results
```

GPST uses `scripts/gpst/evaluate_document_ppl.py` with the same document-result
flags. `scripts/merge_native_document_ppl.py` can independently validate and
merge an atomic document directory. Exact coverage requires
`--expected-documents`; old shard totals without per-document IDs can still be
combined only when this exact-coverage option is omitted.

## Audit decisions

The integration did not copy the remote tree wholesale. Current `main` already
has the more general `compute_depth_rows_gpu()` implementation, so the remote
`compute_last_depth_rows_gpu()` duplicate was represented by parity tests rather
than a second API. Current v1/v2 protocol metadata and corpus paths were kept.

Two remote defects were corrected during integration:

- `scripts/merge_native_document_ppl.py` referenced `protocol` outside the
  function where it was created;
- `scripts/profile_native_document_ppl.py` used `prefix` in the Pushdown branch
  without initializing it and used only the immediately preceding GPST sentence
  instead of the evaluator's bounded candidate-0 history.

The RTX3090-only `--mem=2M`, two-CPU allocation, and thread-count changes were
not copied as repository defaults because those values are cluster-policy and
host specific. The portable allocator override, configurable output directory,
explicit checkpoint/data settings, and recovery behavior were retained.

## Verification boundary

CPU regression tests cover cached/full-prefix token, attachment, and joint NLL
parity for both v1 and v2, tied and untied output heads, context-window rebuilds,
OOM retry bookkeeping, GPST/Pushdown document skip behavior, depth-row parity,
atomic writes, run-fingerprint rejection, and strict merge failures. A real
multi-GPU full-corpus rerun is still required before replacing any registered
paper value; this implementation change alone is not result evidence.
