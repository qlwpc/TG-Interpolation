# FSDP downstream evaluation: persistent risk record

Status: **open / do not treat multi-rank FSDP downstream evaluation as safe**
Recorded: 2026-08-14

## Scope already safe

The current TreeReg and Pushdown syntax-evaluation campaign uses DDP. Its
per-batch barriers have been removed, and downstream metrics perform one
count-insensitive final reduction/gather after each rank finishes its local
partition. This record does not invalidate those DDP results.

## Unresolved FSDP hazards

1. `DistributedEvalSampler` intentionally does not pad or truncate. Rank-local
   sample and batch counts may therefore differ.
2. An FSDP forward is not rank-local: pre-forward parameter unsharding performs
   collectives. If one rank performs an extra forward while another enters the
   metric collective, collective order diverges and the job may deadlock.
3. `Trainer._summon_params_ctx()` currently wraps custom beam/generation calls.
   Those calls start model forwards inside `FSDP.summon_full_params()`. The
   installed PyTorch API explicitly says that forward/backward cannot be
   started inside that context; this path is unsupported even when ranks have
   equal batch counts.
4. Variable-work generation paths may execute different numbers of model calls
   per example or per rank, so merely balancing top-level DataLoader batches is
   insufficient for custom FSDP beam search.
5. A metric's count-insensitive final `compute()` only helps if every rank
   reaches it. It cannot repair an earlier FSDP collective mismatch or a
   rank-local exception.

## Affected evaluator paths

- Ordinary FSDP forward: terminal/gold BLiMP, standard ICL tasks, terminal SG,
  TG sentence/document PPL.
- Unsupported custom-forward context: Pushdown SG beam, BLiMP beam,
  beam-search ICL, Pushdown downstream parsing/generation, XSUM generation, and
  TG word-synchronous generation.

## Operational rule

- Use DDP/full-model replicas for downstream evaluation unless a path has a
  dedicated multi-GPU FSDP integration test.
- Do not use `summon_full_params()` as a context for inference forwards.
- Increasing NCCL timeouts is diagnostic only and does not fix collective
  ordering.

## Required remediation before claiming FSDP support

1. Add a fail-fast validation for unsupported FSDP evaluator branches.
2. For ordinary teacher-forced evaluation, give every rank the same number of
   FSDP forwards while attaching a validity mask so padded work never affects
   metrics. Group boundaries for gold-K BLiMP and document PPL must remain
   intact.
3. Move custom beam/generation evaluation to replicated full models, or redesign
   it so every rank executes exactly the same distributed-forward schedule.
4. Add real two-or-more-GPU Slurm tests covering unequal dataset partitions,
   final metric collection, early rank failure, and custom generation.

## Evidence locations

- Repository sampler: `olmo/data/util.py::DistributedEvalSampler`
- Evaluation loop and custom context: `olmo/train.py::Trainer.eval` and
  `Trainer._summon_params_ctx`
- Final metric collectives: `olmo/eval/downstream.py::_all_reduce_tensor` and
  `_gather_list`
- Installed PyTorch restriction:
  `/home/wangpch/.conda/envs/LLM/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py::summon_full_params`
