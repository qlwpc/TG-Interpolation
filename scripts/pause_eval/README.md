# Pause checkpoint evaluation campaign

This workflow runs the canonical evaluation package for a SEP-backed Pause
checkpoint:

- SG, BLiMP, and terminal-document PPL;
- XSum finetuning and full-test ROUGE for five fixed seeds;
- BoolQ finetuning and validation accuracy for the same five seeds.

The driver freezes checkpoint-derived configs and records hashes before any
Slurm submission. A four-GPU smoke job performs one real optimizer step and a
two-batch evaluation for both downstream data paths. For already-trained
finetune checkpoints, prefer the dependency-free eval-only array workflow below:
each array element receives its own GPU allocation directly from Slurm.

All entry points honor `PAUSE_WORKSPACE`, `PAUSE_SCRIPT_ROOT`, and
`PAUSE_RUNTIME_BIN`.  This allows a campaign-owned source snapshot on another
cluster to run without editing or overwriting that cluster's existing worktree.

## Prepare a new Pause-2 campaign

Run from the repository root.  Replace the two example paths and label only:

```bash
python scripts/pause_eval_campaign.py prepare \
  --checkpoint saved_models/Pause2_MODEL/stepNNNNN-unsharded \
  --campaign-dir artifacts/experiment/pause2_sep50261_full_eval_YYYYMMDD \
  --label pause2_sep50261

python scripts/pause_eval_campaign.py verify \
  --campaign-dir artifacts/experiment/pause2_sep50261_full_eval_YYYYMMDD

```

For SIST RTX6000D nodes, use the measured runtime profile below. BoolQ can use
eval batch 32. Pause XSum must remain at eval batch 1:

```bash
python scripts/pause_eval_campaign.py prepare \
  --checkpoint saved_models/Pause2_MODEL/stepNNNNN-unsharded \
  --campaign-dir artifacts/experiment/pause2_sep50261_full_eval_YYYYMMDD \
  --label pause2_sep50261 \
  --xsum-train-microbatch 10 --boolq-train-microbatch 10 \
  --xsum-eval-batch 1 --boolq-eval-batch 32

```

The precision, global batch, epochs, learning rates, datasets, and metrics stay
canonical. XSum pipeline version 2 passes the checkpoint context length and
pause-token ID into the finetune dataset, expands the raw summary supervision
mask with the token stream, and uses phase-constrained pause generation with a
KV cache. Device eval batch size remains 1 because generation is intentionally
implemented one document at a time.

Completed XSum directories created by an older pipeline are not reusable. The
finetune and eval runners require a matching `training_contract.json` with
`xsum_pipeline_version: 2`; an absent or stale contract exits with status 42.
This prevents a successful-looking marker from silently reusing a checkpoint
trained with the old label mask or pause convention.

The older `pause_eval_campaign.py submit` path creates cross-job `afterok` /
`aftercorr` chains and the packed runner manually partitions GPU indices. Keep
it only for reproducing legacy submissions. On SIST, use the native phase and
array wrappers below.

## SIST native eval-only arrays

Use this after all five task checkpoints have valid `TRAIN_DONE` and
`final_checkpoint.txt` records. It submits one Slurm array with five elements;
each element independently requests four GPUs, and `%2` caps total concurrent
usage at eight GPUs. There are no cross-job dependencies and the runner never
overrides Slurm's `CUDA_VISIBLE_DEVICES`.

```bash
export PAUSE_WORKSPACE=/path/to/campaign-owned/code
export PAUSE_SCRIPT_ROOT="$PAUSE_WORKSPACE/scripts"
export PAUSE_RUNTIME_BIN=/public/home/wangpch/venvs/LLM-sm120/bin
export PAUSE_SBATCH_BIN=/opt/gridview/slurm/bin/sbatch
export PAUSE_ARRAY_MAX_PARALLEL=2
export PAUSE_GPUS_PER_RUN=4
export PAUSE_TOTAL_GPU_BUDGET=8

bash "$PAUSE_SCRIPT_ROOT/pause_eval/submit_eval_only_array.sh" \
  /path/to/campaign boolq

# Submit XSum only after BoolQ is complete when task order matters.
bash "$PAUSE_SCRIPT_ROOT/pause_eval/submit_eval_only_array.sh" \
  /path/to/campaign xsum
```

Pending elements with reason `JobArrayTaskLimit` are expected: they are the
array's `%2` concurrency gate, not a dependency or priority failure. For XSum,
the wrapper forces eval batch 1 and disables per-example prediction logging.

For a fresh campaign, use three explicit phases without cross-job dependencies:

```bash
# Phase 1: SG + BLiMP + terminal doc PPL use 2 + 2 + 1 GPUs concurrently.
bash "$PAUSE_SCRIPT_ROOT/pause_eval/submit_base_jobs.sh" /path/to/campaign

# Phase 2: each BoolQ array element finetunes and evaluates one seed.
bash "$PAUSE_SCRIPT_ROOT/pause_eval/submit_finetune_eval_array.sh" \
  /path/to/campaign boolq

# Phase 3, submitted after BoolQ completes: five XSum seed elements.
bash "$PAUSE_SCRIPT_ROOT/pause_eval/submit_finetune_eval_array.sh" \
  /path/to/campaign xsum
```

To interleave fresh Pause-1 and Pause-2 XSum runs while keeping the same eight
GPU ceiling, submit the combined 10-element array. Adjacent array elements use
the same seed, first Pause-1 and then Pause-2, so `%2` compares both variants
without giving one campaign a full-array head start:

```bash
sbatch --array=0-9%2 \
  --export=ALL,PAUSE1_CAMPAIGN_DIR=/path/to/pause1,PAUSE2_CAMPAIGN_DIR=/path/to/pause2 \
  "$PAUSE_SCRIPT_ROOT/slurm/pause_dual_xsum_finetune_eval.sbatch"
```

On SIST Shanghai, the QOS counts array elements and currently caps submitted
jobs at five. Use paired mode there: each of five elements owns one seed and
runs both models serially on four GPUs; `%2` still caps aggregate use at eight
GPUs. Odd seeds run Pause-2 first and even seeds run Pause-1 first.

```bash
sbatch --array=0-4%2 \
  --export=ALL,PAUSE_DUAL_PAIR_MODE=1,PAUSE1_CAMPAIGN_DIR=/path/to/pause1,PAUSE2_CAMPAIGN_DIR=/path/to/pause2 \
  "$PAUSE_SCRIPT_ROOT/slurm/pause_dual_xsum_finetune_eval.sbatch"
```

The phase boundary is an explicit operator decision based on durable markers,
not an `afterok` chain. Within a phase, Slurm owns placement, GPU visibility,
queuing, retries, and the `%2` eight-GPU concurrency cap.

`prepare` requires exactly five seeds; the default canonical list is
`6198 13171 31723 42 2026`.  It also requires the checkpoint grammar to begin
with `pause`, the checkpoint `pause_token_id` to be `50261`, and tokenizer token
`<|SEP|>` to have that same ID.  If Pause-2 intentionally changes the SEP ID,
pass `--expected-pause-token-id ID` and use a matching tokenizer before
submitting.

## Collect results

Collection is safe to run while jobs are active; missing values remain visible
until their durable `TRAIN_DONE` / `EVAL_DONE` markers appear.

```bash
python scripts/pause_eval_campaign.py collect \
  --campaign-dir artifacts/experiment/pause2_sep50261_full_eval_YYYYMMDD
```

The campaign directory contains frozen YAML configs, per-run logs, checkpoint
hashes, `results.json`, `results.csv`, and `REPORT.md`. Treat the campaign as
complete only after the collector verifies all 13 canonical metric records. Finetune
checkpoints retain `model.pt` and `config.yaml`; their optimizer and trainer
states are removed after a successful final-model hash to control disk usage.

If a submission attempt needs to be recreated, first inspect `jobs.json` and
Slurm accounting.  Use `--resubmit` only after resolving the failed attempt;
run markers make already completed task runners idempotent.
Successful submissions are checkpointed after every `sbatch` call in
`jobs.partial.json`.  If Slurm rejects a later job, fix the resource request and
resume without duplicating earlier jobs by passing `--resume-partial` with the
same scheduler/profile arguments.
