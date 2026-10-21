# Pretraining Reproduction Workflow

New BBC experiments default to **dedup train + reserved-clean dev/test**. When reproducing the paper's historical settings, explicitly select the historical generator and the corresponding corpus; the two are not the same experiment. For model and result descriptions, see [Paper Results](../results/paper.md); for external asset status, see [Public Materials](../../reproducibility/README.md).

## 1. Environment and Source Code

Create the `LLM` environment and build the extensions according to the root [README](../../README.md#environment).
Build `tg_mask` and `cppbackend` with the target Python; typed layout also requires a C++17 compiler.
Record the source code commit and any uncommitted patches/newly added source files, configs, dependency package versions, GPU/driver, Torch/CUDA/ABI, and the hashes of the actually loaded dynamic libraries and extensions. Package version numbers cannot substitute for the actual dynamic library identity.
The public environment is defined by [environment.yml](../../environment.yml); the dedicated installation and library fixes on the local 5090 are not prerequisites for the public installation. New hardware requires independent validation of kernels, NCCL collectives, and DDP.

## 2. First Freeze Data and Model Identity

| Data | Default Location | Number of Documents |
|---|---|---:|
| dedup train | `dataset/bbc-news-parsed-dedup/{format}/train.npy` | 1,466,132 |
| clean dev/test | `dataset/bbc-news-reserved-clean-v1/{format}/{dev,test}.npy` | 1,000 / 5,000 |
| NoONT clean dev/test | `dataset/bbc-news-parsed-dedup/tree_noont/` | 1,000 / 5,000 |

NoONT must be obtained by removing ONT from the same version of clean Tree; do not use the old dev/test just because the path is under the dedup directory.
For each representation, retain a manifest: source, split, dtype, shape, tokens, bytes, number of documents, SHA-256.
NPY must be read from the header; BBC is `uint16`, Qwen3/FineWeb is `uint32`. dev/test offsets are
strictly increasing `uint64`, length N+1, with the first entry 0 and the last entry equal to the token count.

The BBC tokenizer vocabulary size is 50320, and EOS/PAD/SEP=50256/50258/50261. The public generator uses the
[frozen tokenizer reference](../../reproducibility/tokenizer-reference.json) to verify byte identity or full
JSON equivalence. The same vocabulary size does not prove token-ID consistency. For newly transferred inputs, compute the full hash; when reusing an existing trusted
train hash, record that it was not recomputed this time, and still fully verify dev/test.
For the scope of data contamination conclusions, see [Data Description](../protocols/bbc_data.md).

<a id="22-四模型的实际路由合同"></a>
### TG Routing

TGTree is LIN2 + causal; TG is LIN2 + STACK/COMPOSE. Non-aug TGNomask and aug are different models.
The public generator's `tg/tgnomask/tgnomask_mix_tg` uses typed kernels:
`tg_typed_attention=true`, Flash/Flex=false; for train/dev/test,
`generate_attention_mask=false`, `generate_doc_lengths=false`, native layout,
`cuda_prefetch=false`. Mix is 8 TG + 8 TGNomask heads. NoONT is an independent causal model.
Check the `use_typed_tg` routing for train/dev/test separately, and validate masks, labels, and gradients on real data.
For details, see [TG Input Pipeline](../models/tg.md).

## 3. Parameters

500M: 16 layers, hidden 1408, 16 heads, MLP ratio 8, RoPE, pre-RMSNorm, SwiGLU,
shared embedding, sequence length 2048, seed 6198, BF16, trained from scratch for 1 epoch.
AdamW betas=(0.9,0.95), eps=1e-8, weight decay=0.1, gradient clipping 1.0; cosine,
2000-step warmup, minimum LR=1e-5.

```text
peak_lr = 1.79 × N^(-0.713) × D^(0.307)
token_batch = 0.58 × D^(0.571)
```

N is the non-embedding parameter count; the measured N for 500M is 507,896,576. D is the
actual train array length of that representation including structural/special tokens. The global batch is rounded according to sequence length and world size; see
[Step Law implementation](../../scripts/step_law.py). Microbatch only adjusts memory and gradient accumulation and must not silently change
data, model, or global batch; save the actual accumulation count and the trailing small batch.

## 4. Select a Generator

| Scope | Command / Guide |
|---|---|
| New BBC dedup 500M | [Public generator parameters](bbc_dedup_500m.md) |
| The paper's historical 27 default runs | `python scripts/prepare_paper_pretraining.py --campaign-dir artifacts/experiment/NEW_PAPER_RUN`; [details](historical_pretraining.md) |
| SEP Pause | `scripts/submit_pause_sep_pretrain.py`; [Pause protocol](../models/pause.md) |

```bash
python scripts/prepare_bbc_dedup_500m.py \
  --models tg tgnomask tgnomask_mix_tg tree_noont \
  --gpus 2 --fixed-microbatch 4 \
  --output-dir artifacts/experiment/NEW_CAMPAIGN
```

## 5. Inspect the Generated Configuration

The output must be a new directory. Inspect base/config/smoke, protocol, and launcher: paths are reachable, and grammar,
tokenizer, and representation are consistent; the formal run performs a full clean dev/test every 3000 steps, W&B offline, and separate run/save
directories. Training LM loss is different from top-300 document PPL. The generator's probe status remains pending.
When using `--runtime-root` to map the target repository, real files, link targets, hashes, and schemas still need to be verified on the remote side.
For the scope of the independent frozen recipes for Pause-1/2 and the new TGTree-Mix-TG experiments, see [New Results](../results/bbc_dedup_500m.md);
do not substitute the CLI key of a similar model.

## 6. Remote Transfer, Environment Validation, and Slurm Submission

First perform file/hash checks, then build extensions, import/schema checks, and kernel forward/gradient with real shapes,
then run a full training/short-evaluation smoke with the formal world size, BF16 autocast, batch, and workers.
Success with a single-GPU kernel cannot substitute for real multi-process, multi-GPU training. The validation wrapper must propagate failure exit codes.
GPU and heavy computation run in an `sbatch` allocation; do not run Python/torchrun directly in the body, do not use
`srun/salloc`, and do not compile or trial-run on the login node. Set account, partition, GPU/CPU, time limit, and log paths
according to the target cluster, run `sbatch --test-only` first, then submit, and save the Job ID and actual resources.

Before starting Python, set a short, exclusive TMPDIR to avoid DataLoader pathname socket length limits:

```bash
set -euo pipefail
export TMPDIR=$(mktemp -d /tmp/tg.XXXXXX)
python scripts/pretraining/check_ipc.py --output "$record/ipc.json"
# Then run the smoke.sh / launch.sh generated this time; record must be a new directory for this run.
```

`check_ipc.py` checks filesystem sockets, spawn workers, and tensor transfer, and refuses to overwrite old credentials.
Caches and logs may use long paths, but the IPC temporary directory must not reuse the long campaign path. The native builder's cache
may change with the allocation, and training nodes must also have a compiler. Do not concurrently modify the shared environment of a running job.

## 7. Microbatch Validation and Formal Training

- `search_max`: search for the usable microbatch while keeping the global batch; for each candidate, run 3 optimizer
  steps and a short dev/test evaluation from scratch, and run 10 steps to confirm the largest candidate. Reduce the batch only on an explicit CUDA OOM;
  environment errors, timeouts, and non-finite values fail directly.
- `fixed`: run real training validation at the explicitly specified value, and a trailing small batch is allowed. Record `fixed_value_validated`,
  and do not record it as the maximum. Write the outer timeout and allowed cost into the protocol.

Smoke does not save formal checkpoints and does not inherit their weights; after passing, reinitialize formal training and restore the full evaluation,
saving, and stopping policy. A short smoke cannot guarantee capacity for any subsequent batch; after adjusting the microbatch, record the reason,
and maintain the global batch/LR/data and progress.

## 8. Resuming and Completion

After the queue disappears, verify `sacct`, exit codes, logs, optimizer step, and checkpoint.
A zero-step failure can be fixed and resubmitted from scratch as a new attempt, with the original evidence kept locally; resuming from an existing checkpoint keeps
seed, LR, batch, scheduler, data, and progress, using `restore_dataloader=true`,
`reset_optimizer_state=false`, `reset_trainer_state=false`, a new continuation output directory,
and does not re-enter the from-scratch probing entry point. Do not use overwrite flags to mask conflicts.

Completion requires epoch/step evidence, a successful exit, a resumable checkpoint, and a final evaluation.
Successful submission, a successful probe, and reaching the time limit are not completion. Cost accumulation includes setup, failures, probes, and resuming.

## 9. Result Registration

The local original campaign saves configs, data/source/environment identity, phase receipts, checkpoints, and logs.
Public results extract only valid final metrics, necessary configs/identity, and completion scope; failure processes and queue status remain local.
New results are backfilled into existing result pages and their machine-readable evidence, avoiding a second manually maintained rolling table.

## 10. Public Reproduction Deliverables and Acceptance

The [Public Materials Inventory](../../reproducibility/README.md) distinguishes: included source code/config/result evidence,
data/weights that still need to be downloaded, and validation that has actually been executed. Hashes cannot substitute for download or build entry points.
Check links, config sources, and CPU tests in a public file copy that does not contain the local archive; completing
builds, GPU smoke, and full retraining from a new external environment is another acceptance level and cannot be inferred from the above checks.
