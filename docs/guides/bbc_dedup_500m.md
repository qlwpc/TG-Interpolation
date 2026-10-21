# 500M Pretraining Configuration for Deduplicated BBC Data

Formal pretraining must follow the [pretraining workflow](pretraining.md). The public generator
`scripts/prepare_bbc_dedup_500m.py` contains eight model recipes and workflow configurations.
The default remains `terminal/tree/tgtree/tgnomask_aug`; newly added models must be explicitly selected via `--models`:

| CLI model | Data representation | grammar / attention |
|---|---|---|
| `terminal` / `tree` / `tgtree` | terminal / tree / tg | Grammar of the same name, causal FlashAttention |
| `tgnomask_aug` | tg | Keeps the original aug + Flex recipe, distinct from non-aug |
| `tg` / `tgnomask` | tg | Grammar of the same name, typed TG / TGNomask |
| `tgnomask_mix_tg` | tg | `mixing`, 8 TG + 8 TGNomask heads, typed |
| `tree_noont` | tree_noont | `tree_noont`, causal FlashAttention |

Generate a brand-new campaign from the repository root, in the training environment:

```bash
python scripts/prepare_bbc_dedup_500m.py \
  --models tg tgnomask tgnomask_mix_tg tree_noont \
  --gpus 2 --hardware RTX5090 --fixed-microbatch 4 \
  --output-dir artifacts/experiment/NEW_UNIQUE_CAMPAIGN
```

Training reads `dataset/bbc-news-parsed-dedup/{format}/train.npy` by default; dev/test read the complete base arrays (1000/5000 documents) under `dataset/bbc-news-reserved-clean-v1/{format}/` by default.
Tree-NoONT clean dev/test defaults to `--data-dir/tree_noont/`, and can also be specified via
`--noont-eval-dir ROOT` as `ROOT/tree_noont/`; it cannot be replaced with the old test set.
A unified directory can be passed to `--data-dir`, `--eval-dir`, and `--noont-eval-dir` at the same time.

Each model outputs `base.yaml`, `config.yaml`, `smoke.yaml`, `protocol.json`, `launch.sh`,
and `smoke.sh`. The formal configuration includes a full clean dev/test evaluation every 3000 steps, W&B offline, DDP/BF16,
1 epoch, checkpoint retention policy, a 44h Trainer time budget, and independent run/save paths; smoke runs separately
for 3 steps and a short evaluation, with `--smoke-timeout` defaulting to 1800 seconds and not changing the formal training budget. The launcher sets `WANDB_MODE=offline` and a short TMPDIR,
uses torchrun with the real world size; it only runs within an already allocated GPU job and is not responsible for requesting resources or submitting.

- `--max-microbatch` defaults to 1 and only generates initially divisible candidates; the protocol marks it as `search_max/pending_gpu_probe`.
- `--fixed-microbatch N` fixes the candidate even if it does not divide the local batch, and records the actual ceil accumulation count and the tail batch;
  it cannot change the global batch and must still be verified on the target machine. Neither option is a claim that a GPU memory probe has passed.
- `--runtime-root /absolute/target/repo` maps in-repository inputs, tokenizer, outputs, and launcher paths
  to the target repository; this mode requires these local paths to be inside the repository. Local assets are checked at generation time; remote existence is verified separately.
- `--campaign` defaults to the output directory name, and together with `--hardware` and the GPU count generates run names and the W&B group;
  `--wandb-entity` can adjust ownership. The output directory must be brand new.
- `--tokenizer-reference` defaults to the already public frozen JSON; if the byte hash differs, it must pass validation via the reference hash and full
  JSON equivalence, and no longer merely warns and continues generation.
- By default it checks the train NPY header/payload and the manifest, and reuses the train SHA-256 from the manifest;
  `--verify-train-hash` can recompute the full training array hash. dev/test recompute the full hash, check
  the manifest, document counts, and uint64 offsets; compatible with the clean total manifest and per-split manifests.

The generator performs a TrainConfig schema check for all models, and calls `use_typed_tg` one by one to verify train,
dev, and test routing; the protocol saves data and configuration identity, Step Law derivation, run settings, and pending-verification status.
After generation, the workflow's remote path, extension/numerical, IPC/DDP, microbatch, and submission-receipt verifications must still be completed.
LM loss evaluation during training is not equivalent to top-300 document PPL.

The protocol is based on the [paper training settings](pretraining.md#3-parameters): 16 layers, hidden size 1408,
16 heads, MLP ratio 8, sequence length 2048, seed 6198, trained from scratch for 1 epoch,
AdamW (0.9, 0.95), weight decay 0.1, gradient clipping 1.0, cosine schedule,
2000-step warmup, minimum LR 1e-5. The non-embedding parameter count is actually computed from the meta model.
The learning rate and batch are recomputed using the existing Step Law implementation based on the number of new training tokens for each representation; the global batch
is taken as the nearest multiple of the GPU count, and the microbatch is the largest integer within the upper limit that divides the per-GPU batch.
With `--fixed-microbatch`, the specified value and the tail small batch are retained; no GPU peak memory is verified at the generation stage.

The exclusion scope of the current clean manifest is still marked as the historical training set; this time no complete overlap audit between the new training set and the clean
dev/test was re-performed. Configuration generation and CPU validation do not mean that training has run or that paper metrics have been reproduced.
