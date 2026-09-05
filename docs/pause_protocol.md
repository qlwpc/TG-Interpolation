# Paper Pause protocol

BBC 100M Pause-1 and Pause-2 in the final paper's Table 4 are **dedicated-SEP**
models. They use the learned `<|SEP|>` token with ID **50261**. The historical
`pause_token_id: null` checkpoints repeat a terminal token in each pause slot;
they are separate historical controls and cannot be converted into the paper
models by editing their YAML.

FineWeb-Edu 1B Pause models retain their existing Qwen3 SEP ID **151673**.

## Pretraining

The default `scripts/prepare_paper_pretraining.py` campaign and its `bbc-100m`
group select the two BBC SEP models. Generate just these runs with:

```bash
python scripts/prepare_paper_pretraining.py \
  --models bbc_100m_pause1_sep bbc_100m_pause2_sep \
  --campaign-dir artifacts/experiment/paper_pause_pretraining
```

| Model | Sequence length | Peak LR | Global batch | GPUs | Microbatch per GPU | Accumulation | Final step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pause-1 | 2048 | 0.006585 | 216 | 8 | 9 | 3 | 45487 |
| Pause-2 | 2049 | 0.007458 | 272 | 8 | 17 | 2 | 54156 |

Both consume the BBC terminal stream, insert SEP positions online, and use
seed 6198, one epoch, causal FlashAttention, and `flex_attention: false`.
The generated launcher preserves the recorded eight-GPU world size. Its
`protocol.json` binds the source, output hashes, and final checkpoint identity.
Use `--validate-data` after preparing the corpus and tokenizer, before launching.

The submitted training configs for SIST jobs 988670/988671 are preserved
byte-for-byte in [`train_configs/paper_sources/`](../train_configs/paper_sources/).
Their SHA-256 values match the original `pause_sep_100m_sist_20260828` submission
manifest. They are explicitly labelled `submitted_training_config`; they are
not exported final-checkpoint configs. This permits config generation without
local model weights. Final checkpoint status and metrics remain in
[`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md).
Pause-2's recorded final checkpoint is on SIST; generation does not download it.

[`scripts/submit_pause_sep_pretrain.py`](../scripts/submit_pause_sep_pretrain.py)
is the original cluster submission/Step Law tool. Use the pinned paper campaign
above when reproducing the recorded runs; changing GPU count in the original
submission tool can change its rounded global batch.

Historical controls are opt-in:

```bash
python scripts/prepare_paper_pretraining.py --groups bbc-100m-historical
```

Their IDs remain `bbc_100m_pause1_repeat` / `bbc_100m_pause2_repeat`, with the
original source configs and `pause_token_id: null`. The default campaign has
27 runs; the complete inventory contains 29 including these two controls.

## Evaluation and finetuning

Use the dedicated campaign for each paper checkpoint:

```bash
python scripts/pause_eval_campaign.py prepare \
  --checkpoint saved_models/pretrain_pause1_100M_SEP50261_steplaw/step45487-unsharded \
  --campaign-dir artifacts/experiment/paper_pause1_eval \
  --label pause1_sep50261
python scripts/pause_eval_campaign.py verify \
  --campaign-dir artifacts/experiment/paper_pause1_eval
```

For Pause-2, use
`saved_models/pretrain_pause2_100M_SEP50261_steplaw/step54156-unsharded`,
`paper_pause2_eval`, and label `pause2_sep50261` on the workspace holding that
checkpoint. Preparation requires the weights and the canonical evaluation data.

- SG and BLiMP score terminal positions after checkpoint-specific pause expansion.
- Document PPL uses document-global pause phase, resets at document boundaries,
  and excludes pause targets from its terminal/EOS denominator.
- XSum requires pipeline v2: aligned summary supervision, phase-constrained
  KV-cache generation, and evaluation batch size 1. Old v1 finetune checkpoints
  and completion markers cannot be reused.
- XSum and BoolQ use seeds 6198, 13171, 31723, 42, and 2026; report the mean and
  sample standard deviation. Checkpoint pause ID and context length are inherited.

[`scripts/pause_eval/README.md`](../scripts/pause_eval/README.md) documents
execution and collection. The generic generator's `pause1` / `pause2` aliases
now identify SEP checkpoints and reject incompatible checkpoint configs or
identity-changing overrides. It directs XSum/BoolQ runs to the dedicated v2
campaign. Explicit `pause1-repeat` / `pause2-repeat` aliases preserve historical
weights for diagnostics; those results do not populate the paper Pause rows.
