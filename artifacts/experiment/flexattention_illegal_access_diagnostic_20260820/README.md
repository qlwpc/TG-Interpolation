# FlexAttention illegal-access diagnostic

## Question

Does the deterministic CUDA illegal memory access in
`tg_100m_xsum_seed6198` come from the PyTorch 2.7.1 FlexAttention backward
tail path for sequence lengths that are not divisible by the sparse block size
128?

## Fixed inputs

- Source config: `../scaleup_nonfineweb_multiseed_20260815/runs/tg_100m_xsum_seed6198/train_config.yaml`
- Source checkpoint: `saved_models/TG_test/step55457-unsharded`
- Seed: 6198
- World size: 4
- Device microbatch: 5
- Global batch: 40
- Maximum duration: one optimizer step
- FlexAttention stages: 1
- `CUDA_LAUNCH_BLOCKING=1`

The historical resource retry fails during its first optimizer step and emits a
FlexAttention backward kernel with `Q_LEN=KV_LEN=738`, `IS_DIVISIBLE=False`,
and 128-token sparse blocks.

## Controlled runs

| Job | Script | Sequence handling | Dependency |
| --- | --- | --- | --- |
| 3411 | `exact_repro.sbatch` | Original dynamic length | failed as expected: illegal access in backward |
| 3412 | `padded_repro.sbatch` | Pad attention computation to a multiple of 128, then slice before logits | succeeded: step 1, optimizer update, and checkpoint |
| 3431 | `validate_pad128_correctness.sbatch` | Dense/operator/model numerical parity | succeeded: all 6 cases passed |
| 3432 | `pad128_100step_smoke.sbatch` | Pad real TG fine-tuning batches to 128 multiples | succeeded: 100 steps and final checkpoint |

The padding path is diagnostic-only and requires
`OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128`. Padded query rows attend only to
themselves; original query rows cannot attend to padded keys. The environment
variable is absent from production jobs, so their behavior is unchanged.

## Controlled result

The A/B experiment completed on 2026-08-21.

- The four original per-rank sequence lengths were 716, 738, 721, and 791.
  All are non-divisible by 128.
- Job 3411 completed all 12 FlexAttention forward layers on all ranks. With
  `CUDA_LAUNCH_BLOCKING=1`, rank 3 then failed inside the compiled
  `flex_attention_backward` launcher for `Q_LEN=KV_LEN=791`.
- The failing generated kernel had `IS_DIVISIBLE=False`, sparse Q/KV block size
  128, head dimension 64, and forward/backward `num_stages=1`.
- Job 3412 padded the same rank-local lengths to 768, 768, 768, and 896. Its
  generated kernels had `IS_DIVISIBLE=True`.
- Job 3412 completed both gradient-accumulation microbatches, optimizer step 1,
  and its final checkpoint with finite loss and gradients.

This is strong controlled evidence that the runtime illegal access is in
PyTorch 2.7.1 / Triton 3.3.1 FlexAttention's non-divisible backward path, not
NCCL, ordinary CUDA OOM, or the earlier compile-time shared-memory limit. The
experiment localizes the faulty generated kernel and branch; it does not yet
identify the exact erroneous instruction inside that Triton template.

## Correctness validation

Jobs 3431 and 3432 completed on 2026-08-21 with Slurm state `COMPLETED` and
exit code `0:0`.

- Job 3431 tested lengths 63, 127, 129, and 191, spanning both sides of the
  128-token boundary, with both head-independent and per-head masks. All four
  operator cases compared padded FlexAttention forward and q/k/v gradients
  against an unpadded fp32 dense masked-attention reference.
- Padded versus exact FlexAttention forward had relative L2 error from 0 to
  0.0006202. Padded versus dense-reference q/k/v gradient relative L2 error
  was 0.00250--0.00298, with cosine similarity at least 0.9999955.
- Two complete two-layer OLMo cases compared padded FlexAttention against an
  identically initialized dense-SDPA model. The maximum loss absolute
  difference was 1.86e-5; parameter-gradient relative L2 error was at most
  0.00403 and cosine similarity was at least 0.9999919.
- Job 3432 then trained the original 12-layer TG model for 100 optimizer steps
  on 4 RTX 3090 GPUs with dynamic rank-local lengths padded to 768 or 896. It
  completed with finite final loss 0.8768, total gradient norm 0.8024, peak GPU
  memory 9054 MB, and saved `step100-unsharded`.

These results validate that pad-to-128 preserves the intended attention and
gradient semantics within expected bf16/fused-kernel numerical error, while
removing the failing non-divisible backward path. The multiseed fine-tuning
entry points `run_finetune.sh` and `run_smoke.sh` now export
`OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128`; evaluation entry points are left
unchanged because the observed failure is training-backward-specific.

## Interpretation

- Exact run fails in backward and padded run succeeds: strong evidence for the
  non-divisible tail path.
- Both fail in the same backward kernel: investigate BlockMask metadata,
  batch/head broadcasting, and QKV strides next.
- Exact run succeeds: the historical failure is not reproducible under the
  current source/cache state; repeat with a captured batch/kernel cache.
- Padded run fails for a different reason: first repair the diagnostic padding
  path before interpreting it.

## Scheduler changes

Pending resource-retry arrays 3349 and 3350 were cancelled with user approval
so the diagnostic jobs become the next GPU work. Running evaluations were not
interrupted.

On 2026-08-21 the controlled jobs were still blocked by the older, higher-
priority evaluation array 2981. Its pending elements were held with
`scontrol hold 2981`; the already-running element 2981_112 was not interrupted.
`release_eval_array.sbatch` is submitted after the padded diagnostic and
releases array 2981 automatically.

Before jobs 3431/3432 ran, scaleup evaluation element 2981_120 was cancelled
to release resources. At the user's direction, the remaining pending scaleup
evaluation elements, newly started 2981_121, and collector 2995 were then
cancelled. Running element 2981_114 was explicitly retained.

## Additional production-scale evidence

The remote RTX3090 XSum run `tgnomask_mix_tg_100m_xsum_seed31723` (job 45203)
also failed with CUDA illegal memory access near the end of three-epoch
fine-tuning despite device microbatch 1. Its report is
[`remote_rtx3090_job45203_report.md`](remote_rtx3090_job45203_report.md). This
is corroborating evidence for the failure class, not proof that it has the same
kernel-level trigger as the controlled `tg_100m` reproduction.
