# Active plan: profile and optimize the Pushdown pretrain pipeline

## Objective and contract

- Run id: `pushdown-gpu-opt-20260802`
- Tier: auxiliary/dev performance optimization with matched DDP training runs.
- Research question: which Pushdown-specific stage limits 300M pretraining throughput, and which equivalence-preserving change removes the largest verified cost?
- Required metrics: tokens/s/device, batches/s/device, per-step wall/GPU time, peak memory, component control deltas, loss/output/gradient equivalence.
- Requested local contract: 8×RTX3090, sequence length 2048, global batch 32, device microbatch 4, DDP, AMP BF16, seed 6198, the production `train_pushdown_unary` data, and `num_workers: 0`.
- Completed matched contract: 4×RTX3090, global batch 16, with every other item above unchanged. The 8-GPU allocation was repeatedly requeued; this fallback preserved per-device work and produced a completed causal before/after pair.
- Scheduler contract: every submission requests one CPU (`-c 1`) and 1 MiB (`--mem=1M`). Slurm rejects the literal spelling `-c=1`, so the accepted equivalent is used.
- Cross-platform caveat: RTX3090 absolute throughput cannot be compared causally with the reported 4×A800 throughput; matched before/after RTX3090 deltas can identify algorithmic overhead.

## Hypotheses and controls

1. The custom `_DepthBiasGradP` path is a large cost because it recomputes dense FP32 attention scores, probabilities, output gradients, and a depth scatter in every layer.
2. The attachment head constructs a full 2048×2048 FP32 matrix, but its measured forward is only 4.6 ms and is not the lead bottleneck.
3. Oracle target derivation is a major serialized bubble: the original GPU→CPU transfer and Python stack loop costs about 215 ms per measured one-GPU step.
4. Dataloading is not expected to dominate with memory-mapped input, but remains part of the workers=0 contract.

Matched controls retain the same model/data/batch shapes: `detach` removes only the manual depth-gradient recovery, `no_bias` removes the full depth-bias path, and `no_attachment` removes only the attachment objective. These are diagnostic controls, not candidate training semantics.

## Code-change map

| Path | Purpose | Control |
| --- | --- | --- |
| `train_configs/pushdown_profile_3090.yaml` | fixed benchmark contract | same overrides for every candidate |
| `scripts/slurm/profile_pushdown_3090.sbatch` | auditable baseline/control/profile launcher | explicit CPU/memory/GPU requests |
| `scripts/analyze_pushdown_profile.py` | durable log parser | same warmup exclusion and metrics |
| `olmo/pushdown.py` | depth-bias candidate after phase attribution | compare outputs, losses, and depth-embedding gradients |
| `olmo/attachment.py` | sentence-local attachment candidate after phase attribution | compare dense reference loss and all input/parameter gradients |
| `tests/test_pushdown_attachment.py` and Pushdown depth tests | regression gates | randomized plus packed-sentence cases |

## Execution design

- Baseline/control runs: 22 steps; exclude steps 1–2 and use steps 3–22 as the 20-step measurement window.
- Focused run: four steps with per-layer custom-backward timing and rank-0 forward/backward probes.
- Promotion gate: finite training, exact objective contract, matching outputs/losses, gradients within stated tolerance, and a matched throughput gain outside ordinary run noise.
- Stop/abandon: OOM, non-finite metrics, objective drift, missing gradients, or a candidate slower than the incumbent.
- Durable outputs: `logs/pushdown_profile_*.log` and `artifacts/experiment/pushdown-gpu-opt-20260802/`.

## Revision log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-08-02 | Initial Pushdown profiling contract | User requested the same analysis/optimization workflow used for TreeReg. |
| 2026-08-02 | Switched the completed matched pair to 4 GPUs without changing per-device work | The four 8-GPU controls remained requeued until 2026-08-06; four GPUs were available for backfill. |
| 2026-08-02 | Promoted vectorized oracle plus exact-mask/output-assisted depth gradient | One-GPU attribution showed 211.5 ms and 73.3 ms incremental step reductions, respectively. |
| 2026-08-02 | Accepted the optimized incumbent | Matched 4-GPU throughput improved 5,328→11,344 tokens/s/device (2.13×), a repeat was within 0.8% step time, and all Pushdown regression tests passed. |
