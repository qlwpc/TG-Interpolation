# Pushdown GPU optimization record (2026-08-02)

## Outcome

The accepted implementation combines the corrected vectorized oracle with the
exact-mask, output-assisted depth-gradient path. In the completed matched 4×RTX 3090 DDP
comparison it reached 11,344 tokens/s/device, versus 5,328 tokens/s/device for the
diagnostic legacy control (2.13×). A repeated optimized run differed by 0.8% in median step
time.

The requested contract was 8×RTX 3090, sequence length 2048, global batch 32, device
microbatch 4, BF16, seed 6198, production `train_pushdown_unary` data, and zero data-loader
workers. The 8-GPU jobs remained requeued, so the completed matched comparison used four
GPUs and global batch 16 while preserving all per-device work. Absolute throughput is not
comparable to A800 runs; only matched RTX 3090 deltas support causal conclusions here.

## Attribution

| Change or control | Result | Decision |
| --- | --- | --- |
| Corrected vectorized oracle | saved about 211.5 ms/step in the one-GPU attribution run | accepted |
| Exact-mask, output-assisted depth gradient | saved another 73.3 ms/step | accepted |
| LSE-reuse depth-gradient branch | 659.45 ms/step versus 608.55 ms/step for the incumbent | abandoned after correctness repair because it remained slower |
| Sentence-local ragged attachment head | dense reference forward was about 4.6 ms | rejected for this campaign; complexity outweighed expected gain |
| Fused depth-gradient kernel | optimized training remained about 295 ms/step behind the detach diagnostic | deferred; requires a correctness-matched CUDA or Triton implementation |
| Input prefetch | incompatible with the fixed zero-worker benchmark contract | held |

The legacy, detach, no-bias, and no-attachment variants were diagnostic controls. They do
not define alternative training semantics.

## Correctness gates

Promotion required finite training, unchanged objective semantics, output and loss parity,
and depth-embedding gradient agreement within the campaign tolerance (`atol=rtol=2e-6`).
The oracle and depth-gradient paths were covered by independent reference tests before the
optimized branch was accepted.

## Evidence locations

- `train_configs/pushdown_profile_3090.yaml`: fixed benchmark configuration.
- `scripts/slurm/profile_pushdown_3090.sbatch`: matched launcher and diagnostic switches.
- `scripts/analyze_pushdown_profile.py`: warmup exclusion and metric parsing.
- `tests/test_pushdown_attachment.py` and the Pushdown depth tests: semantic regression
  gates.
- `artifacts/experiment/pushdown-gpu-opt-20260802/`: local campaign artifacts.

This report replaces the completed root-level `PLAN.md`, `plan.md`, `CHECKLIST.md`,
`OPTIMIZE_CHECKLIST.md`, and `CANDIDATE_BOARD.md` files.
