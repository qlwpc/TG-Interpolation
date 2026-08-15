# Pushdown GPU optimization map

- Anchor: production Pushdown 300M model, sequence length 2048, gold span tape, learned depth embeddings, and attachment loss weight 1.0.
- Requested comparison surface: 8×RTX3090 DDP, global batch 32, device microbatch 4, seed 6198, data workers 0, and a 20-step steady-state window. The completed matched fallback uses 4 GPUs/global batch 16 with identical per-device work because full-node jobs were requeued.
- Measured edges: original → vectorized corrected oracle saves 211.5 ms/step; vector oracle → exact-mask/output-assisted depth gradient saves another 73.3 ms/step.
- Accepted incumbent: vectorized oracle plus optimized depth gradient, 11,344 tokens/s/device on the matched 4-GPU run versus 5,328 for the diagnostic legacy control.
- Remaining frontier: fuse the manual depth-gradient recovery; it accounts for a 295 ms/step gap to the 4-GPU detach diagnostic.
- Reporting edge: draw causal conclusions only from matched RTX3090 runs; do not compare absolute throughput with A800 results.
