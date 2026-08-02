# Pushdown GPU optimization run

The requested 8×RTX3090 jobs use workers=0, sequence 2048, global batch 32, and device batch 4. Because the full-node jobs were repeatedly requeued, the completed matched evidence uses 4×RTX3090, global batch 16, and the same device work. Parsed metrics are under `metrics/`, and the consolidated result is in [RESULT.md](RESULT.md).
