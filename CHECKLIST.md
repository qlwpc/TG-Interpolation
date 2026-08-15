# Pushdown experiment checklist

- [x] Audit the production config, data representation, attention gradient path, attachment head, and existing dirty-tree changes.
- [x] Freeze the workers=0 requested 8×RTX3090 benchmark and diagnostic controls.
- [x] Record one-GPU original, detach, no-attachment, vector-oracle, and focused phase evidence.
- [x] Record a completed matched 4-GPU legacy/optimized/detach DDP comparison after the 8-GPU jobs were requeued.
- [x] Select the lead candidates from measured component deltas.
- [x] Add independent oracle and depth-gradient reference tests before promotion.
- [x] Implement and test the lead candidates without reverting user changes.
- [x] Repeat the optimized 22-step benchmark; the two optimized medians differ by 0.8%.
- [x] Parse logs, compare peak memory and throughput, and document limitations.
- [x] Leave durable manifests, metrics, and a concise final report.
