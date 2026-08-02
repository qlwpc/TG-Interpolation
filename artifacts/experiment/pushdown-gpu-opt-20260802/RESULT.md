# Pushdown pretrain profiling and optimization result

## Outcome

The accepted implementation raises matched 4×RTX3090 DDP throughput from **5,328 to 11,344 tokens/s/device (2.13×)**. Median step time over steps 3–22 falls from **1,527.95 to 715.45 ms (−53.2%)**, with unchanged measured peak memory of 12,360 MiB. A separate optimized repeat measured 709.80 ms and 11,422 tokens/s/device, within 0.8% of the accepted run's median step time.

| 4-GPU run | Job | Median step (ms) | tokens/s/device | batches/s/device | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy oracle + legacy depth gradient | 1893 | 1,527.95 | 5,328 | 0.6504 | 12,360 |
| Optimized incumbent | 1894 | 715.45 | 11,344 | 1.385 | 12,360 |
| Optimized repeat | 1891 | 709.80 | 11,422 | 1.394 | 12,360 |
| Detach diagnostic ceiling | 1892 | 420.45 | 19,010 | 2.321 | 12,062 |

All four runs use sequence length 2048, device batch 4, AMP BF16, workers=0, the same seed/data, and DDP. The legacy run activates diagnostic switches that reproduce the original CPU oracle and original manual depth-gradient algorithm. It is a matched performance control, not the desired objective: profiling exposed correctness bugs in both legacy paths.

## Attribution

One-GPU phase probes separate the two promoted changes:

| Variant | Median step (ms) | tokens/s/device | LM + attachment-loss phase (ms) | Backward (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original | 893.35 | 9,085 | 224.75 | 553.40 |
| Vectorized corrected oracle only | 681.85 | 11,983 | 9.55 | 556.30 |
| Final optimized | 608.55 | 13,410 | 9.50 | 483.95 |
| Vector oracle + detached depth gradient | 276.95 | 29,293 | 9.55 | 151.65 |

- The original oracle copied spans to CPU and ran Python stack loops every microbatch. The on-device fixed-shape sort/scatter-reduce implementation saves 211.5 ms/step in the isolated run; the measured loss phase drops by 215.25 ms.
- The attachment head itself takes only 4.6 ms. Although it creates dense 2048² logits, ragged sentence-local scoring is not the lead optimization target.
- The depth-gradient change saves another 73.3 ms/step. It caches the exact causal+padding+document support once for all 12 layers and uses the actual detached FlexAttention output for the softmax derivative, avoiding a dense `g_attn * pT` reduction.
- A manual depth-gradient cost remains: the optimized 4-GPU run is 295 ms/step slower than the detach control. Removing it requires a fused/custom attention-depth backward rather than another Python-level allocation tweak.

## Correctness repairs

The old oracle closed spans before shifting the current token. Singleton/preterminal spans `(k,k)` could therefore clear the earlier stack and turn later reduces into shift-only targets. The corrected literal stack reference shifts first, and the vectorized oracle matches it on randomized tests and 32 sampled production rows.

The old manual depth backward reconstructed only causal+padding support, while forward FlexAttention also applies document boundaries. It could send depth-embedding gradients through cross-document cells that did not exist in the forward computation. The new backward reuses the exact support, handles all-pad query rows without NaNs, and matches dense autograd at `atol=rtol=2e-6`.

The profiled working-tree Pushdown/parse test set reports 66 passed and 6 skipped; the final targeted rerun after adding diagnostic legacy switches reports 30 passed. An exported copy of the exact optimization commit, excluding unrelated local changes, reports 58 passed and 7 skipped. The final one-GPU phase run found zero parameters with non-finite gradients, and both optimized 4-GPU runs completed all 22 steps.

## Rejected and deferred branches

FlexAttention LSE reuse initially produced NaNs on empty rows. After finite-row handling, the branch was stable but slower (659.45 ms/step) than the accepted softmax reconstruction (608.55 ms), so it was abandoned. A ragged attachment head was rejected for this pass because its forward is only 4.6 ms. A fused depth backward is the next meaningful frontier.

## Scheduler limitation

The requested 8-GPU jobs were submitted with one CPU and 1 MiB. This Slurm accepts `-c 1` but rejects the literal `-c=1`; `--mem=1M` is used verbatim. The initial legacy 8-GPU run (1873) encountered non-finite gradients after its first compiled step. Jobs 1874–1877 were still pending/requeued with predicted starts on 2026-08-06 when this report closed. Therefore no completed 8-GPU performance number is claimed; the accepted evidence is the completed matched 4-GPU pair. Absolute RTX3090 values should not be compared directly with A800 throughput.
