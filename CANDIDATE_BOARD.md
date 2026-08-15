# Pushdown optimization candidate board

| Candidate ID | Level | Strategy/family | Status | Expected gain | Semantic/engineering risk | Decision gate |
| --- | --- | --- | --- | --- | --- | --- |
| `pd-oracle-vectorize` | incumbent | exploit / on-device sort and scatter-reduce | accepted | saved 211.5 ms/step in the one-GPU matched control | low implementation risk; analysis exposed and repaired a singleton close-order bug in the old oracle | exact equality to a corrected literal stack reference on randomized and 32 production rows |
| `pd-depth-grad-output-mask` | incumbent | exploit / exact shared mask plus output-assisted softmax derivative | accepted | saved another 73.3 ms/step and repaired cross-document gradients | medium; manual gradient must equal dense autograd | gradient parity at `atol=rtol=2e-6`, finite DDP training, repeated throughput gain |
| `pd-depth-grad-lse` | branch | exploit / FlexAttention LSE reuse | abandoned | stable candidate measured 659.45 ms versus 608.55 ms for incumbent | empty query rows initially produced NaNs; finite handling repaired correctness but remained slower | do not promote |
| `pd-attachment-ragged` | brief | exploit / sentence-local sparsity | rejected for this pass | dense head forward measured only 4.6 ms | medium; exact sentence-local CE and all gradients must be preserved | not worth lead-candidate complexity at current profile |
| `pd-depth-grad-fused` | brief | exploit / fused attention backward | deferred frontier | optimized 4-GPU step is still 295 ms slower than detach control | high; custom CUDA/Triton kernel and exact depth-embedding gradient required | revisit for a second optimization campaign |
| `pd-depth-tape-sync` | brief | exploit / tensorization | held | removes small host synchronizations in tape construction | low, expected small because tape is cached across layers | revisit after fused backward feasibility |
| `pd-prefetch` | brief | explore / input pipeline | held | may hide mmap latency | conflicts with the requested workers=0 experimental condition | do not promote in this campaign |

The incumbent is the vectorized corrected oracle plus exact-mask/output-assisted depth gradient. Diagnostic legacy behavior is available only behind explicit environment switches used by the matched control.
