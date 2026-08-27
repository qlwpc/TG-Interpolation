# Pushdown XSum evaluation speed diagnosis (2026-08-27)

## Observed bottleneck

- Old formal jobs: array `3570`, actual jobs `3571` and `3572`, each requested
  4 GPUs but only 2 CPUs.
- `ps` showed all eight Python ranks at approximately 49.7--49.8% CPU: each
  rank received only half a CPU core.
- A 10-second `nvidia-smi dmon` sample showed only 3--9% SM utilization and
  roughly 6.4--8.1 GiB memory per GPU. The decoder was host/launch-bound, not
  GPU-compute-bound. Slurm exposed four allocated GPUs inside each job; there
  was no evidence that accidental CUDA device overlap caused the slowdown.
- The old formal throughput was about 68--75 seconds per rank step.

## Code-level cause and correction

`pushdown_generate` rebuilt the beam span tensor by assigning every span into a
CUDA tensor inside nested Python loops. At beam 6 and about 700 prompt spans this
created roughly 4,200 tiny CUDA writes per generated token. Cached decoding also
rebuilt a square depth tape even though it needed only the final query row.

Measured on an allocated RTX 3090 (batch 6, 700 spans, length 1200):

- old per-span CUDA collation: 668.365 ms
- one-shot batched collation: 2.840 ms (235.4x faster)
- full square depth tape: 5.025 ms
- incremental final depth row: 2.310 ms (2.2x faster)

The incremental row is tested for exact equality with the final row of the full
matrix. Cached logits and attachment scores also have numerical-parity tests.

## End-to-end validation and deployment

- Optimized smoke job `3575`, real XSum checkpoint, 4 samples:
  - evaluation started 11:06:49
  - steps completed at 11:06:55, 11:06:59, 11:07:04, 11:07:08
  - generation average about 4.76 seconds/sample
  - `R-AVG=0.1407`, identical to the earlier smoke
- 32 focused tests passed.
- New formal array `3576` uses 2 GPUs + 2 CPUs per job, at most four concurrent
  jobs, and disables full source/prediction logging without changing ROUGE state.
- Replacement task 0 is job array `3582` (the first `3576_0` attempt collided
  with smoke TCP port 17700 and was resubmitted after the smoke completed).
- Initial optimized formal throughput across tasks 1--4 is about 7--10 seconds
  per rank step. Optimized GPU SM utilization rose to about 12--25%.
