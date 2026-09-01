# FlexAttention head-dimension-128 control — 2026-09-01

> **Correction:** this experiment followed an initial misinterpretation of the
> incident. The relevant boundary is sequence length `N < 128`, not
> `head_dim=128`. The measurements below remain valid as a separate control,
> but they do not diagnose the reported failure. See
> `2026-09-01-flex-attention-short-sequence-upgrade-summary.md` for the corrected
> investigation and environment decision.

## Outcome

Compiled PyTorch FlexAttention forward and backward passed on the allocated
RTX 3090 for both the current and upgraded environments. The test used BF16
Q/K/V tensors of shape `(1, 16, 2048, 128)`, a causal `BlockMask`, default
kernel options, a random output gradient, and explicit CUDA synchronization.
All output values and all Q/K/V gradients were finite.
Both Slurm jobs finished in state `COMPLETED` with exit code `0:0` on node
`rtx3090b` (20 seconds for job 3869 and 15 seconds for job 3874).

| Job | Environment | PyTorch | Triton | Result | Compile + run time |
|---|---|---|---|---|---:|
| 3869 | `LLM` | 2.7.1+cu126 | 3.3.1 | passed | 11.653 s |
| 3874 | `LLM-flex213` | 2.9.1+cu126 | 3.5.1 | passed | 7.058 s |

The upgraded run was about 39% faster in this single cold run, but this timing
includes compilation and is not a steady-state benchmark.

The historical backward failure does not reproduce in the current 2.7.1
environment, so the evidence does not support the causal claim that upgrading
was required to repair it. It does show that the isolated 2.9.1 candidate also
handles the reported shape and is a viable FlexAttention-specific upgrade
candidate.

## Gap found in the earlier task report

Job 3868 validated the third-party `flash-attn` package with tensors shaped
`(1, 16, 2, 64)`. It did not invoke
`torch.nn.attention.flex_attention` and therefore did not test the 128-wide
compiled backward kernel. The dedicated validator added here is
`verify_flex_attention_128.py`, with reusable Slurm entry point
`verify_flex_attention_128.sbatch`.

## Isolated upgrade

The working `LLM` environment was cloned before any package replacement. The
clone retains the name `LLM-flex213` because PyTorch 2.13 was the initial
target. The official CUDA 12.8 index has no 2.13 wheel for this environment's
CPython 3.10; the newest installable candidates listed there stopped at 2.11.
PyTorch 2.11/cu128 was evaluated next, but its main wheel download was
impractically slow through the cluster proxy. The tested candidate was
therefore moved to the smaller-risk CUDA-12.6 line:

- torch 2.9.1+cu126
- torchvision 0.24.1+cu126
- torchaudio 2.9.1+cu126
- Triton 3.5.1
- cuDNN 9.10.2.21
- cuSparseLt 0.7.1
- NCCL 2.27.5
- NVSHMEM 3.3.20

All downloaded wheels matched the SHA256 values published by the official
PyTorch/NVIDIA indexes. After installation, `python -m pip check` reported no
broken requirements and `conda doctor -n LLM-flex213` reported no altered or
missing Conda-owned files.

The repository's extension/parse regression passed in the candidate: 17 tests
passed. The GPST `cppbackend` binary in the repository was compiled against
torch 2.7 and cannot load against torch 2.9 (`c10::SymInt::sym_ne` is missing).
For validation it was rebuilt under `/tmp/tg-cppbackend-flex291` with torch
2.9; the repository binary was deliberately not overwritten. The existing
`tg_mask` binary imported in the candidate, but it should also be rebuilt for
a production migration.

## Promotion decision (superseded by the short-sequence investigation)

Do not replace the production `LLM` environment or update the main
`environment.yml` yet. The current environment already passes the exact
FlexAttention-128 reproducer, while the candidate intentionally lacks the old
FlashAttention 2.8.2 wheel because that wheel is tied to torch 2.7. Paths that
require `flash_attn_varlen_func` or the fused FlashAttention cross-entropy have
not been validated in the candidate.

Before promotion:

1. build both native extensions against the target torch version without
   overwriting binaries required by the current environment;
2. install or build a FlashAttention wheel compatible with the target torch,
   or explicitly verify every configuration that can run without it;
3. run one real checkpoint forward/backward step for the affected 1B config,
   then run the normal multi-GPU smoke test;
4. create a separate Python 3.11 migration if PyTorch newer than 2.11 is
   required, rather than upgrading Python in place.

## Evidence

- Baseline structured result:
  `results/flex-attention-128-torch271-default_3869.json`
- Upgraded structured result:
  `results/flex-attention-128-torch291-default_3874.json`
- Baseline stdout/stderr: `flex-attention-128_3869.out` / `.err`
- Upgraded stdout/stderr: `flex-attention-128_3874.out` / `.err`
- Validator: `verify_flex_attention_128.py`
- Slurm entry point: `verify_flex_attention_128.sbatch`

Relevant upstream issue:
<https://github.com/pytorch/pytorch/issues/133254> documents an Ampere/Ada
shared-memory failure during compiled FlexAttention backward with `D=128`.
The issue remains open, so the passing local runs are workload-specific
evidence, not a general proof that every 128-wide score/mask modification is
fixed.
