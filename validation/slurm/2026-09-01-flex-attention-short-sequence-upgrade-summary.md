# FlexAttention short-sequence (`N < 128`) upgrade investigation — 2026-09-01

> **Correction after the real-mask audit:** the broad outcome originally
> recorded below is superseded. Jobs 3907/3908 show non-finite FlexAttention
> Q/K/V gradients for the repository's real TG and tgnomask masks at `N=127`
> under torch 2.7.1; jobs 3913/3914 reproduce the same failures under torch
> 2.9.1. The synthetic per-head and Pushdown cases in this report remain valid,
> but they do not cover the failing head-broadcast TG path. Padding the real
> masks to 128 passes in job 3919. See
> `2026-09-01-flex-attention-local-path-performance-audit.md` for the corrected
> task-level conclusion and performance routing recommendation.

## Corrected incident scope

The remembered boundary is sequence length, not attention head dimension:
`N=128` is the working boundary and the historical failure was below it. The
local source supports that interpretation: `datatools/test_flexattn.py`
special-cases `Q_LEN < 128`, and the repository's production regression
`test_pushdown_flex_backward` deliberately runs FlexAttention backward at
`N=16`.

The earlier `(1, 16, 2048, 128)` experiment remains a valid head-dimension
control, but it did not test this incident. Its report is retained as an audit
record and marked as superseded for incident diagnosis.

## Original outcome (superseded as a general conclusion)

The current production environment passes the two narrow short-sequence
reproducers below. The isolated upgrade candidate passes those same narrow
cases too. Later real-mask tests demonstrate that these passes cannot be
generalized to all local short-sequence paths.

| Job | Environment | PyTorch / Triton | Test | Result |
|---|---|---|---|---|
| 3879 | `LLM` | 2.7.1+cu126 / 3.3.1 | upstream `N=127`, per-head BlockMask, both APIs compiled, plus backward | passed |
| 3880 | `LLM-flex213` | 2.9.1+cu126 / 3.5.1 | same | passed |
| 3885 | `LLM` | 2.7.1+cu126 / 3.3.1 | repository Pushdown FlexAttention `score_mod` backward at `N=16` | passed |
| 3886 | `LLM-flex213` | 2.9.1+cu126 / 3.5.1 | same | passed |

Jobs 3879 and 3880 use `(B,H,N,D)=(64,8,127,32)`, BF16 tensors,
`H_BlockMask=8`, a compiled `create_block_mask`, a compiled `flex_attention`,
a random output gradient, CUDA synchronization, and finite checks for the
output and all Q/K/V gradients. They reproduce the shape and compilation
conditions from PyTorch issue 147267; that upstream issue was fixed and closed
in March 2025. Both jobs completed on `rtx3090b` with exit code `0:0`.

Jobs 3885 and 3886 run the repository's existing
`tests/test_pushdown_flex_parity.py::test_pushdown_flex_backward`. This is the
more important local evidence: it exercises the production Pushdown
FlexAttention `score_mod`, causal/padding BlockMask, depth-bias gradient, and a
full model logits backward at `N=16`. Both passed; their JUnit XML records one
test and zero failures/errors.

The first boundary matrix also passed with eager BlockMask construction:
`N=127` and `N=128` in both environments (jobs 3875–3878). Those runs are
supporting controls rather than the primary evidence because the historical
upstream reproducer compiled BlockMask construction.

## Separate open GQA issue

PyTorch issue 160018 is a different `N < 128` failure: float32 `D=128`, GQA
(`8` query heads / `4` KV heads), `Q_LEN=100`, `KV_LEN=700`, and an Eagle3
mask. Jobs 3881 and 3882 extended that sample through backward, but both hit
the 10-minute job limit before producing a result. Follow-up forward-only jobs
3883 and 3884 also timed out; their unbuffered phase logs show that BlockMask
construction completes and all remaining time is spent in compiled
FlexAttention forward/config selection. A scheduler timeout is inconclusive:
it is neither a passing kernel result nor the reported `No valid triton
configs` exception.

This GQA result is not used to decide the repository migration. In the local
Pushdown FlexAttention implementation, K/V are expanded to the query-head
count before calling FlexAttention, so it does not call this GQA kernel shape.
The upstream issue also remains open and therefore cannot be treated as a
generally fixed short-sequence case.

## Environment decision

Do not replace `LLM` with `LLM-flex213` for this incident:

1. both environments fail the newly added real TG/tgnomask `N=127`
   head-broadcast cases, so torch 2.9.1 does not repair the local incident;
2. the candidate intentionally has no `flash-attn`, while production has
   FlashAttention 2.8.2 and some repository paths require it;
3. the repository `cppbackend` binary was compiled against torch 2.7 and must
   be rebuilt for torch 2.9; the temporary rebuild used during candidate
   validation was not promoted;
4. `pip check` is clean in both environments, and the production environment
   was not modified.

Keep the isolated candidate for later end-to-end migration work. Promotion
would require rebuilding native extensions, restoring a torch-2.9-compatible
FlashAttention package, and running a real checkpoint training step plus the
normal multi-GPU smoke test.

For the reproduced real-mask failure, the established workaround is padding
the Flex computation to 128 via the repository's
`OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128` option, or using the same structured
mask through SDPA for short workloads. A head-broadcast BlockMask (`H=1`) is
part of the newly failing shape and is not a workaround for this case. The
`OLMO_FLEX_ATTENTION_NUM_STAGES=1` control is for a different shared-memory
configuration failure and should only be used after reproducing that error.

## Evidence

- Validator: `verify_flex_attention_short_sequence.py`
- Boundary Slurm entry point: `verify_flex_attention_short_sequence.sbatch`
- Repository regression Slurm entry point:
  `verify_pushdown_flex_short_backward.sbatch`
- Exact `N=127` results:
  `results/flex-attention-short-sequence-torch271-n127-h8-compiledmask_3879.json`
  and
  `results/flex-attention-short-sequence-torch291-n127-h8-compiledmask_3880.json`
- Repository `N=16` JUnit results:
  `results/pushdown-flex-short-backward-torch271-n16_3885.xml` and
  `results/pushdown-flex-short-backward-torch291-n16_3886.xml`
- Repository test: `tests/test_pushdown_flex_parity.py`
- Upstream issues:
  <https://github.com/pytorch/pytorch/issues/147267> and
  <https://github.com/pytorch/pytorch/issues/160018>
