# `tree_spans` split contamination audit

Date: 2026-08-20

Claim type: computed

## Scope and method

The audit compares the existing precomputed TreeReg `spans.npy` with spans
freshly parsed after replacing object-identity keys in `tree_spans` with
occurrence-unique keys. It reproduces the TreeReg span eligibility checks to
separate raw split changes from changes that actually enter the auxiliary loss.

- Development and test splits were audited in full.
- The 4,947,089-chunk training split was audited with two independent,
  20,000-chunk stratified block samples (64 blocks across the complete split).
- Training estimates below pool the two nonidentical samples (40,000 chunks,
  81,818,379 spans, and 42,861,520 intended TreeReg decisions).

## Results

| Split | Coverage | Chunks with any changed split | Changed spans / all spans | Chunks with affected TreeReg decisions | Corrupted TreeReg decisions | Wrong gold | Dropped decisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 1,620 / 1,620 | 36.7901% | 0.029417% | 20.1852% | 0.025631% | 0.021772% | 0.003859% |
| test | 1,637 / 1,637 | 35.9194% | 0.031760% | 21.2584% | 0.034737% | 0.030466% | 0.004271% |
| train | 40,000 / 4,947,089 sampled | 35.6050% | 0.029179% | 21.8950% | 0.027710% | 0.023149% | 0.004561% |

The two independent training estimates for corrupted TreeReg decisions were
0.028565% and 0.026855%. Their agreement is a sampling-stability check, not a
formal confidence interval.

The apparently large chunk rate and small decision rate use different
denominators. A 2048-token packed chunk contains roughly two thousand spans, so
a sparse span-level defect is distributed across many chunks.

## Validation

All three audits passed the following invariants:

- precomputed and fixed span counts match for every audited chunk;
- no audited `(left, right)` pair changed;
- every observed span difference is confined to `split`.

These checks support two separate conclusions:

1. Pushdown consumers remain unchanged because they use only `(left, right)`.
2. TreeReg supervision is contaminated because it consumes `split`; most
   affected intended decisions receive a wrong gold split, while a smaller
   portion is dropped by TreeReg's eligibility filters.

## Evidence

- `tree_spans_contamination_dev.json`
- `tree_spans_contamination_test.json`
- `tree_spans_contamination_train_sample_20000.json`
- `tree_spans_contamination_train_sample_20000_seed20260820.json`
- `../audit_tree_spans_contamination.py`
