# Explanation of Paper Results and Supplementary Experiments

This document distinguishes three types of results: frozen-manuscript original values, clean re-measurements of old checkpoints, and new dedup training. This document preserves the identity of the original values;
revisions to the paper's experimental data are based on [R-08 discrepancy verification](protocol_comparison.md),
and the paper was not modified this time. For training configuration identity, see the [configuration manifest](../../train_configs/paper_pretraining_manifest.json);
for evaluation definitions, see [Evaluation](../guides/evaluation.md).

## Frozen-manuscript BBC main table and clean re-measurements

The paper column is a direct transcription of the frozen manuscript's `tab:exp1-results`, not an old submission version. XSum/BoolQ are the
mean ± sample standard deviation over five fine-tuning seeds; the rest are single evaluations. The main-text manuscript and the locally registered old rerun have the same values for this version; the old submission column differs.
The machine-readable [paper table](../../reproducibility/evidence/paper_bbc_table.json) preserves the manuscript hash;
the [clean results](../../reproducibility/evidence/old_checkpoint_clean.json) preserve exact NLL, coverage, and weight identity.

| Model | Paper XSum | Paper BoolQ | Paper SG | Paper BLiMP | Paper old DocPPL | Old-model clean DocPPL |
|---|---:|---:|---:|---:|---:|---:|
| Terminal-100M | 22.02 ± 0.06 | 66.70 ± 0.36 | 69.80 | 70.77 | 9.89 | 15.369225 |
| Tree-100M | 22.84 ± 0.03 | 68.07 ± 0.35 | 81.40 | 80.95 | 10.01 | 14.736097 |
| TGTree-100M | 23.03 ± 0.02 | 68.12 ± 0.62 | 79.47 | 81.46 | 10.21 | 14.709097 |
| TG-100M | 20.15 ± 0.04 | 64.50 ± 0.40 | 79.19 | 83.28 | 11.02 | 16.405236 |
| TGNomask-100M | 21.91 ± 0.04 | 66.66 ± 0.45 | 78.62 | 81.78 | 9.97 | 14.658209 |
| TGNomask-Aug-100M | 22.12 ± 0.02 | 67.01 ± 0.28 | 78.68 | 83.63 | 10.08 | 14.700370 |
| Tree-NoONT-100M | 22.41 ± 0.04 | 68.67 ± 0.32 | 85.38 | 81.00 | 9.95 | 15.061134 † |
| Tree-Compress-100M | 22.70 ± 0.05 | 68.48 ± 0.49 | 78.27 | 81.40 | 9.91 | 14.730862 † |
| Tree-TripleCNT-100M | 22.43 ± 0.04 | 68.09 ± 0.12 | 81.57 | 83.42 | 10.26 | 14.801293 |
| Tree-Shuffle-100M | 21.56 ± 0.09 | 68.08 ± 0.33 | 76.56 | 67.77 | 13.41 | 17.852117 |
| TGNomask-Mix-TG-100M | 21.87 ± 0.04 | 66.13 ± 0.54 | 75.77 | 83.14 | 10.01 | 14.572762 |
| TGTree-Mix-TG-100M | 22.16 ± 0.07 | 67.91 ± 0.36 | 78.84 | 82.75 | 10.00 | 14.481718 |
| Pause-1-100M | 22.38 ± 0.05 | 62.04 ± 0.19 | 75.11 | 70.85 | 9.77 | 14.916121 |
| Pause-2-100M | 22.25 ± 0.06 | 65.70 ± 3.45 | 76.97 | 72.59 | 10.32 | 15.116468 |
| Pushdown-100M | 21.06 ± 0.05 | 65.54 ± 0.29 | 74.86 | 74.97 | 13.29 | 19.316305 |
| Terminal-500M | 20.75 ± 0.06 | 69.69 ± 0.60 | 71.11 | 64.74 | 2.83 | 17.302676 |
| Tree-500M | 22.25 ± 0.05 | 70.72 ± 0.35 | 63.13 | 76.09 | 3.18 | 16.001136 |
| TGTree-500M | 22.01 ± 0.08 | 70.63 ± 0.32 | 65.95 | 77.65 | 3.28 | 15.894648 |
| TGNomask-Aug-500M | 21.77 ± 0.06 | 70.78 ± 0.74 | 57.58 | 75.97 | 3.19 | 15.483865 |

† The clean values for NoONT/Compress are retained only as historical identity: on 2026-10-07 it was confirmed that projection-based deduplication is missing,
so they do not satisfy the [current scoring protocol](../protocols/clean_document_ppl.md) and must not be used for current rankings or paper revisions.
The public JSON remains a byte-for-byte snapshot of the original and does not indicate that these values have been re-validated; replacement values must come from a complete re-measurement in a new directory.

The erroneous grammar results for NoONT/Compress/TripleCNT in the original summary have been excluded; the table above preserves the
corrected grammar values for the three models; the final verification at that time did not certify the projection deduplication requirement added later. Tree-Shuffle uses a masked checkpoint; Pause uses a dedicated SEP,
which cannot be replaced by a repeat-token control. TreeReg and the BBC 1B additional models are preserved in the clean JSON and are not mixed into the paper main table.
The receipt of the original 22-model final verification remains `failed`, and the scope of the subsequent successful final receipt is the three grammar-corrected models;
for the distinction between the valid per-row summary and the acceptance scope, see [R-08 §7](protocol_comparison.md#completion-status).

## How to interpret the differences

Of the 4,966 documents in the historical BBC DocPPL test, 4,752 have complete token copies in the actual old train
(95.6907%). The clean test is a different set of 5,000 documents, and structural scoring also uses valid candidates and the model-best
history. The table above is therefore a descriptive comparison in which both the data and the scoring protocol changed, and the entire difference cannot be attributed to decontamination.
For the complete boundaries and verifiable counts, see [Data description](../protocols/bbc_data.md).

In the clean measurements of these old checkpoints, Tree-100M/500M have lower DocPPL than Terminal;
for 100M, the two mixed-attention models are the lowest, and for 500M, TGNomask-Aug is the lowest. The original manuscript's statements of "small LM cost" and
that mixed attention "provides no benefit" cannot be generalized to this protocol. Rankings of close values have not been tested for significance.
The clean PPL of old 500M being higher than the corresponding old 100M also cannot be used to infer that parameter scale generally harms generalization, or to replace downstream
task scaling evidence. Pushdown/Tree-Shuffle use their own protocols and are not included in the same controlled-family ranking.

## FineWeb-Edu 1B

The current manuscript uses the Qwen3 tokenizer and the 11-task average of OLMES completion/cloze; the Tree/TGTree
main values use `tree_eval_type=terminal`. The following is a summary that has been registered and checked against the frozen manuscript; the complete per-sample logs
remain an external asset pending release, and this document organization is not written as a new evaluation acceptance.

| Model | BoolQ (%) | 11-task AVG (%) |
|---|---:|---:|
| Terminal | 58.65 | 53.20 |
| Tree | 63.85 | 53.56 |
| TGTree | 67.55 | 55.88 |
| Pause-1 | 65.41 | 55.17 |
| Pause-2 | 63.39 | 55.80 |

SG/BLiMP are independent tasks and are not part of the 11-task average. The models with the same names in BBC 1B cannot substitute for the FineWeb-Edu weights.

## New experiments and reproduction scope

The [dedup 500M results page](bbc_dedup_500m.md) covers ten models obtained with a new training set;
the training token count, LR, global batch, and checkpoint all differ from the table above. The five seeds are fine-tuning
seeds; a single pretraining seed cannot demonstrate pretraining stability. The ten-model evaluation tasks and full DocPPL coverage
were both completed in the 2026-09-29 snapshot; among them, NoONT's DocPPL must be re-measured according to the projection deduplication protocol,
and the old completion receipt cannot certify this new requirement. For the scope of the final verification of local input/weight identity after evaluation, see
the [final verification summary](../../reproducibility/results/dedup500m/final_status.json).

The differences between the paper description and Pushdown's XSum reduce upper bound, BoolQ supplied-span scoring, and BLiMP token-only candidate scoring are preserved in [Evaluation §8](../guides/evaluation.md#pushdown-paper-discrepancies).
The cause of the parsing of the old 91 structural differences remains unknown and cannot be attributed to a uniform top-1/top-2 swap;
after review on 2026-10-02, R-03 was closed by user decision, retaining the [historical reconstruction limitations](../protocols/bbc_data.md#provenance).
These limitations affecting interpretation are publicly retained; the step-by-step troubleshooting process remains local.

The download entries for data/weights have not yet been completed, and a complete retraining has not been completed from an external clean environment; see [Asset status](../../reproducibility/README.md).
