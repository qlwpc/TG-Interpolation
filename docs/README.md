# Documentation index

This directory contains current protocols, implementation records, and dated diagnostic
material. Experimental values and checkpoint identity have one authoritative registry:
[`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md). Before citing an
older report, check [`REPOSITORY_CLEANUP_MEMORY.md`](../REPOSITORY_CLEANUP_MEMORY.md) for known
corrections and scope limits.

## Primary entry points

| Document | Purpose |
| --- | --- |
| [`README.md`](../README.md) | repository setup, data preparation, training, and evaluation entry points |
| [`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md) | model identity, run status, and result registry |
| [`Evaluation.md`](../Evaluation.md) | current evaluator, data-entry, configuration, and protocol routing; results remain in the registry |
| [`pretraining_reproduction.md`](pretraining_reproduction.md) | reproducible pretraining data and campaign workflow |
| [`pretraining_data_pipeline_repair.md`](pretraining_data_pipeline_repair.md) | supplied BBC split fingerprints, data-integrity fixes, offline tests, and full-corpus verification limits |
| [`pause_protocol.md`](pause_protocol.md) | final-paper SEP Pause identities, pretraining configs, v2 evaluation, and explicit historical controls |
| [`pushdown_word_atom_strict_binary_document_ppl_protocol.md`](pushdown_word_atom_strict_binary_document_ppl_protocol.md) | current Pushdown fixed-word-atom Document-PPL protocol |
| [`FSDP_DOWNSTREAM_EVAL_RISKS.md`](FSDP_DOWNSTREAM_EVAL_RISKS.md) | unresolved multi-rank evaluation risks and operating rule |

## Evaluation and data contracts

| Document | Status and scope |
| --- | --- |
| [`gpst_binary_pushdown_document_ppl_protocol.md`](gpst_binary_pushdown_document_ppl_protocol.md) | current GPST strict-binary to Pushdown evaluation contract |
| [`native_model_topk_300_v2_format.md`](native_model_topk_300_v2_format.md) | current model-native top-300 storage format |
| [`native_binary_storage.md`](native_binary_storage.md) | native-binary corpus storage contract |
| [`pushdown_vs_original_repo.md`](pushdown_vs_original_repo.md) | semantic comparison with the original Pushdown repository; use current protocol documents for final values |
| [`native_nary_300_format.md`](native_nary_300_format.md) | legacy shared n-ary top-300 format; not the current model-native format |

## Implementation and active design work

| Document | Status and scope |
| --- | --- |
| [`gpst_implementation.md`](gpst_implementation.md) | completed GPST architecture and implementation record |
| [`PLAN_pushdown_ppl_2x2.md`](PLAN_pushdown_ppl_2x2.md) | active auxiliary 2x2 comparison associated with the current uncommitted evaluator work |
| [`pushdown_gold300_document_ppl_design.md`](pushdown_gold300_document_ppl_design.md) | historical design-stage record; not a directly runnable current protocol |
| [`superpowers/specs/2026-05-22-robust-contributions-design.md`](superpowers/specs/2026-05-22-robust-contributions-design.md) | dated repository design record |

## Diagnostics and historical reports

| Document | Scope |
| --- | --- |
| [`tree300_vs_test_boundary_report.md`](tree300_vs_test_boundary_report.md) | `tree_300` and test-corpus boundary audit |
| [`diagnostics/2026-08-20-tree300-eval-failure-report.md`](diagnostics/2026-08-20-tree300-eval-failure-report.md) | full tree-300 failure diagnosis |
| [`reports/pushdown_gpu_optimization_20260802.md`](../reports/pushdown_gpu_optimization_20260802.md) | consolidated outcome of the completed Pushdown GPU optimization campaign |
| [`reports/`](../reports/) | dated audits and run reports; they do not replace the result registry |

Completed campaign plans and checklists are intentionally removed once their durable
architecture, protocol, or failure evidence has been incorporated into the documents above.
