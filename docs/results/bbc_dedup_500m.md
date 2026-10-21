# Additional BBC dedup 500M experiments

These models were trained from scratch on the new deduplicated training set; they do not replace the [frozen paper table](paper.md).
SG/BLiMP/BoolQ and XSum R-AVG are percentages; ± denotes the sample standard deviation across five fine-tuning seeds.
Fine-tuning seeds: 42, 2026, 6198, 13171, 31723. These are not five pretraining seeds.
DocPPL uses the full reserved-clean v1 test: 5,000 documents, 129,085 sentences and 3,068,713 scored tokens.
Terminal/Pause use a single path; structural models sum valid top-300 joint probabilities and use model-best history.
As of 2026-09-29, each of the ten models had completed 22/22 evaluation tasks.

The 2026-10-07 review found possible duplicate candidates after NoONT projection. Its historical DocPPL is marked † below
and must not be used for current rankings; reevaluation under the [projection-deduplication protocol](../protocols/clean_document_ppl.md) is required.
Original JSON and acceptance summaries retain their historical bytes and completion status; they do not certify the later deduplication requirement.

## Results

| Evaluation site | Model | Completed tasks | SG ↑ | BLiMP ↑ | clean DocPPL ↓ | XSum R-AVG ↑ | BoolQ ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX3090B | Terminal | 22/22 | 80.8848 | 72.3149 | 14.028763 | 23.1149 ± 0.0735 | 70.5382 ± 0.5065 |
| RTX3090B | Tree | 22/22 | 83.4316 | 82.7552 | 13.863770 | 23.5731 ± 0.0649 | 70.0489 ± 0.5797 |
| RTX3090B | TGTree | 22/22 | 80.6760 | 83.7209 | 13.946850 | 23.5886 ± 0.0503 | 68.4281 ± 0.2609 |
| RTX3090B | Pause-1 | 22/22 | 79.0012 | 70.8478 | 14.019490 | 23.4231 ± 0.0560 | 70.2630 ± 0.4060 |
| RTX3090B | Pause-2 | 22/22 | 78.1798 | 72.6851 | 14.030113 | 23.3832 ± 0.0214 | 68.8624 ± 0.3232 |
| RTX3090B | Tree-NoONT | 22/22 | 83.6717 | 81.1030 | 13.938351 † | 23.3025 ± 0.0312 | 70.2385 ± 0.2643 |
| SIST | TG | 22/22 | 79.5455 | 83.3164 | 15.465373 | 20.6733 ± 0.0719 | 65.5474 ± 0.3047 |
| SIST | TGNomask | 22/22 | 81.5836 | 83.2090 | 13.995203 | 22.1355 ± 0.0713 | 69.5352 ± 0.6079 |
| SIST | TGTree-Mix-TG | 22/22 | 79.7618 | 83.3672 | 13.784714 | 23.3052 ± 0.0419 | 69.4067 ± 0.3255 |
| SIST | TGNomask-Mix-TG | 22/22 | 80.1115 | 83.2224 | 13.972621 | 22.3746 ± 0.0556 | 69.0581 ± 0.2896 |

## Evidence and scope

The following JSON files contain per-seed values, SG/BLiMP submetrics, document coverage and original collection times.
They are public copies of the original experiment summaries. Aggregation checked task identity, successful exits,
complete evaluation and DocPPL document/sentence/token coverage.
Post-evaluation input and checkpoint SHA-256 hashes were checked against the original manifests.
SIST checks used a local synchronized mirror; remote storage was not rescanned and evaluation was not rerun.

- [bbc_dedup_500m_local_20260919](../../reproducibility/results/dedup500m/bbc_dedup_500m_local_20260919/results.json): `complete`, collected at `2026-09-29T05:45:17.155150+00:00`.
- [bbc_dedup_500m_pause_noont_20260922](../../reproducibility/results/dedup500m/bbc_dedup_500m_pause_noont_20260922/results.json): `complete`, collected at `2026-09-29T05:45:18.561605+00:00`.
- [bbc_dedup_500m_sist_20260922](../../reproducibility/results/dedup500m/bbc_dedup_500m_sist_20260922/results.json): `complete`, collected at `2026-09-29T05:45:20.429761+00:00`.

Full training configurations and identities are in the [additional experiment configs](../../reproducibility/README.md#dedup-configs).
The public dedup CLI covers eight recipes. Frozen configs for this campaign's Pause-1/2 and TGTree-Mix-TG are stored separately;
TGNomask-Aug or similarly named recipes cannot substitute for them. Data, weights and full-log availability are listed in the
[public asset inventory](../../reproducibility/README.md#assets).

This page is generated from the JSON above by `scripts/render_dedup_results.py`. All three batches, aggregation and local acceptance were completed;
counts, result hashes and scope are recorded in the [acceptance summary](../../reproducibility/results/dedup500m/final_status.json).
