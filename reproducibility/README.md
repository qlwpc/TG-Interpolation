# Public Reproducibility Materials

This directory contains only the fixed inputs, result snapshots and identity summaries needed for reproducibility. Daily work records, failed attempts, queue status, remote synchronization scripts and full logs are kept locally and are not the default entry point for readers.

| Content | Entry point |
|---|---|
| Historical paper models and 29 fixed configurations | [manifest](../train_configs/paper_pretraining_manifest.json), [generation guide](../docs/guides/historical_pretraining.md) |
| BBC tokenizer validation reference | [reference JSON](tokenizer-reference.json) |
| Revised manuscript BBC main table | [25-row table](evidence/paper_bbc_table_revised.json), [explanation](../docs/results/paper.md); manuscript updated 2026-10-20, with completed NoONT/Compress DocPPL corrections |
| Earlier frozen draft | [Archived 19-row table](evidence/paper_bbc_table.json); retained as historical evidence, not the current manuscript |
| Clean reevaluation of old models | [22-model historical summary](evidence/old_checkpoint_clean.json), plus the [completed projection-deduplicated results](evidence/projected_docppl_20261007.json) superseding NoONT/Compress DocPPL on 2026-10-08 |
| BBC overlap audit | [old data exact](evidence/historical_overlap.json), [clean fuzzy](evidence/clean_overlap.json) |
| Subsequent FineWeb data repair and collection | [Selected audit fields](evidence/fineweb_data_audit.json), [known issues](../docs/protocols/fineweb_edu_data.md); includes confirmed, unrepaired ADJ leakage, separate from the paper experiments |
| Extracted sources and content verification | [evidence manifest](evidence/manifest.json); `local_origin` only identifies the original source and is not a download link |
| Evaluation of the ten dedup models | [results page](../docs/results/bbc_dedup_500m.md), [per-seed/sub-item JSON](results/dedup500m), [historical final-validation summary](results/dedup500m/final_status.json); NoONT DocPPL uses the completed October correction, while its BLiMP candidate-uniqueness check remains pending |

<a id="dedup-configs"></a>
## New experiment configurations

The [ten-model configuration manifest](configs/dedup500m/manifest.json) records checkpoint step, configuration SHA-256 and original-byte YAML. The configurations are identity materials for the actual checkpoints and retain the paths at the time; new runs must use the generator or be adapted to a new directory and cannot directly reuse the original save directories. Eight recipes including `tg/tgnomask/tgnomask_mix_tg/tree_noont` use the [common CLI](../docs/guides/bbc_dedup_500m.md); Pause uses the [SEP entry point](../docs/models/pause.md). TGTree-Mix-TG currently has no unified dedup CLI key; the frozen configuration provides the concrete recipe, and re-running still requires completing the identity and GPU verification in the [workflow](../docs/guides/pretraining.md).

<a id="assets"></a>
## Availability of data, weights and logs

| Asset | Included content | External reproducibility status |
|---|---|---|
| Code, C++/Triton sources, configurations and protocols | Current source tree, build commands, generators | Reviewable; extensions/GPU must be verified in the target environment |
| BBC historical splits, sources and tokenizer reference | Split JSON, data tools, configuration manifest | Full byte-level reconstruction of historical training data has not yet been demonstrated; cannot promise that re-downloading/parsing yields the same arrays |
| dedup train / clean dev/test / top-300 | Build/filter programs, versions and necessary verification summaries | Public download locations for the frozen arrays and candidates are to be added |
| BBC/FineWeb tokenizer runtime assets | Build entry points; the BBC reference is a loadable complete frozen JSON | The BBC file is included and can be copied to `dataset/bbc-news/TG_GPT2_tokenizer.json`; public release of the frozen Qwen3 file with the data is pending |
| Downstream processed data and parse sidecars | The repository already contains some SG and OLMES data and task definitions | The remaining processed files are released directly according to the layout specified in Evaluation; download entry points are to be added, and rebuilding the pipeline is not required |
| Historical and new dedup checkpoints | Fixed configurations; weight hashes attached for old-model clean and ten new models | Full weight download locations are to be added; configurations do not contain weights |
| Per-document results, raw logs, historical run source snapshots | Selected valid summaries, fingerprints and source hashes | Full external archive entry points are to be added; the current working tree is not automatically equivalent to the source code of each historical run |
| FineWeb 11-task per-sample results | Summaries in the paper description | Complete per-sample evidence and bootstrap artifacts are to be released |
| Repaired FineWeb source and filtered paired 20B subset | Repair/filter/collection tools, English protocols and selected receipt fields with source hashes | Full arrays, tokenizer, deltas, journals and diagnostic outputs remain external; public download locations are pending. FWE-014 remains unrepaired |

Hashes verify identity and cannot substitute for acquisition addresses. Until the above gaps are closed, we only state that the source code, recipes and selected evidence have been organized; we do not state that external readers can already fully retrain or reproduce bit by bit. No complete retraining in an external clean environment has been performed.

## Checking the public file copy

```bash
python scripts/check_public_release.py
python scripts/check_public_release.py --export /tmp/tg-public-review
cd /tmp/tg-public-review
python scripts/check_public_release.py
python scripts/prepare_paper_pretraining.py --list
python scripts/render_dedup_results.py
```

The export reads publishable files from the current working tree, including uncommitted modifications, and is not equivalent to a `git archive` of a committed commit. The checks cover Markdown links/anchors in the public file set and configuration and evidence hashes; files that exist locally but are ignored do not count as passing. The export does not include local work records or `.git`. Before release, the required files must also be committed and the actual clean checkout verified; this command does not commit, push or upload assets.
