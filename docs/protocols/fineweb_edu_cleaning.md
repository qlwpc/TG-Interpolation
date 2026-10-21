# FineWeb-Edu cleaning diagnostics and deduplication

This page records the 2026-10-12 diagnostic study and conservative sample backup.
It does not claim a full 100BT cleaning run or improved model quality. The later
[20B collection](fineweb_clean_20b.md) has its own filter and split protocol;
the [bracket/internal-EOS repair](fineweb_bracket_repair.md) has a separate scope.
The [known-issues register](fineweb_edu_data.md) includes unresolved ADJ leakage.

## Upstream processing and local scope

FineWeb was already processed with DataTrove, including quality filters and
MinHash within each Common Crawl dump; FineWeb-Edu adds educational-quality
selection. See the [FineWeb paper](https://arxiv.org/html/2406.17557v2) and
[FineWeb-Edu data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#dataset-curation).
The local corpus has since been parsed and Qwen-tokenized and lacks complete
upstream document metadata. A local filter rejection therefore does not by
itself prove an upstream filtering error.

The diagnostic used DataTrove 0.10.1 from commit
`1977fbb0f3c163d43cace334b073dda17fa3ef26` in an isolated environment. Relevant
fixed source examples are the [FineWeb pipeline](https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/examples/fineweb.py)
and [four-stage MinHash pipeline](https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/examples/minhash_deduplication.py).
This snapshot is not authenticated as identical to the 2024 release environment.

[Dolma](https://github.com/allenai/dolma), [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)
and [ftfy](https://ftfy.readthedocs.io/en/latest/) were also considered. Only
DataTrove and the small ftfy encoding trial were executed in this study. Those
alternatives are not dependencies of this repository's training recipe.

## Observed compatibility limits

A standalone reproduction of FineWeb's line-ending punctuation rule rejected
7,175 of 7,872 samples as stored, but only 23 after scoring with line-end
`rstrip()`; 7,152 changed decisions. This isolates a representation artifact,
not the full pipeline's rejection rate. The upstream sequence first runs C4,
whose stripping removes trailing whitespace before the FineWeb rule.
C4 may also remove actual text, so its transformed preview cannot be paired with
an unchanged historical Tree array.

The [DataTrove diagnostic](../../scripts/campaigns/fineweb_datatrove_check.py)
ran Gopher/C4/FineWeb text modules and the official text-filter order on the same
7,872 samples. URL/WARC extraction and language filtering were not rerun.

| Diagnostic | Recorded result |
|---|---|
| Four-module text chain | 6,760 passed; 1,112 rejected |
| First rejection due to alphabetic-word ratio | 1,063 records |
| Original formula-keep subset | 6,674 passed; 1,059 rejected out of 7,733 |
| Independent C4 text changes | 7,871 records, including 22 with non-whitespace changes |
| Four-stage MinHash, fixed order and four reader ranks | One already-known exact duplicate pair; no new near-duplicate group |

Manual inspection found medical and ecological articles rejected by the
alphabetic-word rule while some comments, advertising-like text and the known
formula false negative passed. These 1,112 quality hits were retained as
annotations, not adopted as an automatic rejection list. An alternative ratio
excluding pure-symbol words was explanatory only, not a replacement threshold.
MinHash disabled number normalization; the duplicate pair had actual 5-gram
overlap 1. The sample outcome cannot estimate full-corpus duplication.

The ftfy 6.3.1 `fix_encoding` trial used one source-bound excerpt from each of 13
mojibake candidates. All changed; eight had readable local repairs and five still
had residual or questionable substitutions. Recovering `WÃ¼ster` to `Wüster`
does not reconstruct interleaved OCR sentences. This was neither a full-document
repair nor an accuracy estimate. Source text and paired tokens were not rewritten.

## Conservative paired sample backup

The reviewed keep list removes one exact duplicate and the manually confirmed
`u(x ) = ln(x )` false negative from the original 7,733 formula-keep records,
leaving 7,731 records and 34,915,292 source Qwen terminal tokens. Original filter
outputs remain unchanged. This subset is still not certified formula-free.

The [paired backup utility](../../scripts/campaigns/fineweb_paired_backup.py)
proposes matches by shard/record ordinal and requires exact terminal bytes after
Tree structural-ID removal, paired labels, source-span hashes and output readback.
The same selection applies to both streams; terminal offsets cannot be reused
as Tree offsets. Historical token bytes, including their original bracket forms,
are retained. Native-bracket JSONL and C4 previews are separate representations,
not newly encoded paired training inputs.

<a id="backup-scope"></a>
### Backup contents and availability

| Local artifact | Contents |
|---|---|
| `filtered_text_backup/` | Original keep/drop/review JSONL copies |
| `reviewed_selection_records/` | 7,731 selected and 141 unselected records, with original fields and new reasons |
| `paired_source_backup/` | All 7,872 original paired samples across 246 shards |
| `paired_keep_subset/` | The 7,731-record paired selection across 246 shards |
| `paired_keep_selection_reviewed.json`, `manifests/`, `receipt.json` | IDs, independent coordinates, hashes, reasons and matching evidence |
| `quality/`, `minhash/`, `validation.json` | Diagnostics, duplicate mappings and reconciliation |

Backups use standard NPY `<u4`, unlike the headerless source files, while keeping
the historical EOS-only/no-initial-BOS token convention. Missing upstream IDs,
URLs and scores were not fabricated. The historical source scan found 93,584,621
EOS segments in each stream before internal-EOS repair; all sampled pairs passed
projection checks. This did not validate every source record's projection.
All 1,669 files in the delivered sample backup matched their recorded SHA-256.

Local provenance is
`artifacts/analysis/fineweb_edu_scope_20261012/datatrove_check_20261012/`.
Full diagnostics, source snapshots, reviewed examples and backups remain local
or external assets; this path is not a public download. Public source code does
not alone reproduce these historical diagnostic outputs.

## Requirements for a larger cleaning experiment

1. Read source format and boundaries explicitly, and record source hashes,
   ordinals and independent terminal/Tree spans. Deduplicate exact token or
   normalized-text hashes globally across shards with stable representatives.
2. Keep normalization used for matching separate from training text. Validate
   encoding, whitespace, quotation and HTML handling with traceable examples.
   Formula-rule changes and other rejection rules require their own version.
3. Evaluate false rejections before applying quality rules broadly. Topic labels,
   tables, code or ordinary numeric measurements are not by themselves defects.
4. If using MinHash, share one global signature/matching space and preserve the
   same input order and task partition in stages 1 and 4. The example's 5-word
   shingles and 14 buckets × 8 hashes are starting parameters, not locally
   validated thresholds. Default number/punctuation/accent normalization may
   erase distinctions in measurements; evaluate candidate pairs explicitly.
5. If training text changes, jointly re-encode both streams and reparse where
   needed. Do not combine changed terminal text with old trees. A new tokenizer
   requires a separately measured token budget.
6. Benchmark complete shards before estimating full-corpus time, memory and
   scratch needs; the clustering stage may be a single-task memory bottleneck.
   No such full-shard benchmark or full-corpus MinHash run was completed here.

Use an allocated CPU batch job with the target cluster's actual resource settings.
The fixed DataTrove Slurm example defaults to `with_srun=True`; the historical
SIST workflow instead required a single-node `with_srun=False` executor or an
`sbatch` wrapper around LocalPipelineExecutor. Historical partition/task counts
are not portable defaults. No GPST training or model-effect evaluation is
established by these diagnostic runs.
