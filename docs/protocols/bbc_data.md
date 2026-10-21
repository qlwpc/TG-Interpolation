# BBC Data Versions and Result Interpretation

New experiments by default use dedup train + reserved-clean dev/test. The training set used in historical papers and the old test set share a large amount of full-text duplication, so the old DocPPL cannot serve as independent held-out generalization evidence. This document retains the conclusions that affect reproduction and interpretation; per-run troubleshooting, source-line tracing, and old task records are kept only locally.
For the rules on old vs. new training parameters, deployment paths, result differences, and subsequent revisions, see
[R-08 verification report](../results/protocol_comparison.md).

<a id="versions"></a>
## Data identity

| Data version | Document count | Sentence count | Scored terminal/EOS count | Purpose |
|---|---:|---:|---:|---|
| Historical index test | 5,025 | — | — | Original split index, not equal to the old DocPPL aligned array |
| Historical DocPPL aligned test | 4,966 | 148,836 | 3,284,061 | Old BBC DocPPL of the frozen draft |
| reserved-clean-v1 dev | 1,000 | — | — | New training validation |
| reserved-clean-v1 test | 5,000 | 129,085 | 3,068,713 | Clean re-evaluation of old models and new experiments |
| dedup train | 1,466,132 | — | Counted separately by representation | New 500M training |

The clean test base array contains 3,073,713 tokens; each document's BOS serves only as context, and EOS is scored.
The early test-1000 subset cannot replace the full 5,000 documents. Different data versions cannot have their results mixed even if their paths all end in `test.npy`;
bind the manifest, tokenizer, candidate, and checkpoint identities.

<a id="overlap"></a>
## Historical training overlap

A full scan of the actual old `terminal/train.npy`: 15,187,326 documents, 10,061,025,584 tokens.
In the old DocPPL test, 4,752/4,966 documents have a training copy with identical full tokens (95.6907%);
this involves 91,394 training document instances and 91,631 matching pairs. Splitting rows into separate splits did not eliminate cross-shard identical text.
The [public summary](../../reproducibility/evidence/historical_overlap.json) retains the definition, training hash, counts, and verification status.
This is not a contamination conclusion for all downstream tasks, nor does it alone explain all differences between models.

<a id="raw-census"></a>
## Training assembly and reserved range

The historical download/parse pool is 94 shards; the training assembly entry point is limited to
[89 shards](../../datatools/parse_pretrain_data/bbc_train_shards.txt), rejecting five
`CC-MAIN-2023-{06,14,23,40,50}` reserved shards. Whole-shard reservation may still contain identical-text copies,
so it cannot be directly regarded as clean. New training also needs to use the deduplicated array; ordinary historical `assemble` does not automatically generate dedup train.
The assembly protection does not change the training experience of historical checkpoints.

<a id="clean-audit"></a>
## clean filtering and audit scope

The [build program and freezing rules](../../datatools/reserved_clean/README.md) use exact, 5-gram near,
and 13-gram local reuse filtering; the computation representation applies NFKC/casefold/PTB/whitespace normalization, and the scoring text is not rewritten.
Near-duplicates use 5-gram Jaccard or qualified containment rate ≥0.8; the latter requires at least 50 tokens and 20 shared
5-grams on both sides. When the union of shared 13-gram positions covers ≥30%, the document is quarantined; this local rule masks the frozen high-frequency templates.

An independent re-check of the final 5,000 documents against the specified historical train found no full-text duplication and no
near-duplication among those checked (Jaccard/qualified containment rate ≥0.6). However, shared fragments exist: when templates are not masked, 41 documents have local union coverage
reaching 30%; after masking templates per the freezing rules and merging old dev/test references, the threshold hit count is 0.
This cannot be used to claim "zero text overlap". The [public quantitative summary](../../reproducibility/evidence/clean_overlap.json)
retains the full/subset definitions, training hash, and limitations.

The above near-duplication and fragment audits are relative to the fixed historical train and its thresholds. On 2026-10-07, an independent full scan was run on
the dedup train's 1,466,132 documents / 800,848,405 terminal tokens; the exact overlap of full-token documents with clean test's 5,000 documents
and dev's 1,000 documents was 0 in both cases. The training array SHA256 is consistent with the NoONT 500M
pretraining record, and the terminal projection of the full NoONT train is also consistent with the scanned terminal train.
Evidence is in `artifacts/evaluation/projected_docppl_20261007/training_exact_audit.json` and
`training_data_binding.json`. This item strictly includes BOS/EOS, whitespace, and token IDs; it does not replace normalized
text, near-duplication, and local fragment screening, and cannot be broadened to "zero text overlap". The remaining items for the new dedup train are still pending audit.

<a id="provenance"></a>
## Limitations of historical reconstruction

The differences between the old index's 5,025 documents and DocPPL's 4,966 documents include assembly range, boundaries, and body text revisions.
Historical byte reconstruction requires a fixed recipe and cannot be directly inferred from re-downloading/re-parsing today. The 403 changed sentences of the 91 documents
include 345 sentences with constituent span changes, 32 sentences with label-only changes, and 26 sentences with unary depth changes; the differences already exist in the original
top300 text and its shard arrays. 167 old trees exist among candidates 1–299 (82 in candidate 1),
and 236 are not among the exported candidates; this cannot be attributed to a uniform top-1/top-2 swap.

The 2026-10-02 local re-check reproduced the above structural counts, the historical stage comparison, and the candidate hit/minimum rank for all 403 sentences.
The local post-processing array has 4 sentences with additional duplicate hit slots, which does not change the above counts and cannot be conflated slot-by-slot with the original export.
The parser weight identity, full configuration, and candidate scores of that run still cannot be confirmed; the parsing cause of the structural changes remains unknown.
Per the user's decision, R-03 is closed and no longer tracked as a pending fix item; closure does not mean the cause is resolved, nor does it change the historical
data or result identities. Subsequent new experiments continue to use dedup train + reserved-clean dev/test.
A single re-encoding of the historical train did not match the byte hash even though the counts were consistent; no claim is made that the old training array has been exactly reconstructed.

<a id="paper-impact"></a>
## Impact on paper conclusions

The [paper result notes](../results/paper.md) present the frozen draft alongside clean re-evaluations of old models, including three correctly re-evaluated grammar
values. Observations such as Tree outperforming Terminal under clean and mixed attention leading in 100M PPL apply only to
the corresponding checkpoint, data, and scoring protocol; there is no significance test, and the old–new differences cannot all be attributed to leakage.
The new dedup 500M is a separate batch of training experiments and cannot be placed in the same row as the old model re-evaluations.

<a id="reproduction"></a>
## Reproduction entry points

- [Historical data construction](../guides/historical_pretraining.md): download, parsing, 89-shard assembly, and data version verification.
- [clean filtering](../../datatools/reserved_clean/README.md), [candidate generation](../../datatools/parse_test_docppl_data/README.md).
- [Audit programs](../../diagnostics/README.md): actual train exact scan, raw census, fuzzy scan, and regression.
- [Public asset list](../../reproducibility/README.md#assets): acquisition status of data arrays, weights, and full logs.

The public summaries are sufficient to verify the counts and definitions listed on this page; re-running the full data audit still requires the original data assets.
