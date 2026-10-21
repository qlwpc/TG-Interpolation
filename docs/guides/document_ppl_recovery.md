# Native DocPPL: Resumption, Caching, and Strict Merging

The current Pushdown history policy is [model-best](../protocols/document_history.md); the history policy of GPST-model itself is not defined by this change. For data version selection, see [Evaluation](evaluation.md).

## Reusable Behavior

- The Pushdown cache stores the selected history's K/V, final hidden, tokens, and sentence IDs; the cache is rebuilt after a full-sentence sliding window. `--no-kv-cache` retains the full-prefix correctness reference, and v1/v2 normalization remain unchanged for each.
- The candidate batch is limited by both the token/action budget and the secondary attention budget. On Pushdown OOM, retry with the same candidate interval halved, down to a minimum of 1; do not skip candidates, and do not reuse partial results from failed batches.
- Complete documents are written atomically to individual JSON files; `--resume-document-results` reuses them only when the run fingerprint is identical. The fingerprint is bound to the checkpoint, corpus, tokenizer, model type, and scoring settings; changing settings such as the history policy or cache uses a new directory. `--max-sentences` may truncate documents and cannot be used together with resumable document output.
- The launcher reads the document/candidate counts from the finalized manifest and waits for all workers; if anything fails, no summary is published.
- merge rejects non-finite NLL, invalid counts, duplicate IDs, filename/ID mismatches, protocol or candidate count mismatches, and coverage gaps; after all checks complete, the aggregate is published atomically. The same filename does not mean the same run identity.

## Entry Point

```bash
PYTHONPATH=. python scripts/evaluate_pushdown_document_ppl.py \
  --checkpoint /path/to/pushdown-checkpoint \
  --native-data dataset/bbc-news-reserved-clean-v1/native_model_topk_300_v2/test \
  --start-document 0 --end-document 5000 \
  --attachment-normalization stack_legal \
  --document-result-dir /path/to/new-run/documents --resume-document-results
```

Divide into complete document intervals according to the device budget. `scripts/run_native_document_ppl_shards.sh` supports an explicit corpus, checkpoint, output directory, and `DOCUMENT_BOUNDS`; the old default paths must be verified before overriding them.
The GPST implementation has been migrated to the separate project `~/GPST`; its `scripts/gpst/evaluate_document_ppl.py` supports a document-result argument of the same name.
Only `scripts/merge_native_document_ppl.py --expected-documents N` requires exact full coverage; when that option is omitted, legacy shard totals can be aggregated, but this does not mean per-document completeness was checked. The clean campaign uses its dedicated [input audit/final validation contract](../protocols/clean_document_ppl.md).

Regression coverage includes cache/full-prefix equivalence, sliding window, OOM retry, fingerprint rejection, and strict merging.
For interpretation of results, see [paper results notes](../results/paper.md).
