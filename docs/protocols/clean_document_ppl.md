# reserved-clean Document-PPL Reproduction Protocol

Complete test: 5,000 articles, 129,085 sentences, 3,068,713 scored terminal/EOS tokens.
For valid results from the old checkpoint, see [paper results](../results/paper.md); for the new dedup model, see
[supplementary experiments](../results/bbc_dedup_500m.md). This page preserves the reusable protocol.

**Completed correction:** the full projected-unique DocPPL reevaluation of
NoONT-100M, Compress-100M and dedup NoONT-500M completed on 2026-10-08.
Their respective results are 15.078992496103785, 14.748081113054113 and
13.954969500128307, and the revised manuscript reports 15.079, 14.748 and 13.955.
The [result/receipt extract](../../reproducibility/evidence/projected_docppl_20261007.json)
records coverage and identities. BLiMP was deferred and remains separately flagged.

## Inputs and Models

The root directory is `dataset/bbc-news-reserved-clean-v1/`. Validate the base manifest, tokenizer,
canonical boundaries, and terminal projection. Tree/TG read `testppl/{tree300,tg300}` and score only the
original valid candidates specified by `valid_counts.npy`; padding slots do not enter the sum. They then convert according to the model grammar
and, within each sentence, again deduplicate the complete, unpadded token sequence, keeping the first occurrence of each candidate.
**Uniqueness of the original tree does not guarantee uniqueness of the sequence after NoONT/Compress conversion.** The same sequence contributes joint
probability only once and is not weighted by the multiplicity of the original tree; the deduplicated K determines when the end-of-sentence history is committed.
TG should equal, token by token, the conversion of the Tree closing
symbol after in-place repetition. Native candidates are separately checked for word boundaries, valid counts, uniqueness, and padding.
Terminal input is `evaluation_terminal/`; the sentence-length/document-sentence-count index is derived from canonical.

The historical configurations of NoONT, Compress, and TripleCNT have a residual `tgtree` grammar. The model and the converted data
**must** be bound to `tree_noont/tree_compress/tree_triplecnt` respectively **before** construction. A shared underlying tree300 does not mean that the
grammar can be shared. The three items with the old grammar error are permanently excluded. On 2026-10-07 it was further confirmed that the earlier grammar
correction retest still did not handle NoONT/Compress projection duplication.
Those old PPL values were superseded by the complete 2026-10-08 reevaluation in a
new output directory. They must not be relabeled as deduplicated results.
Pause uses a dedicated SEP; Tree-Shuffle uses the masked checkpoint, and the base evaluation uses terminal.

## Probabilities and Numerics

Terminal/Pause use a single path; structural models use the joint probability of the valid candidates per sentence as a truncated sum,
without dividing by K. Tree/TG and standalone Pushdown select the candidate with the minimum joint NLL for this sentence to commit to subsequent history;
see [model-best](document_history.md) for details. Pushdown uses native n-ary,
`stack_legal` v1; different protocols such as binary/v2 must not be mixed in.

The shared Trainer clean retest uses FP32/SDPA with TF32 disabled; Pushdown standalone is also FP32.
Each micro-batch of the same sentence uses a common maximum padding length; history commits only the true length of the selected candidate, and an overlong single sentence raises an error.
BOS serves only as context; the cache is reset at each article boundary, and history continues between sentences. The scoring denominator does not include structural/Pause positions.

## Running and Merging

After checking the arguments from `--help`, run with explicit checkpoint, grammar, input, and a new output directory:

```bash
python scripts/audit_reserved_docppl_inputs.py --help
python scripts/audit_projected_docppl.py --help
python scripts/evaluate_reserved_document_ppl.py --help
python scripts/merge_reserved_docppl.py --help
python scripts/finalize_reserved_docppl.py --help
```

For a complete example of a Pushdown invocation, see [recovery protocol](../guides/document_ppl_recovery.md).
The NoONT/Compress runner must pass `--projection-audit`, pointing to the full projection audit directory;
verify per sentence the unique candidate count, first-occurrence indices, the mapping from original candidates to unique candidates, and the token denominator.
The three models of 2026-10-07 were merged using `scripts/campaigns/projected_docppl_20261007.py`,
validating the results with projected counts; the old merge entry point that checks only the original `valid_counts.npy` cannot be used.
Partition the `[start,end)` intervals by complete document; document results are saved atomically. The fingerprint binds weights, configuration,
data, tokenizer, source code, extensions, and probability protocol; differing identities cannot resume. Only redo the current document that was interrupted.

Strict merge checks: exactly cover documents 0–4999, sentence continuity, per-document token denominator, finite NLL,
candidate range and history counts, and consistency of all scoring fingerprints. Final validation additionally verifies input/weights/source identity.
Missing shards, wrong grammar, or identity conflicts cannot pass; Slurm 0:0, file existence, or complete coverage each alone is insufficient.

The public [clean JSON](../../reproducibility/evidence/old_checkpoint_clean.json) is a historical merge of the original results and the three grammar-correction
results, retaining NLL and weight hashes; its NoONT/Compress entries predate projection
deduplication. Their replacements are in the separate completed October result extract,
so the historical JSON is preserved without silently changing its protocol identity.
The original per-document files and complete run credentials have not yet been released with the source code;
see [asset status](../../reproducibility/README.md#assets). This reorganization did not rerun the evaluation.
The original total final receipt for 22 models is `failed`, and the scoped final receipts for the three subsequent grammar-correction models
are `complete`; the three-model final validation cannot be expanded into a successful total final validation for all old models. For valid results and acceptance boundaries, see
[R-08 §7](../results/protocol_comparison.md#completion-status).
