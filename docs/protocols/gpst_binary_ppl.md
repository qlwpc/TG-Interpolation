# Pushdown DocPPL Contract for GPST Strict-Binary Candidates

Input v2 here refers to the `native_model_topk_300_v2` storage format, which is a separate version axis from v1/v2 in attachment normalization. The current history-selection policy is **model-best**; see [history selection](document_history.md). Older candidate-0 measurements are collected in the [word-atom protocol](pushdown_ppl.md).

## Inputs and structure

Implementation: `olmo/eval/gpst_binary_pushdown_document_ppl.py`; run/merge:
`scripts/evaluate_gpst_binary_pushdown_document_ppl.py`, `scripts/merge_gpst_binary_pushdown_document_ppl.py`.

- The GPST axis reads `terminal_tokens`, `content_bounds`, `document_ids`, `gpst_valid_counts`,
  `gpst_merge_orders`; it does not read the Pushdown axis and does not treat proposal scores as model probability weights.
- Only the first K valid candidates are consumed; 300 is the physical capacity, and padding does not enter the forward pass, summation, or counting.
- The merge order of the T content tokens must be a permutation of gaps `0..T-2`. Maintain the left and right boundaries of the active interval and merge the constituents on both sides of the gap each time, outputting `(left, split, right)` in original terminal coordinates.
  Each tree has exactly T−1 spans and finally covers all content; conversion complexity is O(KT).
- Recover literal stack actions using closing counts. Valid positions are, in order, the current token, then the right endpoints from stack top to bottom; the gold target is determined by the number of reduces for this token. At the end of the sentence, only one root remains.
  Control positions with `sentence_ids=-1` have no attachment target but may still require LM scoring.
- Multi-BPE words keep a fixed right-recursive intra-word subtree, consistent with the training representation of that checkpoint; they must not be changed to BPE-spliced.

## Scoring, history, and counts

Use the same history for all valid candidates of the same sentence, compute `joint_nll = token_nll + attachment_nll`,
`sentence_ll = logsumexp(-joint_nll)`, without dividing by K and without multiplying by proposal weights.

| Probability axis | Definition |
|---|---|
| v1 `stack_legal` | attachment softmax is normalized only over positions currently legal on the stack |
| v2 `sentence_causal` | normalized over causal positions within the complete sentence; consistent with the support of attachment training CE |

Under a fixed path, v2 NLL ≥ v1, but this cannot be used to compare different search supports. The candidate with the smallest joint NLL for this sentence is selected to commit the history, with ties going to the earliest; document boundaries clear the state. For context, only complete historical sentences are dropped, and KV/hidden and coordinates are rebuilt after the sliding window; an error is raised if the current single sentence exceeds the model length. When record-level BOS removal is applied to non-first sentences, structural coordinates are shifted accordingly.

Historically named fields such as `joint_document_perplexity_v1` must be interpreted together with their normalization metadata.
`candidate0_structured_terminal_perplexity` accumulates only the token NLL of **candidate 0 of the current sentence**,
whose conditioning history is still model-best; it is affected by tree structure and is not a flat terminal baseline.
Both metrics share the same denominator: BOS is not counted, and ordinary terminals/EOS are scored.

Merging must verify: identical protocol/history policy/data/tokenizer/checkpoint hashes; full-document ranges are contiguous with no duplicates and no gaps;
NLL is finite; `model_candidate_forwards == valid_candidate_count`;
`candidate_slots == sentence_count × max_candidates_per_sentence`. `--max-sentences` is for smoke testing only
and does not merge partial-document results into a full run. Candidate batch/OOM retries must not skip or double-count.

## Run boundaries

From the repository root, explicitly set the checkpoint, corpus, and new output path. For example, the clean corpus:

```bash
python scripts/evaluate_gpst_binary_pushdown_document_ppl.py \
  --checkpoint /path/to/pushdown-checkpoint \
  --native-data dataset/bbc-news-reserved-clean-v1/native_model_topk_300_v2/test \
  --candidate-source gpst-strict-binary \
  --attachment-normalization stack_legal \
  --max-sequence-length 2048 --output /path/to/new-result.json
```

The script's old BBC default paths do not represent the current clean entry point; shards are partitioned by complete-document `[start,end)`.
`--disable-kv-cache` is the full-prefix reference path. The merger's `--require-full-bbc` **specifically refers to the old 4,966 articles**
and cannot be used to validate the clean 5,000 articles. For clean data, summaries, and completion gates, see
[clean evaluation protocol](clean_document_ppl.md).

The old GPST-axis full invariants are 4,966 documents, 148,836 sentences, 3,284,061 scored tokens, 37,227,054 valid candidates,
and 44,650,800 physical slots. For K=1/2/5/14/42/132/300, the sentence counts are
9,517 / 2,740 / 3,575 / 3,916 / 3,673 / 3,806 / 121,609 respectively; do not apply these to other corpora.

The regression entry point `tests/test_gpst_binary_pushdown_document_ppl.py` covers conversion/literal-stack cross-checks,
v1 sparse/dense equivalence, poison padding, dual metrics, KV/full-prefix, sliding window, microbatch, prefetch order, and strict merging.
These gates do not prove that a new full run has been completed; all results are still registered according to the actual run identity.
