# Document PPL: model-selected history

As of 2026-09-06, `parse_tree` (`tg_doc` in `olmo/train.py`) and both standalone
Pushdown document evaluators select history from the tested model's scores.

All candidates for a sentence are scored against the preceding selected
history. After the entire candidate set (normally 300 slots) is evaluated, the
candidate with the smallest total NLL becomes the history for the next sentence.
Ties select the earliest candidate. Candidate multiplicities affect the existing
PPL marginalization only; they do not weight history selection.

- `parse_tree`: use the same summed, label-masked sequence NLL as document PPL,
  including the first-token probability from the preceding sentence. Commit the
  winning candidate's KV state, final next-token distribution, and attention-mask
  state together. Document mask generation happens in the evaluation step so
  prefetched data cannot commit an unselected tree. Retain each selected
  candidate's actual length, excluding batch padding from its KV state.
- Pushdown: minimize joint token + attachment NLL under the configured attachment
  normalization. Explicit token-only runs minimize token NLL. With KV caching,
  each microbatch returns its best row's cache; the evaluator retains the global
  winner across all microbatches. This applies to gold300/native n-ary and binary
  Pushdown evaluation, including alternate binary candidate corpora.

The existing PPL candidate aggregation and token denominators are preserved.
The binary Pushdown diagnostic `candidate0_structured_terminal_perplexity` still
scores the current sentence's candidate0, conditioned on model-selected history.
Standalone GPST-model evaluation is outside this change.

## Selection metrics

`non_candidate0_ratio = non_candidate0_count / sentence_count` (range 0–1).
Each fully evaluated sentence contributes once, including document-final
sentences; document boundaries reset history. Stable tree compression keeps the
first occurrence, so selecting a duplicate of candidate0 does not count as a
change from its tree.

- Trainer output: `eval/downstream/<label>_non_candidate0_ratio`. The count and
  candidate scores are reduced across ranks; only complete candidate sets enter
  this ratio.
- Standalone Pushdown JSON and per-document callbacks: `non_candidate0_count`
  and `non_candidate0_ratio`. Shard/document merging sums counts and divides by
  the total sentence count, rather than averaging document ratios.

Pushdown outputs and resumable-run contracts now identify
`prefix_policy="model_best"`. Merging rejects mixed `candidate0` and `model_best`
results, and a new run cannot resume the old history policy's result directory.
Historical candidate0 PPL values must be rerun to obtain this protocol's values.
