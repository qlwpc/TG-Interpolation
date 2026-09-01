# Pushdown PPL 2×2 auxiliary comparison

## Question

For one trained Pushdown checkpoint, how much of the reported sentence-level
joint PPL is explained by (a) the source of the retained parse support and (b)
the normalization used for attachment probabilities?

This is an `auxiliary/dev` experiment.  It is deliberately sentence-local so
that the existing incremental Pushdown beam and teacher-forced scorer have the
same context contract.  It is not a replacement for the production
candidate-0 document-PPL evaluator.

## Matrix

| parse support | v1: `stack_legal` | v2: `sentence_causal` |
|---|---|---|
| `beam_search` | model-generated complete top-300 parses; legal logits renormalized | same search, but legal expansions retain their probability under the full causal row |
| `teacher_forced` | supplied 300 parse slots, converted to unique Pushdown structures; legal logits renormalized | same supplied structures, scored under the full causal row |

The checkpoint and token probabilities are fixed.  v1/v2 change attachment
probabilities and, for beam search, may also change which parses survive beam
pruning.

## Shared probability contract

For sentence `s`, every cell computes a truncated joint mass

```text
sentence_ll[s] = logsumexp_y log p_model(x_s, y)
```

over that cell's retained **unique** parse support.  There is no `-log(K_s)`:
the attachment factors already make each path a joint `p(x,y)`.  Corpus PPL is

```text
exp(-sum_s sentence_ll[s] / total_terminal_count)
```

where `total_terminal_count` counts sentence-content terminals only.  The
per-document BOS, inter-sentence whitespace/newline, and any synthesized
context token are excluded in this auxiliary sentence-local comparison.

Teacher-forced serialized trees may collide after labels/unaries are removed;
such collisions are one Pushdown latent structure and are counted once.  Beam
search starts from BOS as LM context but an empty structural stack, excludes
BOS from the attachment softmax with sentence IDs, and retains only parses
whose final stack covers the complete sentence content.

## Protocol definitions

Let `C_k` be all finite sentence-local causal targets and `L_k` the targets
reachable from the current stack (`L_k` is a subset of `C_k`).

```text
v1: log p(r_k) = log_softmax(logits[L_k])[r_k]
v2: log p(r_k) = log_softmax(logits[C_k])[r_k], r_k in L_k
```

Thus, for the same fixed path, `NLL_v1 <= NLL_v2`.  This inequality need not
hold between the final beam cells because the two protocols can retain
different approximate top-300 supports.

## Hypotheses and diagnostics

- Null: changing support source and normalization has no material effect after
  fixing tokens and the terminal denominator.
- Alternative: v1 reports a lower teacher-forced PPL by conditioning away
  illegal causal mass, while v2 may substantially change beam pruning and the
  retained structural mass.
- Required outputs: four PPLs and log likelihoods, terminal/sentence counts,
  requested beam width, supplied slots, unique supplied structures, checkpoint,
  training-protocol metadata, and the explicit aggregation/denominator labels.
- Assertions: all four metrics are finite; all four cells share counts; the two
  teacher-forced token streams are identical; teacher-forced v1 joint NLL is no
  larger than teacher-forced v2 for every fixed candidate.

## Minimal code-change map

1. `olmo/attachment.py`: canonical protocol names and target log-probability
   helper.
2. `olmo/model.py`: explicit beam attachment normalization plus an opt-in
   sentence-local, complete-parse mode; preserve existing defaults.
3. `olmo/eval/pushdown_document_ppl.py`: explicit teacher-forced normalization
   and durable protocol metadata; preserve existing callers.
4. `scripts/evaluate_pushdown_ppl_2x2.py`: run the four sentence-local cells
   from one checkpoint and one supplied-tree corpus.
5. Tests: exact v1/v2 log-probabilities, fixed-path inequality, complete
   sentence-local beam behavior, and result aggregation/denominator checks.

## Evidence ladder

- Minimum: unit tests and a toy CPU matrix pass.
- Solid: one real checkpoint on a frozen small sentence subset, with durable
  JSON and exact command.
- Maximum: repeat on a larger subset, then separately implement a document-
  context beam if a full document-PPL comparison is required.
