# Pushdown word-atom strict-binary Document-PPL protocol

## Conclusion and scope

This is the binary-tree representation that matches the actual preprocessing
of `saved_models/pushdown_terminalonly/step34354-unsharded`.

```text
parser word with multiple BPEs -> fixed right-recursive preterminal subtree
word-level constituency tree   -> unary collapse + deterministic right-CNF
```

It supports two external proposal axes:

- `gpst-strict-binary`: direct word-level strict-binary CKY top-K, already
  stored in the fixed-word-atom representation;
- `pushdown-nary-word-atom-right-binary`: native n-ary top-K, right-binarized
  at word level and then expanded with the same fixed word atoms.

The first has 37,227,054 candidates. The second has 33,007,452 unique binary
topologies after stable collision deduplication. The second is nearly a subset
of the first because both start from the same Benepar span chart but retain
different external top-K supports.

## Direct evidence from the training pipeline

The checkpoint config points to:

```text
dataset/bbc-news/parse_aligned/train_pushdown_unary_terminals
```

Its `preprocessing.json` records right binarization and unary collapse. The
dataset was converted from `train_treereg` by
`scripts/convert_treereg_to_pushdown_terminals.py`, which copies the already
right-CNF spans and removes only singleton spans. In `olmo/data/parse_align.py`,
`collapse_unary_tree` stops at a preterminal whose children are BPE integer
leaves, and `binarize_tree(direction="right")` right-binarizes that retained
preterminal. Thus multi-BPE words are fixed atoms, not spliced siblings of the
word's parent.

The corpus generator independently records the same contract as
`fixed-per-word-right-recursive-v1`; its `expand_candidate_to_subwords` builds
the word atom before materializing the word-level CKY tree.

## Native n-ary conversion

For each native candidate:

1. use stored `word_starts` to map real BPE-interval constituents back to word
   intervals;
2. deterministically right-binarize the n-ary tree in word coordinates;
3. map each word-level binary node `(l,s,r)` to
   `(start[l], end[s], end[r])` in BPE coordinates;
4. for every multi-BPE word `[a..b]`, add fixed word-atom nodes
   `(g,g,b)` for `g=a..b-1`;
5. canonicalize by BPE split gap and stably deduplicate equal binary topologies.

The implementation is
`right_binarize_native_nary_spans_with_word_atoms` and
`NativeNaryWordAtomRightBinarizedPushdownCorpus` in
`olmo/eval/gpst_binary_pushdown_document_ppl.py`.

On the first 200 sentences, all 46,005 converted n-ary topologies occur in the
49,887 direct-GPST support and candidate 0 matches in 200/200 sentences,
including all 112 sentences containing a multi-BPE word. The completed full
structural audit (`3789`) finds 33,007,316/33,007,452 converted n-ary topologies
in direct GPST, zero support-disjoint sentences, and candidate-0 equality in
148,834/148,836 sentences. The two exceptions have the same complete
300-topology support and only swap the top two under a proposal-score near tie.
Evidence:
`artifacts/analysis/nary_word_atom_right_binary_candidate_audit_20260831.json`.

## Probability protocols

Both protocols use the same checkpoint, terminal stream, binary candidates,
candidate-0 document history, 2,048-token complete-sentence truncation, and
terminal denominator. Every sentence uses:

```text
log p_topK(x | h) = logsumexp_y [log p_token(x | y,h)
                                 + log p_attachment(y | x,h)]
```

There is no division by candidate count and proposal scores are not probability
weights.

- v1 `stack_legal` reproduces the historical Table-4 evaluator convention and
  is required for comparison with 13.293598.
- v2 `sentence_causal` is the checkpoint's actual attachment training
  cross-entropy support. It is a different probability and must be reported
  separately.

For a fixed path, v2 attachment NLL is never below v1 because its softmax
denominator contains every sentence-local causal key rather than only currently
stack-legal keys.

## Results and completed full runs

The 200-sentence gate (4,701 terminals) produced:

| support | normalization | candidates | joint PPL | candidate-0 token PPL |
|---|---|---:|---:|---:|
| native n-ary → word-atom right-CNF | v1 | 46,005 | 15.342437 | 14.705170 |
| native n-ary → word-atom right-CNF | v2 | 46,005 | 15.525468 | 14.705168 |
| direct strict-binary GPST | v2 | 49,887 | 15.523951 | 14.705170 |

For candidate 0, historical native n-ary versus the matching word-atom binary
tree loses 507.72 log-likelihood on this subset: 408.28 (80.4%) comes from
attachment actions and 99.44 (19.6%) from structured token probabilities.
Across the full corpus, native candidate 0 is already binary at word level in
all 148,836 sentences. The conversion adds exactly 222,282 fixed word-internal
BPE reductions (7.61% of 2,919,989 binary reductions) and zero candidate-0
word-level right-CNF nodes. The candidate-0 discrepancy is therefore isolated
to omitted multi-BPE preterminal topology.

The completed full results are:

```text
n-ary/word-atom v1 joint LL  = -9,127,769.3338
n-ary/word-atom v1 joint PPL = 16.10959864
n-ary/word-atom v1 candidate-0 token PPL = 16.05088377

n-ary/word-atom v2 joint LL  = -9,160,975.3534
n-ary/word-atom v2 joint PPL = 16.27331339
n-ary/word-atom v2 candidate-0 token PPL = 16.05087503

direct GPST v1 joint LL  = -9,124,537.0987
direct GPST v1 joint PPL = 16.09375107
direct GPST v1 candidate-0 token PPL = 16.05081721

direct GPST v2 joint LL  = -9,157,799.3091
direct GPST v2 joint PPL = 16.25758293
direct GPST v2 candidate-0 token PPL = 16.05086226
```

The n-ary/word-atom results are close to the direct strict-binary GPST results
under both normalizations.  The direct-support advantage is 0.01585 PPL under
v1 and 0.01573 under v2; it comes overwhelmingly from its additional retained
tail candidates, while candidate-0 token PPL is effectively unchanged.

```text
n-ary/word-atom v1: array 3783, strict merge 3786, COMPLETE
n-ary/word-atom v2: array 3784, strict merge 3787, COMPLETE
direct GPST v2:      array 3785, strict merge 3788, COMPLETE
```

The n-ary/word-atom v2 merge passed every full-corpus invariant.  Its v2
likelihood is 33,206.0196 below v1, or 0.0101113 nat per terminal, and all
4,966 document likelihoods obey the expected v2 <= v1 monotonicity.
The direct-GPST v2 merge also passed every invariant and an independent sum of
all eight shards and 4,966 per-document files.  Relative to direct v1, v2 loses
33,262.2105 log-likelihood, or 0.0101284 nat per terminal.

Every strict merge requires 4,966 documents, 148,836 sentences, 3,284,061
scored terminals, contiguous document ranges, matching checkpoint/data/tokenizer
hashes, the expected candidate count, matching normalization/source metadata,
finite likelihoods, and exact model-forward counts.

## Preserved alternative-protocol diagnostics

The following results are retained because they answer a different protocol
question; they are **not** checkpoint-training-representation results.

| structure protocol | scope | normalization | joint PPL | candidate-0 token PPL | interpretation |
|---|---:|---|---:|---:|---|
| native n-ary candidate, splice every word's BPE pieces into its parent, then right-CNF | first 200 sentences / 4,701 terminals | v1 `stack_legal` | 13.716485 | 14.285710 | controlled BPE-topology ablation only |
| native n-ary candidate, word-level right-CNF, then restore fixed right-recursive word atoms | same 200 sentences | v1 `stack_legal` | 15.342437 | 14.705170 | matching training representation on the same subset |

The BPE-spliced scorer's probability implementation was separately checked
against a dense reference: maximum joint-NLL error was `2.4263e-5` for full
prefix scoring and `7.3142e-5` for KV-cache scoring. This validates the scorer
under that alternative tree definition; it does not validate the tree
definition as checkpoint-training aligned.

The full BPE-spliced run was not retained after the preprocessing audit. Only
the bounded 200-sentence diagnostic above remains reportable. Evidence is in
`artifacts/analysis/nary_right_binary_candidate_audit_20260830.json` and
`artifacts/analysis/gpst_binary_pushdown_scorer_equivalence_20260830/rightbinary_v2_result.json`.

## Interpretation boundary

The paper Table-4 value 13.293598 remains the historical native n-ary v1 number.
It evaluates fewer/non-binary reductions than the checkpoint saw during
pretraining and therefore is not the training-representation likelihood.
Binary results around 16 are not evidence of a preprocessing mismatch. They
measure a different, stricter latent structure in which word-atom and artificial
right-CNF reductions also contribute attachment decisions and depth trajectories.
