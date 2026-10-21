# Pushdown fixed-word-atom: Structure, Probabilities, and Historical Comparison

The current history-selection policy is [model-best](document_history.md); for input/conversion details, see
[binary support contract](gpst_binary_ppl.md). Historical candidate-0 and the current clean/model-best are different protocols.

## Training representation and candidate space

The input for `saved_models/pushdown_terminalonly/step34354-unsharded` is
`dataset/bbc-news/parse_aligned/train_pushdown_unary_terminals`.
`preprocessing.json`, `scripts/convert_treereg_to_pushdown_terminals.py`, and
`olmo/data/parse_align.py` consistently indicate that multi-BPE words first retain a fixed right-recursive intra-word subtree, and the word-level tree is then
unary collapse + deterministic right-CNF. The conversion only deletes singleton spans and does not merge intra-word BPE into the parent node.
The generator records this contract as `fixed-per-word-right-recursive-v1`.

| Candidate axis | Structure |
|---|---|
| `gpst-strict-binary` | Word-level strict-binary CKY top-K, then expand fixed word atoms |
| `pushdown-nary-word-atom-right-binary` | Native n-ary top-K at word-level right-CNF, then expand the same word atoms |
| Native n-ary raw evaluation | Does not complete intra-word/manual binary reductions; it is not the training-representation likelihood of this checkpoint |

The n-ary conversion uses `word_starts` to map constituents to word intervals, converting word-level `(l,s,r)` to
`(start[l],end[s],end[r])`; for each multi-BPE word `[a,b]`, it adds `(g,g,b)` (g=a..b−1),
normalizes by split gap, and stably deduplicates. The same topology is counted only once.
The implementation is `right_binarize_native_nary_spans_with_word_atoms` / `NativeNaryWordAtomRightBinarizedPushdownCorpus`.

## Probabilities

`sentence_ll = logsumexp(token_ll + attachment_ll)`, without dividing by the number of candidates and without multiplying proposal weights.
v1 `stack_legal` normalizes at stack-legal positions; v2 `sentence_causal` normalizes at causal positions within the complete sentence,
corresponding to the support of the training attachment CE. For a fixed path, v2 NLL ≥ v1; the two must be registered separately.
Both use the actual number of scored terminal/EOS tokens as the PPL denominator; the history-selection policy is another independent protocol axis.

## Relationship to the paper's results

The paper's native n-ary v1 measurement does not complete all intra-word/manual binary reductions; the fixed-word-atom
binary measurement includes these attachment factors and changes the depth trajectory. One cannot conclude from the difference in the two values alone that the implementation
is wrong, nor can the binary/v2 comparison be relabeled as the paper's main value. The case-by-case comparison for the old data/candidate-0 is kept locally.
For valid clean results and the paper boundary, see [Paper Results Notes](../results/paper.md).
