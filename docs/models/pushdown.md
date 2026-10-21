# Pushdown document PPL: A Stable Comparison Between This Repository's Protocol and the Original Repository

> [!NOTE]
> This document only explains the differences in evaluation semantics between the original repository and this repository, and is not a summary of results. The 2026-08-31 checkpoint preprocessing audit confirmed that the training representation retains the fixed right-recursive multi-BPE word atom; the current complete v1/v2 numbers and interpretation are governed by
> [`pushdown_word_atom_strict_binary_document_ppl_protocol.md`](../protocols/pushdown_ppl.md).
> v1 and v2 are two well-defined but mutually incompatible probability protocols.

## 1. Conclusion and Scope

The Pushdown document PPL in this repository is a **document-level, externally top-K structurally truncated marginal metric** defined on local OLMo Pushdown checkpoints. It is not a reproduction of the evaluation script from the original authors' `MurtyShikhar/Pushdown-Layers`.

The original repository version compared here is fixed at
[`MurtyShikhar/Pushdown-Layers@6a4c543`](https://github.com/MurtyShikhar/Pushdown-Layers/tree/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30).
This repository has implemented two probability protocols: v1 `stack_legal` and v2 `sentence_causal`; the corresponding implementation commit is
[`47564d9`](https://github.com/qlwpc/TG-Interpolation/commit/47564d9bb91dffab3f45b8c3b7012ce1e77f6c4b).
The current evaluation entry point aligned with the training representation is:

```text
checkpoint: saved_models/pushdown_terminalonly/step34354-unsharded
candidate data: direct strict-binary GPST, or native n-ary converted to word-atom right-CNF
entry point: scripts/evaluate_gpst_binary_pushdown_document_ppl.py
structure_source: fixed-word-atom strict-binary external top-K
attachment_normalization: stack_legal (v1) or sentence_causal (v2)
prefix_policy: candidate0
```

Therefore, results should be named "local Pushdown fixed-word-atom strict-binary top-K truncated-marginal document PPL (with v1/v2 indicated)", and must not be directly called "original repository Pushdown PPL" or "original repository gold-300 reproduction".

## 2. The Three Probability Paths That Actually Exist in the Original Repository

When discussing attachment probabilities, the original repository's training, given-tree evaluation, and beam decoding must be kept separate.

### 2.1 Training

The original repository feeds the attachment head output directly into cross-entropy, masking only padding / queries with no target; it does not renormalize over the current stack's legal action set before CE. In other words, the training target is the gold attachment probability over the current causal support.

Evidence: original repository
[`interfaces/lm_interface.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/interfaces/lm_interface.py#L120-L162).

### 2.2 Teacher-forced evaluation given a gold tree

In the original repository's `gold` mode, for each query it first performs `log_softmax` over the causal attachment logits, then reads the gold target. It likewise does not first remove already-reduced positions, nor does it perform stack-legal conditioning.

Evidence: original repository
[`pushdown_util.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/pushdown_util.py#L16-L34)
and
[`eval_utils/eval_pushdown_model.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/eval_utils/eval_pushdown_model.py#L176-L197).

### 2.3 Beam decoding

The beam path first performs `log_softmax` over all current causal attachment logits, and only afterward sets already-reduced positions to `-inf`. This step is used to prohibit illegal beam expansions, but it does not renormalize the remaining actions. So it retains the original log probability from the training distribution, not
`p(action | action is stack-legal)`.

Evidence: original repository
[`models/pushdown_transformer_lm.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/models/pushdown_transformer_lm.py#L326-L336).

Conclusion: the original repository indeed uses "teacher forcing for training, beam for some tests", but neither redefines attachment probability as a conditional probability within the stack-legal set. Training and gold-tree scoring use the full causal support; beam only prohibits illegal expansions after softmax.

## 3. Local v1/v2 Probability Protocols

### 3.1 protocol v1: historical Table-4 evaluator convention

The local old evaluator used to mask non-stack-legal actions before CE and then normalize over the remaining actions:

```text
p_v1(r_k) = p(r_k | r_k is stack-legal)
```

This is not the same probability as the local checkpoint's training target or the original repository's given-tree scoring, but it is the historical evaluator convention for the paper's Table 4. Historical native n-ary v1 is retained only under this old structure protocol; the current fixed-word-atom v1 has been independently run in full and can be used for comparisons under the same normalization convention. Neither structure representation may be mixed or merged with v2 numbers.

### 3.2 protocol v2: training CE convention

The current evaluator performs CE directly after the attachment head's existing causal, padding, and sentence-local masks:

```text
p_v2(r_k) = softmax(sentence-causal attachment logits)[r_k]
```

Stack legality is used only to:

- verify whether a given gold/candidate target represents a legal stack transition;
- prevent illegal expansions during beam search.

It does not participate in renormalizing the teacher-forced likelihood. See the implementation at
[`olmo/eval/pushdown_document_ppl.py`](../../olmo/eval/pushdown_document_ppl.py);
the protocol metadata is fixed as:

```text
protocol_version=2
attachment_normalization=sentence_causal
```

Resume-from-checkpoint and result merging check these fields, thereby rejecting old results that lack protocol metadata. The complete v1/v2 numbers are in the current protocol document; this comparison does not duplicate a second set of result tables.

## 4. Item-by-Item Comparison of Evaluation Protocols

| Dimension | Original repository | This repository's fixed-word-atom document PPL |
|---|---|---|
| Main purpose | Sentence PPL, parsing, BLiMP, SyntaxGym surprisal | Document-level PPL for BBC-News |
| Model and checkpoint | Original authors' 16-layer Pushdown LM / BLLIP-LG checkpoint | Local OLMo Pushdown / `saved_models/pushdown_terminalonly/step34354-unsharded` |
| attachment head | `StackPredictor`: based on `PushdownAttention`, with depth embedding/composer, q/k MLP, and relative position term | Two-layer MLP inspired by the paper's formula + a single bilinear `W`; not the original repository's `StackPredictor` module, see [`olmo/attachment.py`](../../olmo/attachment.py) |
| Basic evaluation unit | Independent sentences, explicit SOS/EOS | Sentences in document order, retaining token and structure context across sentences |
| Structure source | A given single gold tree, or the model's own beam | Up to 300 candidates produced and ranked by an external parser; no Pushdown beam is run |
| Structure representation | Unlabeled binary tree after GPT-2 tokenization | unary-collapse, word-level right-CNF, multi-BPE words use fixed right-recursive atom |
| attachment normalization | causal support; beam prohibits illegal actions after softmax | v1=stack-legal conditioning; v2=sentence-local causal support |
| Candidate aggregation | No aggregation in gold mode; beam may apply `logsumexp` over retained beams | `logsumexp` over the model joint probabilities of unique external top-K candidates |
| Candidate weights | Beam scores come from the model's own joint probability | Parser proposal scores are used only for selection and ranking, not multiplied into model probabilities |
| Cross-sentence prefix | None | All candidates share history; each sentence's candidate 0 updates the subsequent prefix |
| Long context | The original repository's sentence-level path requires no document truncation | When too long, truncate from the left by complete history sentences, maximum length 2048 |
| Acceleration | Original repository's incremental beam state | candidate 0 K/V cache and final-hidden cache |
| Denominator | Defined by the original repository according to its sentence tokens / attachment steps | Number of terminal tokens; document BOS not counted, ordinary whitespace/newline tokens counted |
| Output meaning | gold-tree joint PPL or model beam approximation | Truncated structurally marginalized document PPL given external top-K support |

The original repository's entry point and public evaluation description are in
[`eval_utils/eval_pushdown_model.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/eval_utils/eval_pushdown_model.py)
and the original repository's
[`README.md`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/README.md).

## 5. Key Comparability Limitations That Still Exist

### 5.1 The local model is not a direct port of the original authors' checkpoint

This repository uses the OLMo backbone and a local attachment head; the network structure, training implementation, and checkpoint all differ from the original repository. Therefore, protocol v2 only shows that the attachment's probability normalization semantics are consistent with the training target / the original authors' likelihood path; it does not show that the entire model implementation reproduces the original repository.

### 5.2 Closed: candidate structure inconsistent with the checkpoint training representation

The old native n-ary path did not restore the artificial binary nodes and multi-BPE word atoms used during checkpoint training, so it is retained only as a historical/ablation result under a different structure protocol. The current production protocol maps direct strict-binary proposals, or word-level right-CNF conversion of native n-ary proposals, uniformly to the fixed-word-atom representation used in training. See the current protocol document for conversion and support-set audits.

### 5.3 top-K `logsumexp` is a truncated sum, not a complete sentence probability

The current main value is:

```text
log p_topK(x) = logsumexp_y_in_external_topK log p_model(x, y)
```

It covers only the valid unique candidates given by the external proposal and does not guarantee exhaustion of the model's structural space. The valid `K` for different sentences can be less than 300. The current results also report the uniform-mixture result after subtracting `log(K)`, but that value is only a diagnostic convention, not the original repository's definition.

### 5.4 The candidate 0 document prefix is a local extension

Currently all candidates for each sentence share the same history, and the next sentence's context is always updated using candidate 0's structure. This is a deterministic definition made to align with this repository's other document-PPL evaluators, and does not come from the original repository's sentence-level test standard.

## 6. Conclusions That Can and Cannot Be Claimed

Can be claimed:

- protocol v2's attachment CE no longer performs stack-legal conditioning;
- this normalization is semantically consistent with the local training objective and the original repository's given-tree likelihood;
- the current results are local document-level truncated marginal metrics obtained on external fixed-word-atom strict-binary top-K support.

Cannot be claimed:

- the current native document PPL is an item-by-item reproduction of the original repository's evaluation;
- external top-K candidates are equivalent to the original repository's model beam;
- external top-K support exhausts all legal structures of the model;
- protocol v1 historical results can be directly compared with or merged with protocol v2.

If the goal is to reproduce the original repository, use the original repository's code, checkpoint, tokenizer / binary-tree preprocessing, and its `gold` or `beam` entry points. If the goal is this repository's training-representation-aligned top-K metric, use
[`pushdown_word_atom_strict_binary_document_ppl_protocol.md`](../protocols/pushdown_ppl.md);
it is still a local OLMo Pushdown metric, not a reproduction of the original authors' checkpoint.

## 7. This Repository's Implementation Entry Points

- evaluator: [`olmo/eval/pushdown_document_ppl.py`](../../olmo/eval/pushdown_document_ppl.py)
- attachment head: [`olmo/attachment.py`](../../olmo/attachment.py)
- CLI: [`scripts/evaluate_gpst_binary_pushdown_document_ppl.py`](../../scripts/evaluate_gpst_binary_pushdown_document_ppl.py)
- merge/protocol check: [`scripts/merge_gpst_binary_pushdown_document_ppl.py`](../../scripts/merge_gpst_binary_pushdown_document_ppl.py)
