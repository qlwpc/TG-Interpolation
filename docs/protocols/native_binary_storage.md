# Native-binary evaluation storage

This early common-structure storage sketch is superseded by format v2 for
production. See `native_model_topk_300_v2_format.md`: GPST and Pushdown now have
independent valid counts, proposal scores, and structural arrays.

## Contract

Use a fixed physical candidate axis (`K_slots = 300`) inside each sentence,
but retain a separate `valid_count`. A short sentence therefore remains easy to
reshape and batch without pretending that duplicated padding trees are extra
probability mass.

Each sentence with `T_s` terminal tokens occupies contiguous blocks:

```text
tokens:          [T_s]
merge_orders:    [300, T_s - 1]       int16
spans:           [300, T_s - 1, 3]    int16
proposal_scores: [300]                float32
valid_count:     scalar               uint16
```

The first `valid_count` rows are unique candidates sorted by proposal score.
Unused rows contain a structurally valid copy of row zero, while their proposal
score is `-inf`. Evaluators must use `[:valid_count]` or an equivalent mask and
must never recover padding multiplicities.

## Corpus files

Use sentence-ragged, contiguous arrays rather than global maximum-length
padding:

```text
metadata.json
tokens.npy                    # flat uint16/uint32
token_offsets.npy             # uint64, num_sentences + 1
merge_orders.npy              # flat int16
spans.npy                     # (-1, 3) int16
structure_offsets.npy         # uint64, num_sentences + 1; units are structures
proposal_scores.npy           # (num_sentences, 300) float32
valid_counts.npy              # (num_sentences,) uint16
document_counts.npy           # existing BBC document convention
```

For sentence `s`:

```python
t0, t1 = token_offsets[s:s + 2]
z0, z1 = structure_offsets[s:s + 2]
T = t1 - t0

tokens = tokens_flat[t0:t1]
orders = merge_orders_flat[z0:z1].reshape(300, T - 1)
spans = spans_flat[z0:z1].reshape(300, T - 1, 3)
K_valid = int(valid_counts[s])

gpst_batch = orders[:K_valid]
pushdown_batch = spans[:K_valid]
```

This needs one mmap slice per structure array and no bracket parsing, unary
collapse, binarization, tree deduplication, or per-candidate terminal check at
evaluation time. Tokens are stored once per sentence rather than 300 times.

## Semantics

Evaluation computes:

```python
sentence_log_probability = logsumexp(model_log_joint[:K_valid])
```

There is no division by `K_valid`, no use of Benepar proposal scores in the
model probability, and no contribution from physical padding rows. Document
prefix selection must use the model-best valid candidate rather than physical
slot zero.
