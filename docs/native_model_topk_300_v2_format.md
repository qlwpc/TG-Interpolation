# GPST / Pushdown model-native top-300 (format v2)

## Evaluation boundary

The canonical BBC document-PPL corpus contains 4,966 documents and 148,836
sentences. The updated `tree/test.npy` and `testppl_tree` audit has
`document_offset = 0`; all 4,966 documents have exactly equal in-tree terminal
sequences and there are no exceptions.

Input SHA-256 values used for production are:

```text
tree_300.npy          0eac90749f854df5e7fad016ffe4aae761f686392bedfa8a5bd3283df14489b2
tree_sent_index.npy   87589fce96748506d6a3751b6603198125e7c0c7f2dccaf1b0c12f43f6417ae4
tree_doc_index.npy    69e7cbbc51356fe728b154b3c10bce0e786d0d06d89ecb9a6660d4af616d0c05
tree/test.npy         fcf10cd1be5dc11ba4e00afc9477a27e73573792456599562d07cc62e1679339
```

## Candidate semantics

GPST and Pushdown share only terminals, document/sentence boundaries, and the
label-marginalized Benepar span chart. They do not share candidate structures,
candidate counts, rankings, or candidate IDs.

- GPST directly runs strict-binary CKY K-best decoding. Its candidate score is
  the sum of the marginalized real-span scores at every binary internal node.
  Every parser word first receives one fixed, unscored right-recursive BPE
  subtree; the CKY word tree is then materialized over those subtrees. Thus
  every CKY split lands exactly on its parser-word BPE boundary.
  There is no n-ary intermediate and no collision aggregation.
- Pushdown runs the canonical unary-free unlabeled n-ary K-best DP. It stores
  only real n-ary constituents mapped to BPE terminal intervals. It does not
  receive the artificial nodes needed by a binary GPST merge trajectory.

For a sentence of `n` parser words, GPST has up to the Catalan number of valid
trees, while Pushdown has up to the little Schröder number. Thus short-sentence
valid counts differ, for example:

```text
n words          1   2   3   4   5    6    7    8
GPST binary       1   1   2   5  14   42  132  300
Pushdown n-ary    1   1   3  11  45  197  300  300
```

Every sentence has 300 physical rows for fast batching. Rows at or beyond the
model-specific valid count repeat candidate zero as shape-safe padding and have
proposal score `-inf`; evaluators must slice by the corresponding valid count.

## Shard arrays

Shared sentence data:

```text
terminal_tokens.npy / terminal_offsets.npy
content_bounds.npy
word_starts.npy / word_offsets.npy
global_sentence_ids.npy / document_ids.npy
```

Pushdown-specific data:

```text
pushdown_valid_counts.npy
pushdown_proposal_scores.npy
pushdown_span_counts.npy
pushdown_spans.npy / pushdown_offsets.npy
pushdown_completed.npy
```

GPST-specific data:

```text
gpst_valid_counts.npy
gpst_proposal_scores.npy
gpst_merge_orders.npy / gpst_offsets.npy
gpst_completed.npy
```

The manifest records the adapter contract as
`fixed-per-word-right-recursive-v1`; finalization rejects any explicitly
different shard version.

A completed 5,329-sentence shard was additionally reconstructed from stored
merge orders. Across all valid rows there were no duplicate trajectories or
invalid gap permutations. For 14,813 top/middle/last candidates, every
multi-BPE word had its fixed atom subtree and every cross-word node split at a
word-end BPE boundary (zero violations).

`olmo.eval.native_model_topk_corpus.NativeModelTopKCorpus` exposes sentence
views without concatenating shards or parsing bracket strings at evaluation
time.

## Verified Pushdown reuse after BOS/EOS normalization

The updated tree300 differs from the production source only in tree-external
document framing. Across all 148,836 sentences, both the in-tree terminal hash
and parser-word/BPE grouping hash are identical. Therefore Benepar charts,
Pushdown n-ary topologies, proposal ranks, and scores are unchanged.

The old absolute-span coordinates were audited before reuse:

```text
coordinate delta 0: 143,958 sentences
coordinate delta 1:   4,878 sentences
valid spans checked: 791,917,407
invalid before translation: 0
invalid after translation:  0
```

All three Pushdown coordinates `(left, right, right)` are translated together.
Each shard writes `pushdown_reuse.json`, and finalization aggregates this
provenance into the corpus manifest.

## Tree-to-TG conversion

`datatools/parse_data/tree_to_tg.py` duplicates every closing non-terminal in
place, updates every candidate-record length, and preserves the document index.
The current normalized corpus was checked exhaustively:

```text
Tree tokens:                 2,595,842,732
duplicated closing tokens:     804,567,316
TG tokens:                   3,400,410,048
candidate records checked:      44,650,800
TG BLAKE2b: 6d1ea284c0ed9e90095a006928aab75175e76810a16eb4e108f7036ef4efe6c0
```

Every token, sentence-index item, and document-index item matched exactly.

## Final production result

The finalized RTX3090 corpus is
`dataset/bbc-news/native_model_topk_300_v2` (6.6 GB):

```text
documents:                   4,966
sentences:                 148,836
shards:                         28
physical slots/sentence:       300
GPST valid candidates:  37,227,054
Pushdown valid candidates: 38,581,363
```

All 28 GPST runs started after the recorded adapter reset and exited with code
zero. A cross-shard stratified check sampled 896 sentences and reconstructed
2,499 stored candidates: missing fixed word atoms = 0 and cross-word splits off
word boundaries = 0.
