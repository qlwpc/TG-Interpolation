# Parse-data production tools

This directory is the canonical home of the BBC document-PPL parse-data
pipeline. The historical paths under `scripts/` and `datatools/` remain thin
compatibility entry points.

## Model-native top-300

`generate_native_topk.py` uses one label-marginalized Benepar score chart but
decodes two independent candidate spaces:

- GPST: strict-binary CKY top-K. Every parser word receives one fixed, unscored
  BPE subtree before the CKY word tree is materialized, so every CKY split stays
  on its exact word boundary. There is no n-ary projection or collision
  aggregation.
- Pushdown: unary-free unlabeled native n-ary top-K. Only real constituent spans
  are stored; artificial binarization nodes are never introduced.

Both use 300 physical rows per sentence and a model-specific `valid_count`.
Their candidate IDs must not be joined. The mmap loader is
`olmo.eval.native_model_topk_corpus`.

```bash
python datatools/parse_test_docppl_data/generate_native_topk.py audit
python datatools/parse_test_docppl_data/generate_native_topk.py reuse-pushdown \
  --reuse-pushdown-from dataset/bbc-news/native_nary_300 \
  --shard-id 0 --num-shards 28
python datatools/parse_test_docppl_data/generate_native_topk.py generate \
  --component gpst --shard-id 0 --num-shards 28
python datatools/parse_test_docppl_data/generate_native_topk.py finalize \
  --num-shards 28
```

`native_topk.py` contains the exact binary/n-ary K-best algorithms and adapters;
`validate_native_topk.py` runs a real Benepar chart smoke test.

## Tree/TG document-PPL data

`generate_labeled_topk.py` generates exact labeled top-300 candidates directly
from frozen canonical sentence/word/BPE boundaries. `labeled_topk.py` replaces
the historical autograd KMax decoder with lazy canonical A/B CKY enumeration;
`audit_labeled_topk.py` independently checks every candidate. The complete
reserved-clean matching rules and build stages are documented in the
[clean corpus builder](../reserved_clean/README.md). Use
`python datatools/parse_test_docppl_data/generate_labeled_topk.py --help`
for candidate-generation options.

`tree_to_tg.py` converts `testppl_tree` by duplicating every closing
non-terminal token in place. It streams multi-GB arrays through mmap, updates
every candidate-record length, preserves the document index, publishes files
atomically, and writes `manifest.json`.

```bash
python datatools/parse_test_docppl_data/tree_to_tg.py
python datatools/parse_test_docppl_data/tree_to_tg.py --validate-only
```

`normalize_document_boundaries.py` is the one-time BOS/EOS normalization tool.
`build_aligned_test.py` derives aligned terminal, Tree, and TG `test.npy`
streams from candidate zero. Both retain their old `scripts/` entry points.

`reproduce_bbc_test.py` extracts indexed archived parsed rows (`--split dev`
or `--split test`), compares the
current pretraining conversion with the historical GPT-2 encoding, audits
terminal differences, and rebuilds the current test streams. It also supports
a frozen delta recipe so raw-derived reconstruction does not require tree300
after the recipe has been saved. Run
`python datatools/parse_test_docppl_data/reproduce_bbc_test.py --help`
for reconstruction commands; each run records its output hashes in its manifest.
`build-raw --encoding legacy` reconstructs the selected dev/test split;
`verify-raw` checks all three arrays against archived reference files.
Add `--normalize-adj` to `build-raw` to map ADJ constituents to real ADJP
nonterminals before encoding, preserving the text leaves. Omit it when
reproducing the unchanged historical dev/test files.

## Production safety

- Run `audit` before parsing and `finalize` only after every component mask is
  complete.
- Never overwrite the legacy `native_nary_300` directory; it is the provenance
  source for verified Pushdown reuse.
- Keep `valid_count` when batching. Physical padding rows repeat candidate zero
  for shape safety and carry no probability mass.
