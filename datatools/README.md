# Data Tools Entry Point

| Task | Entry point |
|---|---|
| New BBC dedup + clean pretraining configuration | [Configuration generator](../scripts/prepare_bbc_dedup_500m.py); use dedup train for training data, and clean dev/test for evaluation data |
| BBC / FineWeb-Edu download, parsing, and assembly | `python -m datatools.parse_pretrain_data.build_pretrain_data`; [data build details](../docs/guides/historical_pretraining.md) |
| clean dev/test filtering and freeze rules | [reserved_clean/](reserved_clean/README.md) |
| DocPPL candidate production, boundary alignment, historical reconstruction | [parse_test_docppl_data/](parse_test_docppl_data/README.md) |

The top level retains compatibility wrappers for some historical paths, such as `native_binary.py` and
`gen_tgppl_fromtree.py`; they are still used by existing tests/callers and are not to be treated as dead code based on their filenames.

Current training assembly is performed by `assemble_streams.py`, which rejects the five reserved shards from entering train;
it is scheduled by `build_pretrain_data`'s stage/task-index. Disabled source code is kept locally for archival purposes only and is not a runtime dependency.

The replacement relationships between current entry points and old tools are as follows:

| Old tool | Current entry point |
|---|---|
| `gen_final_train.py`, `sample_testset.py` | `build_pretrain_data`'s split/assemble stages; new clean filtering uses `reserved_clean` |
| `parse_testppl*.py`, `tokenize_testppl.py`, `parse_test.sh` | `parse_test_docppl_data/generate_labeled_topk.py`, `generate_native_topk.py`; historical byte reconstruction uses `reproduce_bbc_test.py` |
| Old parsing queue, manual launch scripts, reset and status | `build_pretrain_data parse --corpus ... --task-index ...` |
| `process_bbc.py` | `parse_pretrain_data/make_tree_variant.py` |

Downstream data adopts the approach of directly publishing already-processed files, loaded according to the directories in [Evaluation](../docs/guides/evaluation.md).
Existing sidecar/legacy-format compatibility tools still have callers; `rsync.sh` is referenced by configuration tools, and `save_datasets.py`
exports MMLU-Redux. These are used separately from the formal pretraining data entry points in the table above.
