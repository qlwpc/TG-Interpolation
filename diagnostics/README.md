# Data Diagnostics Entry

The current BBC traceability and contamination audits use `audit_bbc_*`, `diagnose_bbc_split_overlap.py`,
`find_bbc_terminal_sources.py`, `read_bbc_structure_sources_remote.py`, `verify_bbc_*`,
`audit_reserved_fuzzy.*`, and the corresponding `summarize_*`. Fixed inputs, definitions, and evidence are given in
[BBC Data Traceability](../docs/protocols/bbc_data.md). These are audit procedures, not new data production entry points.

[data/bbc_historical_shard_counts.json](data/bbc_historical_shard_counts.json) retains the old parsed 94-shard counts and source hashes.
The raw census and split-cause audit read this JSON directly; they do not depend on executing or parsing the source of historical sampling scripts.

For new candidate generation/independent checks, see [DocPPL Data Entry](../datatools/parse_test_docppl_data/README.md).
The old patching scripts and per-run diagnostics are kept only locally. For public count evidence, see [Data Version Notes](../docs/protocols/bbc_data.md).

`audit_dataset_versions.py` performs a full structural check on the small BBC dev/test streams, sums the top-300 indices,
and performs deterministic sampling on candidate text and precomputed caches; it reads data only and writes JSON to stdout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python diagnostics/audit_dataset_versions.py \
  --root dataset --host 3090B > /tmp/dataset-legality.json
```

Requires NumPy; it does not invoke the parser/GPU and does not scan the entire training set. Passing the sampling does not prove that the entire large file is legal,
nor does it replace a full SHA or contamination audit. Specific version determinations are recorded locally in
`reports/dataset_migration_20261002.md` (not released with the public source code).

## Model and Performance Tools

Apart from the top-level BBC data audit, tools are organized by inspection target. Commands are run from the repository root,
using the runtime environment specified in the README and `PYTHONPATH=.`.

| Directory | Purpose |
|---|---|
| `models/` | checkpoint integrity, XSum/Pause generation, Tree-Shuffle terminality, Pushdown/GPST scorer, and protocol equivalence |
| `candidates/` | GPST BPE, n-ary, and word-atom binary candidate audits |
| `training/` | training data inspection and dataloader checks |
| `performance/` | Pushdown/TreeReg profile analysis, depth microbenchmarks, DocPPL profiler, and stress runner |
| `data/` | fixed historical shard counts; contains audit inputs only |

For example:

```bash
PYTHONPATH=. python diagnostics/models/check_checkpoint_health.py --help
PYTHONPATH=. python diagnostics/candidates/audit_gpst_training_bpe_candidates.py --help
```

Performance and stress tools may require a GPU; some old tools contain specialized inputs or default paths; read the arguments and source first.
GPU benchmarks are not included in the lightweight source check. The clean input audit before formal runs
is still provided by `scripts/audit_reserved_docppl_inputs.py`.
