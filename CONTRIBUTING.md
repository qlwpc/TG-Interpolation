# Working on this repository

Run commands from the repository root. The public tree contains source code,
reproduction recipes and selected evidence. Data and checkpoints are listed in
[asset availability](reproducibility/README.md#assets).

## Where changes belong

| Change | Location |
|---|---|
| Model, training or scoring implementation | `olmo/` |
| OLMES task loading and bundled task assets | `olmo_data/` |
| Supported training/evaluation CLI | `scripts/`; model-specific drivers in `scripts/pause_eval/` |
| Experiment-specific collection or configuration | `scripts/campaigns/`; machine-specific launch examples in `scripts/slurm/` |
| Data construction or representation conversion | `datatools/` |
| Data audits, model diagnostics or performance investigations | `diagnostics/` |
| Configuration templates and frozen paper inputs | `train_configs/`; task-specific evaluation examples also live in `evaluation/` |
| Regression tests | `tests/` |
| User procedures, model notes, protocols or result interpretation | `docs/guides/`, `docs/models/`, `docs/protocols/`, `docs/results/` |
| Public result evidence and frozen identities | `reproducibility/` |

Keep run-specific outputs under ignored `artifacts/` and local work records under
`.local/`. Existing ignored histories remain local. Do not add a new top-level
folder or a dated script when an existing category serves the same purpose.

When moving an entry point, update its callers, repository-root discovery,
documentation and generated output. Frozen configuration bytes and identity hashes
are evidence: retain their historical paths inside the files. A new run uses a new
configuration and output directory.

## Fast source checks

Python 3.10 or newer and Bash are sufficient; these checks need no GPU, model,
private dataset, third-party Python package or network access:

```bash
python scripts/check_public_release.py --syntax
python scripts/render_dedup_results.py --check
```

The first command checks the public file boundary, local Markdown links and
anchors, frozen-input hashes (including the historical BBC split indices), Python syntax and Bash syntax. The second checks
that the dedup result page and JSON formatting match their generator. The
[source-check workflow](.github/workflows/source-checks.yml) runs these checks
on pushes and pull requests and repeats them in an exported public copy.
It uses the official [checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python) actions.

To inspect the public copy locally, choose a destination that does not yet exist:

```bash
python scripts/check_public_release.py --syntax --export /tmp/tg-public-review
cd /tmp/tg-public-review
python scripts/check_public_release.py --syntax
python scripts/render_dedup_results.py --check
```

Exports reflect the working tree, including uncommitted changes. They are not a
release tag or a clean installation of the training environment.

## Runtime checks

Use [environment.yml](environment.yml) and the extension build instructions in
[README](README.md#environment). Run the existing tests relevant to the changed
behavior; for data preparation and reproduction configuration, for example:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_prepare_bbc_dedup_500m.py tests/test_paper_pause_protocol.py \
  tests/test_parse_pipeline_smoke.py tests/test_pretraining_data_integrity.py
```

Source checks do not establish runtime, GPU or full-retraining correctness.
Training changes require the corresponding stages of the
[pretraining workflow](docs/guides/pretraining.md). Report the environment,
checks actually run, and any untested hardware or unavailable assets.
