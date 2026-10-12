# TG-Interpolation

Research code for *A Scaled-Up Empirical Study of Syntactic Language Models*.
Includes BBC / FineWeb-Edu data preparation, Terminal / Pause / Tree / TG training,
and syntactic and downstream evaluation.

**Release status:** source code, frozen configurations and selected result evidence
are available. Full data and checkpoint downloads are still pending; see
[asset availability](reproducibility/README.md#assets) before attempting reproduction.

| Start here | Contents |
|---|---|
| [Reproduce an experiment](docs/guides/pretraining.md) | Environment, data, configuration, training and validation |
| [Understand the paper results](docs/results/paper.md) | Frozen paper values, clean-set reevaluation and limits on interpretation |
| [Additional BBC dedup 500M experiments](docs/results/bbc_dedup_500m.md) | Ten trained models, completed evaluations and per-seed results |
| [Run evaluation](docs/guides/evaluation.md) | Task entry points, model identity and scoring protocols |
| [Get data and checkpoints](reproducibility/README.md) | Included evidence and external assets still awaiting release |

**BBC data versions matter.** The historical BBC test has substantial overlap with
the historical training set. New experiments use deduplicated training data and
reserved-clean dev/test. Clean reevaluation of old checkpoints and newly trained
models are separate experiments; see the [data statement](docs/protocols/bbc_data.md).

## Check the source release

With Python 3.10+ and Bash, no training environment or data download is needed:

```bash
python scripts/check_public_release.py --syntax
python scripts/render_dedup_results.py --check
```

These commands verify public files, links, frozen inputs, source syntax and generated
result tables. Runtime tests and public-copy validation are described in
[CONTRIBUTING](CONTRIBUTING.md).

## Environment

```bash
conda env create -f environment.yml
conda activate LLM
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cmake -S olmo/data/tgmasking -B olmo/data/tgmasking/build -G Ninja \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build olmo/data/tgmasking/build --parallel 2
cp olmo/data/tgmasking/build/tg_mask*.so olmo/data/
```

The public environment is [environment.yml](environment.yml). GPU support and
extension compatibility must be checked on the target machine; configuration
checks alone do not establish a successful training run.

## Generate a configuration

The frozen BBC tokenizer is included. Place it at the canonical data path:

```bash
mkdir -p dataset/bbc-news
cp reproducibility/tokenizer-reference.json dataset/bbc-news/TG_GPT2_tokenizer.json
```

After obtaining the required data:

```bash
python scripts/prepare_bbc_dedup_500m.py \
  --models terminal tree tgtree --gpus 2 --max-microbatch 1 \
  --output-dir artifacts/experiment/NEW_CAMPAIGN
```

See the [dedup generator](docs/guides/bbc_dedup_500m.md) for supported models.
For historical paper settings, use `python scripts/prepare_paper_pretraining.py --list`
and the [historical reproduction guide](docs/guides/historical_pretraining.md).
Both generators write new run directories; they do not submit jobs.

GPST training and model code have moved to the independent `~/GPST` workspace.
Shared candidate formats and Pushdown comparison evaluators remain here.

## Repository layout

| Directory | Purpose |
|---|---|
| `olmo/` | Model, trainer, attention kernels and scoring implementation |
| `olmo_data/`, [evaluation/](evaluation/README.md) | OLMES loaders, bundled task assets and evaluation examples |
| [scripts/](scripts/README.md) | Training/evaluation entry points; experiment-specific tools in `campaigns/` |
| [datatools/](datatools/README.md) | Data preparation and representation conversion |
| [train_configs/](train_configs/README.md) | Generator templates and frozen paper configuration inputs |
| [docs/](docs/README.md) | `guides/`, `models/`, `protocols/` and `results/` |
| [reproducibility/](reproducibility/README.md) | Compact result evidence, identities and asset availability |
| `tests/`, [diagnostics/](diagnostics/README.md) | Regression checks, data/model audits and performance tools |

Large training arrays, weights, raw logs and local job histories are excluded from
the public source tree; some processed evaluation task assets are included.
Necessary scientific limitations are
retained in the result and protocol documentation. No external clean-environment
full retraining is claimed; asset download locations remain to be supplied.
