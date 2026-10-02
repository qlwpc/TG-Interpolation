# TG-Interpolation

Research code for *A Scaled-Up Empirical Study of Syntactic Language Models*.
Includes BBC / FineWeb-Edu data preparation, Terminal / Pause / Tree / TG training,
and syntactic and downstream evaluation.

| Start here | Contents |
|---|---|
| [Reproduce an experiment](docs/pretraining_workflow.md) | Environment, data, configuration, training and validation |
| [Understand the paper results](docs/paper_results.md) | Frozen paper values, clean-set reevaluation and limits on interpretation |
| [Additional BBC dedup 500M experiments](docs/bbc_dedup_500m_evaluation_results_20260925.md) | Ten trained models, completed evaluations and per-seed results |
| [Run evaluation](Evaluation.md) | Task entry points, model identity and scoring protocols |
| [Get data and checkpoints](reproducibility/README.md) | Included evidence and external assets still awaiting release |

**BBC data versions matter.** The historical BBC test has substantial overlap with
the historical training set. New experiments use deduplicated training data and
reserved-clean dev/test. Clean reevaluation of old checkpoints and newly trained
models are separate experiments; see the [data statement](docs/bbc_data_provenance.md).

## Environment

```bash
conda env create -f environment.yml
conda activate LLM
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cmake -S olmo/data/tgmasking -B olmo/data/tgmasking/build -G Ninja \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build olmo/data/tgmasking/build --parallel 2
cp olmo/data/tgmasking/build/tg_mask*.so olmo/data/
MAX_JOBS=2 python olmo/gpst/cpp_extension/setup.py build_ext --inplace
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

See the [dedup generator](docs/bbc_dedup_500m_pretraining.md) for supported models.
For historical paper settings, use `python scripts/prepare_paper_pretraining.py --list`
and the [historical reproduction guide](docs/pretraining_reproduction.md).
Both generators write new run directories; they do not submit jobs.

## Repository layout

| Directory | Purpose |
|---|---|
| `olmo/`, `olmo_data/` | Model, trainer, attention kernels and evaluation implementation |
| [scripts/](scripts/README.md), [datatools/](datatools/README.md) | Training/evaluation entry points and data preparation |
| [train_configs/](train_configs/README.md) | Generator templates and frozen paper configuration inputs |
| [docs/](docs/README.md) | Reproduction and scientific protocol reference |
| [reproducibility/](reproducibility/README.md) | Compact result evidence, identities and asset availability |
| `tests/`, [diagnostics/](diagnostics/README.md) | Regression checks and reproducible data audits |

Local job histories, failed attempts, old drafts, data, weights and raw logs are
excluded from the public source tree. Necessary scientific limitations are
retained in the result and protocol documentation. No external clean-environment
full retraining is claimed; asset download locations remain to be supplied.
