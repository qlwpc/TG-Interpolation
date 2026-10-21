# Configuration Entry Points

New runs start from the generator; parameters and the validation procedure are described in [the pretraining workflow](../docs/guides/pretraining.md).

| Experiment | Entry point |
|---|---|
| BBC dedup + clean 500M | `scripts/prepare_bbc_dedup_500m.py`; [eight recipes and parameters](../docs/guides/bbc_dedup_500m.md) |
| Historical paper settings | `scripts/prepare_paper_pretraining.py`; 27 runs by default, plus 2 explicit repeat-token controls |
| dedicated-SEP Pause | `scripts/submit_pause_sep_pretrain.py`; [model identity and evaluation](../docs/models/pause.md) |

- `templates/`: generator inputs. Paths, learning rates, batch sizes and stopping conditions in the templates are overridden by the generator.
- `paper_pretraining_manifest.json`: models, corpora, configuration sources, checkpoint identities and measured hyperparameters.
- `paper_sources/`: 29 original-byte configurations, each pinned by a SHA-256 in the manifest; generating historical configs does not require weights or a local `saved_models/`.
- `eval_per_metric/`, `*_eval.yaml`: evaluation configs; selected by checkpoint, tokenizer and protocol.
- `diagnostics/`: test, smoke and profile inputs; these cannot be used directly as paper training entry points.

The historical generator sets `max_duration=1ep`, `stop_at=2000000000`, `stop_after=null`
to prevent old stopping steps from truncating training from scratch. The output config, launcher and protocol are written to a new directory;
successful generation does not mean the inputs are complete, GPU validation has passed, or training has finished.
