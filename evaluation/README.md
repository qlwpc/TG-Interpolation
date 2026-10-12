# Evaluation task assets and examples

The supported model/task combinations and scoring rules are in the
[evaluation guide](../docs/guides/evaluation.md). Scoring code lives in `olmo/eval/`;
training and evaluation drivers live in [scripts/](../scripts/README.md).

| Location | Contents |
|---|---|
| `SG/` | Syntactic generalization task data and tokenizer conversion |
| `eval_configs/`, `xsum_configs/` | Task-specific YAML examples, including historical and smoke settings |
| `eval_scripts/`, `boolq/`, `CB/`, `RTE/` | Shell launch examples with machine/checkpoint assumptions |

Generate a fresh run configuration using the evaluation guide. Inspect any example's
paths, tokenizer, checkpoint and protocol before use. Additional model-specific
evaluation configurations are indexed in [train_configs](../train_configs/README.md).
The asset table in [reproducibility](../reproducibility/README.md#assets) distinguishes
included task files from data still awaiting a public download location.
