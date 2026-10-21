# Script Entry Points

Choose the task and data protocol first, then generate an independent output directory. Machine parameters and old campaign defaults must not be inherited directly.

| Work | Current entry point |
|---|---|
| New BBC dedup pretraining configuration | `prepare_bbc_dedup_500m.py`; already covers typed TG / TGNomask / 8+8 Mix / NoONT; see [generator documentation](../docs/guides/bbc_dedup_500m.md) for parameters and validation boundaries |
| Pre-training multi-process IPC check | `pretraining/check_ipc.py --output <new-receipt.json>`; run under the target Python/TMPDIR, cannot replace GPU/DDP/model smoke |
| Reproduction of explicitly selected historical paper configurations | `prepare_paper_pretraining.py` + `train_configs/paper_pretraining_manifest.json` |
| Dedicated SEP Pause pretraining configuration | `submit_pause_sep_pretrain.py`; generates only by default, see [Pause](../docs/models/pause.md) for protocol |
| Clean document PPL | `evaluate_reserved_document_ppl.py`, `merge_reserved_docppl.py`, `finalize_reserved_docppl.py`; [run contract](../docs/protocols/clean_document_ppl.md) |
| Native document PPL and resume | `evaluate_pushdown_document_ppl.py`, `run_native_document_ppl_shards.sh`; [recovery contract](../docs/guides/document_ppl_recovery.md) |
| Binary support probability comparison | `evaluate_gpst_binary_pushdown_document_ppl.py`; [probability protocol](../docs/protocols/gpst_binary_ppl.md) |
| Pause downstream and five-seed campaign | `pause_eval_campaign.py`, [pause_eval/](pause_eval/README.md); v2 contract and explicit model identity still in use |
| DocPPL candidate generation | [datatools/parse_test_docppl_data](../datatools/parse_test_docppl_data/README.md); thin wrapper under scripts still remains as compatibility entry point |

`init_cfg_and_sbatch.py` is still used by BBC finetune/eval tests and is retained for now; its FineWeb routing restrictions are documented in
[Evaluation](../docs/guides/evaluation.md). Its `datatools/rsync.sh` dependency is also retained; archived old configurations have not been reconnected to this entry point.

Supplementary result rendering: `python scripts/render_dedup_results.py`, rebuilds the existing results page from public JSON.
Public release file check: `python scripts/check_public_release.py`.

## Organized by purpose

| Location | Purpose |
|---|---|
| Top level of this directory | General training, configuration generation, evaluation, merging, data compatibility entry points, and public release checks |
| [campaigns/](campaigns/README.md) | Scale-organized SG configurations, terminal-format topic validation, grammar re-testing, and result collection |
| `pretraining/` | Pre-submission runtime environment checks |
| [pause_eval/](pause_eval/README.md) | Model-specific workflows |
| `slurm/` | Launch examples for specific clusters; adjust according to actual resources and paths |
| [diagnostics/](../diagnostics/README.md) | Data auditing, model diagnostics, training data checks, and performance analysis |

Diagnostic scripts have been moved from this directory to `diagnostics/{models,candidates,training,performance}/`,
with filenames unchanged; corresponding launchers use the new paths. Topic-specific scripts have been moved to `campaigns/`.
Commonly used training, pretraining generator, and formal evaluation entry points remain at their original paths.
New scripts should be placed in existing directories according to purpose; run [source checks](../CONTRIBUTING.md#fast-source-checks) before release.
