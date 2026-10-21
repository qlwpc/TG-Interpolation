# Campaign Experiment Tools

These tools serve specific experiments and result collection, and some contain historical campaign paths or model lists.
Start new training and general evaluation from the [parent entry](../README.md); before using the tools here, verify the input configuration,
checkpoint, data protocol, and output location.

| Tool | Scope |
|---|---|
| `make_sg_*_eval_configs.py` | Generate SG configs by 100M / 500M / 1B model set |
| `make_terminal_format_validation_configs.py`, `audit_terminal_format_validation.py` | FineWeb terminal/full focused validation and input audit |
| `summarize_terminal_format_validation*.py` | Summarize the results of the above focused runs |
| `prepare_variant_grammar_retest.py`, `validate_variant_grammar_retest.py`, `finalize_variant_grammar_retest.py` | Retest workflow after variant grammar corrections |
| `collect_finetune_campaign.py`, `collect_syntax_eval_results.py` | Result collection for existing fine-tuning and syntax experiments |
| `fineweb_edu_topic_locator.py`, `fineweb_slice_quality_audit.py` | Source-bound samples and read-only quality annotations; [data identity and known issues](../../docs/protocols/fineweb_edu_data.md) |
| `fineweb_formula_filter.py`, `fineweb_datatrove_check.py`, `fineweb_paired_backup.py` | Formula screening, sampled cleaning diagnostics and aligned backups; [scope and limitations](../../docs/protocols/fineweb_edu_cleaning.md) |
| `fineweb_bracket_repair.py`, `fineweb_parquet_text_search.py` | Journaled paired repair and provenance lookup; [repair and recovery protocol](../../docs/protocols/fineweb_bracket_repair.md) |
| `fineweb_collection_filter.py`, `fineweb_paired_collection.py`, `fineweb_collection_unpack.py` | Campaign-specific paired 20B collection and input conversion; [storage and split protocol](../../docs/protocols/fineweb_clean_20b.md) |

Run commands from the repository root, for example:

```bash
PYTHONPATH=. python scripts/campaigns/collect_finetune_campaign.py --help
```

Some historical generators have no argument parsing; running them directly writes to the campaign specified in the source; `--help`
cannot be regarded as a preview method supported by all scripts. Focused diagnostics cannot replace the paper's main protocol.
For evaluation identity and supported scope, see the [evaluation guide](../../docs/guides/evaluation.md).
