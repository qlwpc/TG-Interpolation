# 脚本入口

先选任务和数据协议，再生成独立输出目录。机器参数和旧 campaign 默认值不能直接继承。

| 工作 | 现行入口 |
|---|---|
| 新 BBC dedup 预训练配置 | `prepare_bbc_dedup_500m.py`；已覆盖 typed TG / TGNomask / 8+8 Mix / NoONT；参数和验证边界见[生成器说明](../docs/bbc_dedup_500m_pretraining.md) |
| 训练前多进程 IPC 检查 | `pretraining/check_ipc.py --output <new-receipt.json>`；目标Python/TMPDIR中运行，不能替代GPU/DDP/model smoke |
| 明确选择的历史论文配置复现 | `prepare_paper_pretraining.py` + `train_configs/paper_pretraining_manifest.json` |
| 专用 SEP Pause 预训练配置 | `submit_pause_sep_pretrain.py`；默认只生成，协议见 [Pause](../docs/pause_protocol.md) |
| clean 文档 PPL | `evaluate_reserved_document_ppl.py`、`merge_reserved_docppl.py`、`finalize_reserved_docppl.py`；[运行合同](../docs/bbc_reserved_docppl_sist_20260907.md) |
| native 文档 PPL 及续跑 | `evaluate_pushdown_document_ppl.py`、`gpst/evaluate_document_ppl.py`、`run_native_document_ppl_shards.sh`；[恢复合同](../docs/native_document_ppl_recovery.md) |
| binary support 概率对照 | `evaluate_gpst_binary_pushdown_document_ppl.py`；[概率协议](../docs/gpst_binary_pushdown_document_ppl_protocol.md) |
| Pause 下游与五 seed campaign | `pause_eval_campaign.py`、[pause_eval/](pause_eval/README.md)；v2 contract 与显式模型身份仍在使用 |
| DocPPL 候选生成 | [datatools/parse_test_docppl_data](../datatools/parse_test_docppl_data/README.md)；scripts 下薄 wrapper 仍为兼容入口 |

`init_cfg_and_sbatch.py` 仍被 BBC finetune/eval 测试使用，暂保留；其 FineWeb 路由限制见
[Evaluation](../Evaluation.md)。它的 `datatools/rsync.sh` 依赖也保留，未将归档的旧配置重新接回该入口。

补充结果渲染：`python scripts/render_dedup_results.py`，从公开 JSON 重建现有结果页。
发布文件检查：`python scripts/check_public_release.py`。
