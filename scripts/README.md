# 脚本入口

先选任务和数据协议，再生成独立输出目录。机器参数和旧 campaign 默认值不能直接继承。

| 工作 | 现行入口 |
|---|---|
| 新 BBC dedup 预训练配置 | `prepare_bbc_dedup_500m.py`；已覆盖 typed TG / TGNomask / 8+8 Mix / NoONT；参数和验证边界见[生成器说明](../docs/guides/bbc_dedup_500m.md) |
| 训练前多进程 IPC 检查 | `pretraining/check_ipc.py --output <new-receipt.json>`；目标Python/TMPDIR中运行，不能替代GPU/DDP/model smoke |
| 明确选择的历史论文配置复现 | `prepare_paper_pretraining.py` + `train_configs/paper_pretraining_manifest.json` |
| 专用 SEP Pause 预训练配置 | `submit_pause_sep_pretrain.py`；默认只生成，协议见 [Pause](../docs/models/pause.md) |
| clean 文档 PPL | `evaluate_reserved_document_ppl.py`、`merge_reserved_docppl.py`、`finalize_reserved_docppl.py`；[运行合同](../docs/protocols/clean_document_ppl.md) |
| native 文档 PPL 及续跑 | `evaluate_pushdown_document_ppl.py`、`run_native_document_ppl_shards.sh`；[恢复合同](../docs/guides/document_ppl_recovery.md) |
| binary support 概率对照 | `evaluate_gpst_binary_pushdown_document_ppl.py`；[概率协议](../docs/protocols/gpst_binary_ppl.md) |
| Pause 下游与五 seed campaign | `pause_eval_campaign.py`、[pause_eval/](pause_eval/README.md)；v2 contract 与显式模型身份仍在使用 |
| DocPPL 候选生成 | [datatools/parse_test_docppl_data](../datatools/parse_test_docppl_data/README.md)；scripts 下薄 wrapper 仍为兼容入口 |

`init_cfg_and_sbatch.py` 仍被 BBC finetune/eval 测试使用，暂保留；其 FineWeb 路由限制见
[Evaluation](../docs/guides/evaluation.md)。它的 `datatools/rsync.sh` 依赖也保留，未将归档的旧配置重新接回该入口。

补充结果渲染：`python scripts/render_dedup_results.py`，从公开 JSON 重建现有结果页。
发布文件检查：`python scripts/check_public_release.py`。

## 按用途存放

| 位置 | 用途 |
|---|---|
| 本目录顶层 | 通用训练、配置生成、评测、合并、数据兼容入口和公开发布检查 |
| [campaigns/](campaigns/README.md) | 按规模组织的 SG 配置、terminal-format 专题验证、grammar 重测和结果收集 |
| `pretraining/` | 提交前运行环境检查 |
| [pause_eval/](pause_eval/README.md) | 模型专用工作流 |
| `slurm/` | 特定集群的启动示例；按实际资源和路径调整 |
| [diagnostics/](../diagnostics/README.md) | 数据审计、模型诊断、训练数据检查与性能分析 |

诊断脚本已从本目录移到 `diagnostics/{models,candidates,training,performance}/`，
文件名保持不变；相应 launcher 使用新路径。专题脚本移到 `campaigns/`。
常用训练、预训练生成器和正式评测入口保持原路径。
新增脚本按用途放入现有目录，发布前执行 [源码检查](../CONTRIBUTING.md#fast-source-checks)。
