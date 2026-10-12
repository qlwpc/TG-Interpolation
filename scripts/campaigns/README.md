# 专题实验工具

这些工具服务于特定实验和结果收集，部分含历史 campaign 路径或模型列表。
新训练和通用评测从[上级入口](../README.md)开始；使用这里的工具前核对输入配置、
checkpoint、数据协议和输出位置。

| 工具 | 范围 |
|---|---|
| `make_sg_*_eval_configs.py` | 按 100M / 500M / 1B 模型集合生成 SG 配置 |
| `make_terminal_format_validation_configs.py`、`audit_terminal_format_validation.py` | FineWeb terminal/full 专题验证与输入审计 |
| `summarize_terminal_format_validation*.py` | 上述专题结果汇总 |
| `prepare_variant_grammar_retest.py`、`validate_variant_grammar_retest.py`、`finalize_variant_grammar_retest.py` | 变体 grammar 修正后的重测流程 |
| `collect_finetune_campaign.py`、`collect_syntax_eval_results.py` | 已有微调、句法实验的结果收集 |

命令从仓库根目录执行，例如：

```bash
PYTHONPATH=. python scripts/campaigns/collect_finetune_campaign.py --help
```

部分历史生成器没有参数解析，直接运行会写入源码指定的 campaign；不能把 `--help`
视为所有脚本都支持的预览方式。专题诊断不能替代论文主协议。
评测身份及支持范围见[评测指南](../../docs/guides/evaluation.md)。
