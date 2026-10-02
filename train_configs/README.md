# 配置入口

新运行从生成器开始，参数与验证流程见[预训练工作流](../docs/pretraining_workflow.md)。

| 实验 | 入口 |
|---|---|
| BBC dedup + clean 500M | `scripts/prepare_bbc_dedup_500m.py`；[八种配方及参数](../docs/bbc_dedup_500m_pretraining.md) |
| 历史论文设置 | `scripts/prepare_paper_pretraining.py`；默认 27 个 run，另有 2 个显式 repeat-token 对照 |
| dedicated-SEP Pause | `scripts/submit_pause_sep_pretrain.py`；[模型身份和评测](../docs/pause_protocol.md) |

- `templates/`：生成器输入。模板中的路径、学习率、batch 和停止条件由生成器覆盖。
- `paper_pretraining_manifest.json`：模型、语料、配置来源、checkpoint 身份和实测超参数。
- `paper_sources/`：29 份原字节配置，每份均由 manifest 的 SHA-256 固定；生成历史配置不需要权重或本地 `saved_models/`。
- `eval_per_metric/`、`*_eval.yaml`：评测配置；按 checkpoint、tokenizer 和协议选择。
- `diagnostics/`：测试、smoke 和 profile 输入，不能直接当作论文训练入口。

历史生成器设置 `max_duration=1ep`、`stop_at=2000000000`、`stop_after=null`，
避免旧停止步数截断从零训练。输出的 config、launcher 和 protocol 写入新目录；
生成成功不代表输入齐全、GPU 验证通过或训练完成。
