# 论文结果与补充实验的解释

本文区分冻结稿原值、旧 checkpoint 的 clean 重测、新 dedup 训练三类结果。本文保留原值身份；
后续论文实验数据修订以 [R-08 差异核定](r08_experiment_protocol_reassessment_20261002.md)为依据，
本次未修改论文。训练配置身份见[配置 manifest](../train_configs/paper_pretraining_manifest.json)，
评测定义见[Evaluation](../Evaluation.md)。

## 冻结稿 BBC 主表与 clean 重测

论文列直接转录冻结稿 `tab:exp1-results`，不是旧投稿版本。XSum/BoolQ 为五个微调 seed 的
均值 ± 样本标准差，其余为单次评测。正文稿与本地旧重跑登记的这一版数值相同；旧投稿列不同。
机器可读[论文表](../reproducibility/evidence/paper_bbc_table.json)保留稿件 hash；
[clean 结果](../reproducibility/evidence/old_checkpoint_clean.json)保留精确 NLL、覆盖与权重身份。

| 模型 | 论文 XSum | 论文 BoolQ | 论文 SG | 论文 BLiMP | 论文旧 DocPPL | 旧模型 clean DocPPL |
|---|---:|---:|---:|---:|---:|---:|
| Terminal-100M | 22.02 ± 0.06 | 66.70 ± 0.36 | 69.80 | 70.77 | 9.89 | 15.369225 |
| Tree-100M | 22.84 ± 0.03 | 68.07 ± 0.35 | 81.40 | 80.95 | 10.01 | 14.736097 |
| TGTree-100M | 23.03 ± 0.02 | 68.12 ± 0.62 | 79.47 | 81.46 | 10.21 | 14.709097 |
| TG-100M | 20.15 ± 0.04 | 64.50 ± 0.40 | 79.19 | 83.28 | 11.02 | 16.405236 |
| TGNomask-100M | 21.91 ± 0.04 | 66.66 ± 0.45 | 78.62 | 81.78 | 9.97 | 14.658209 |
| TGNomask-Aug-100M | 22.12 ± 0.02 | 67.01 ± 0.28 | 78.68 | 83.63 | 10.08 | 14.700370 |
| Tree-NoONT-100M | 22.41 ± 0.04 | 68.67 ± 0.32 | 85.38 | 81.00 | 9.95 | 15.061134 |
| Tree-Compress-100M | 22.70 ± 0.05 | 68.48 ± 0.49 | 78.27 | 81.40 | 9.91 | 14.730862 |
| Tree-TripleCNT-100M | 22.43 ± 0.04 | 68.09 ± 0.12 | 81.57 | 83.42 | 10.26 | 14.801293 |
| Tree-Shuffle-100M | 21.56 ± 0.09 | 68.08 ± 0.33 | 76.56 | 67.77 | 13.41 | 17.852117 |
| TGNomask-Mix-TG-100M | 21.87 ± 0.04 | 66.13 ± 0.54 | 75.77 | 83.14 | 10.01 | 14.572762 |
| TGTree-Mix-TG-100M | 22.16 ± 0.07 | 67.91 ± 0.36 | 78.84 | 82.75 | 10.00 | 14.481718 |
| Pause-1-100M | 22.38 ± 0.05 | 62.04 ± 0.19 | 75.11 | 70.85 | 9.77 | 14.916121 |
| Pause-2-100M | 22.25 ± 0.06 | 65.70 ± 3.45 | 76.97 | 72.59 | 10.32 | 15.116468 |
| Pushdown-100M | 21.06 ± 0.05 | 65.54 ± 0.29 | 74.86 | 74.97 | 13.29 | 19.316305 |
| Terminal-500M | 20.75 ± 0.06 | 69.69 ± 0.60 | 71.11 | 64.74 | 2.83 | 17.302676 |
| Tree-500M | 22.25 ± 0.05 | 70.72 ± 0.35 | 63.13 | 76.09 | 3.18 | 16.001136 |
| TGTree-500M | 22.01 ± 0.08 | 70.63 ± 0.32 | 65.95 | 77.65 | 3.28 | 15.894648 |
| TGNomask-Aug-500M | 21.77 ± 0.06 | 70.78 ± 0.74 | 57.58 | 75.97 | 3.19 | 15.483865 |

原始汇总中 NoONT/Compress/TripleCNT 的错误 grammar 结果已排除，上表使用三模型完成
全量终验的修正值。Tree-Shuffle 使用 masked checkpoint；Pause 使用 dedicated SEP，
不能换成 repeat-token 对照。TreeReg 及 BBC 1B 附加模型保存在 clean JSON 中，未混入论文主表。
原 22 模型总终验回执仍为 `failed`，后续成功总回执的范围是三个 grammar 修正模型；
有效逐行汇总与验收范围的区别见 [R-08 §7](r08_experiment_protocol_reassessment_20261002.md#7-完成状态证据可信度与未解决项)。

## 如何解释差异

历史 BBC DocPPL test 的 4,966 篇中，有 4,752 篇在实际旧 train 中有完整 token 副本
（95.6907%）。clean test 是另一套 5,000 篇数据，结构评分还改用有效候选和 model-best
历史。因此上表是数据与评分协议均有变化的描述性比较，不能把全部差值归因于去污染。
完整边界及可核对数量见[数据说明](bbc_data_provenance.md)。

在这些旧 checkpoint 的 clean 测量中，Tree-100M/500M 的 DocPPL 均低于 Terminal；
100M 两个混合注意力模型最低，500M 的 TGNomask-Aug 最低。原稿“少量 LM cost”、
混合注意力“没有收益”的表述不能推广到此协议。接近数值的排名未经显著性检验。
旧 500M 的 clean PPL 高于对应旧 100M，也不能推出参数规模普遍损害泛化，或代替下游
任务的 scaling 证据。Pushdown/Tree-Shuffle 使用各自协议，不纳入同一受控家族排名。

## FineWeb-Edu 1B

当前稿件采用 Qwen3 tokenizer、OLMES completion/cloze 的 11-task 平均，Tree/TGTree
主值为 `tree_eval_type=terminal`。以下为已登记并与冻结稿核对的汇总；完整逐样本日志
仍是待发布外部资产，不将本次文档整理写成新的评测验收。

| 模型 | BoolQ (%) | 11-task AVG (%) |
|---|---:|---:|
| Terminal | 58.65 | 53.20 |
| Tree | 63.85 | 53.56 |
| TGTree | 67.55 | 55.88 |
| Pause-1 | 65.41 | 55.17 |
| Pause-2 | 63.39 | 55.80 |

SG/BLiMP 是独立任务，不属于 11-task 平均。BBC 1B 的同名模型不能代替 FineWeb-Edu 权重。

## 新增实验与复现范围

[dedup 500M 结果页](bbc_dedup_500m_evaluation_results_20260925.md)是使用新训练集得到的
十个模型，训练 token 数、LR、global batch 和 checkpoint 均不同于上表。五 seed 是微调
seed；单个预训练 seed 不能证明预训练稳定性。十模型评测任务与 DocPPL 全量覆盖
均已完成；评测后的本地输入/权重身份终验通过，范围见
[终验摘要](../reproducibility/results/dedup500m/final_status.json)。

Pushdown 的 XSum reduce 上限、BoolQ supplied-span 评分、BLiMP token-only 候选得分与
论文描述的区别保留在 [Evaluation §8](../Evaluation.md#pushdown-paper-discrepancies)。
旧 91 篇结构差异的解析原因仍未知，不能归因于统一的 top-1/top-2 交换；
2026-10-02 复核后按用户决定关闭 R-03，保留[历史重建限制](bbc_data_provenance.md#provenance)。
这些影响解释的限制公开保留，逐次排错经过留在本地。

数据/权重的下载入口尚未补齐，也未从外部干净环境完成完整重训；见[资产状态](../reproducibility/README.md)。
