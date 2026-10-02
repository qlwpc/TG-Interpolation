# BBC dedup 500M 补充实验

这些模型使用新 dedup train 重新训练；不替换[冻结论文表](paper_results.md)。
SG/BLiMP/BoolQ 为百分数，XSum 为 R-AVG 百分数；± 为五个微调 seed 的样本标准差。
微调 seed：42、2026、6198、13171、31723；并非五个预训练 seed。
DocPPL 使用 reserved-clean v1 完整 test：5,000 篇、129,085 句、3,068,713 个计分 token。
Terminal/Pause 是单路径；结构模型是 valid top-300 joint sum、model-best 历史。
截至 2026-09-29，十模型各 22/22 项测试完成；五 seed 指微调重复，并非预训练重复。

## 结果

| 执行地 | 模型 | 完成任务 | SG ↑ | BLiMP ↑ | clean DocPPL ↓ | XSum R-AVG ↑ | BoolQ ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX3090B | Terminal | 22/22 | 80.8848 | 72.3149 | 14.028763 | 23.1149 ± 0.0735 | 70.5382 ± 0.5065 |
| RTX3090B | Tree | 22/22 | 83.4316 | 82.7552 | 13.863770 | 23.5731 ± 0.0649 | 70.0489 ± 0.5797 |
| RTX3090B | TGTree | 22/22 | 80.6760 | 83.7209 | 13.946850 | 23.5886 ± 0.0503 | 68.4281 ± 0.2609 |
| RTX3090B | Pause-1 | 22/22 | 79.0012 | 70.8478 | 14.019490 | 23.4231 ± 0.0560 | 70.2630 ± 0.4060 |
| RTX3090B | Pause-2 | 22/22 | 78.1798 | 72.6851 | 14.030113 | 23.3832 ± 0.0214 | 68.8624 ± 0.3232 |
| RTX3090B | Tree-NoONT | 22/22 | 83.6717 | 81.1030 | 13.938351 | 23.3025 ± 0.0312 | 70.2385 ± 0.2643 |
| SIST | TG | 22/22 | 79.5455 | 83.3164 | 15.465373 | 20.6733 ± 0.0719 | 65.5474 ± 0.3047 |
| SIST | TGNomask | 22/22 | 81.5836 | 83.2090 | 13.995203 | 22.1355 ± 0.0713 | 69.5352 ± 0.6079 |
| SIST | TGTree-Mix-TG | 22/22 | 79.7618 | 83.3672 | 13.784714 | 23.3052 ± 0.0419 | 69.4067 ± 0.3255 |
| SIST | TGNomask-Mix-TG | 22/22 | 80.1115 | 83.2224 | 13.972621 | 22.3746 ± 0.0556 | 69.0581 ± 0.2896 |

## 可核对证据与范围

以下 JSON 包含逐 seed、SG/BLiMP 子项、文档覆盖和原收集时间。它们是原实验汇总的
公开副本；严格汇总已核对任务身份、成功退出、完整评测和 DocPPL 文档/句/token 覆盖。
评测后的输入和 checkpoint 文件 SHA-256 已按原 manifest 再次核对；
SIST 使用本地同步镜像，本次未重新扫描远端存储或重新评测。

- [bbc_dedup_500m_local_20260919](../reproducibility/results/dedup500m/bbc_dedup_500m_local_20260919/results.json)：`complete`，收集于 `2026-09-29T05:45:17.155150+00:00`。
- [bbc_dedup_500m_pause_noont_20260922](../reproducibility/results/dedup500m/bbc_dedup_500m_pause_noont_20260922/results.json)：`complete`，收集于 `2026-09-29T05:45:18.561605+00:00`。
- [bbc_dedup_500m_sist_20260922](../reproducibility/results/dedup500m/bbc_dedup_500m_sist_20260922/results.json)：`complete`，收集于 `2026-09-29T05:45:20.429761+00:00`。

完整训练配置及身份见[补充实验配置](../reproducibility/README.md#dedup-configs)。
公共 dedup CLI 覆盖八种配方；本批 Pause-1/2 与 TGTree-Mix-TG 的冻结配置另行保存，
不能用 TGNomask-Aug 或其他近名配方代替。数据、权重及完整日志的获取状态见
[公开资产清单](../reproducibility/README.md#assets)。

本页由 `scripts/render_dedup_results.py` 从上述 JSON 生成。三批任务、汇总与本地终验均完成；
核对数量、结果哈希和范围见[终验摘要](../reproducibility/results/dedup500m/final_status.json)。
