# 公开复现材料

本目录只放复现所需的固定输入、有效结果和身份摘要。日常工作记录、失败尝试、队列状态、
远端同步脚本及完整日志保留在本地，不作为读者的默认入口。

| 内容 | 入口 |
|---|---|
| 历史论文模型与 29 份固定配置 | [manifest](../train_configs/paper_pretraining_manifest.json)、[生成指南](../docs/pretraining_reproduction.md) |
| BBC tokenizer 校验参考 | [reference JSON](tokenizer-reference.json) |
| 冻结稿 19 行 BBC 结果 | [机器可读表](evidence/paper_bbc_table.json)、[解释](../docs/paper_results.md) |
| 旧模型 clean 重测 | [22 模型有效结果](evidence/old_checkpoint_clean.json)，三项 grammar 修正已替换无效值 |
| BBC 重叠审计 | [旧数据 exact](evidence/historical_overlap.json)、[clean fuzzy](evidence/clean_overlap.json) |
| 提取来源及内容校验 | [证据 manifest](evidence/manifest.json)；`local_origin` 仅标识原始来源，不是下载链接 |
| 新增十模型评测 | [结果页](../docs/bbc_dedup_500m_evaluation_results_20260925.md)、[逐 seed/子项 JSON](results/dedup500m/)、[终验摘要](results/dedup500m/final_status.json) |

<a id="dedup-configs"></a>
## 新增实验配置

[十模型配置 manifest](configs/dedup500m/manifest.json)记录 checkpoint step、配置 SHA-256
及原字节 YAML。配置是实际 checkpoint 的身份材料，保留当时的路径；新运行需要用生成器
或适配到新目录，不能直接复用原保存目录。`tg/tgnomask/tgnomask_mix_tg/tree_noont`
等八种配方使用[公共 CLI](../docs/bbc_dedup_500m_pretraining.md)；Pause 使用
[SEP 入口](../docs/pause_protocol.md)。TGTree-Mix-TG 暂无统一 dedup CLI key，冻结配置
提供具体配方，重新运行仍需完成[工作流](../docs/pretraining_workflow.md)的身份和 GPU 验证。

<a id="assets"></a>
## 数据、权重和日志的可获取性

| 资产 | 已包含内容 | 外部复现状态 |
|---|---|---|
| 代码、C++/Triton 源码、配置与协议 | 当前源码树、构建命令、生成器 | 可审阅；需在目标环境验证扩展/GPU |
| BBC 历史 split、来源与 tokenizer 参考 | split JSON、数据工具、配置 manifest | 完整历史训练字节重建尚未证实；不能承诺重新下载/解析得到同一数组 |
| dedup train / clean dev/test / top-300 | 构建/筛选程序、版本和必要校验摘要 | 冻结数组及候选的公开下载位置待补 |
| BBC/FineWeb tokenizer 运行资产 | 构建入口；BBC reference 是可加载的完整冻结 JSON | BBC 文件已包含，可复制到 `dataset/bbc-news/TG_GPT2_tokenizer.json`；Qwen3 冻结文件随数据发布 |
| 下游处理后数据及 parse sidecars | 仓库已有部分 SG、OLMES 数据与任务定义 | 其余按 Evaluation 指定布局直接发布已处理文件；下载入口待补，不要求重新构造 pipeline |
| 历史与新 dedup checkpoint | 固定配置；旧模型 clean 及十个新模型附权重 hash | 完整权重下载位置待补；配置不含权重 |
| 逐文档结果、原始日志、历史运行源码快照 | 已选有效汇总、fingerprint 与来源 hash | 全量外部存档入口待补；当前工作树不自动等同于每次历史运行源码 |
| FineWeb 11-task 逐样本结果 | 论文说明中的汇总 | 完整逐样本证据与 bootstrap 产物待发布 |

hash 是校验身份，不能代替获取地址。上述缺口未关闭前，只声明源代码、配方和选定证据
已整理，不声明外部读者已经可以完整重训或逐 bit 复现。未进行外部干净环境完整重训。

## 检查公开文件副本

```bash
python scripts/check_public_release.py
python scripts/check_public_release.py --export /tmp/tg-public-review
cd /tmp/tg-public-review
python scripts/check_public_release.py
python scripts/prepare_paper_pretraining.py --list
python scripts/render_dedup_results.py
```

export 读取当前工作树中可发布的文件，包含未提交的修改，不等同于已提交 commit 的
`git archive`。检查包括公开文件集合中的 Markdown 链接/锚点、配置与证据 hash；
文件在本地存在但被忽略时不算通过。导出不包含本地工作记录或 `.git`。
发布前还需提交所需文件并核对实际干净检出；本命令不会 commit、push 或上传资产。
