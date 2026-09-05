# 仓库整理记忆：实验与测试纠错索引

更新：2026-09-05。

本文是仓库整理期间的**纠错与风险记忆**，用于回答“哪些旧记录不能直接相信、正确
口径在哪里、整理时应保留什么”。实验数值与 checkpoint 身份仍以
[`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md) 为唯一人工总表；
本文不复制完整结果表，避免产生第二套数值真源。

## 1. 标记规则

| 标记 | 含义 | 整理动作 |
|---|---|---|
| 🔴 错误/作废 | 已有反证、错误实现或错误模型身份 | 删除错误数值与重复任务，只保留根因和修正入口 |
| 🟠 有条件有效 | 结果本身可复核，但协议、模型或数据范围不同 | 必须连同限定条件引用 |
| 🟡 待验证 | 尚缺完整数据、重跑或跨环境证据 | 不写成已完成结论 |
| 🟢 当前口径 | 已有明确协议和完成证据 | 仍需记录 checkpoint、数据、分母和 run id |

## 2. 已确认错误与替代口径

| ID | 级别 | 错误或危险旧结论 | 当前纠正 | 证据 / 当前入口 |
|---|---|---|---|---|
| M-01 | 🔴 | 把旧 100M Pause-1/2 checkpoint 当作论文式 learned pause token | 两个旧权重的 `pause_token_id=null`，实际复制前一真实 token；dedicated-SEP 50261 是 2026-08-28 独立重训，身份和结果必须分行 | `EXPERIMENT_REPRODUCTION_RECORD.md` §2、§2A.2、§3 |
| M-14 | 🟢 | 默认论文预训练清单和通用 `pause1`/`pause2` 别名仍选择旧 repeat-token 权重 | 默认清单已改为 SEP50261 两模型并保留实际 8-GPU LR/batch；原始提交 YAML 与 SHA-256 已收录。旧模型仅经 `bbc-100m-historical` 或 `pause1-repeat`/`pause2-repeat` 显式选择；XSum/BoolQ 路由至 Pause v2 campaign | [`docs/pause_protocol.md`](docs/pause_protocol.md)、`train_configs/paper_pretraining_manifest.json` |
| M-15 | 🟢 | 默认数据 plan 重抽 split、组装排序索引、tokenization 跳坏行；Benepar 单树被误当候选列表 | 固定已补齐 dev/test 原文件（4,980/5,025 文档），保留列表顺序；增加解析/三路 shard/最终组装完成与哈希门禁。118 项离线回归通过，未重建全量历史语料，不把新 test 当 4,966 文档 DocPPL | [`docs/pretraining_data_pipeline_repair.md`](docs/pretraining_data_pipeline_repair.md) |
| M-02 | 🔴 | 引用旧 Pause XSum v1 输出作为模型能力 | v1 同时存在 pause convention、label mask 和生成相位错误；错误数值已删除。SEP50261 v2 五 seed R-AVG 为 Pause-1 `22.378 ± 0.046`、Pause-2 `22.250 ± 0.061` | [`reports/pause_xsum_pipeline_audit_20260829.md`](reports/pause_xsum_pipeline_audit_20260829.md) |
| M-03 | 🔴 | 用 `Tree_shuffle_pretrain` unmasked checkpoint 承载论文 Tree-Shuffle 行 | 论文行恢复映射为 `treeshufflemask_pretrain/step49440-unsharded`；旧 unmasked 结果只作历史对照，二者 checkpoint 和 protocol 不得混写 | `EXPERIMENT_REPRODUCTION_RECORD.md` §2、§3 |
| M-04 | 🔴 | 把 BBC/GPT-2 1B 与 FineWeb-Edu/Qwen3 1B 当成同一模型，或用错误 shard 失败覆盖另一模型 | 两套模型须使用 corpus-qualified key；BBC `terminal-bbc-1B` 的 model-only 结果不属于 FineWeb-Edu Table 6 | `EXPERIMENT_REPRODUCTION_RECORD.md` §2、§4 |
| M-05 | 🔴 | `tree_spans` 的 `id()` 缺陷同时破坏 Pushdown 的深度偏置和 attachment gold action | 审计显示变化只在 `split`，所有已审计 `(left,right)` 均不变。GPST merge-order 会崩溃；TreeReg split 监督受轻微污染；只消费 `(left,right)` 的 Pushdown 路径不受此缺陷影响。Pushdown job 45195 另有候选 terminal 分组数据错误，不能与此混为一因 | [`docs/diagnostics/2026-08-20-tree300-eval-failure-report.md`](docs/diagnostics/2026-08-20-tree300-eval-failure-report.md)、[`diagnostics/results/tree_spans_contamination_summary.md`](diagnostics/results/tree_spans_contamination_summary.md) |
| M-06 | 🔴 | 把 BPE-spliced n-ary/right-binary 转换称为 checkpoint-training aligned | checkpoint 训练保留 fixed right-recursive multi-BPE word atom；纯错误协议文档和未完成全量记录已删除。正确实现的 200-sentence BPE 拓扑消融仍作为**不同测试协议**保留 | [`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md) “Preserved alternative-protocol diagnostics” |
| M-07 | 🟠 | 把所有 Pushdown protocol v1 都视为无效，或把 v1/v2 数值直接混合 | v1 `stack_legal` 是历史 Table-4 evaluator 口径；v2 `sentence_causal` 对齐 attachment 训练 CE。两者都是定义明确但不同的概率，必须分开报告；当前训练表示使用 fixed-word-atom binary 协议 | [`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md) |
| M-08 | 🔴 | 使用 2026-08-27 的旧 Pushdown BoolQ 数值 | 旧 attachment loss 除以 LM label-token 数，分母错误；旧值已删除。修正后五 seed validation accuracy 为 `65.54 ± 0.29` | `EXPERIMENT_REPRODUCTION_RECORD.md` §3 “Pushdown” |
| M-09 | 🔴 | 仅凭 Slurm 状态把 TreeReg-layer9 job 3561 判为计算失败 | 148,836 条评测已完成并得到 PPL `12.37`；退出码来自计算结束后的 grep metric key 错误 | `EXPERIMENT_REPRODUCTION_RECORD.md` §5 |
| M-10 | 🔴 | 认为 model-only 评测必须存在 `optim.pt`/`train.pt`，因而把 Tree-Shuffle 3450 当作最终失败 | model-only restore 可重置 optimizer/trainer state；3467/3473/3474 已替代该失败任务。旧 unmasked 结果仍不能填论文 masked 行 | `EXPERIMENT_REPRODUCTION_RECORD.md` §3、§5 |
| M-11 | 🟠 | 把 Terminal/Pause、Tree/TG 和 Pushdown 的 Doc-PPL 当作完全相同的指标直接排序 | 它们分别可能是单路径 terminal、tree-marginalized 近似或外部 top-K 截断联合概率；比较时必须同时给出候选 support、attachment normalization、BOS/EOS 和分母 | `EXPERIMENT_REPRODUCTION_RECORD.md` §1、各 protocol 文档 |
| M-12 | 🔴 | 把 TreeReg-layer6 的早期 Doc-PPL 启动记录当成评测结果 | 该配置在进入 evaluation 前把 dataloader 错指到 `terminal/train.npy`，没有产生指标；任务行已从主表删除，只保留此根因 | 本文 |
| M-13 | 🔴 | 把尚未进入模型执行的 Slurm 启动任务保留为实验记录 | 非交互 shell 未进入固定环境会找不到 `torchrun`；此类任务没有产生指标，编号与失败记录已删除，只保留环境根因 | 本文 |

## 3. 文档可信度与阅读顺序

### 🟢 当前入口

1. [`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md)：模型身份、论文值、重跑值、任务状态的人工总表。
2. [`Evaluation.md`](Evaluation.md)：当前 evaluator/data/config 协议导航；后半仍保留并明确标记 2026-08-23 历史附录。
3. [`docs/pretraining_reproduction.md`](docs/pretraining_reproduction.md)：当前预训练数据与 campaign 操作入口。
4. [`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md)：当前 Pushdown fixed-word-atom v1/v2 协议。
5. [`reports/pause_xsum_pipeline_audit_20260829.md`](reports/pause_xsum_pipeline_audit_20260829.md)：Pause XSum v1 失败根因与 v2 结果。
6. [`docs/FSDP_DOWNSTREAM_EVAL_RISKS.md`](docs/FSDP_DOWNSTREAM_EVAL_RISKS.md)：仍然开放的 FSDP 下游评测风险。

### 🟠 专题与历史材料；只能按所写范围引用

| 文档 | 限定 |
|---|---|
| [`reports/document_ppl_validation_20260823.md`](reports/document_ppl_validation_20260823.md) | 只保留当日已完成的 terminal 结果、正确性检查与性能观察；失败/排队快照已删除 |
| [`reports/pretraining_pipeline_paper_vs_repo_20260823.md`](reports/pretraining_pipeline_paper_vs_repo_20260823.md) | 已按 8 月 31 日状态回写；仍是预训练专题报告，不是结果总表 |
| [`reports/finetune_pipeline_audit_20260825.md`](reports/finetune_pipeline_audit_20260825.md) | 数据资产审计仍有用；“当前标准协议”仅指当时 TreeReg campaign，不能覆盖后来的 Pause/Pushdown 专用协议 |
| [`docs/pushdown_vs_original_repo.md`](docs/pushdown_vs_original_repo.md) | 原仓库与本地协议的语义对照；数值始终以 fixed-word-atom 当前协议文档和主表为准 |
| [`docs/pushdown_gold300_document_ppl_design.md`](docs/pushdown_gold300_document_ppl_design.md) | 设计阶段文档，不是当前可直接运行的正式协议 |
| 本机忽略文件 `analysis-output/SG_BLIMP_FULL_RESULTS.md` | 原始数值摘录缺 run id、日志和 protocol 元数据，只能作线索，不能回填 FineWeb-1B 正式表 |
| 本机忽略文件 `analysis-output/FINDINGS.md` / `FINDINGS_zh.md` | 7 月的机制诊断与抽样实验；不能用局部/替代评分结果覆盖 full-suite 复现值；默认不会随普通 `git add` 发布 |

### 🔴 已删除的误导性文档

- `docs/tree_300_eval_failure.md`：完整报告的截短重复副本。
- `docs/gpst_strict_binary_training_bpe_pushdown_document_ppl_protocol.md`：文件名和核心
  “training BPE” 假设均错误。
- `docs/nary_right_binary_pushdown_document_ppl_protocol.md` 与
  `docs/nary_right_binary_pushdown_document_ppl_v2_protocol.md`：把 BPE-spliced tree
  错称为训练表示。

上述文档的错误原因保留在 M-05/M-06。定义明确的 BPE 拓扑消融数值已迁移到当前
Pushdown 协议文档，不随错误文档删除。

## 4. 仍未关闭的风险

| ID | 级别 | 风险 | 整理建议 |
|---|---|---|---|
| R-01 | 🟡 | FineWeb-Edu 1B 的 11-task OLMES（含 BoolQ）已有完整 per-example/config/log 证据并回填；SG/BLiMP 旧摘录仍缺 run/protocol 元数据，XSum/Doc-PPL 不属于该合同 | OLMES 以总登记表 §4 为准；其余列保留 `—`，不用 BBC 1B 或无元数据摘录代填 |
| R-02 | 🟡 | 多 rank FSDP 下游评测可能因 forward/collective 次数不一致死锁，自定义生成在 `summon_full_params()` 内也不受支持 | 正式下游评测继续使用 DDP/full replica，直至真实多 GPU 集成测试通过 |
| R-03 | 🟠 | `testppl_tree` 与 `tree/test.npy` 有固定 `+59` 文档偏移、48 句总数差和 6 个异常文档 | 以 `testppl_tree` 边界为 Doc-PPL 真值；保留异常清单，不做静默顺延 |
| R-04 | 🟡 | `olmo/data/tg_mask.py` 权威 Python 源码仍缺失，仅有编译 `.so`/旧副本线索 | 开源前找回或重建并增加等价性测试 |
| R-05 | 🟡 | XSum、SuperGLUE 解析树 sidecar 的从原始数据再生脚本仍不完整，MultiRC/ReCoRD 仍为存根 | 按 `reports/finetune_pipeline_audit_20260825.md` 的 G1--G3 处理 |
| R-06 | 🟠 | 审计时工作树已有大量未提交修改与未跟踪实验文件 | 整理前先按“代码 / 协议 / 结果 / 外部 artifact”分批建清单；禁止批量 reset、覆盖或移动未知改动 |
| R-07 | 🟠 | 多份当前 protocol/report 仍是未跟踪文件，`analysis-output/` 下的历史报告还被 Git 忽略 | 提交本记忆文档前同步决定哪些证据文档进入版本控制，避免新 clone 出现悬空链接或只剩结论没有证据 |
| R-08 | 🟠 | Tree/TG `tg_doc` 已登记运行实际使用 candidate-0 document history，当前稿件文字却描述 model-greedy history | 在登记表和 `Evaluation.md` 显式披露；修正稿件或对齐实现并重跑，不得静默改变已运行指标的语义 |
| R-09 | 🟠 | camera-ready Pushdown 下游描述与已登记主 run 不完全一致：XSum 主 run 是 beam=6、`max_reduce=null`，BoolQ 是 gold-span teacher-forced terminal-format scoring | 以总登记表的 run 证据为实际协议；修正稿件中残留的 `max_reduce=4`/beam 文字后再解除风险 |
| R-10 | 🟡 | `scripts/init_cfg_and_sbatch.py` 暴露 `*-fwedu-1B` key，但当前仍重建为 BBC 数据与 GPT-2 tokenizer | FineWeb 正式评测只用 Qwen3 专用 config；通用生成器在变为 corpus-aware 前不是 FineWeb 入口 |

## 5. 整理时的最小规则

1. 错误实现产生的数值和无结果失败任务不进入主结果表；根因只在本文登记一次。
2. 正确但口径不同的结果保留为 protocol comparison，并写清适用范围，不能混入主值。
3. 任何结果离开 artifact 目录进入论文表前，至少绑定 checkpoint、数据版本、protocol、分母、run id 和完成状态。
4. `saved_models/`、集群绝对路径和本机 artifact 不等于可发布资产；整理时另做“存在性、哈希、许可、可再生性”清单。
5. 表格中的 `—` 只表示未按该口径重跑，不表示 0、失败或可从别处推断。

## 6. 本次整理改动

- 新增 [`docs/README.md`](docs/README.md) 作为协议、实现记录与历史诊断的分类入口。
- 删除已完成的 Pushdown 优化 campaign 根目录 plan/checklist/candidate board，以及已经
  落地的 GPST/Pushdown 实施计划；将性能证据收敛为
  [`reports/pushdown_gpu_optimization_20260802.md`](reports/pushdown_gpu_optimization_20260802.md)，
  将仍有长期价值的 GPST 架构说明收敛为
  [`docs/gpst_implementation.md`](docs/gpst_implementation.md)。
- 在主登记表和 README 增加本记忆入口。
- 保留完整 `tree_300` 根因报告，删除其截短重复副本。
- 给早期 Evaluation、Doc-PPL、预训练、finetune 与 Pushdown 对照文档加历史/废弃提示。
- 把无元数据的 FineWeb-1B 数值摘录及 7 月机制分析标为历史/诊断材料。
- 删除纯错误协议文档；把仍有意义的 BPE 拓扑消融结果迁移到当前协议对照。
- 重写 `Evaluation.md` 的当前协议层，集中数据入口、配置分层、模型族路由、完整性
  门禁和未关闭差异；2026-08-23 内容降为带警告的历史附录。
- 核对并回填 FineWeb-Edu 1B 五模型 11-task OLMES 产物；保留无 run/log/protocol
  元数据的 SG/BLiMP 摘录为历史线索。
- 登记 Tree/TG Doc-PPL candidate-0 history、Pushdown 下游稿件偏差与 FineWeb 通用配置
  生成器语料混用风险。
