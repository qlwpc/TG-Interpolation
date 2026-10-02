# Evaluation：数据、模型身份与评分协议

本文说明模型、任务、数据和配置路径。有效结果与论文解释见
[论文结果](docs/paper_results.md)及[新增实验](docs/bbc_dedup_500m_evaluation_results_20260925.md)，
可获取材料见[公开资产清单](reproducibility/README.md)。历史值不能改贴为当前协议。

> [!IMPORTANT]
> 本文的“当前”指当前可复核的 evaluator/data/config 合同，不表示所有模型都共享同一
> 概率、候选集合或分母。Terminal/Pause、Tree/TG、Tree-Shuffle 和 Pushdown 的
> Document-PPL 是四类协议；正确报告方式是并列注明协议，不是把它们改写成同一种指标。

## 1. 先按模型族选协议

| 模型族 | Document-PPL | SG | BLiMP | XSum / BoolQ 或 OLMES |
|---|---|---|---|---|
| BBC Terminal | `terminal_doc`，单 terminal 路径 | terminal teacher-forced | terminal，K=1 | BBC 五 seed task-specific finetune |
| BBC dedicated-SEP Pause | `terminal_doc`；document-global pause phase，pause target 不进分母 | pause-expanded teacher-forced | pause-expanded terminal，K=1 | 只走 Pause v2 campaign；XSum phase-constrained KV-cache generation |
| Tree/TG 与其线性化变体 | `tg_doc`；至多 300 个有效 CRF proposals，truncated sum | word-synchronous DFS beam，beam=300 | 读取已有 300 parses 后对 joint tree loss 做 `logsumexp` | BBC 五 seed；保留模型自己的 tree/TG 表示 |
| Tree-Shuffle（masked 论文行） | terminal-only，K=1 | terminal-only teacher-forced | terminal-only，K=1 | checkpoint 保持 `tree_shuffle_mask`；不能换成 unmasked checkpoint |
| TreeReg-layer9 | `terminal_doc` | terminal teacher-forced | terminal，K=1 | 论文外架构对照；下游须区分 legacy FT 与 parse-aligned auxiliary-loss FT |
| Pushdown | native attachment candidates；clean 已登记协议为 n-ary v1 `stack_legal` / model-best | incremental attachment beam=300 | supplied gold300 attachment candidates | XSum 使用 source gold spans + ROOT-free summary stack；BoolQ 使用 corrected gold-span scoring |
| FineWeb-Edu 1B | 不用 BBC Doc-PPL 入口代填 | 当前 SG 框架 + Qwen3 tags | 当前 BLiMP 框架 + Qwen3 top-300；按 grammar 选择 K | Qwen3 tokenizer；OLMES completion/cloze，多 shot，primary=`tree_eval_type: terminal` |

`Treeterm` 与 `TGTreeterm` 是 Tree/TGTree checkpoint 的 terminal-score 评测名，不是独立
权重。100M 旧 `pause_token_id=null` 权重是 repeat-token compute control，也不是
dedicated-SEP Pause。Tree-Shuffle 的论文行使用
`saved_models/treeshufflemask_pretrain/step49440-unsharded`；
`saved_models/Tree_shuffle_pretrain/step49440-unsharded` 只保留为 unmasked 历史对照。

BBC 100M `pause1` / `pause2` 公共别名现在绑定 SEP50261 checkpoint；历史权重通过
`pause1-repeat` / `pause2-repeat` 显式选择。通用配置生成器核对 checkpoint 原始 pause ID，
拒绝通过 override 改写模型身份；论文 XSum/BoolQ 必须走 Pause v2 campaign。
预训练参数与完整命令见 [`docs/pause_protocol.md`](docs/pause_protocol.md)。

## 2. 数据入口

路径均相对仓库根目录。存在文件只说明本机有资产；进入结果登记前仍须绑定哈希、split、
checkpoint 和 run id。

### BBC Document-PPL：先选择数据版本

当前 clean 全量评测见 [reserved-clean 运行协议](docs/bbc_reserved_docppl_sist_20260907.md)。
根目录为 `dataset/bbc-news-reserved-clean-v1/`：

| 表示 | 根目录下路径 | 合同 |
|---|---|---|
| Terminal / Pause / TreeReg / Tree-Shuffle terminal | `evaluation_terminal/{test.npy,test_sent_index.npy,test_doc_index.npy}` | 正文来自 `terminal/test.npy`，索引由输入审计从 canonical 边界派生 |
| Tree / TG | `testppl/tree300/`、`testppl/tg300/` | 每句至多 300 个有效候选；按 `valid_counts.npy` 计分，填充不计概率质量 |
| GPST / Pushdown native | `native_model_topk_300_v2/test/` | 两模型独立候选空间与有效数量，不能共享 candidate ID |

完整 test 为 **5,000 篇、129,085 句、3,068,713 个计分 terminal/EOS token**。
基础数组含 BOS 为 3,073,713 tokens。固定 test-1000 子集另行登记，不能冒充完整 test。

**历史数据仅用于明确选择的旧结果复现**：`dataset/bbc-news/terminal/`、
`testppl_tree/`、`testppl_tg/` 和 `testppl/native_model_topk_300_v2/` 对应旧 4,966 篇、
148,836 句与 3,284,061 个计分 tokens；它们不是 clean split。旧索引 test 的 5,025 篇
又是另一版本，见 [版本溯源](docs/bbc_data_provenance.md#versions)。

### 其他评测数据

| 任务 | 当前数据入口 | 必须核对的结构 |
|---|---|---|
| SG | `evaluation/SG/tokenized/*.json`；Qwen3 用 `evaluation/SG/tokenized/qwen3/*.json` | 当前 32 项、6 类；`nn-nv-rpl` 不计入 |
| BLiMP | `dataset/BLiMP/tree300/blimp_{terminal,tree_300,tg_300,tree_300_qwen}.npy` | full suite=67 tasks × 1,000 pairs；terminal K=1，结构模型 K=300 |
| XSum | `dataset/Xsum/` | filtered train 由 `save_ids.json` 决定；full test=11,333；source、summary、gold JSONL 必须成套 |
| BoolQ | `dataset/SuperGLUE/BoolQ/` | train=9,427，validation=3,270；parsed passage/question sidecars须与 JSONL 同版 |
| FineWeb-Edu OLMES | `olmo_data/oe_eval_tasks/`、`olmo_data/hf_datasets/` 与任务 registry | tokenizer=`dataset/TG_QWEN3_tokenizer.json`；不得回落到 BBC GPT-2 tokenizer |

下游数据采用直接开源已处理文件的交付方式（含所需解析树 sidecar），按上述目录加载；
不要求另行提供数据构造 pipeline。MultiRC/ReCoRD 保留现有 stub，不计划实现，
不列入已支持的下游评测任务。

BBC GPT-2 evaluator 使用 `dataset/bbc-news/TG_GPT2_tokenizer.json`。FineWeb-Edu/Qwen3
模型使用 `dataset/TG_QWEN3_tokenizer.json`，并需相应 Qwen3 SG/BLiMP 数据。任何配置同时
出现 FineWeb checkpoint 与 BBC tokenizer，或 BBC checkpoint 与 Qwen3 task sidecar，都应
在提交前失败，而不是让 evaluator 自动猜测。

## 3. 配置分层与入口

### 3.1 四层合同

1. **checkpoint config**：只负责已经训练出的架构、grammar、tokenizer identity、context
   length 和 pause id；checkpoint 中后来被评测覆盖的 `data.paths` 不是训练来源证据。
2. **protocol config**：显式选择 evaluator label/type、数据、`structure_mode`、
   `tree_eval_type`、候选数、分母和 task split。
3. **runtime config**：设备、DDP/FSDP、microbatch、worker 和输出路径。改变 runtime 不得
   改变第 2 层概率语义。
4. **result record**：记录 checkpoint、config、数据/哈希、seed、job/run id、日志、完成
   状态和主协议/诊断身份。

test-only 运行必须从 checkpoint model config 重建，并设置
`eval_on_load=true`、`eval_no_save=true`、`max_duration=0`、
`reset_optimizer_state=true`、`reset_trainer_state=true`、`try_load_latest_save=false`。
不要继承 checkpoint 的训练 shards、旧 evaluator、`stop_at` 或 optimizer/trainer state。

### 3.2 可执行入口

| 范围 | 首选入口 | 边界 |
|---|---|---|
| BBC 常规模型的 SG/BLiMP、历史 Doc-PPL 与普通 XSum/BoolQ campaign | `scripts/init_cfg_and_sbatch.py` | BBC key 绑定 GPT-2；`terminal-1B` / `tree-1B` 仍是 BBC 兼容别名 |
| FineWeb-Edu 1B SG/BLiMP | `scripts/init_cfg_and_sbatch.py`，`*-fwedu-1B` key | 绑定 Qwen3 tokenizer / tags / BLiMP 数组；启动方式见 §5.1；不复用 BBC 的 DocPPL/微调配方 |
| dedicated-SEP Pause 全套评测 | `scripts/pause_eval_campaign.py`、`scripts/pause_eval/README.md` | XSum 必须是 pipeline v2 且 eval batch=1；旧 v1 checkpoint/marker 不可复用 |
| TreeReg/Pushdown SG 与 BLiMP 协议对照 | `scripts/make_syntax_eval_configs.py` | `structure_mode` 必须显式；Pushdown 主值为 SG `beam`、BLiMP `gold` |
| reserved-clean 完整 Document-PPL | `scripts/evaluate_reserved_document_ppl.py`；Pushdown 使用 standalone evaluator | 按 [运行协议](docs/bbc_reserved_docppl_sist_20260907.md) 的输入审计、独立配置与严格 finalizer 执行 |
| Pushdown 当前 Document-PPL | `scripts/evaluate_pushdown_document_ppl.py` 与 `docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md` | 论文主值、v1/v2、direct/n-ary 和 topology diagnostic 分开登记 |
| masked Tree-Shuffle | `scripts/init_cfg_and_sbatch.py`，`tree_shuffle_mask` key | base 三项用 runtime `terminal`；XSum/BoolQ 保持 checkpoint grammar=`tree_shuffle_mask` |
| FineWeb-Edu 11-task OLMES | `evaluation/eval_configs/a800_bootstrap_{terminal,tree,pause}.yaml` | 使用 Qwen3 tokenizer；Tree/TGTree primary score 必须 `tree_eval_type: terminal` |
| FineWeb terminal/full 验证 | `scripts/make_terminal_format_validation_configs.py` | Validation-8 decomposition 与 Validation-10 不得冒充 11-task test 结果 |

`evaluation/eval_configs/` 和 `train_configs/eval_per_metric/` 中仍有历史、smoke 和专题 YAML。
文件名含 `smoke`、`profile`、`decomp` 或旧绝对路径的配置不能仅凭存在就升级为主协议。
优先从已核对 checkpoint 生成新 run 配置；冻结后的 campaign config 与 manifest 一起保存。

### 3.3 `structure_mode` 是当前结构协议开关

`EvaluatorConfig.structure_mode` 取值为：

| 值 | 语义 |
|---|---|
| `auto` | 保留模型族默认；只适合已被本表明确覆盖的普通 Tree/TG 路径 |
| `terminal` | terminal teacher-forced；不提供 parse/stack tape |
| `gold` | 当前用于 Pushdown BLiMP；消费 supplied 300 parses/spans |
| `beam` | 由模型增量推断 latent structure；Pushdown 与 Tree/TG 使用各自搜索器 |

旧 `beam_search: true` 只作兼容；新配置同时写冲突的 `structure_mode` 会报错。论文主协议
不得依赖隐含 `auto` 来选择 Tree-Shuffle 或 Pushdown 分支。

## 4. 当前 Document-PPL 合同

### 4.1 Terminal、Pause、TreeReg 与 Tree-Shuffle terminal projection

evaluator 为 `label=terminal_doc_ppl, type=terminal_doc`，batch=1、worker=0。每篇文档只在
开头提供 BOS；BOS 不计分，普通 terminal 与 EOS 计分。分母由所选数据版本决定：
clean 全量为 3,068,713，旧 4,966 篇版本为 3,284,061。DataLoader 按完整文档分配到 rank，每句必须恰好评测一次；metric 对
全局 NLL 与计数做归并，并拒绝 partial evaluation。

同一文档内句间延续 KV cache。下一句首 token 必须由上一句冻结的末位分布计分，不能用
当前句提交缓存后的分布覆盖。超过 context length 时只裁历史 cache。

Pause 使用 document-global phase；跨句继续、换文档重置。插入位置随 checkpoint 的
`pause_token_id` 决定：`null` 表示 repeat-token control，50261/151673 表示 dedicated SEP。
插入位置通过 `label_mask` 排除，因此分子和分母仍与 terminal projection 对齐。

### 4.2 Tree/TG 的 CRF-300 truncated marginal

evaluator 为 `type=tg_doc`，Tree label=`txl_approx_doc`，TG label=`tg_approx_doc`，当前
物理槽位上限为 300；clean 入口只对有效候选的 joint tree NLL 做
`logsumexp(-NLL)`，总和再除以 terminal/EOS 分母，不除以 K。

当前实现对本句全部有效候选评分后，选择总 NLL 最小的候选提交为后续文档的历史，
连同 KV cache、下一 token 分布和 mask 状态一起更新；并列取最早候选。
见 [model-best 历史策略](docs/document_ppl_model_best_history.md)。旧 candidate-0 结果
仍保留原协议身份，不能凭代码更新重标记。

### 4.3 Pushdown native top-K

当前 standalone Pushdown 使用 `prefix_policy="model_best"`，依据当前协议下的
joint token + attachment NLL 选择历史；token-only 诊断按 token NLL 选择。
reserved-clean 已登记运行采用 native n-ary / v1 `stack_legal`，见
[运行协议](docs/bbc_reserved_docppl_sist_20260907.md)。每句对唯一有效结构的联合概率
作 truncated sum，不减 `log K_s`。

fixed-word-atom n-ary/direct strict-binary、v2 `sentence_causal`、uniform-average、
token-only 和 BPE-spliced topology 属于不同协议或诊断。定义及早期测量的历史范围见
[Pushdown 协议](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md)。
该文保留的旧数据/candidate-0 测量不能与当前 clean/model-best 结果合并；续跑与合并须核对
`prefix_policy`、数据哈希、checkpoint 和完整文档覆盖。旧论文值见[论文结果](docs/paper_results.md)。

## 5. SG 与 BLiMP

SG 使用 32 项、6 类公式，按 target-region surprisal 判断并汇总 category average。
Terminal/Pause/Tree-Shuffle/TreeReg 使用 teacher-forced logits；Tree/TG 使用
word-synchronous DFS beam=300，`nc=max(term_len,5)`、`pc=3`、
`max_length=max(6*term_len,10)`；Pushdown 主协议使用 attachment beam=300。

BLiMP full suite 为 67×1,000 minimal pairs。Terminal/Pause/Tree-Shuffle/TreeReg 每句 K=1；
Tree/TG 读取已有 300 parses 并对 joint tree likelihood 做 truncated `logsumexp`；Pushdown
主协议把 supplied gold300 trees 转成 terminal-coordinate spans/action candidates 后边缘化。
terminal-only 或 inferred-beam 结果是 protocol diagnostic，不能替换 registered gold300
主值。任何 tied/non-finite pair 数也应随结果登记。

### 5.1 FineWeb-Edu 1B SG/BLiMP 当前启动方式

FineWeb-Edu 1B 的 SG/BLiMP 使用当前代码测试框架和 Qwen3 tokenizer。
在仓库根目录，按 [environment.yml](environment.yml) 创建并激活 `LLM` 环境后运行：

```bash
conda activate LLM
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/init_cfg_and_sbatch.py \
  --device RTX3090 \
  --models terminal-fwedu-1B tree-fwedu-1B tgtree-fwedu-1B \
           pause1-fwedu-1B pause2-fwedu-1B \
  --tasks SG blimp --seeds 6198 \
  --campaign-dir artifacts/evaluation/fineweb_1b_syntax \
  --submit
```

去掉 `--submit` 只生成配置和启动脚本。每个任务默认两卡 DDP；`--device` 选择目标集群
配置。输出为 `runs/<model>_<task>_seed6198/config.yaml`、同目录 `.sh` 和总目录
`jobs.json`，checkpoint 使用 `saved_models/A800_models/` 下对应的最终权重。

生成器验证 checkpoint 的词表大小、EOS/PAD 和 Pause SEP 身份，并绑定
`dataset/TG_QWEN3_tokenizer.json`（Pause SEP=151673）。SG 从
`evaluation/SG/tokenized/qwen3/` 读取对齐 tags；BLiMP 从
`dataset/BLiMP/tree300/blimp_tree_300_qwen.npy` 读取 uint32 数据，Terminal/Pause 每句
取 candidate 0 并转换为 terminal，Tree/TGTree 使用 300 候选及对应表示转换。
显式 terminal/beam 模式也从 Qwen 数组转换，不读取 GPT-2 的 `blimp_terminal.npy`。
生成器同时写入小型 `config.eval-only.npy`，仅供 Trainer 构造未使用的训练 loader；
保留它与 YAML 同目录。正式 SG/BLiMP 不对该占位数组计分，也不依赖 FineWeb 训练分片。

BBC 模型使用相同生成器、BBC model key 与 GPT-2 数据。FineWeb 的 11-task OLMES
继续使用 §6.2 的专用配置；SG/BLiMP 不并入其平均值。

## 6. XSum、BoolQ 与 FineWeb-Edu OLMES

### 6.1 BBC XSum / BoolQ

普通 BBC checkpoint 从预训练权重重置 optimizer/trainer 后独立微调。XSum 为 3 epochs、
LR `6e-5`、global batch 40；完整 11,333-example test 报 ROUGE-1/2/L 及三者均值 R-AVG。
BoolQ 为 5 epochs、LR `3e-4`、global batch 40；完整 validation 报 accuracy。论文表使用
seeds `42, 2026, 6198, 13171, 31723` 的 mean ± sample SD。

Pause 只使用 v2：训练 supervision mask 与 expanded summary 对齐，生成保持 pause phase，
使用 KV cache 且 eval batch=1。任何缺少匹配 `training_contract.json` 或
`xsum_pipeline_version: 2` 的旧产物都不可复用。

Pushdown XSum 的已登记主 run 使用 source gold spans、ROOT-free summary stack、beam=6、
`max_reduce=null`；prompt spans 只作 attention history，不成为 summary attachment target。
Pushdown BoolQ 使用修正后的 attachment-loss 分母和 gold-span teacher-forced terminal-format
MC scoring。旧错误分母产生的 BoolQ 和 root-containing/max-reduce=4 XSum 均不进入主表。

### 6.2 FineWeb-Edu OLMES

11 tasks 使用 completion/cloze，而不是显示选项字母的分类。HellaSwag、WinoGrande、MMLU、
MMLU-Redux、OpenBookQA 为 5-shot；BoolQ、ARC-Easy、ARC-Challenge、PIQA、SocialIQA、
CommonsenseQA 为 3-shot。每个 textual component 使用 Benepar 1-best；不使用 300 proposals。

Tree/TGTree 的 primary metric 是 continuation terminal-token log-prob 之和
(`tree_eval_type: terminal`)；full score 只作 sensitivity analysis。11-task test、
Validation-8 decomposition 和 Validation-10 是三套不同数据合同，不得互相代填。
已完成的 11-task per-example evidence 与 bootstrap 汇总位于
`analysis-output/bootstrap/`；公开汇总时绑定 checkpoint 和完成证据；完整逐样本产物获取位置尚待补齐。

## 7. 分布式与完整性门禁

- evaluator 使用无 padding duplication、无 tail truncation 的 `DistributedEvalSampler`。
  Document-PPL 按完整文档分 rank；K>1 BLiMP 按完整 sentence group 分 rank。
- `eval_subset_num_batches=-1` 才是 full run。smoke/partial 指标不得填主表；terminal
  Document-PPL metric 会主动拒绝记录数不完整的运行。
- 当前自定义生成和部分结构搜索不支持在 FSDP shard 上直接调用 module 方法。正式下游评测
  继续使用 DDP/full replica，直至真实多 GPU 集成测试关闭
  [`docs/FSDP_DOWNSTREAM_EVAL_RISKS.md`](docs/FSDP_DOWNSTREAM_EVAL_RISKS.md) 中的风险。
- model-only checkpoint 可以没有 `optim.pt`/`train.pt`；只要显式关闭两类 state restore，
  这不是失败证据。
- 输出进入结果页前必须核对 finite metrics、完整样本数、checkpoint/config/data identity、
  run/job id 和日志完成标记。主协议与 alternative-protocol diagnostic 分行。

## 8. 论文已知差异与协议边界

### 8.1 DocPPL 的历史结果与当前协议

当前 Tree/TG 与 standalone Pushdown 已使用 model-best；旧 candidate-0 结果与新数据结果须分开。
论文正文/附录残留差异及重测对结论的影响见 [2026-09-11 审计](docs/bbc_data_provenance.md#paper-impact)。
论文已冻结，后续差异和结论适用范围在仓库文档中说明，不再以修改论文为待办。

<a id="pushdown-paper-discrepancies"></a>

### 8.2 Pushdown：冻结论文与实际运行的已知差异

以下根据已核对的运行记录说明冻结论文与实现的差距，
不更改历史指标，也不将当前实现自动视为历史运行现场的源码。
论文版本与数值转录见[论文结果](docs/paper_results.md)。

| 项目 | 冻结论文的描述 | 已登记主 run / 代码行为 | 解释与证据 |
|---|---|---|---|
| XSum reduce 限制 | 下游附录，第 708 行：`at most 4 consecutive reductions` | 主 run 为 `beam=6,max_reduce=null`；source parser spans 只作 attention history，summary 使用空的、句内 ROOT-free attachment stack | 明确的参数描述差异。`beam=6` 一致，无额外连续 reduce 上限。来自 sentence-local 五 seed run 的已核对记录；本地 campaign 的 `run_eval.sh` 也显式设置 `null` |
| BoolQ 搜索方式 | 同段：`candidate scoring with beam size 20` | 主 run 使用 supplied-span teacher-forced terminal-format multiple-choice scoring；仅缺少 `tree_spans` 时才走 beam=20 fallback | 明确的评分路径差异。来自 corrected BoolQ run 的已核对记录，以及 `Trainer.pushdown_icl_eval_step` |
| BLiMP 概率定义 | BLiMP 附录，第 749、753 行：以 `sum_k p(x,y_k)` 解释树边缘化，并称 Pushdown 对 300 个候选树边缘化 | 当前 gold300 路径仅对给定树下的 token log-likelihood 做 `logsumexp`，未加入 attachment log-probability；邻近历史提交 `5ee0e8f` 也是这一实现 | 当前实现不能按联合 token–attachment 概率边缘化解释。历史 job 2521 的配置和 74.97% 日志已核对，但其精确运行源码身份尚未确认；不能不经核实或重测将旧分数改贴为 joint marginal |
| 推理结构来源 | Setup，第 346 行：attachment parser 的预测产生推理时的 stack-depth bias | BoolQ、BLiMP 及 XSum source 使用提供的解析树；SG 和 XSum summary 使用模型搜索的 attachment 结构 | 表述范围过宽；阅读时须按任务区分 supplied structure 与 inferred structure |
| `gold` 的含义 | 下游附录，第 708 行：`gold spans / gold parse` | 树来自解析器，属于 silver trees / supplied parser spans | 术语歧义；不代表人工 gold 标注。Setup 中已有 silver parse trees 的说明 |

XSum/BoolQ 主 run 从 `pushdown_terminalonly/step34354-unsharded` 独立微调。
SG/BLiMP 使用相同预训练权重，SG attachment beam=300。完整历史运行日志仍为本地
证据，获取状态见[资产清单](reproducibility/README.md#assets)。尤其历史 BLiMP 的精确
运行源码身份未确认，当前代码测试不代替当时完整评测。

评分实现证据为 [Trainer](olmo/train.py) 的 `model_forward`、`eval`、
`pushdown_icl_eval_step`，以及 [BLiMPMetric](olmo/eval/downstream.py) 的 `update/compute`。
BLiMP gold 路径在 eval 模式下不计算 attachment logits，metric 只接收 LM logits。
本次在 `LLM` Conda 环境执行了 Pushdown beam 概率、给定 prompt spans、ROOT-free stack、
attachment-loss 分母及下游数据转换相关 CPU 检查，11 项通过；这些检查不替代历史全量重跑。

**DocPPL 的适用边界：**论文 Pushdown 的 13.29 与附录第 758 行的历史 candidate-0
协议相符，本身不是数值与协议矛盾。它不能改贴为当前 clean/model-best 结果。
论文正文第 366 行的 model-selected history 与附录对一般 CRF 树模型所写的
parser-selected history 另有内部表述差异；Pushdown 在附录中已作为 candidate-0
例外单列。当前协议见 [model-best 合同](docs/document_ppl_model_best_history.md)，
历史与 clean 测量各自保留原数据、checkpoint、候选集合和历史策略身份。

## 9. 评测正确性检查

以下边界作为评测正确性要求保留；排错过程仅在本地归档。

| 检查项 | 当前要求 |
|---|---|
| Pause BoolQ 答案 mask | 先以 `len(full_query)-len(continuation)` 定义原始答案边界，再随 Pause 展开 mask；每个选项必须有有效计分 token |
| Pause PPL 分母 | 只使用 terminal/EOS NLL 与计数，排除插入的 Pause 位置 |
| 跨句 cache | 提交当前句前保留前一句末端预测分布；文档边界重置，句间延续受长度预算约束的历史 |
| 结构变体 | 配置显式绑定 grammar、数据和 tokenizer；未知类型不得隐式回退 |


Pause XSum 的 label 展开、生成 phase 和 checkpoint 兼容要求见
[Pause v2 协议](docs/pause_protocol.md)。
model-only 恢复及分布式限制见 §7；native 续跑、完整文档落盘和严格合并见
[恢复合同](docs/native_document_ppl_recovery.md)。
