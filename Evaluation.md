# Evaluation：当前协议、数据入口与配置边界

更新：2026-09-02。

本文是评测层的当前导航与协议说明，回答“该模型在该任务上应走哪条数据和配置路径”。
它不保存第二份结果表：checkpoint 身份、run 状态和可引用数值仍只登记在
[`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md)，已知错误与待关闭
风险见 [`REPOSITORY_CLEANUP_MEMORY.md`](REPOSITORY_CLEANUP_MEMORY.md)。论文方法表述以
[`camera_ready/paper.tex`](camera_ready/paper.tex) 为准；若论文、登记表和代码不一致，必须
分别记录“论文声称”“实际运行”“checkpoint 身份”，不能静默合并。

> [!IMPORTANT]
> 本文的“当前”指当前可复核的 evaluator/data/config 合同，不表示所有模型都共享同一
> 概率、候选集合或分母。Terminal/Pause、Tree/TG、Tree-Shuffle 和 Pushdown 的
> Document-PPL 是四类协议；正确报告方式是并列注明协议，不是把它们改写成同一种指标。

## 1. 先按模型族选协议

| 模型族 | Document-PPL | SG | BLiMP | XSum / BoolQ 或 OLMES |
|---|---|---|---|---|
| BBC Terminal | `terminal_doc`，单 terminal 路径 | terminal teacher-forced | terminal，K=1 | BBC 五 seed task-specific finetune |
| BBC dedicated-SEP Pause | `terminal_doc`；document-global pause phase，pause target 不进分母 | pause-expanded teacher-forced | pause-expanded terminal，K=1 | 只走 Pause v2 campaign；XSum phase-constrained KV-cache generation |
| Tree/TG 与其线性化变体 | `tg_doc`；CRF proposal K=300 truncated sum | word-synchronous DFS beam，beam=300 | 读取已有 300 parses 后对 joint tree loss 做 `logsumexp` | BBC 五 seed；保留模型自己的 tree/TG 表示 |
| Tree-Shuffle（masked 论文行） | terminal-only，K=1 | terminal-only teacher-forced | terminal-only，K=1 | checkpoint 保持 `tree_shuffle_mask`；不能换成 unmasked checkpoint |
| TreeReg-layer9 | `terminal_doc` | terminal teacher-forced | terminal，K=1 | 论文外架构对照；下游须区分 legacy FT 与 parse-aligned auxiliary-loss FT |
| Pushdown | native attachment candidates，当前论文值用 n-ary v1 `stack_legal` truncated joint sum | incremental attachment beam=300 | supplied gold300 attachment candidates | XSum 使用 source gold spans + ROOT-free summary stack；BoolQ 使用 corrected gold-span scoring |
| FineWeb-Edu 1B | 不用 BBC Doc-PPL 入口代填 | GPT-2/Qwen 数据不能混用 | 同左 | Qwen3 tokenizer；OLMES completion/cloze，多 shot，primary=`tree_eval_type: terminal` |

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

| 任务 | 当前数据入口 | 必须核对的结构 |
|---|---|---|
| terminal Document-PPL | `dataset/bbc-news/terminal/{test.npy,test_sent_index.npy,test_doc_index.npy}` | 148,836 句、4,966 文档；三个文件必须来自同一版本 |
| Tree Document-PPL | `dataset/bbc-news/testppl_tree/{tree_300.npy,tree_sent_index.npy,tree_doc_index.npy}` | 每句 300 proposals；使用 bos/eos-normalized testppl 版本 |
| TG Document-PPL | `dataset/bbc-news/testppl_tg/{tg_300.npy,tg_sent_index.npy,tg_doc_index.npy}` | 同上，表示为 LIN2/TG |
| Pushdown Document-PPL | `dataset/bbc-news/testppl/native_model_topk_300_v2/` | model-native candidates、文档边界和候选数必须由 finalizer 校验 |
| SG | `evaluation/SG/tokenized/*.json`；Qwen3 用 `evaluation/SG/tokenized/qwen3/*.json` | 当前 32 项、6 类；`nn-nv-rpl` 不计入 |
| BLiMP | `dataset/BLiMP/tree300/blimp_{terminal,tree_300,tg_300,tree_300_qwen}.npy` | full suite=67 tasks × 1,000 pairs；terminal K=1，结构模型 K=300 |
| XSum | `dataset/Xsum/` | filtered train 由 `save_ids.json` 决定；full test=11,333；source、summary、gold JSONL 必须成套 |
| BoolQ | `dataset/SuperGLUE/BoolQ/` | train=9,427，validation=3,270；parsed passage/question sidecars须与 JSONL 同版 |
| FineWeb-Edu OLMES | `olmo_data/oe_eval_tasks/`、`olmo_data/hf_datasets/` 与任务 registry | tokenizer=`dataset/TG_QWEN3_tokenizer.json`；不得回落到 BBC GPT-2 tokenizer |

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
| BBC 常规模型的 SG/BLiMP/Doc-PPL 与普通 XSum/BoolQ campaign | `scripts/init_cfg_and_sbatch.py` | 当前实现硬编码 BBC terminal/tree/TG 数据与 GPT-2 tokenizer；不要把其中的 `*-fwedu-1B` key 当作 FineWeb 通用入口 |
| dedicated-SEP Pause 全套评测 | `scripts/pause_eval_campaign.py`、`scripts/pause_eval/README.md` | XSum 必须是 pipeline v2 且 eval batch=1；旧 v1 checkpoint/marker 不可复用 |
| TreeReg/Pushdown SG 与 BLiMP 协议对照 | `scripts/make_syntax_eval_configs.py` | `structure_mode` 必须显式；Pushdown 主值为 SG `beam`、BLiMP `gold` |
| Pushdown 当前 Document-PPL | `scripts/evaluate_pushdown_document_ppl.py` 与 `docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md` | 论文主值、v1/v2、direct/n-ary 和 topology diagnostic 分开登记 |
| masked Tree-Shuffle 已完成复现包 | `artifacts/experiment/treeshufflemask_terminal_multiseed_20260830/prepare_campaign.py` | base 三项用 runtime `terminal`；XSum/BoolQ 保持 checkpoint grammar=`tree_shuffle_mask` |
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
开头提供 BOS；BOS 不计分，普通 terminal 与 EOS 计分。当前固定分母为 3,284,061 个
terminal/EOS token。DataLoader 按完整文档分配到 rank，每句必须恰好评测一次；metric 对
全局 NLL 与计数做归并，并拒绝 partial evaluation。

同一文档内句间延续 KV cache。下一句首 token 必须由上一句冻结的末位分布计分，不能用
当前句提交缓存后的分布覆盖。超过 context length 时只裁历史 cache。

Pause 使用 document-global phase；跨句继续、换文档重置。插入位置随 checkpoint 的
`pause_token_id` 决定：`null` 表示 repeat-token control，50261/151673 表示 dedicated SEP。
插入位置通过 `label_mask` 排除，因此分子和分母仍与 terminal projection 对齐。

### 4.2 Tree/TG 的 CRF-300 truncated marginal

evaluator 为 `type=tg_doc`，Tree label=`txl_approx_doc`，TG label=`tg_approx_doc`，当前
结果使用 `SENT_SIZE=300`。每句对 300 个 joint tree NLL 做
`logsumexp(-NLL)`，总和再除以 terminal/EOS 分母；这是截断候选和，不除以 K。

当前实现把每句第一个 proposal（candidate 0）提交为后续文档 prefix 的 KV cache。它不是
对 300 个 proposal 按当前模型重新求 joint argmax。当前登记结果因此必须写
`history=candidate-0`；论文若写 model-greedy history，需先完成协议对齐或重跑，不能用文字
把已运行结果改成另一条路径。

### 4.3 Pushdown native top-K

当前论文 Table-4 的 13.293598 使用历史 native n-ary candidates 和 v1
`stack_legal` attachment normalization；它没有补全 checkpoint 训练时的 fixed-word-atom
与人工 right-CNF reductions，因此不是 training-representation likelihood。当前句计算
`logsumexp(token_ll + attachment_ll)`，不减 `log K_s`；历史句固定 candidate 0，分母仍为
3,284,061。已完成的 fixed-word-atom n-ary/direct strict-binary 与 v2 `sentence_causal`、
uniform-average、token-only、BPE-spliced topology 都是明确定义但不同的协议或诊断，
不能覆盖历史 v1 主行。完整定义以
[`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md)
为准。

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
`analysis-output/bootstrap/`；进入论文/登记表时仍须由总登记表绑定 checkpoint 和完成证据。

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
- 输出进入总登记表前必须核对 finite metrics、完整样本数、checkpoint/config/data identity、
  run/job id 和日志完成标记。主协议与 alternative-protocol diagnostic 分行。

## 8. 当前未关闭的协议差异

1. Tree/TG Document-PPL 的当前代码/已登记运行使用 candidate-0 history，而论文正文描述
   model-greedy history；在对齐或重跑前应显式披露。
2. camera-ready Pushdown 下游文字仍可能残留 `max_reduce=4` 或 BoolQ beam 描述；已登记主 run
   是 XSum `max_reduce=null`、BoolQ gold-span teacher-forced，引用时以 run 证据为准并修正文稿。
3. FineWeb-Edu 11-task OLMES 已有完整 per-example/log/config evidence，但 SG/BLiMP 的旧摘录仍
   缺 run id、日志和数据/protocol 元数据；两者信任级别不同。
4. `scripts/init_cfg_and_sbatch.py` 仍同时暴露 BBC 与 `*-fwedu-1B` model key，却重建为 BBC
   数据/tokenizer。FineWeb 正式 run 只能走专用 Qwen configs，直到通用生成器变为 corpus-aware。

---

# 历史附录：2026-08-23 evaluator 实现快照

> [!WARNING]
> 以下内容保留用于追溯旧 evaluator 分发、当时的行号和错误修复背景。`file:line` 已漂移，
> “当前”“标准”字样只代表当日状态；Pause v2、masked Tree-Shuffle、Pushdown v1/v2、
> corpus-qualified FineWeb 配置和分布式 sampler 修正均以上面的当前协议层为准。历史数值不能
> 绕过总登记表直接引用。

**快照日期**：2026-08-23。
**原用途**：个人实现参考，记录 evaluator 分发、触发机制和早期结果。

---

## 1. Evaluator 类型与分发

### EvaluatorType 枚举 (`olmo/config.py:746-752`)
```python
class EvaluatorType(StrEnum):
    downstream = "downstream"      # ICL 多选 (boolq/cb/copa 等)
    lm = "lm"                      # legacy token-stream PPL
    tg_doc = "tg_doc"              # TG 格式文档级困惑度
    tg_sent = "tg_sent"            # TG 格式句子级困惑度
    terminal_doc = "terminal_doc"  # terminal/pause 文档级 PPL
    rouge = "rouge"                # 摘要 ROUGE (xsum)
    beam_search_icl = "beam_search_icl"
```

### 三处分发

**1. 构造分发** (`olmo/eval/__init__.py:166-196`): `build_evaluator` 按 `eval_config.type` 分发
- `{tg_doc, terminal_doc, tg_sent, downstream, rouge, beam_search_icl}` → `build_downstream_evaluator`
- `lm` → `build_eval_dataloader` + `MeanMetric`
- 其他 → `ValueError`

**2. Metric 类分发** (`olmo/eval/__init__.py:90-143`): `build_downstream_evaluator` 内按 type/label 选 metric
- `tg_sent` → `TGPerplexitySentenceLevelMetric`
- `tg_doc` → `TGPerplexityDocumentLevelMetric`
- `terminal_doc` → `TerminalDocumentPerplexityMetric`（FP64 NLL，严格检查每句恰好一次）
- label `"syntactic_generalization"` → `SyntacticGeneralizationMetric`
- label `"BLiMP"` → `BLiMPMetric`
- type `rouge` → `RogueMetric`
- type `beam_search_icl` → `BeamSearchICLMetric`
- label 以 `"_decomp"` 结尾 → `DecomposedICLMetric`
- else → `ICLMetric`

**3. 运行时分发** (`olmo/train.py:1301-1311`): `eval()` 方法按 batch 分发到不同 eval_step
```python
if evaluator.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc):
    self.TG_doc_eval_step(eval_batch, evaluator)
elif evaluator.label == "syntactic_generalization":   # SG 按 label 分发 (type 是 downstream)
    self.SG_eval_step(eval_batch, evaluator)
elif evaluator.type == EvaluatorType.rouge:
    self.summarization_eval_step(eval_batch, evaluator)
elif evaluator.type == EvaluatorType.beam_search_icl:
    self.beam_search_icl_eval_step(eval_batch, evaluator)
else:   # lm, tg_sent, downstream (ICL 多选)
    self.eval_step(eval_batch, evaluator)
```
注意: SG 按 **label** 分发 (其 type 是 `downstream`)。catch-all `eval_step` 处理 `lm`、`tg_sent`、`downstream`。

### 任务 → Evaluator 映射总表

| Task | EvaluatorConfig label | Type | Metric 类 | Dataset 类 |
|------|----------------------|------|-----------|------------|
| pretrain | TG-ppl-validation | lm | MeanMetric | MemMapDataset |
| docppl (terminal/pause，快照当日标准) | terminal_doc_ppl | terminal_doc | TerminalDocumentPerplexityMetric | TerminalDocumentPerplexityDataset |
| docppl (terminal/pause，旧口径) | TG-ppl-validation + -test | lm | MeanMetric | MemMapDataset |
| docppl (tg/tree) | tg_approx_doc / txl_approx_doc | tg_doc | TGPerplexityDocumentLevelMetric | TGPerplexityApproximationDataset |
| boolq/cb/copa/... | {task} | downstream | ICLMetric | ICLMultiChoiceTaskDataset 子类 |
| SG | syntactic_generalization | downstream | SyntacticGeneralizationMetric | SGDataset |
| blimp | BLiMP | downstream | BLiMPMetric | BLiMPApproximationDataset |
| xsum | xsum | rouge | RougeMetric | XsumDataset |

---

## 2. Eval 触发机制

### eval_on_load (`olmo/train.py:1452-1455`)
`fit()` 开始时，训练前:
```python
if self.cfg.eval_on_load:
    eval_metrics = self.eval()
    wandb.log(eval_metrics, step=self.global_step)
```
在 `global_step=0` (或恢复的 step) 运行 eval。之后 line 1458 设回 `train()` 模式。

### stop_at 钳制 (`olmo/train.py:1388-1395, 1476`)
```python
# stop_after 可设定相对步数
if self.cfg.stop_after is not None:
    self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after) if self.cfg.stop_at else self.global_step + self.cfg.stop_after
if self.cfg.stop_at is None:
    self.cfg.stop_at = self.max_steps + 10
# line 1476: 钳制到 max_steps
stop_at = self.cfg.stop_at if self.cfg.stop_at <= self.max_steps else self.max_steps
```

### 训练中 eval 触发 (`olmo/train.py:1603-1616`)
```python
if not cancel_initiated and (
    self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at
):
    eval_metrics = self.eval()
    ...
    self.dist_model.train()   # eval 后切回训练模式
```
触发条件: `global_step` 是 `eval_interval` 的倍数 **或** `global_step >= stop_at` (训练结束)。

### 训练循环退出 (`olmo/train.py:1623-1624`)
`if self.global_step >= stop_at: break`

### 最终 checkpoint (`olmo/train.py:1652`)
`if save_checkpoints and not self.cfg.eval_no_save:` — `eval_no_save` 抑制保存 (test_only 运行用)。

---

## 3. test_only vs finetune 参数

定义于 `scripts/init_cfg_and_sbatch.py:122-123`:
```python
test_only_params = {
    "eval_on_load": True, "eval_no_save": True, "max_duration": 0,
    "reset_optimizer_state": True, "reset_trainer_state": True,
}
finetune_params = {"reset_optimizer_state": True, "reset_trainer_state": True, "eval_interval": 1000000}
```

### test_only_params (用于 docppl, xsum_test, blimp, SG, hellaswag, winogrande)
- `eval_on_load=True`: 训练前在 step 0 eval
- `eval_no_save=True`: 不保存 checkpoint
- `max_duration=0`: 0 训练步，eval 通过 `eval_on_load` 触发
- `reset_optimizer_state=True` / `reset_trainer_state=True`: 恢复时只加载
  `model.pt`，不读取 `optim.pt` 或 `train.pt`
- **行为**: 加载 model-only checkpoint → step 0 eval → 无训练 → 无保存

### finetune_params (用于 boolq, cb, copa, multirc, record, rte, wic, wsc, xsum_finetune)
- `reset_optimizer_state=True`: 清空 Adam 动量，finetune 从零开始优化器状态
- `reset_trainer_state=True`: 清空 trainer 状态 (步数计数器等)
- `eval_interval=1000000`: 训练中禁用周期性 eval (1M 步远超任何运行)
- **行为**: 加载 checkpoint → 重置优化器/trainer → 训练 N epochs → 通过 `global_step >= stop_at` 触发 eval → 保存

### xsum_finetune 特殊 (`scripts/init_cfg_and_sbatch.py:130-131`)
额外设 `eval_interval: 1000000` + `device_eval_batch_size: 1` + finetune_params。

---

## 4. document-level PPL（快照当日可比口径）

`task="docppl"` 对 terminal 和所有 pause grammar 现在统一产生：

```python
EvaluatorConfig(
    label="terminal_doc_ppl", type=EvaluatorType.terminal_doc,
    device_eval_batch_size=1,
)
```

它读取 `dataset/bbc-news/terminal/{test.npy,test_sent_index.npy,test_doc_index.npy}`，一条路径一条句子地延续 KV cache；每篇文章从 BOS 开始、以 EOS 结束。BOS 只提供上下文，所有普通 terminal 与 EOS 都计分。当前数据集固定为 **148,836 句、4,966 篇文档、3,284,061 个计分 token**。

pause 模型按其 grammar 在同一文档内连续插入 pause（下一篇文档相位重置），并以 `label_mask` 排除插入位置；因此分子和分母与普通 terminal 完全对齐，能直接比较 PPL。`terminal_doc` 强制 batch=1、num_workers=0，防止乱序破坏 KV cache。

句子间首 token 使用上一句冻结的末位分布计分；该修正不能用当前句缓存提交后的分布覆盖。

### 本机完整运行（2026-08-23）

| 模型 | checkpoint | Slurm job | 状态 |
|---|---|---:|---|
| terminal | `saved_models/Terminal-lr005-bs144/step34115-unsharded` | 3444 | 完成：PPL 9.88981（148,836 句） |
| pause1 | `saved_models/pretrain_pause1_100M/step45487-unsharded` | 3445 | 完成：PPL 9.83125（148,836 句） |

评测脚本必须显式进入固定的 `LLM` conda 环境，避免非交互 Slurm shell 缺少
`torchrun`。失败任务及其编号不作为实验记录保留。

生成的可复现配置、脚本和日志在 `artifacts/evaluation/terminal_doc_ppl_20260823/{terminal,pause1}/`。结果从各自 `logs/slurm-*.out` 的 `eval/downstream/terminal_doc_ppl_doc_ppl` 读取；两次运行均完成于 2026-08-23。

### 旧的 lm 路径（仅保留作历史对照）

两条路径取决于 `input_format`:

### 路径 A: terminal-input docppl (`scripts/init_cfg_and_sbatch.py:284-291`)
当 `task=="docppl" and input_format=="terminal"` (或 `task[:8]=="pretrain"`):
```python
Evallist = [EvaluatorConfig(label="TG-ppl-validation"), EvaluatorConfig(label="TG-ppl-validation-test")]
Evallist[0].data.paths = [dev.npy]
Evallist[1].data.paths = [test.npy]
Evallist[0].data.generate_doc_lengths = Evallist[1].data.generate_doc_lengths = (input_format!="tg")
grammar_type = cfg.model.transformer_grammar_type
cfg.model.flex_attention = (
    grammar_type == "tg"
    or grammar_type.startswith("tgproximal")
    or grammar_type.startswith("tgnomask")
)
```
- type 默认 `lm` → `build_eval_dataloader` → `MemMapDataset`
- `generate_doc_lengths=True` (非 tg 格式): `MemMapDataset` 预计算文档边界长度
- TG/tgproximal/tgnomask 才允许进入长度感知路由：短/动态结构化输入走
  SDPA，达到评测阈值的重复长序列才走 Flex；非 128 整数倍的 Flex 形状
  自动补齐。mixing 在真实逐头基准完成前保持关闭
- terminal/tree 格式不生成 TG bias，保持 `flex_attention=False`，文档边界
  路径使用 `flash_attn_varlen_func`
- `MeanMetric` (`olmo/eval/__init__.py:177-178`) 累积每实例 CE loss
- `compute_metrics` (`olmo/eval/evaluator.py:51-77`) 输出 `eval/{label}/CrossEntropyLoss` 和 `Perplexity = exp(CE)`

### 路径 B: TG-format docppl (`scripts/init_cfg_and_sbatch.py:292-293`)
当 `task=="docppl"` 但 `input_format != "terminal"`:
- label 改为 `"tg_approx_doc"` (tg) 或 `"txl_approx_doc"` (tree)
- type `tg_doc`, `device_eval_batch_size=60`
- `TGPerplexityDocumentLevelMetric` (`downstream.py:1090-1129`): 单设备 (`sync_on_compute=False`)，`loglikelihoods` buffer 形状 `(dataset_length // samples_per_sent, samples_per_sent=300)`
- `compute`: `ppl = exp(-logsumexp(-loglikelihoods, dim=1).sum() / data_numwords)`，`data_numwords = sum(term_length)` (term_length 只数 terminal token, `downstream.py:1320`)

### TG_doc_eval_step (`train.py:979-1054`)
顺序处理文档 + KV cache，每 300 token 快照 KV cache (`doc_kv_cache`)。CE 在所有位置计算 (line 1045-1048)，减去前一块边界的首 token log-prob 修正 (line 1049-1050)。

### 旧 lm 路径的 pause 稀释
- `TG-ppl-validation[-test]` 的 `lm` 路径仍会把 pause 展开位置纳入平均 CE，不能与 terminal 或 `terminal_doc` PPL 比较。
- 不要用简单的展开倍数换算其 PPL；新的 `terminal_doc_ppl` 是唯一报告用的 terminal/pause 文档 PPL。

---

## 5. 下游 ICL 多选 (boolq/cb/copa 等)

### ICLMetric.update (`downstream.py:60-167`)
对 batch 中每个样本，在 `[ctx_len-1 : ctx_len+cont_len-1]` 位置收集 continuation log-likelihood (line 71-73):
```python
lm_cont_logits = lm_logits[idx][ctx_len-1 : ctx_len+cont_len-1]
lm_log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens).squeeze()
```
对 `metric_type=="acc"` (line 111-113): `log_likelihood = lm_log_likelihood.sum()`

### ICLMetric.compute (`downstream.py:169-248`)
按 `doc_id` 分组 log-likelihood，填 `[num_continuations]` 张量；对 acc: `correct = argmax(loglikelihoods) in label_dict[doc_id]` (line 221)。预测 = 总 log-likelihood 最高的 continuation。返回 per-group 准确率 + `"_"` 总平均。

### prep_examples 流程 (`downstream.py:554-709`)
1. `continuations = self.doc_to_continuations(doc)` — continuation 字符串列表 (boolq: `[" yes", " no"]`)
2. `ctx = self.token_encode(doc_text)` — `encode_TG_string` + `convert_grammar_input` (line 846-849)
3. `continuation = self.token_encode(continuation_str)` (line 577)
4. `query = ctx + continuation` (line 582)
5. `query = query[-model_ctx_len:]` — 左截断 (line 589)
6. `query = self.convert_grammar_input(query)` (line 590)
7. `full_query = query` — 保存未截断副本 (line 599)
8. **split != "train" 截断**: `query = query[:-1]` (line 602) — 丢最后一个 token 让模型预测它
9. `actual_ctx_len = len(query) - len(continuation) + 1` (line 606) — `+1` 补偿 `[:-1]` 截断

### PAUSE 路径 (line 620-673，当 `self.ispause` 为真)
- `ctx_real = len(full_query) - len(continuation)` (line 654) — **不用** `actual_ctx_len` (它带 `+1` 偏移)
- `query = pause_input_ids(full_query, self.pause_token_id, pause_num=gtype)` (line 656) — 展开**完整** ctx+cont
- `split = pause_expanded_len(ctx_real, p, q)` (line 658) — 第一个 continuation real token 的展开位置
- `trim = pause_trailing_trim(ctx_real + cont_real, p, q)` (line 659) — 末尾 pause
- `continuation = query[split : len(query) - trim]` (line 660) — 评分的 continuation 是展开 query 的连续切片
- `actual_ctx_len = split` (line 661) — 覆盖，使 ICLMetric.update 在正确展开位置收集

### shots_num
- BoolQ 默认 `shots_num=3` (`downstream.py:2018`)
- 当 `train_config.finetune_task is not None`，`build_downstream_evaluator` 强制 `shots_num=0` (`olmo/eval/__init__.py:54-55`)
- shots 仅在 `shots_num != 0 and split != "train"` 时准备 (line 488)。finetune 时 split="train" 且 shots_num=0，无 shots。

---

## 6. BLiMP

### BLiMPApproximationDataset (`downstream.py:1800-1991`)
加载 `blimp_{dataset_name}.npy`，`dataset_name`:
- `"terminal"` — terminal/pause 类型
- `"tree_300"` — tree 类型
- `"tg_300"` — TG 类型 (lines 1852-1857)

### SENT_SIZE (`downstream.py:1835-1847`)
- terminal/pause: `SENT_SIZE = 1` (300 棵树转换成相同 terminal 序列，每句 1 样本)
- tree/tg: `SENT_SIZE = samples_per_sent = 300`

### TASK_SIZE (line 1848)
`TASK_SIZE = 2 * pair_per_task(1000) * SENT_SIZE`
- terminal/pause: 2000
- tree/tg: 600000

### device_eval_batch_size (`scripts/init_cfg_and_sbatch.py:294-298`)
- terminal/pause (`[:8]=="terminal"` 或 `[:5]=="pause"`): 100
- 其他 (tree/tg): 150

### Pause 展开 (`downstream.py:1927-1930`)
`__getitem__` 中 `self.ispause` 时:
```python
input_ids = pause_input_ids(input_ids, self.pause_token_id, pause_num=self.transformer_grammar_type)
```
BLiMP 句子在数据加载时 pause 展开，CE 在所有展开位置计算。

### BLiMPMetric (`downstream.py:1669-1797`)
- `loglikelihoods` buffer: `SENT_SIZE==1` → 1D `(dataset_length,)`；`SENT_SIZE>1` → 2D `(dataset_length//SENT_SIZE, SENT_SIZE)`
- `update` (line 1712): 所有位置 CE (左移 labels)，可选 mask 非 terminal (`tree_eval_type=="terminal"`)，存每句 CE
- `compute` (line 1759): 
  - `SENT_SIZE>1`: `logsumexp(-loglikelihoods, dim=1)` (tree-marginalized) 或 `-mean(...)` (terminal eval)
  - `SENT_SIZE==1`: `-loglikelihoods` (负 CE = log-prob)
  - 对 67 个 task pair (每 task 1000 pair) 比较 `p_good > p_bad`，返回 per-task/category 准确率 + `overall/overall`

### 整除断言 (`olmo/eval/__init__.py:145-149`)
`SENT_SIZE % eval_batch_size == 0 or TASK_SIZE % eval_batch_size == 0`

---

## 7. SG (Syntactic Generalization)

### SG_eval_step (`train.py:1056-1096`)
- terminal/pause 模型 (`grammar_type[:8]=="terminal" or ispause`, line 1067): 前向 `model_forward`，`score = sum(ce_loss[0] * tag[0])` (line 1071) — tag mask 选择贡献 surprisal 的位置
- tree/tg 模型: `word_sync_beam_search` (line 1077-1090)，`beam_size=samples_per_sent=300`，`nc=max(int(sg_nc_ratio*term_len), 5)`，`max_len=max(3*term_len, 10)`，`pc=sg_pc=3`

### prep_examples pause 处理 (`downstream.py:1575-1609`)
```python
if self.ispause:
    if is_gpt2:
        sent["tag"][0] = [0] + sent["tag"][0]   # BOS 前置
    sent["tag"][0] = pause_input_ids(sent["tag"][0], pause_token_id=None, pause_num=gtype)[1:]
    sent_paused = pause_input_ids(sent["input_ids"][0], self.pause_token_id, pause_num=gtype)
    sent["input_ids"] = sent_paused.unsqueeze(0)
```
- tag (0/1 mask) 用 `pause_token_id=None` (广播模式) 展开，每个 real token 的 tag 值重复到其 pause 槽
- 前导 BOS tag 项 `[1:]` 去掉
- input_ids 用实际 pause token 展开
- `score = sum(ce_loss * tag)` 只在 real-token 位置求和 (pause 位置 tag 值 = 前一 real 的值，但模型预测 pause 容易，贡献可忽略；tag 展开保证对齐)

### SyntacticGeneralizationMetric (`downstream.py:1466-1519`)
6 类 (`test_suite_dict`, lines 1455-1464):
1. **Agreement**: number_orc, number_prep, number_src
2. **Center_Embedding**: center_embed, center_embed_mod
3. **Garden_Path_Effects**: mvrr, mvrr_mod, npz_ambig, npz_ambig_mod, npz_obj, npz_obj_mod
4. **Gross_Syntactic_Expectation**: subordination, subordination_orc-orc, subordination_pp-pp, subordination_src-src
5. **Licensing**: npi_orc_any/ever, npi_src_any/ever, reflexive_orc/prep/src_fem/masc
6. **Long_Distance_Dependencies**: fgd_subject/object/pp, fgd-embed3/4, fgd_hierarchy, cleft, cleft_modifier

- `update` (line 1486): 从 `formula_dict` 查公式，代入 condition surprisal，求值 (如 `[ (%plaus%) ] < [ (%implaus%) ]`)，布尔结果加入类别列表
- `compute` (line 1506): 返回 per-category 准确率 + `avg` (排除 `nn-nv-rpl`)

### 关键数值
- 32 tasks in `task_list`，6 categories
- `samples_per_sent=300` (beam_size 默认)，`sg_nc_ratio=1.0`，`sg_pc=3`

---

## 8. xsum (rouge)

### summarization_eval_step (`train.py:1157-1191`)
- **terminal 模型** (line 1160-1168): `self.dist_model.module.generate(input_ids, max_steps=MAX_SUMMARY_LENGTH=150, beam_size=6)`，标准自回归生成，`predictions[:, 0, :]` 取最佳 beam
- **非 terminal 模型** (line 1169-1186): `word_sync_beam_search`，`beam_size=6`，`max_length=150`，`max_word_steps=75`，预测 = `predictions[0]["input_ids"]`
  - **Pause 修复** (line 1182-1184): 若 `grammar_type[:5]=="pause"`:
    ```python
    p, q = self.cfg.model.pause_spec
    predictions = extract_real_tokens(predictions, p, q, skip_first=True)
    ```
    去除生成的 pause token，`skip_first=True` 丢前导 BOS
  - 然后 `predictions = vocab.convert_treenpy_to_terminal(predictions)` (line 1185) 转 terminal token

### RougeMetric
- `update` (`downstream.py:1057-1066`): 打印 `<New Passage>: {passage} {prediction}` (line 1062)，存 prediction 和 reference token id
- `compute` (`downstream.py:1072-1088`): HuggingFace `evaluate.load('rouge')`，`rouge_types=['rouge1','rouge2','rougeL']`，`use_stemmer=True`，`use_aggregator=True`，返回 rouge1/rouge2/rougeL + `R-AVG` (三者均值)

### extract_real_tokens (`olmo/data/util.py:456-487`)
反转 `pause_input_ids` (pause_token_id=None 版)。遍历 real-token 索引 `j=0,1,2,...`，emit `paused[j + (j//q)*p]`。`skip_first=True` 从 `j=1` 开始 (丢 BOS real token)。

### 关键数值
- `MAX_SUMMARY_LENGTH=150`，`beam_size=6`
- xsum test set: 11,333 句 (慢的 beam-search eval ~6h on 4 GPU)
- xsum_finetune: 3 epoch 训练 + eval

---

## 9. BoolQ Eval Bug (已修复)

### 根因
bug 在 `ICLMultiChoiceTaskDataset.prep_examples` 的 pause 路径。BoolQ finetune eval 用 `split="train"` (finetune task 构造设定)，所以 `query[:-1]` 截断 (line 602) **被跳过** (条件是 `if self.split != "train"`)。但 `actual_ctx_len` (line 606) 仍按 `len(query) - len(continuation) + 1` 计算，带 `+1` 偏移 (本为补偿 `[:-1]` 截断，但该截断未运行)。

pause 路径旧代码用 `actual_ctx_len` (对 train split = `len(ctx)+1`) 作为 `ctx_real`。BoolQ continuation 是单 token (`" yes"`/`" no"`)，`full_query = ctx + continuation` 的 `len = len(ctx)+1`。buggy `ctx_real = actual_ctx_len = len(ctx)+1 = len(full_query)`，使展开 split 点 `pause_expanded_len(ctx_real, p, q)` 落在展开 query 末尾或之后 → continuation 切片 `query[split:end]` 为空 `[]`。

空 continuation → 两选项 ("yes"/"no") 都得 `log_likelihood = 0.0` (零元素求和) → `torch.argmax` 平局返回 0 ("yes") → BoolQ 多数类 ~62% "yes" → 始终预测 "yes" → 准确率 **0.6217**。

### 修复 (`downstream.py:654`)
```python
ctx_real = len(full_query) - len(continuation)
```
正确计算 `full_query = ctx + continuation` 中真实 ctx token 数，无 `+1` 偏移。注释 (lines 646-653) 说明为何不能用 `actual_ctx_len`。同时展开 `full_query` (未截断 ctx+cont) 而非 trimmed `query`，使 continuation token 保留在展开序列中。`actual_ctx_len = split` (line 661) 覆盖为正确展开 split 点。

### 验证
- pause1 boolq: 0.6217 → 0.6780
- 未微调 pause1 boolq: 0.6217 → 0.4413 (eval 不再退化)
- terminal boolq: 0.6835 (从未受影响)
- 所有 pause 模型 boolq 重跑后均在 0.67-0.69

### Terminal (非 pause) 不受影响
pause 路径 (line 620) 仅当 `self.ispause` 为真时执行。terminal `pause_spec=(0,1)`，`ispause=0`，走标准路径 `actual_ctx_len = len(query) - len(continuation) + 1` (正确)。

---

## 10. 配置生成 (init_cfg_and_sbatch.py)

### Evaltasks dict (`scripts/init_cfg_and_sbatch.py:182-199`)
```python
"pretrain": [EvaluatorConfig(label="TG-ppl-validation")]                    # lm
"docppl": [EvaluatorConfig(label="tg_approx_doc", type=tg_doc, device_eval_batch_size=60)]
"xsum_test"/"xsum_finetune": [EvaluatorConfig(label="xsum", type=rouge)]
"SG": [EvaluatorConfig(label="syntactic_generalization", type=downstream)]
"blimp": [EvaluatorConfig(label="BLiMP", type=downstream, device_eval_batch_size=100)]
"boolq"/"cb"/...: [EvaluatorConfig(label="{task}", type=downstream)]
"hellaswag"/"winogrande": [EvaluatorConfig(label="{task}", type=downstream, device_eval_batch_size=5)]
```

### generate_config (`scripts/init_cfg_and_sbatch.py:258-312`)
加载目标 checkpoint 的 `config.yaml` 作为**模型结构基线** → 应用
`Models[modelname]` + `train_params[task]` 覆盖 → 从 `INPUTFORMAT` 解析
`input_format`。随后重建 workspace、tokenizer、训练占位数据、load path、
evaluator 与状态控制；不继承 checkpoint 的数据 shards、`try_load_latest_save`、
`stop_at` 或 optimizer/trainer state。

这避免了 1B checkpoint 的 FineWeb-Edu shard 列表污染 BBC News 测评，也使仅有
`model.pt` 的 checkpoint（例如已知的 Tree-Shuffle）能够进行 test-only 评测。

### INPUTFORMAT 解析 (lines 70-77, 276-280)
```python
"terminal": ["terminal", "pause1", "pause1_label", "pause1/2", "pause1/2_label", "pause2", "pause3", "pause1/3", "pause1/4"]
"tree": ["tree", "tree_shuffle", "tree_shuffle_mask"]
"tg": ["tg", "mixing", "tgnomask", "tgnomask_aug", "tgtree"]
"tree_compress", "tree_noont", "tree_triplecnt": 各自独立
```
决定数据路径: `train_path = {workspace}/dataset/bbc-news/{input_format}/train.npy` (line 281)。
**注意**: 新 grammar type 必须加入对应 INPUTFORMAT 列表，否则 `input_format=None` → 数据路径 `bbc-news/None` 失败。

### docppl/terminal-input 分支
当 `task=="docppl" and input_format=="terminal"`（含 pause）时，创建唯一的 `terminal_doc_ppl` / `terminal_doc` evaluator，batch=1。`pretrain*` 仍保留旧的 `TG-ppl-validation` lm 监控，不应当作上述正式比较指标。

### docppl 非 terminal 分支 (lines 292-293)
`task=="docppl"` 但 `input_format != "terminal"`: label 改 `tg_approx_doc` (tg) 或 `txl_approx_doc` (tree)，type `tg_doc`，batch 60。

### blimp batch size 分支 (lines 294-298)
```python
if grammar_type[:8]=="terminal" or grammar_type[:5]=="pause":
    device_eval_batch_size = 100
else:
    device_eval_batch_size = 150
```

### Tree-Shuffle terminal syntax protocol

Tree-Shuffle 的 SG/BLiMP 若要与普通 terminal 模型比较，必须显式设置：

```yaml
structure_mode: terminal
tree_eval_type: terminal
beam_search: false
```

代码分支也已硬编码支持该协议：SG 在 `structure_mode=terminal` 时走
teacher-forced causal logits，不调用 `word_sync_beam_search`；BLiMP 通过
`force_terminal` 使用 terminal 数据且 `SENT_SIZE=1`，不加载 gold300。

最后 `cfg.evaluators = Evaltasks[task]` (line 306)。

---

## 关键数值速查

| 项目 | 值 |
|------|-----|
| BLiMP pair_per_task | 1000 |
| BLiMP SENT_SIZE | 1 (terminal/pause) / 300 (tree/tg) |
| BLiMP TASK_SIZE | 2000 / 600000 |
| BLiMP task 数 | 67 |
| BLiMP category 数 | 12 |
| SG task 数 | 32 |
| SG category 数 | 6 |
| SG samples_per_sent (beam_size) | 300 |
| SG sg_nc_ratio | 1.0 |
| SG sg_pc | 3 |
| docppl tg_doc SENT_SIZE | 300 |
| docppl tg_doc batch | 60 |
| terminal_doc batch / workers | 1 / 0 |
| terminal_doc 测试集 | 148,836 句 / 4,966 文档 / 3,284,061 tokens |
| xsum MAX_SUMMARY_LENGTH | 150 |
| xsum beam_size | 6 |
| xsum test set | 11,332 句 |
| BoolQ shots_num 默认 | 3 |
| BoolQ shots_list | [0,1,7,11,3,4,5] |
| BoolQ metric_type | acc |
| BoolQ split 默认 | val |
| BoolQ 多数类准确率 | ~0.6217 (bug 退化值) |
| finetune shots_num | 0 (强制) |
| test_only_params | eval_on_load=True, eval_no_save=True, max_duration=0 |
| finetune_params | reset_optimizer_state=True, reset_trainer_state=True, eval_interval=1000000 |
| stop_at 钳制 | min(cfg.stop_at, max_steps) (line 1476) |

## 运行方式

1. 用 `scripts/init_cfg_and_sbatch.py` 生成 config + sbatch 脚本:
   ```python
   Device = "RTX3090"
   modelnames = ["pause1"]  # 或其他
   tasks = ["docppl", "SG", "blimp", "boolq", "xsum_finetune"]  # 
   ```
2. sbatch 脚本需注入 `conda activate LLM` (在 `cd ${workspace}` 后)
3. `sbatch run_folder/{model}/{model}_{task}_test.sh`
4. 结果在 `slurm-{jobid}.out`:
   - terminal/pause docppl: `eval/terminal_doc_ppl/Perplexity=`
   - SG: `avg=`
   - BLiMP: `overall/overall=`
   - boolq: `eval/downstream/boolq_acc__=`
   - xsum: `rouge1=` / `rouge2=` / `rougeL=`

## 已知坑

1. **新 grammar type 必须加入 INPUTFORMAT** — 否则数据路径解析为 None
2. **不要混用旧 lm docppl** — 它的 pause CE/PPL 被插入位置稀释；报告 terminal/pause 对比时只使用 `terminal_doc_ppl`
