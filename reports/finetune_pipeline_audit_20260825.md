# BoolQ / XSum（及 SuperGLUE 家族）Finetune 管线审计与开源数据清单

> [!WARNING]
> 本文是 2026-08-25 的数据与管线快照。“当前标准协议”仅指当时的 TreeReg campaign，
> 不是此后 Pause/Pushdown 的统一协议。Pause XSum v1 后来确认存在 label mask、pause
> convention 和生成相位错误，旧结果已作废；见
> [`pause_xsum_pipeline_audit_20260829.md`](pause_xsum_pipeline_audit_20260829.md) 和
> [`../REPOSITORY_CLEANUP_MEMORY.md`](../REPOSITORY_CLEANUP_MEMORY.md) M-02。

**日期**: 2026-08-25
**范围**: XSum 摘要 finetune+评测、BoolQ finetune+ICL 评测、SuperGLUE 其余任务（cb/copa/multirc/record/rte/wic/wsc）
**目的**: 为开源整理管线文档，并摘录各任务训练/评测所需的具体数据集
**行号基准**: `olmo/eval/downstream.py` 当前工作区状态（5813 行）；`olmo/data/__init__.py:281-314`

---

## 0. 结论速览

| 维度 | XSum | BoolQ | SuperGLUE 其余 |
|---|---|---|---|
| 任务性质 | SFT 摘要生成 + ROUGE 生成式评测 | SFT finetune + 零样本 val 评测；亦可 3-shot ICL 评测 | 与 BoolQ 同构（ICLMetric 多选） |
| 训练数据 | `dataset/Xsum/xsum_train.txt`(树) + `xsum_train_summary.txt`(树) + `gold_train_summary.jsonl`，运行时经 `save_ids.json` 过滤 → **17,904 条**（已分离至 `train_filtered/`） | `dataset/SuperGLUE/BoolQ/{train}.jsonl`（9,427）+ `{split}_{passage,question}.txt` PTB 解析树 | 各任务 `./dataset/SuperGLUE/{Task}/` 同构 |
| 评测数据 | `xsum_test.txt` + `gold_test_summary.jsonl`（11,333 条）；另有 validation 11,327 | `val.jsonl`（3,270）；`test.jsonl`（3,244，官方无标签） | 各任务 `val.jsonl` |
| 实现入口 | `XsumDataset` downstream.py:1156 | `BoolQ` downstream.py:2838 | downstream.py:2935-3473 |
| 配置 | `evaluation/xsum_configs/*.yaml` | `evaluation/boolq/*.yaml` | `evaluation/{RTE,CB}/` 等 |
| 开源硬缺口 | **无脚本**从 HF 原始数据生成 `xsum_*.txt` 树文件 | **无脚本**生成 `*_passage.txt` 等解析树 sidecar | 同左 |

两个任务的 finetune 共享同一机制：`TrainConfig.finetune_task` 一旦设置，
`build_train_dataloader` 完全忽略 memmap 预训练数据路径
（`olmo/data/__init__.py:289-314`），改为实例化
`label_to_task_map[finetune_task]` 数据集类、以 `split="train"` 构造 SFT 样本；
若任务属于 `Super_GLUE` 字典则把 collator 换成数据集自带的 `collate_fn`
（`__init__.py:313-314`）。损失为普通 LM 交叉熵但只在真实 loss token 上
归一化（`olmo/train.py:918-933`），优化器与 trainer 状态被重置
（`reset_optimizer_state/reset_trainer_state=true`）。

最终任务注册表：`label_to_task_map = {**TG_task_map, **label_to_task_map,
**label_to_task_map_new, **Super_GLUE}`（downstream.py:5808-5813）。
`"xsum"` 来自 `TG_task_map`（:5019），SuperGLUE 任务来自 `Super_GLUE` 字典
（:5023-5035）。

---

## 1. XSum 管线

### 1.1 数据来源与准备链

```
EdinburghNLP/xsum (HF hub) ──inspect_Xsum.py──▶ gold_{validation,test}_summary.jsonl
lighteval/summarization "xsum" train ──filter_xsum_train_id.py──▶ save_ids.json
(外部解析管线, 不在 repo) ──▶ xsum_{train,validation,test}.txt + xsum_train_summary.txt + gold_train_summary.jsonl
        │
XsumDataset 运行时过滤 (id ∈ save_ids) ──▶ 真正的训练集 17,904 条
        │
separate_xsum_train_subset.py (新增) ──▶ dataset/Xsum/train_filtered/* （物理分离产物）
```

各环节细节：

| 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `datatools/inspect_Xsum.py:63-72` | HF `EdinburghNLP/xsum` 的 validation/test | `gold_{validation,test}_summary.jsonl`，每行 `{"summary","id"}` | 跳过空 document → 11,333 test / 11,327 validation（非官方全量）。注意 `prepare_gold_summary()` 被注释掉，需手工调用 |
| `datatools/filter_xsum_train_id.py:6-21` | HF `lighteval/summarization` xsum train + `EdinburghNLP/xsum` train | `save_ids.json`（17,907 个 id，去重后 17,904） | 用 strip 后的原文精确字符串匹配映射 lighteval 文章 → 原始 id；任何归一化不一致会 KeyError |
| `datatools/separate_xsum_train_subset.py`（本次新增） | 四个平行文件 | `train_filtered/` 三件套 + MANIFEST | 复刻 `XsumDataset.__init__` 的运行时过滤（downstream.py:1199-1214），离线物化 |

**磁盘现状（`dataset/Xsum/`）**：

| 文件 | 大小 | 行数 | sha256（前 8 位） |
|---|---:|---:|---|
| `xsum_train.txt` | 1.24 GB | 204,017 | `15ba7397` ✅ 与 campaign pin 一致 |
| `xsum_train_summary.txt` | 69.3 MB | 204,017 | `b0eb7e73` |
| `gold_train_summary.jsonl` | 31.1 MB | 204,017 | `89a2779d` |
| `save_ids.json` | 210 KB | 17,907 条 id | `5c72e9a7` |
| `xsum_test.txt` | 71.6 MB | 11,333 | `48ac8f6e`（campaign pin） |
| `xsum_validation.txt` | 68.9 MB | 11,327 | — |
| `gold_test_summary.jsonl` | 1.8 MB | 11,333 | — |
| `gold_validation_summary.jsonl` | 1.8 MB | 11,327 | — |
| **`train_filtered/`（新增）** | 44 MB 合计 | **17,904** | 见 §1.5 |

三个平行文件行数严格对齐（204,017 = 204,017 = 204,017），zip 无静默截断；
2026-08-15 重生成的 `xsum_train.txt` 与 Sep 2025 其余文件经哈希验证一致。

### 1.2 数据格式与序列组装

- 文章与摘要均以**句法树文本**存储：PTB 风格括号树，换行为 `(Ċ Ċ)` 对；
  加载时由 `convert_TG_format` / `pformat_flat`（`olmo/data/util.py:151-199`）
  展平为 `<(S>...<S)>` 内联 TG 括号标记。
- 固定指令 prompt（downstream.py:1177）：
  ```
  " \n<(S><(VP> Summarize<(NP> the above article<NP)><(PP> in<(NP> 1 sentence<NP)><PP)><VP)> .<S)> \n"
  ```
- `__getitem__` 组装（downstream.py:1238-1317）：
  `[bos] + passage[:ctx-len(summary)-len(prompt)-2] + prompt [+ summary + eos]`
  - loss token = summary + eos，`label_mask` 尾部对齐；
  - pause 模型按 (p,q) 把 ctx 预算缩为 `q/(q+p)`（:1182-1187）；
  - grammar 相关转换在 :1274-1305（terminal→terminal ids；tree*→tree 数组；
    pause*→pause 展开；其余走外部 `generate_TG_attention_bias`）。
- 左 padding collator（:1171-1172）。

### 1.3 Finetune 配置

五个手写配置（`evaluation/xsum_configs/`）共享 OLMo-300M 骨干、ctx 2048、
global batch 40（micro 10）、3 epoch、amp_bf16 DDP、`finetune_task: xsum`、
evaluator `[label: xsum, type: rouge]`、`device_eval_batch_size: 1`；
`load_path` 全为 null，由启动脚本 CLI 注入。差异：

| 配置 | grammar | LR | warmup/min_lr | 备注 |
|---|---|---|---|---|
| terminal.yaml | terminal | **5.0e-5** | 50 / 1e-5 | |
| TG.yaml | tg | 6e-5 | 100 / 1e-6 | |
| tree.yaml | tree | 6e-5 | 50 / 1e-5 | |
| nomask.yaml | tgnomask | 6e-5 | 100 / 1e-6 | flex_attention=false |
| mix_nomask.yaml | mixing（tg×6 + tgnomask×6 heads） | 6e-5 | 100 / 1e-6 | flex_attention=false |

程序化生成器口径（`scripts/init_cfg_and_sbatch.py:155-156`）：统一 3ep /
LR 6e-5 / warmup 100 / min_lr 1e-6 / batch 40。多 seed campaign
（treereg_layer9_multiseed_20260824）即采用此口径并 pin 了输入哈希。

### 1.4 评测（ROUGE）

分发：`EvaluatorType.rouge` → `Trainer.summarization_eval_step`
（`olmo/train.py:1625-1703`），按 grammar 分三种生成方式：

| grammar | 生成方式 |
|---|---|
| terminal / pushdown / treereg | 自回归 `generate(max_steps=150, beam_size=6)` |
| pause1_label | 确定性 `pause_label_generate` |
| tg / tgtree/tree / tgnomask / mixing / pause* | `word_sync_beam_search`（beam 6、max_word_steps=75、max_length=150；pause 在 NT range 屏蔽 NT 发射），pause 再经 `extract_real_tokens` 去 pause |

打分：`RougeMetric`（downstream.py:1325-1379）— HF `evaluate.load('rouge')`,
R1/R2/RL + stemmer + aggregator，自定义 **R-AVG = 三者均值**；预测经
repo 的 TG-GPT2 tokenizer 往返解码后与纯文本 gold 比较。测试集 11,333 条，
beam search eval 约 6 h / 4 GPU。

### 1.5 训练子集分离产物（本次完成）

`python datatools/separate_xsum_train_subset.py` → `dataset/Xsum/train_filtered/`：

| 文件 | 行数 | sha256 |
|---|---:|---|
| `xsum_train.txt` | 17,904 | `3234a829…` |
| `xsum_train_summary.txt` | 17,904 | `fa5d2c11…` |
| `gold_train_summary.jsonl` | 17,904 | `9bd9a91e…` |
| `MANIFEST.json` | — | 含全部输入/输出校验和 |

开源时直接发布该目录即可消除对 `lighteval/summarization` 的依赖；
`XsumDataset` 若改用 `train_filtered/` 可跳过运行时过滤（需小改或软链覆盖）。

---

## 2. BoolQ 管线

### 2.1 双重身份

同一 `BoolQ` 类（downstream.py:2838-2931）承担两种用法：

1. **SFT finetune**（论文协议）：`finetune_task: boolq` → 训练集以
   `split="train"` 构造，每条样本只有 **gold continuation**
   `[[" yes"," no"][label]]`（:2918-2919），做 gold-answer LM loss；
   评测强制 `shots_num=0`（`olmo/eval/__init__.py:146-147`）→ 零样本 val acc。
2. **few-shot ICL 评测**（A800 1B 协议）：不设 finetune_task 时默认
   `shots_num=3`，exemplar 取自 train split 的 `shots_list=[0,1,7,11,3,4,5]`
   前 3 条（:2853, :2901-2905）。

### 2.2 数据与 prompt 格式

- 来源：**本地 SuperGLUE BoolQ 官方 jsonl**（非 HF hub 在线加载）：
  `./dataset/SuperGLUE/BoolQ/{train,val,test}.jsonl`
  = 9,427 / 3,270 / 3,244 行（与官方一致）。
- 每条记录 `{"question", "passage", "label": true/false}`；
  `load_local_datasets`（:2883-2899）用 `{split}_passage.txt` /
  `{split}_question.txt`（PTB 解析树，如 `(NP (NP (NN Ethanol)...)...`）
  **按行覆写** passage/question 字段，经 `convert_TG_format` 转内联 TG 括号标记。
  ⚠️ 即使 terminal/pause 模型也依赖这些解析树文件（转换后 NT 在
  `convert_grammar_input` 中才剥离）——它们是所有 grammar 的必需输入。
- prompt（:2907-2908）：
  ```
  {passage} \n<(SQ><(NP> Question<NP)> :{question} ?<SQ)> \n<(S><(NP> The answer<NP)><(VP> is<(NP>
  ```
- continuations `" yes"` / `" no"`；`doc_to_label`: True→0(yes), False→1(no)。
- 打分：`ICLMetric`（:107-370）对每个候选求 continuation 总 log-prob，
  argmax 判定，metric_type="acc"；结果键 `eval/downstream/boolq_acc__`。
- PMI/domain-conditional 分支存在但 `dc_input_ids` 从未在前向中传递——死代码。

### 2.3 Finetune 配置与启动

`evaluation/boolq/` 五个 yaml 全是 finetune 配置：OLMo-300M、ctx 2048、
batch 40、5ep、lr 3e-4、warmup 100、min_lr 1e-6、amp_bf16 DDP、
`finetune_task: boolq`、evaluator `[label: boolq, type: downstream]`、
`stop_at: 10000`、seed 6198。差异：

| yaml/sh | grammar | micro-batch | load_path（CLI 注入） |
|---|---|---|---|
| tg.yaml/tg.sh | tg | 3 | `saved_models/TG_test/step55457-unsharded` |
| tree.yaml/tree.sh | tree | 10 | `saved_models/Tree_test/step49440-unsharded` |
| tgnomask.yaml/tgnomask.sh | tgnomask | 3 | `saved_models/nomask_test/step55853-unsharded` |
| mix.yaml/test_mix.sh | mixing | 3 | `saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded` |
| terminal.yaml/test_term.sh | terminal（缺省） | 10 | `saved_models/Terminal-lr005-bs144/step34115-unsharded` |

启动模板（*.sh）：sbatch `-N1 -n4 -c1 --gres=gpu:4 --mem-per-cpu=1` +
`HF_ENDPOINT=https://hf-mirror.com` + torchrun 4 GPU + `--load_path=`。

程序化生成器（init_cfg_and_sbatch.py:157-158）：boolq 5ep / lr 3e-4 / batch40。
RTE 手写 yaml 则是 5ep / lr **1e-4**（`evaluation/RTE/terminal.yaml`）——
手写 yaml 与生成器超参不同套，使用时须分清。

**多 seed campaign（TreeReg 当时采用的协议，不是全仓库统一标准，`artifacts/experiment/treereg_layer9_multiseed_20260824/`）**：

- 训练：`prepare_campaign.py` 以 checkpoint config 为基线生成 5 seed × {xsum, boolq}
  配置（fp32 DDP 4GPU、batch 40、XSum 3ep/6e-5、BoolQ 5ep/3e-4、
  `evaluators=[]`、wandb disabled）；`run_finetune.sh` 先做 **sha256 校验**
  再 torchrun，产物 checkpoint 记入 `final_checkpoint.txt` 并删 optim/train state。
- 评测：`run_eval.sh` 独立 test_only 运行
  （`--eval_on_load --eval_no_save --max_duration=0 --stop_at=0`），
  XSum evaluator `device_eval_batch_size: 1`；BoolQ 10（OOM 后降 microbatch=1，
  梯度累积保 global batch）。
- 结果（TreeReg-layer9）：XSum R-AVG 17.59±0.10、BoolQ 63.54±0.56
  （EXPERIMENT_REPRODUCTION_RECORD.md §3）。

### 2.4 已知修复过的 bug（背景）

pause 路径曾因 `ctx_real` 带 `+1` 偏移导致空 continuation → 两选项 log-prob
同为 0 → 永远预测 "yes"（准确率恒 62.17% 多数类）。修复于
downstream.py:654（`ctx_real = len(full_query) - len(continuation)`），
修复后 pause1 boolq 0.6217→0.6780（Evaluation.md §9）。

---

## 3. SuperGLUE 其余任务家族

全部走同一 ICLMultiChoiceTaskDataset 机制；数据在
`./dataset/SuperGLUE/{Task}/`，均为「官方 jsonl + 解析树 txt sidecar」双件套，
`load_local_datasets` 按 key 覆写后 `convert_TG_format` 转换：

| 任务 | 类（行号） | txt sidecar 字段 | prompt 要点 | continuations | 默认 split/shots | 实现完整度 |
|---|---|---|---|---|---|---|
| boolq | BoolQ (:2838) | passage, question | 见 §2.2 | yes/no | val / 3-shot | ✅ |
| cb | CommitmentBank (:2935) | premise, hypothesis | premise + Question:{hyp}. True/False or Neither? | True/False/Neither | val | ✅ |
| copa | COPA (:3010) | premise, choice1, choice2 | premise 截尾 + `<(SBAR>` + because/therefore；choice 小写化截尾 | choice1/choice2 | val | ✅ |
| multirc | MultiRC (:3104) | （TODO 空） | `doc_to_text` 抛 NotImplementedError | — | — | ❌ 存根 |
| record | ReCoRD (:3182) | text, query（有转换） | `doc_to_text` 抛 NotImplementedError | — | — | ❌ 存根 |
| rte | RTE (:3254) | premise, hypothesis | premise + True or False? | True/False | val / shots_num=0 | ✅ |
| wic | WiC (:3329) | sentence1, sentence2 | 双句 + word 同义问句 | yes/no | val | ✅ |
| wsc | WSC (:3402) | text | 代词指代问句（span1/span2） | yes/no | val | ✅ |

补充：
- `finetune_params` 生成器里 cb/copa/multirc/record/rte/wic/wsc 都有超参条目
  （cb/rte/wsc 3ep、copa 10ep/lr5e-4/wd0.1、wic 1ep；wic/wsc 键重复定义，无害），
  但 **multirc/record 因类存根实际不可跑**。
- 磁盘另有 `AX-b/`、`AX-g/`（SuperGLUE AX 套件）与 `submit/*.jsonl`
  （提交格式），代码零引用——开源可剔除或注明未用。
- A800 1B 侧另有 OE-eval 版 BoolQ（`boolq_mc_5shot` 等 oe-eval requests，
  `olmo_data/oe_eval_tasks/boolq/`），当前配置未引用。

---

## 4. 开源数据清单（核心交付）

### 4.1 必备数据资产总表

| # | 资产 | 路径 / 来源 | 用途 | 发布形式建议 |
|---|---|---|---|---|
| 1 | TG-GPT2 tokenizer | `dataset/bbc-news/TG_GPT2_tokenizer.json`（4.07 MB） | 所有任务的编码/解码、vocab 50,320 | 直接发布 |
| 2 | XSum 树格式文章 | `dataset/Xsum/xsum_{train,test,validation}.txt`（~1.39 GB） | finetune 训练 + 评测输入 | 直接发布 + sha256（源自 EdinburghNLP/xsum + 外部解析，见缺口 G1） |
| 3 | XSum 摘要 | `dataset/Xsum/xsum_train_summary.txt` + `gold_{validation,test}_summary.jsonl` | SFT 目标 + ROUGE 参考 | 同上 |
| 4 | XSum 去污染 id 表 | `dataset/Xsum/save_ids.json` | 训练集过滤（17,904） | 直接发布（源自 lighteval/summarization，见缺口 G1） |
| 5 | **XSum 真实训练子集** | `dataset/Xsum/train_filtered/`（17,904 条三件套 + MANIFEST） | 替代 #2+#4 即可复现训练 | **优先发布**（最小闭环） |
| 6 | BoolQ 官方数据 | `dataset/SuperGLUE/BoolQ/{train,val,test}.jsonl`（9,427/3,270/3,244） | finetune + ICL 评测 | 发布 + sha256（官方 SuperGLUE 原样） |
| 7 | BoolQ 解析树 sidecar | `dataset/SuperGLUE/BoolQ/{split}_{passage,question}.txt` | **所有 grammar** 的 prompt 构造 | 发布 + 补再生脚本（缺口 G2） |
| 8 | SuperGLUE 其余任务 | `dataset/SuperGLUE/{CB,COPA,RTE,WiC,WSC}/` 双件套 | 同构 finetune/ICL | 同上；MultiRC/ReCoRD 存根不可跑，注明即可 |
| 9 | 预训练 checkpoint 清单 | `EXPERIMENT_REPRODUCTION_RECORD.md` §2（terminal/tree/tg/…×100M/500M/1B） | finetune 起点 | 发布 ckpt 或仅登记 sha256 |
| 10 | BBC News docppl 数据 | `dataset/bbc-news/terminal/{test.npy,*_index.npy}` 等 | 其他指标（非本审计重点） | 已有独立管线，另行处理 |

### 4.2 校验和锚点（来自 campaign 脚本 + 本次 MANIFEST）

```
xsum_train.txt            15ba739782e8829b2c6d15ccb71898156e02798dc20b7a614d91702213f2c5ad  (204,017)
xsum_test.txt             48ac8f6e4ac2204b47dadfd5f7a91f91959cf2a5e06710d696c2dae043541d57  (11,333)
xsum_train_summary.txt    b0eb7e73360ce150b93115df044727bdae5bb5c5827e0b6a814856a630e12337  (204,017)
gold_train_summary.jsonl  89a2779dca51dc95e51473a308d1be664bf24a6e0ba561fc76d1d3f9232e8580  (204,017)
save_ids.json             5c72e9a790dd252f28686c1fc179f1e77d698f2415e29eea9bbd11a2718220ad
BoolQ train.jsonl         5a0cc1d6cb971a7a177b74bde27b8355de4b0f0e4d86d0a8435ec92cfeb63ba6  (9,427)
BoolQ val.jsonl           0c86a5045886e5795fe9052003873f7d94b88ed3028a33007c51d99e44fd66d9  (3,270)

train_filtered/xsum_train.txt           3234a829f1c591c48f5e593b8e13669ac7f5843f35ce230f72a70e99d93fec93  (17,904)
train_filtered/xsum_train_summary.txt   fa5d2c11c1b56a5ce7d0644687bb312cef4b23b37d60a4d5ab1860fe4105c9e5  (17,904)
train_filtered/gold_train_summary.jsonl 9bd9a91ec54a7103f825728dd9374e4ea027259863fbf467541fe71ee9a8e966  (17,904)
```

### 4.3 许可与再分发注意

- `EdinburghNLP/xsum`：CC BY-SA 3.0 + XSum 收集条款；派生物（解析树版本）
  通常允许再分发但需署名并保留许可声明。
- `lighteval/summarization`：仅用于生成 id 列表；发布 `train_filtered/` 后
  该依赖可完全移除。
- SuperGLUE：官方要求申请下载；**解析树 sidecar 是派生物**，稳妥做法是
  让用户自行下载官方 jsonl，仓库只发布解析 sidecar + 再生脚本。

### 4.4 缺口清单（开源前必须补齐）

| 编号 | 缺口 | 影响 | 建议 |
|---|---|---|---|
| G1 | **无脚本**从 HF 原始 XSum 生成 `xsum_*.txt` 树文件（外部 benepar 解析管线未入库；`xsum_train.txt` 曾于 08-15 重生成，靠 sha256 pin 兜底） | 无法从原始数据复现训练语料 | 移植 datatools 的 benepar 管线写一个 `prepare_xsum_trees.py`；或直接只发布 `train_filtered/` 产物 + 测试/验证集 |
| G2 | SuperGLUE `*_passage.txt` 等解析树 sidecar 同样无生成脚本 | 用户拿到官方 jsonl 也无法重建 prompt 所需树 | 复用 `datatools/setup_parse_deps.py` + benepar 写 `parse_superglue.py` |
| G3 | MultiRC / ReCoRD 类存根（doc_to_text NotImplementedError） | 生成器里有 finetune 超参但实际不可跑 | 实现或删除相应条目 |
| G4 | 相对路径 `dataset_path="./dataset/Xsum"` 等依赖 CWD=workspace root | 异地构建易踩坑 | 启动脚本已 `cd ${workspace}`，文档注明即可 |
| G5 | `xsum_term.sh` 引用 `/public/home/...` 陈旧路径（本机不存在） | 该脚本在本机必失败 | 改 `${HOME}/...` |
| G6 | run_name 与实际超参漂移（如 `Tree_finetune_xsum_lr1e-4_warmup50_2ep` 实际 6e-5/3ep；test_term/test_mix 共用 `Terminal_finetune_boolq` 名） | 归档混淆 | 重命名历史 run 或在 README 登记对照表 |
| G7 | `evaluation/boolq/*.yaml` 的 `data.paths` 指向不存在的 `bbc-news/tg/dev.npy` | 目前无害（finetune 忽略 paths），但误导读者 | 清理字段 |

---

## 5. 快速复现命令（供 README 引用）

```bash
# XSum finetune（以 terminal 100M 为例）
sbatch evaluation/eval_scripts/xsum_term.sh          # torchrun scripts/train.py evaluation/xsum_configs/terminal.yaml --load_path=<ckpt>

# BoolQ finetune
sbatch evaluation/boolq/test_term.sh                 # evaluation/boolq/terminal.yaml --load_path=<ckpt>

# TreeReg 多 seed campaign（2026-08-24 历史协议）
python artifacts/experiment/treereg_layer9_multiseed_20260824/prepare_campaign.py
sbatch artifacts/experiment/treereg_layer9_multiseed_20260824/finetune_array.sbatch
sbatch artifacts/experiment/treereg_layer9_multiseed_20260824/eval_array.sbatch
python scripts/collect_finetune_campaign.py          # 汇总 ROUGE/acc

# 分离真实 XSum 训练子集（开源用）
python datatools/separate_xsum_train_subset.py --xsum_dir dataset/Xsum
```

结果读取键：XSum → slurm 日志 `R-AVG=`；BoolQ → `eval/downstream/boolq_acc__=`。

---

## 附：本次审计新增/改动文件

- 新增 `datatools/separate_xsum_train_subset.py`（离线物化运行时过滤）
- 生成 `dataset/Xsum/train_filtered/`（17,904 条 + MANIFEST.json）
