# 论文实验复现登记表

更新：2026-08-28。本文是本仓库的唯一人工维护登记表，用于把论文中的
模型、可加载 checkpoint、论文报告值和本次重跑值放在同一张记录中。

来源：

- 论文：`14901_A_Scaled_Up_Empirical_St.pdf`，Table 3/4/6/7。
- checkpoint：本机 `saved_models/` 的实际目录；均已检查目录存在。
- `artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/STATUS_AND_RESULTS.md`
  只可用于辅助识别模型/目录，**其旧实验数值不录入本表，也不参与比较**。

## 1. 比较规则

左侧“论文”列是从 PDF 解析的历史报告值；右侧“本次统一重跑”只填写本轮
新任务产生的日志/JSON。空白（`—`）表示尚未重跑，绝不表示零或失败。

| 评测 | 论文列 | 本次统一重跑列 | 可比性要求 |
|---|---|---|---|
| XSum | R-AVG | R-AVG | 同一 XSum test、同一 decode 设置 |
| BoolQ | accuracy (%) | accuracy (%) | terminal-format MC scoring、同一 shots |
| SG | accuracy (%) | accuracy (%) | 同一 32 项 SG 数据与聚合方式 |
| BLiMP | accuracy (%) | accuracy (%) | 67×1,000 pairs；结构模型须记录 gold/beam/terminal 协议 |
| Doc-PPL | PPL | PPL | 必须记录路径、候选数、分母与 BOS/EOS 规则 |

当前 terminal/pause 的正式重跑任务使用 `terminal_doc_ppl`：148,836 句、
4,966 篇文档、3,284,061 个计分 terminal/EOS token；BOS 只作上下文、EOS
计分、pause 插入位置由 label mask 排除。旧 `lm` 型 `TG-ppl-validation` 结果
不可填入“本次统一重跑”列。

Tree/TG 的论文 Doc-PPL 是 tree-marginalized 近似，论文 Table 4 也注明 tree
linearization PPL 为 upper bound。因此录入时必须附带 `protocol`；不能把它和
terminal/pause 的 `terminal_doc_ppl` 误标为严格同一指标。

## 2. 论文模型 ↔ 本机 checkpoint

`Treeterm` 与 `TGTreeterm` 不是额外训练的 checkpoint，而分别是 Tree、TGTree
在 terminal-format scoring 下的评测协议。

| 论文模型 | 规模/数据 | 训练变体（Table 3） | 本机 checkpoint | 复现配置 model key | 状态 |
|---|---|---|---|---|---|
| Terminal | 100M / BBC | Terminal, causal | `saved_models/Terminal-lr005-bs144/step34115-unsharded` | `terminal` | 可用 |
| Tree | 100M / BBC | LIN1, causal | `saved_models/Tree_test/step49440-unsharded` | `tree` | 可用 |
| TGTree | 100M / BBC | LIN2, causal | `saved_models/TGtree/step69817-unsharded` | `tgtree` | 可用 |
| TG | 100M / BBC | LIN2, STACK/COMPOSE | `saved_models/TG_test/step55457-unsharded` | `tg` | 可用 |
| TGNomask | 100M / BBC | LIN2, Nomask | `saved_models/nomask_test/step55853-unsharded` | `tgnomask` | 可用 |
| TGNomask-Aug | 100M / BBC | LIN2, Nomask-Aug | `saved_models/TGnomask_aug_pretrain/step55853-unsharded` | `tgnomask_aug` | 可用 |
| Tree-NoONT | 100M / BBC | LIN1−ont, causal | `saved_models/tree_noont/step42440-unsharded` | `tree_noont` | 可用 |
| Tree-Compress | 100M / BBC | LIN1−merge, causal | `saved_models/tree_compress/step45965-unsharded` | `tree_compress` | 可用 |
| Tree-TripleCNT | 100M / BBC | LIN3, causal | `saved_models/tree_triplecnt/step60045-unsharded` | `tree_triplecnt` | 可用 |
| Tree-Shuffle | 100M / BBC | LIN1shuf, causal | `saved_models/Tree_shuffle_pretrain/step49440-unsharded` | `tree_shuffle` | 可用 |
| TGNomask-Mix-TG | 100M / BBC | LIN2, mixed heads | `saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded` | `nomask_mix_tg` | 可用 |
| TGTree-Mix-TG | 100M / BBC | LIN2, mixed heads | `saved_models/tgtree_mix_tg_pretrain/step69817-unsharded` | `tree_mix_tg` | 可用 |
| Pause-1 | 100M / BBC | 1 pause/token | `saved_models/pretrain_pause1_100M/step45487-unsharded` | `pause1` | 可用 |
| Pause-2 | 100M / BBC | 2 pauses/token | `saved_models/pretrain_pause2_100M/step52609-unsharded` | `pause2` | 可用 |
| Terminal | 500M / BBC | Terminal, causal | `saved_models/terminal_500M/step34115-unsharded` | `terminal-500M` | 可用 |
| Tree | 500M / BBC | LIN1, causal | `saved_models/Tree_500M/step49440-unsharded` | `tree-500M` | 可用 |
| TGTree | 500M / BBC | LIN2, causal | `saved_models/TGTree_500M/step55853-unsharded` | `tgtree-500M` | 可用 |
| TGNomask-Aug | 500M / BBC | LIN2, Nomask-Aug | `saved_models/TGnomaskaug_500M/step55853-unsharded` | `tgnomask_aug-500M` | 可用 |
| Terminal（补充） | 1B / BBC | Terminal, causal | `saved_models/terminal_1B/step34115-unsharded` | `terminal-bbc-1B`（兼容旧 key `terminal-1B`） | 可用；GPT-2/BBC，不能冒充论文 FineWeb-Edu 主实验 |
| Tree（补充） | 1B / BBC | LIN1, causal | `saved_models/Tree_1B/step49440-unsharded` | `tree-bbc-1B`（兼容旧 key `tree-1B`） | 可用；GPT-2/BBC，不能冒充论文 FineWeb-Edu 主实验 |
| Terminal | 1B / FineWeb-Edu | Terminal, causal | `saved_models/A800_models/terminal_1B/step94299-unsharded` | `terminal-fwedu-1B` | 可用；checkpoint 内 `data.paths` 已被评测路径污染，语料按论文与训练步数核验 |
| Tree | 1B / FineWeb-Edu | LIN1, causal | `saved_models/A800_models/tree_1B/step137217-unsharded` | `tree-fwedu-1B` | 可用；246 个 FineWeb-Edu-v2 tree shard patterns |
| TGTree | 1B / FineWeb-Edu | LIN2, causal | `saved_models/A800_models/tgtree_1B/step143658-unsharded` | `tgtree-fwedu-1B` | 可用；246 个 FineWeb-Edu-v2 tg shard patterns |
| Pause-1 | 1B / FineWeb-Edu | 1 pause/token | `saved_models/A800_models/pause1_1B_SEP/step116061-unsharded` | `pause1-fwedu-1B` | 可用；专用 SEP pause id 151673 |
| Pause-2 | 1B / FineWeb-Edu | 2 pauses/token | `saved_models/A800_models/pause2_1B_SEP/step141380-unsharded` | `pause2-fwedu-1B` | 可用；专用 SEP pause id 151673，sequence length 2049 |

不属于该论文 Table 3 的补充模型：TreeReg 使用 terminal 输入流和训练期辅助正则。
正式评测 checkpoint 为 `saved_models/treereg_layer9/step34354-unsharded`
（`treereg_layer=9`）；评测时关闭训练期辅助损失，按 terminal-format document
PPL 运行。

Tree-Shuffle 的 checkpoint 配置名虽为 `tree_shuffle`，但实际 logits 诊断显示其在
训练一致的 shuffled tree 输入上也几乎从不以非终结符作为 top-1；证据见
`artifacts/evaluation/tree_shuffle_terminality_20260823.md`。因此本轮将其按
`terminal_doc` 而非 `tg_doc` 重跑；该决策是 checkpoint 行为诊断，不改变论文中
Tree-Shuffle 的历史 Table 4 数值。

## 2A. 每个模型的预训练 config 协议

本节只登记**预训练**。结论的读取优先级为：论文定义语料/变体与“一 epoch”口径；
checkpoint 内 `config.yaml` 决定已经训练出的权重实际使用的架构、优化器、batch 和
step；`train_configs/*.yaml` 只在与 checkpoint 核心字段一致时才可作为重训入口。
checkpoint 内的绝对路径、`run_name`、evaluator 和后续被覆盖的 `data.paths` 不自动视为
训练来源证据。

### 2A.1 论文共同协议

- 架构（论文 Table 8）：100M=`d_model=768, layers=12, heads=12`；
  500M=`1408,16,16`；1B=`2048,16,16`。仓库实际 `mlp_ratio` 分别为 4/8/8，
  SwiGLU、RoPE、pre-RMSNorm、weight tying、sequence length 2048。
- 优化器（论文附录 A.1）：AdamW，`betas=(0.9,0.95)`、`eps=1e-8`、
  `weight_decay=0.1`、global grad clip 1.0；cosine + 2,000 warmup，最低
  learning rate `1e-5`；从头训练一 epoch。
- 学习率/全局 batch（论文 Eq. 8）：
  `lr=1.79*N^(-0.713)*D^0.307`，token batch=`0.58*D^0.571`；`N` 是非 embedding
  参数量，`D` 是该模型**实际线性化/扩展后的训练 token 数**。仓库非 embedding
  参数量为 70,900,224 / 507,896,576 / 1,073,876,992。
- BBC 使用 GPT-2 扩展 tokenizer（vocab/embedding=50320，EOS=50256，PAD=50258，
  `uint16`）；FineWeb-Edu 1B 使用 Qwen3 扩展 tokenizer（151732，EOS=151643，
  PAD=151670，必须 `uint32`）。BBC 约 10.06B terminal tokens；论文 Table 2 的
  LIN1/LIN2 约为 24.7B/32B。FineWeb-Edu-100BT 对应约 98B/233B/301B。
- checkpoint 均为 seed 6198、`amp_bf16`。100M/500M 原运行主要为 DDP，论文附录称
  每个规模使用 4×H800；FineWeb-Edu 1B checkpoint 为 FSDP。device microbatch 只决定
  显存/梯度累积，科学协议应固定下面的 global batch。

下表中的 `D*` 是 `checkpoint_step × global_batch × sequence_length` 的处理量近似，
用于核查一 epoch 与 Step Law；不是把 padding 后 token 当作新的语料统计。

### 2A.2 BBC 100M：逐模型实际协议

| 模型 | 数据/在线变换 | grammar key（训练） | D* / final step | peak LR | global / device microbatch | 关键附加 key |
|---|---|---|---:|---:|---:|---|
| Terminal | `terminal/train.npy` | legacy `null`→`terminal` | 10.061B / 34115 | 0.005000 | 144 / 32 | causal；该 LR 比论文 Step Law 值 0.005323 低约 6.1% |
| Tree | LIN1 `tree/train.npy` | `tree` | 24.706B / 49440 | 0.007000 | 244 / 32 | causal |
| TGTree | LIN2 `tg/train.npy` | `tree`（评测身份为 `tgtree`） | 32.029B / 69817 | 0.007600 | 224 / 28 | causal；global batch 224 低于 Step Law 约 282 |
| TG | LIN2 | `tg` | 32.028B / 55457 | 0.007600 | 282 / 30 | STACK/COMPOSE FlexAttention |
| TGNomask | LIN2 | `tgnomask` | 32.028B / 55853 | 0.007600 | 280 / 30 | full-k Nomask |
| TGNomask-Aug | LIN2 | `tgnomask_aug` | 32.028B / 55853 | 0.007600 | 280 / 30 | `_aug` 后缀使 CNT2 对后续 token 可见 |
| Tree-NoONT | LIN1−ont，旧目录 `bbc-news/1` | `tgtree`（评测 key `tree_noont`） | 17.383B / 42440 | 0.006296 | 200 / 25 | causal；变体身份由数据流承载 |
| Tree-Compress | LIN1−merge，旧目录 `bbc-news/2` | `tgtree`（评测 key `tree_compress`） | 20.710B / 45965 | 0.006644 | 220 / 30 | causal；变体身份由数据流承载 |
| Tree-TripleCNT | LIN3，旧目录 `bbc-news/3` | `tgtree`（评测 key `tree_triplecnt`） | 39.351B / 60045 | 0.008091 | 320 / 30 | causal；变体身份由数据流承载 |
| Tree-Shuffle | LIN1，collator 在线 shuffle NT | `tree_shuffle` | 24.706B / 49440 | 0.007000 | 244 / 28 | shuffle 不是独立 `.npy` |
| TGNomask-Mix-TG | LIN2 | `mixing` | 32.029B / 69817 | 0.007600 | 224 / 28 | `mix_head_type=[tg:6,tgnomask:6]` |
| TGTree-Mix-TG | LIN2 | `mixing` | 32.029B / 69817 | 0.007600 | 224 / 28 | `mix_head_type=[tgtree:6,tg:6]` |
| Pause-1 | terminal 在线 1 pause/token | `pause1` | 20.122B / 45487 | 0.006585 | 216 / 8 | seq=2048，`pause_token_id=null`，实际重复组内 terminal token |
| Pause-2 | terminal 在线 2 pauses/token | `pause2` | 30.183B / 52609 | 0.007595 | 280 / 15 | **seq=2049**，`pause_token_id=null`，实际重复 terminal token |

Pause-1/2-100M 的 `pause_token_id=null` 很重要：当前实现不是插入一个独立可学习的
共享 pause embedding，而是把每组最后一个真实 token 广播到 pause slot。论文把控制
描述为 learned latent token，因此复现论文文字时必须明确：现有 100M checkpoint 是
“repeat-token compute control”；不能静默改成专用 SEP，否则已经变成新实验。

**Checkpoint 级复核（2026-08-28）。** 本机这两个目录各只有一个实际 final step：
`pretrain_pause1_100M/step45487-unsharded` 与
`pretrain_pause2_100M/step52609-unsharded`；`latest-unsharded` 只是指向对应 step 的软链接，
并不存在第二个隐藏的 SEP step。run-root 与 step 内的 YAML 均为
`pause_token_id: null`。更关键的是，直接用权重预测 pause 槽时，Pause-1/2 对“前一个
真实 token”的 top-1 命中率均为 100%，平均概率分别为 0.9844/0.9896；GPT-2 embedding
范围内候选特殊 token 50257--50263 的平均概率仅为约 1e-9--1e-6。对 held-out BBC
片段进行协议似然比较也一致：repeat 序列的 pause NLL 为 0.0063/0.0089，而固定
SEP-50263 序列为 28.674/12.126。因此，本机这两个 **100M 权重本身就是 repeat-token**，
不是“SEP 训练但 YAML 漏写”；真正采用独立 SEP 的补充 checkpoint 是下文 FineWeb-Edu
1B 的 `pause1_1B_SEP` / `pause2_1B_SEP`（id 151673）。若另有 100M SEP checkpoint，
必须另给确切 step 路径与 pause id，不能复用当前两个 checkpoint 身份。

为补齐这一缺口，2026-08-28 已在 SIST `ShangHAI` 分区启动独立 SEP 的 100M
重训 campaign：`pause_sep_100m_sist_20260828`。两者都使用 terminal train stream、
seed 6198、`pause_token_id=50261`、FlashAttention-2 和严格按实际 pause-expanded D
重算的 Step Law。Pause-1 为 LR=0.006585、global/device/micro/accum=
216/27/9/3（job 988670，8×RTX 6000D，RUNNING）；Pause-2 为 LR=0.007458、
272/34/17/2（job 988671，`afterok:988670` 排队）。这是尚未完成的补充预训练，
不能提前作为 checkpoint 或结果引用；配置、公式展开、提交脚本和 job manifest 位于
`artifacts/experiment/pause_sep_100m_sist_20260828/`。

### 2A.3 BBC 500M 与补充 BBC 1B

| 模型 | 数据 | D* / final step | peak LR | global / microbatch | 架构/协议备注 |
|---|---|---:|---:|---:|---|
| Terminal-500M | terminal | 10.061B / 34115 | 0.001308 | 144 / 16 | 1408×16×16，causal |
| Tree-500M | LIN1 | 24.706B / 49440 | 0.001723 | 244 / 16 | 1408×16×16，causal |
| TGTree-500M | LIN2 | 32.029B / 55853 | 0.001866 | 280 / 16 | checkpoint key=`tree`，但数据为 LIN2；causal |
| TGNomask-Aug-500M | LIN2 | 32.029B / 55853 | 0.001866 | 280 / 12 | `tgnomask_aug`，FlexAttention |
| Terminal-1B-BBC（补充） | terminal | 10.061B / 34115 | 0.000766 | 144 / 12 | 2048×16×16，GPT-2/BBC，不属于 FineWeb 主实验 |
| Tree-1B-BBC（补充） | LIN1 | 24.706B / 49440 | 0.001010 | 244 / 12 | 2048×16×16，GPT-2/BBC，不属于 FineWeb 主实验 |

### 2A.4 FineWeb-Edu 1B：论文主实验五模型

| 模型 | 数据/变换 | D* / final step | peak LR | global / microbatch | 必须显式的 key |
|---|---|---:|---:|---:|---|
| Terminal | 98B terminal | 98.880B / 94299 | 0.001546 | 512 / 4 | `terminal`, `uint32`, bias=false |
| Tree | 233B LIN1 | 233.809B / 137217 | 0.002013 | 832 / 5 | `tree`, `uint32`, bias=false |
| TGTree | 301B LIN2 | 301.273B / 143658 | 0.002176 | 1024 / 5 | `tgtree`（仍为 causal），`uint32`, bias=false |
| Pause-1 | terminal + 1 SEP/token | 197.761B / 116061 | 0.002013 | 832 / 5 | `pause1`, seq=2048, **pause id=151673**, `uint32` |
| Pause-2 | terminal + 2 SEP/token | 296.640B / 141380 | 0.002176 | 1024 / 5 | `pause2`, **seq=2049**, **pause id=151673**, `uint32` |

五个 checkpoint 均在 `saved_models/A800_models/`。此前“本机无 Pause-1/2-1B”和
“TGTree-1B 无 model key”的登记已纠正。Terminal-1B checkpoint 的 `data.paths` 当前
指向 HellaSwag，根目录另一个 TGTree-1B config 指向 `fewshot_sources.py`；这是后续评测
配置污染/占位，不能据此否定论文语料。Tree/TGTree/Pause checkpoint 保存的 246 个
`fineweb-edu-v2` shard pattern、Qwen3 tokenizer、`uint32` 和上述一 epoch 处理量构成
更强的训练来源证据。

### 2A.5 checkpoint config ↔ `train_configs/` 对应关系

| 模型组 | 可参考模板 | 是否能精确重训当前 checkpoint |
|---|---|---|
| Terminal/Tree/TG-100M | `terminal.yaml` / `tree.yaml` / `TG.yaml` | 否；这些是 early/truncated 模板，LR、batch、stop 与正式 checkpoint 不同 |
| TGTree、TGNomask、TGNomask-Aug、三种 Tree 数据变体、Tree-Shuffle、Pause-1/2-100M | 无独立正式模板 | 否；以对应 checkpoint 的 `config.yaml` 为唯一核心参数源，重写绝对数据路径 |
| 两个 mixing 模型 | `nomask_and_tg.yaml` | 否；现文件是 `stop_at=6, lr=6e-4` 的 smoke config；必须从 checkpoint 保留完整 `mix_head_type` |
| Terminal/Tree/TGTree-500M | `terminal-500M.yaml` / `tree-500M.yaml` / `tgtree-500M.yaml` | 核心 LR/batch/架构一致；final step 仍按 checkpoint 与一 epoch数据长度 |
| TGNomask-Aug-500M | 无独立正式模板 | 以 checkpoint config 为准 |
| Terminal/Tree-1B-BBC | 无无歧义成对模板 / `tree-1B.yaml` | `tree-1B.yaml` 对应 BBC；`terminal-1B.yaml` 实际是 FineWeb-Edu 草稿，不能用于 BBC Terminal |
| FineWeb-Edu Terminal-1B | `terminal-A800.yaml` 最接近 | batch/micro/data dtype 一致，但模板 LR=0.001536、checkpoint=0.001546；精确复现用 checkpoint |
| FineWeb-Edu Tree/TGTree/Pause-1/Pause-2 | 无独立正式模板 | 以 `saved_models/A800_models/*/step*/config.yaml` 为准 |

实际重训时应复制 checkpoint config 到新的 run config，仅替换 `run_name`、
`save_folder` 和等价的本机数据/tokenizer 路径；不得继承 checkpoint 中后续评测的
evaluator、`load_path`、`data.paths` 或 trainer/optimizer state。已有权重的评测则使用
上表 corpus-qualified model key，避免把 BBC 1B 与 FineWeb-Edu 1B 混在一起。

上述人工核对结果已固化为机器可读清单
`train_configs/paper_pretraining_manifest.json`；`scripts/prepare_paper_pretraining.py`
会逐项比对清单与 checkpoint config，再生成清除旧路径和恢复状态的 27 个重训 run。
清单只负责机器执行，模型身份与协议解释仍以本节为准，避免 README 另存一份易漂移的参数表。

### 2A.6 已消除和仍需显式记录的隐含 key

1. `transformer_grammar_type: null`：旧 Terminal checkpoint 现由
   `TrainConfig.update_legacy_settings` 统一迁移为 `terminal`，不再依赖某个评测脚本私下修补。
2. `mixing`：现强制要求非空 `mix_head_type`，且各项 `n_heads` 之和必须等于模型总 heads；
   单写 `transformer_grammar_type=mixing` 会直接报配置错误。
3. `tree` 与 `tgtree`：预训练都在 causal-only dispatch 集合中；二者的 checkpoint 字段
   不能单独证明 LIN1/LIN2。线性化由训练数据路径和论文变体共同确定，评测 key 再决定
   word-sync grammar 身份。
4. `tgnomask_aug`：Aug 行为由字符串 `_aug` 后缀触发；写成 `tgnomask` 会改变 CNT2
   后续可见性，不是同义别名。
5. Pause：必须同时登记 grammar、`pause_token_id` 和 sequence length。Pause-2 的 2049
   使真实 token budget 683 经三倍扩展后恰为 2049；擅自改为 2048 会落到 682→2046。
6. Qwen3：token id 超过 uint16，FineWeb-Edu 必须 `data.memmap_dtype=uint32`；沿用 BBC
   默认 `uint16` 会截断 token id。
7. 路径不是协议：旧 `/dev/shm`、`/data/home/*`、`/2024233198/*` 可做等价重映射；但目录
   `terminal/tree/tg`、shard 覆盖范围、dtype、tokenizer 和在线变换不得改变。
8. `workspace: ${workspace}[/...]`：部分导出的 1B config 含无效自引用，现由 legacy
   migration 规范为仓库相对路径 `.`，随后由任务生成器覆盖为目标 workspace，避免
   OmegaConf 递归插值失败。
9. Step Law：精确复现已训练 checkpoint 时优先保留表中实际 LR/global batch；为新语料
   或新 token 数设计实验时再按论文 Eq. 8 重算。Terminal-100M、TGTree/mixing-100M 和
   Pause-1-1B 等已存在可量化偏差，不应倒改历史 config 伪装成公式严格一致。

### 2A.7 非论文 Table 3 的补充预训练模型

TreeReg/Pushdown 不应与本文 12 个 SLM 变体混称为论文模型。其 checkpoint 均为 100M、
BBC、seed 6198、AdamW/cosine 共同协议，peak LR 0.005323、global batch 144、microbatch 12、
final step 34354；训练输入由 LIN1 parse 经 unary collapse + right CNF 对齐成 terminal 流。
TreeReg-layer9 还要求 `treereg_layer=9, treereg_n_heads=3, treereg_every_k=10,
treereg_alpha=1`；Pushdown 要求 `pushdown_max_depth=64,
pushdown_attachment_weight=1, pushdown_attachment_layer=-1`。这些设定分别应结合其来源论文，
不能用本论文的 Table 3 为它们背书。

## 3. BBC News：论文 Table 4 与本次重跑

论文值按原样保留。Doc-PPL 的脚注为：Tree linearization 模型是 upper bound。
“重跑来源”必须填写 job id、日志路径和 protocol，避免误混旧结果。

| 模型 | 论文 XSum R-AVG | 论文 BoolQ | 论文 SG | 论文 BLiMP | 论文 Doc-PPL | 本次 XSum | 本次 BoolQ | 本次 SG | 本次 BLiMP | 本次 Doc-PPL | 重跑来源 / protocol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Terminal-100M | 21.99 | 67.40 | 69.98 | 70.77 | 10.19 | — | — | 69.80 | 70.77 | 9.88981 | `SG job=3454; BLiMP job=3455; checkpoint=saved_models/Terminal-lr005-bs144/step34115-unsharded; evaluator=terminal teacher-forced; dataset=full SG suite / full BLiMP (67×1000 pairs); BLiMP warnings=7 tied-or-non-finite pairs; logs=artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal_{SG,blimp}_seed6198/logs/; completed=2026-08-23. Doc-PPL: job=3444; evaluator=terminal_doc/terminal_doc_ppl; dataset=dataset/bbc-news/terminal/test.npy; protocol=K1,BOS-context,EOS-scored,denom=3284061; log=artifacts/evaluation/terminal_doc_ppl_20260823/terminal/logs/slurm-terminal_terminal_doc_ppl_full_20260823-3444.out` |
| Tree-100M | 22.82 | 67.86 | 81.64 | 80.90 | 10.17 | — | — | 81.40 | 10.01 | — Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27| `SG job=3419; checkpoint=saved_models/Tree_test/step49440-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tree_beam300_len6_3419.out; completed=2026-08-21` |
| TGTree-100M | 22.69 | 68.38 | 79.40 | 81.48 | 10.36 | — | — | 79.47 | 10.21 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgtree_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-25| `SG job=3418; checkpoint=saved_models/TGtree/step69817-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgtree_beam300_len6_3418.out; completed=2026-08-21` |
| TG-100M | 20.13 | 64.31 | 78.80 | 83.23 | 11.20 | — | — | 79.19 | 11.02 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tg_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27| `SG job=3413; checkpoint=saved_models/TG_test/step55457-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tg_beam300_len6_3413.out; completed=2026-08-21` |
| TGNomask-100M | 21.60 | 67.25 | 78.47 | 81.77 | 10.14 | — | — | 78.62 | 9.971 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgnomask_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27| `SG job=3414; checkpoint=saved_models/nomask_test/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgnomask_beam300_len6_3414.out; completed=2026-08-21` |
| TGNomask-Aug-100M | 22.04 | 68.81 | 78.44 | 83.64 | 10.24 | — | — | 78.68 | 10.08 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgnomask_aug_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28 [10.08 via flex_attention=false path; flex=true cross-check run tgnomask_aug_docppl_seed6198 still running locally at 2026-08-29 02:44, expected numerically identical]| `SG job=3415; checkpoint=saved_models/TGnomask_aug_pretrain/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgnomask_aug_beam300_len6_3415.out; completed=2026-08-21` |
| Tree-NoONT-100M | 22.29 | 68.13 | 85.37 | 81.00 | 10.08 | — | — | 85.38 | 9.947 | — Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_noont_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-26| `SG job=3475; checkpoint=saved_models/tree_noont/step42440-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=tree_noont; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_noont_beam300_len6_3475.out; completed=2026-08-24` |
| Tree-Compress-100M | 22.63 | 68.04 | 78.18 | 81.44 | 10.09 | — | — | 78.27 | 9.909 | — Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_compress_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-26| `SG job=3476; checkpoint=saved_models/tree_compress/step45965-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=tree_compress; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_compress_beam300_len6_3476.out; completed=2026-08-24` |
| Tree-TripleCNT-100M | 21.18 | 69.02 | 79.83 | 83.38 | 10.41 | — | — | 81.57 | 10.26 | — Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_triplecnt_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-26| `SG job=3479; checkpoint=saved_models/tree_triplecnt/step60045-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=tree_triplecnt; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_triplecnt_beam300_len6_3479.out; completed=2026-08-24` |
| Tree-Shuffle-100M | 21.76 | 67.77 | 76.96 | 67.67 | 13.69 | — | — | 61.59 | 71.68 | 50.91 | `SG job=3473; BLiMP job=3474; Doc-PPL job=3467; checkpoint=saved_models/Tree_shuffle_pretrain/step49440-unsharded; evaluator=terminal-only teacher-forced / terminal_doc; dataset=full SG suite / full BLiMP / BBC terminal test; protocol=model-only restore, input_format=terminal,beam_search=false; logs=artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_{SG,blimp}_seed6198/logs/; legacy SG 3468 and BLiMP 3469 are superseded and excluded; completed=2026-08-23. Independent terminal-only SG confirmation: job=3477, avg=61.59%, log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_shuffle_terminal_only_3477.out; completed=2026-08-23` |
| TGNomask-Mix-TG-100M | 21.86 | 67.49 | 75.96 | 83.15 | 10.16 | — | — | 75.77 | 10.01 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/nomask_mix_tg_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27| `SG job=3416; checkpoint=saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgnomask_mix_tg_beam300_len6_3416.out; completed=2026-08-21` |
| TGTree-Mix-TG-100M | 22.25 | 67.74 | 79.25 | 82.76 | 10.16 | — | — | 78.84 | 10.00 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_mix_tg_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28| `SG job=3478; checkpoint=saved_models/tgtree_mix_tg_pretrain/step69817-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=mixing; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tgtree_mix_tg_beam300_len6_3478.out; completed=2026-08-24` |
| Pause-1-100M | 19.86 | 66.06 | 78.47 | 69.99 | 10.64 | — | — | 76.29 | 72.52 | 9.83125 | `SG job=3460; BLiMP job=3461; checkpoint=saved_models/pretrain_pause1_100M/step45487-unsharded; evaluator=pause1 teacher-forced; dataset=full SG suite / full BLiMP (67×1000 pairs); BLiMP warnings=5 tied-or-non-finite pairs; logs=artifacts/evaluation/terminal_pause_syntax_20260823/runs/pause1_{SG,blimp}_seed6198/logs/; completed=2026-08-23. Doc-PPL: job=3445; evaluator=terminal_doc/terminal_doc_ppl; dataset=dataset/bbc-news/terminal/test.npy; protocol=K1,pause1-document-phase,BOS-context,EOS-scored,pause-masked,denom=3284061` |
| Pause-2-100M（补充） | — | — | — | — | — | — | — | 75.85 | 73.15 | 9.92898 | `SG job=3462; BLiMP job=3463; checkpoint=saved_models/pretrain_pause2_100M/step52609-unsharded; evaluator=pause2 teacher-forced; dataset=full SG suite / full BLiMP (67×1000 pairs); BLiMP warnings=3 tied-or-non-finite pairs; logs=artifacts/evaluation/terminal_pause_syntax_20260823/runs/pause2_{SG,blimp}_seed6198/logs/; completed=2026-08-23. Doc-PPL: job=3446; evaluator=terminal_doc/terminal_doc_ppl; dataset=dataset/bbc-news/terminal/test.npy; protocol=K1,pause2-document-phase,BOS-context,EOS-scored,pause-masked,denom=3284061` |
| Terminal-500M | 20.71 | 69.97 | 71.25 | 64.74 | 3.06 | — | — | 71.11 | 64.74 | 2.82518 | `SG job=3456; BLiMP job=3457; checkpoint=saved_models/terminal_500M/step34115-unsharded; evaluator=terminal teacher-forced; dataset=full SG suite / full BLiMP (67×1000 pairs); BLiMP warnings=7 tied-or-non-finite pairs; logs=artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal-500M_{SG,blimp}_seed6198/logs/; completed=2026-08-23. Doc-PPL: job=3447; evaluator=terminal_doc/terminal_doc_ppl; dataset=dataset/bbc-news/terminal/test.npy; protocol=K1,BOS-context,EOS-scored,denom=3284061; log=artifacts/evaluation/terminal_doc_ppl_20260823/terminal_500m/logs/slurm-terminal_500m_terminal_doc_ppl_full_20260823-3447.out` |
| Tree-500M | 22.05 | 71.90 | 63.26 | 76.07 | 3.32 | — | — | 63.13 | 3.182 | — Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree-500M_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27| `SG job=3437; checkpoint=saved_models/Tree_500M/step49440-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,non-flex,2 GPU; log=analysis-output/sg_500m_nonflex_rtx3090_20260822/logs/sg_500m_tree_beam300_len6_nonflex_3437.out; completed=2026-08-22. Flex reference: A6000 job=38152, 62.89%` |
| TGTree-500M | 21.96 | 70.86 | 66.50 | 77.63 | 3.40 | — | — | 65.95 | 3.279 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgtree-500M_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28| `SG job=3438; checkpoint=saved_models/TGTree_500M/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,non-flex,2 GPU; log=analysis-output/sg_500m_nonflex_rtx3090_20260822/logs/sg_500m_tgtree_beam300_len6_nonflex_3438.out; completed=2026-08-22. Flex reference: A6000 job=38153, 65.59%` |
| TGNomask-Aug-500M | 21.63 | 71.04 | 57.73 | 75.99 | 3.32 | — | — | 57.58 | 3.188 | — Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgnomask_aug-500M_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28| `SG job=3439; checkpoint=saved_models/TGnomaskaug_500M/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,non-flex,2 GPU; log=analysis-output/sg_500m_nonflex_rtx3090_20260822/logs/sg_500m_tgnomask_aug_beam300_len6_nonflex_3439.out; completed=2026-08-22. A6000 flex job=38154 failed during Triton/Inductor compilation; no flex score` |
| Terminal-1B（BBC，补充） | — | — | — | — | — | — | — | 67.41 | 63.31 | 1.705 | `SG job=3471; BLiMP job=3472; Doc-PPL job=3470; checkpoint=saved_models/terminal_1B/step34115-unsharded; evaluator=terminal teacher-forced / terminal_doc; dataset=full SG suite / full BLiMP / BBC terminal test; protocol=model-only restore, reset optimizer/trainer state; logs=artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/terminal-1B_{docppl,SG,blimp}_seed6198/logs/; completed=2026-08-23` |
| TreeReg-layer9（补充；legacy FT 无辅助 loss） | — | — | — | — | — | 17.59 ± 0.10 | 63.54 ± 0.56 | 67.65 | 66.68 | 12.37 | `XSum/BoolQ five-seed mean ± sample-SD: training=3493_[0-4] / 3518_[5-9]; eval=3495_[0-4] / 3519_[5-9]; checkpoint=saved_models/treereg_layer9/step34354-unsharded; protocol=scaleup_nonfineweb_multiseed_20260815 settings, fp32 DDP 4 GPU, global batch=40, XSum 3ep LR=6e-5, BoolQ 5ep LR=3e-4; BoolQ initial microbatch=10 jobs 3493_[5-9] OOM, then completed with microbatch=1 while preserving global batch; evidence=artifacts/experiment/treereg_layer9_multiseed_20260824/runs/; completed=2026-08-25. Although config declared treereg_alpha=1, downstream batches had no tree_spans / word boundaries / sentence ids, so train.py did not execute the auxiliary branch; it also retained TG-token input rather than parse-aligned terminal input. SG job=3464; BLiMP job=3453; evaluator=terminal teacher-forced, no test-time tree input; dataset=full SG suite / full BLiMP (67×1000 pairs); logs=analysis-output/logs/eval_treereg_layer9_{SG,BLiMP}_{3464,3453}.out; completed=2026-08-23. Fresh terminal_doc confirmation: job=3561, evaluator completed all 148836 records and emitted PPL=12.37; job exit=1 only because the post-run grep used the wrong metric key, corrected in artifact script; log=artifacts/evaluation/treereg_layer9_terminal_docppl_20260825/run.log; protocol=K1,BOS-context,EOS-scored` |
| TreeReg-layer9（parse-aligned + auxiliary loss） | — | — | — | — | — | 20.88 ± 0.03 | 65.87 ± 0.31 | — | — | — | `Five-seed parse-aligned fine-tune: train=3528_[0-9], eval=3531_[0-9], all exit=0; checkpoint=saved_models/treereg_layer9/step34354-unsharded; XSum=3ep LR=6e-5, BoolQ=5ep LR=3e-4, fp32 DDP 4 GPU, device microbatch=1, global batch=40. Loader converts local NT-token trees with right binarization + unary collapse and supplies spans, word boundaries, sentence ids; TreeReg layer=9, heads=3, alpha=1, every 10 global optimizer batches. Evidence=artifacts/experiment/treereg_layer9_auxloss_multiseed_20260825/runs/. No post-finetune SG/BLiMP/Doc-PPL run.` |
| Pushdown-100M（gold unary spans） | — | — | — | — | — | 15.87 ± 0.17 | 65.49 ± 0.55 | — | — | — | `Five-seed Pushdown fine-tune: train=3541_[0-9]; valid eval=3576_[1-9] + 3582_0; checkpoint=saved_models/pushdown_terminalonly/step34354-unsharded; XSum=3ep LR=6e-5, BoolQ=5ep LR=3e-4, global batch=40, amp_bf16 DDP. Training converts parsed trees to terminal streams with right binarization + unary collapse and supervises both LM tokens and attachment targets from gold spans. XSum evaluation loads each source prompt's gold parse spans directly; beam search is restricted to the generated summary suffix and jointly searches completion tokens and attachment choices (beam=6,max_reduce=4). BoolQ uses terminal-format candidate scoring with gold unary spans (beam=20). Evidence=artifacts/experiment/pushdown_finetune_5seeds_20260825/runs/; completed=2026-08-28. Original eval task 3576_0 failed before evaluation because of a TCP port collision and is superseded by successful task 3582_0.` |

### TreeReg-layer9 五 seed XSum / BoolQ（2026-08-25）

以下为 `treereg_layer9_multiseed_20260824` 的独立训练、独立 DDP 评测结果。
XSum 为 R-AVG，BoolQ 为 validation accuracy；汇总使用五个 fine-tuning seed 的均值 ± 样本标准差。

| seed | XSum R-AVG | BoolQ accuracy |
|---:|---:|---:|
| 6198 | 17.42 | 63.00 |
| 13171 | 17.67 | 63.12 |
| 31723 | 17.63 | 63.52 |
| 42 | 17.56 | 63.61 |
| 2026 | 17.66 | 64.43 |
| mean ± sample SD | 17.59 ± 0.10 | 63.54 ± 0.56 |

XSum 的训练/评测链为 `3493_[0-4] → 3495_[0-4]`。BoolQ 的初始训练
`3493_[5-9]` 在 device microbatch=10 时发生 CUDA OOM；重试链
`3518_[5-9] → 3519_[5-9]` 仅将 device microbatch 降为 1，并通过梯度累积
保持 global batch=40。十个训练与十个独立评测均以退出码 0 完成；逐 seed 日志在
`artifacts/experiment/treereg_layer9_multiseed_20260824/runs/`。

### TreeReg-layer9：parse-aligned auxiliary-loss 五 seed 对照（2026-08-25）

本组使用同一预训练 checkpoint、seed、epoch、learning rate、global batch=40 和独立
DDP evaluator；训练加载器在 XSum 的 `save_ids.json` 过滤之后、以及 BoolQ 本地
PTB/TG 树读取之后，将 NT-token 树转换为 terminal input、right-binary + unary-collapse
spans、word boundaries、sentence ids。`treereg_alpha=1` 的辅助项每 10 个 global
optimizer batch 生效。所有训练 `3528_[0-9]` 和评测 `3531_[0-9]` 均 exit=0。

| seed | XSum R-AVG | BoolQ accuracy |
|---:|---:|---:|
| 6198 | 20.87 | 65.78 |
| 13171 | 20.89 | 65.54 |
| 31723 | 20.90 | 65.63 |
| 42 | 20.83 | 66.24 |
| 2026 | 20.91 | 66.15 |
| mean ± sample SD | 20.88 ± 0.03 | 65.87 ± 0.31 |
| versus legacy FT above | +3.29 | +2.33 |

该差值是“parse-aligned terminal input + TreeReg auxiliary loss”相对 legacy 下游
pipeline 的端到端对照；不能单独归因于 auxiliary loss，因为 legacy 组没有提供
spans/word-boundaries/sentence-ids，且输入表示也不是 terminal parse-aligned。若需严格
隔离 loss，需另跑同一新数据路径但 `treereg_alpha=0` 的五 seed control。

### Pushdown：gold-unary-span 五 seed XSum / BoolQ（2026-08-28）

本组从 `saved_models/pushdown_terminalonly/step34354-unsharded` 独立微调。训练加载器
将 parsed TG 树转换为 terminal token stream，并对 right-binarized tree 做 unary
collapse；所得 gold spans 同时指导 Pushdown attention/stack tape 和 attachment-head
监督。XSum 评测直接加载 source prompt 的 gold parse tree spans，prompt 部分不做
beam search；仅对待生成的 summary 后缀联合搜索 token 与 attachment，设置为
`beam=6,max_reduce=4`。BoolQ 使用 gold unary spans 下的 terminal-format candidate
scoring，`beam=20`。汇总值为五个 fine-tuning seed 的均值 ± 样本标准差。

| seed | XSum ROUGE-1 | XSum ROUGE-2 | XSum ROUGE-L | XSum R-AVG | BoolQ accuracy |
|---:|---:|---:|---:|---:|---:|
| 6198 | 22.90 | 7.14 | 17.90 | 15.98 | 65.38 |
| 13171 | 22.89 | 7.08 | 17.83 | 15.93 | 64.98 |
| 31723 | 22.56 | 6.94 | 17.59 | 15.70 | 66.27 |
| 42 | 23.06 | 7.14 | 17.96 | 16.05 | 65.02 |
| 2026 | 22.51 | 7.00 | 17.56 | 15.69 | 65.81 |
| mean ± sample SD | 22.78 ± 0.24 | 7.06 ± 0.09 | 17.77 ± 0.18 | 15.87 ± 0.17 | 65.49 ± 0.55 |

训练任务 `3541_[0-9]` 全部完成。正式评测中，XSum 每个 seed 完成 5,667 个
evaluation step，BoolQ 每个 seed 完成 654 个 evaluation step；所有十个目录均有
`TRAIN_DONE`、`EVAL_DONE` 和日志末尾的 `Training complete`，且未发现 traceback、
OOM、NaN 或 fatal error。有效评测任务为 `3576_[1-9]` 与 `3582_0`；原
`3576_0` 在进入正式评测前发生 TCP port collision，已由 `3582_0` 完整替代。
逐 seed 配置、checkpoint 指针和最终日志位于
`artifacts/experiment/pushdown_finetune_5seeds_20260825/runs/`。

### 结构模型 Doc-PPL 补充（2026-08-29，本地 H20 cluster）

论文 Table 4 中 13 个结构模型的 Doc-PPL 于 2026-08-25 起在本机 H20 集群
（4×H20，非 SLURM）统一重跑，全部跑满 744,180 步并输出结果（seed 6198）。
数据为 2026-08-25 从本机重传的 `dataset/bbc-news/testppl_{tg,tree}/`
（bos/eos-normalized 版本，md5 校验）；本地日志在
`artifacts/evaluation/docppl_structural_20260825/runs/<run>_docppl_seed6198/logs/launch.out`
（日志不在本机）。tg-family 语法用 `tg_approx_doc`、tree-family 用
`txl_approx_doc`，均为 tree-marginalized 近似（与论文 Table 4 脚注一致）。

| 模型 | run 名 | evaluator | 本次 Doc-PPL | 完成 |
|---:|---|---|---:|---|
| Tree-100M | tree | txl_approx_doc | 10.01 | 2026-08-27 |
| TGTree-100M | tgtree | tg_approx_doc | 10.21 | 2026-08-25 |
| TG-100M | tg | tg_approx_doc | 11.02 | 2026-08-27 |
| TGNomask-100M | tgnomask | tg_approx_doc | 9.971 | 2026-08-27 |
| TGNomask-Aug-100M | tgnomask_aug | tg_approx_doc | 10.08 | 2026-08-28 |
| Tree-NoONT-100M | tree_noont | txl_approx_doc | 9.947 | 2026-08-26 |
| Tree-Compress-100M | tree_compress | txl_approx_doc | 9.909 | 2026-08-26 |
| Tree-TripleCNT-100M | tree_triplecnt | txl_approx_doc | 10.26 | 2026-08-26 |
| TGNomask-Mix-TG-100M | nomask_mix_tg | tg_approx_doc | 10.01 | 2026-08-27 |
| TGTree-Mix-TG-100M | tree_mix_tg | tg_approx_doc | 10.00 | 2026-08-28 |
| Tree-500M | tree-500M | txl_approx_doc | 3.182 | 2026-08-27 |
| TGTree-500M | tgtree-500M | tg_approx_doc | 3.279 | 2026-08-28 |
| TGNomask-Aug-500M | tgnomask_aug-500M | tg_approx_doc | 3.188 | 2026-08-28 |

注：TGNomask-Aug-100M 的 10.08 来自 flex_attention=false 路径（compile 路径不同、
数值预期等价；flex=true 复核任务 `tgnomask_aug_docppl_seed6198` 于 2026-08-29 凌晨
仍在本地运行，完成后再确认）。
### 本轮 Doc-PPL 结果补充（2026-08-23）

下表是同一轮完整 BBC News test 集评测的最终 Slurm 状态。成功任务均完成
148,836 个 sentence step，使用 `terminal_doc`：K=1、仅 BOS 作上下文、EOS 计分；
`denom=3,284,061`。Pause 模型另使用 document-phase pause 并屏蔽 pause target。

| 模型 | 最终状态 | 本次 Doc-PPL | job / 日志 |
|---|---|---:|---|
| Terminal-100M | COMPLETE | 9.88981 | `3444`; `artifacts/evaluation/terminal_doc_ppl_20260823/terminal/logs/slurm-terminal_terminal_doc_ppl_full_20260823-3444.out` |
| Pause-1-100M | COMPLETE | 9.83125 | `3445`; `artifacts/evaluation/terminal_doc_ppl_20260823/pause1/logs/slurm-pause1_terminal_doc_ppl_full_20260823-3445.out` |
| Pause-2-100M | COMPLETE | 9.92898 | `3446`; `artifacts/evaluation/terminal_doc_ppl_20260823/pause2/logs/slurm-pause2_terminal_doc_ppl_full_20260823-3446.out` |
| Terminal-500M | COMPLETE | 2.82518 | `3447`; `artifacts/evaluation/terminal_doc_ppl_20260823/terminal_500m/logs/slurm-terminal_500m_terminal_doc_ppl_full_20260823-3447.out` |
| TreeReg-layer9 | COMPLETE | 12.36841 | `3451`; `artifacts/evaluation/terminal_doc_ppl_20260823/treereg_layer9/logs/slurm-treereg_layer9_terminal_doc_ppl_full_20260823-3451.out` |
| Terminal-1B | FAILED（未进入 eval） | — | `3448`; 配置继承了缺失的 FineWeb-Edu Arrow shards：`artifacts/evaluation/terminal_doc_ppl_20260823/terminal_1b/logs/slurm-terminal_1b_terminal_doc_ppl_full_20260823-3448.out` |
| Tree-Shuffle-100M | FAILED（未进入 eval） | — | `3450`; checkpoint 缺少 `optim.pt`：`artifacts/evaluation/terminal_doc_ppl_20260823/tree_shuffle/logs/slurm-tree_shuffle_terminal_doc_ppl_full_20260823-3450.out` |

Tree-Shuffle 的输入/输出诊断支持把它按 terminal 模型评测；详见
`artifacts/evaluation/tree_shuffle_terminality_20260823.md`。本轮因 checkpoint
恢复失败尚未取得其 Doc-PPL，不能将该失败项与论文 Table 4 数值混用。

### 新版 model-only 评测结果（2026-08-23）

修复后的配置从 checkpoint model config 建立，并设置
`reset_optimizer_state=true`、`reset_trainer_state=true`，因此以下结果均不读取
`optim.pt`/`train.pt`。Tree-Shuffle 的 Doc-PPL 使用 `terminal_doc`（terminal
`test.npy`、BOS 上下文、EOS 计分、148,836 句）。其 SG/BLiMP 首轮任务曾误走
auto/beam，现已改为 terminal-only teacher-forced protocol，并重新提交 3473/3474。

| 模型 | Doc-PPL | SG | BLiMP | evidence |
|---|---:|---:|---:|---|
| Tree-Shuffle-100M | 50.91（job 3467，COMPLETE） | 61.59%（job 3473，COMPLETE，terminal-only） | 71.68%（job 3474，COMPLETE，terminal-only） | `artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_{SG,blimp}_seed6198/logs/`；旧 3468/3469 为 legacy auto/beam，作废 |
| Terminal-1B（BBC） | 1.705（job 3470，COMPLETE） | 67.41%（job 3471，COMPLETE） | 63.31%（job 3472，COMPLETE） | `artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/terminal-1B_{docppl,SG,blimp}_seed6198/logs/` |

Tree-Shuffle 新任务的运行时 grammar type 已明确设为 `terminal`，并设置
`input_format=terminal`、`structure_mode=terminal`、`tree_eval_type=terminal`、
`beam_search=false`；权重仍来自 `Tree_shuffle_pretrain/step49440-unsharded`。
因此 SG/BLiMP 不再使用 beam search 或 gold300。新任务 SG job 3473 已完成，
`avg=0.6159`（61.59%）；BLiMP job 3474 已完成，`overall/overall=0.7168`
（71.68%）。

Terminal-1B Doc-PPL 日志达到 `eval_step=148836`；SG/BLiMP 分别记录
`avg=0.6741` 与 `overall/overall=0.6331`。Tree-Shuffle 的 SG/BLiMP 此前在
完整 suite 上运行；现已分别由 3473/3474 完成并回填至 §3 总表。

## 4. 1B FineWeb-Edu：论文数据与本次重跑

论文 Table 6 是 11-task OLMES；这里只登记用户要求的 BoolQ，并保留 AVG 作为
定位信息。SG、BLiMP、Doc-PPL、XSum 未在该表以同一设置报告，不填论文值。

| 论文模型 / 评测协议 | 论文 BoolQ | 论文 11-task AVG | 本次 BoolQ | 本次 XSum | 本次 SG | 本次 BLiMP | 本次 Doc-PPL | checkpoint / 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Terminal | 59.11 | 53.23 | — | — | — | — | — | `saved_models/A800_models/terminal_1B/step94299-unsharded`。旧 SG 3458/BLiMP 3459 错用 BBC `terminal_1B/step34115` 并继承缺失 shards，未产生 FineWeb 指标；BBC model-only 结果只列 §3 |
| Tree | 64.83 | 53.93 | — | — | — | — | — | `saved_models/A800_models/tree_1B/step137217-unsharded`；旧 SG job=3435 使用的是 BBC `Tree_1B/step49440`，不能回填本行 |
| TGTree | 68.20 | 54.08 | — | — | — | — | — | `saved_models/A800_models/tgtree_1B/step143658-unsharded`；旧 SG job=3441 NCCL timeout，不记录数值 |
| Treeterm（Tree eval） | 64.83 | 54.72 | — | — | — | — | — | 与上一行 Tree 共用 FineWeb checkpoint；terminal-format scoring，无独立权重 |
| TGTreeterm（TGTree eval） | 68.20 | 55.92 | — | — | — | — | — | 与上一行 TGTree 共用 FineWeb checkpoint；terminal-format scoring，无独立权重 |
| Pause-1 | 63.91 | 54.95 | — | — | — | — | — | `saved_models/A800_models/pause1_1B_SEP/step116061-unsharded` |
| Pause-2 | 61.62 | 55.68 | — | — | — | — | — | `saved_models/A800_models/pause2_1B_SEP/step141380-unsharded` |

## 5. 本轮任务登记与填写格式

| run id | 模型 | 任务 | 状态 | 证据路径 | 结果填写位置 |
|---|---|---|---|---|---|
| `terminal_terminal_doc_ppl_full_20260823` | Terminal-100M | Doc-PPL | COMPLETE, PPL=9.88981 | `artifacts/evaluation/terminal_doc_ppl_20260823/terminal/logs/slurm-terminal_terminal_doc_ppl_full_20260823-3444.out` | §3 Terminal-100M |
| `pause1_terminal_doc_ppl_full_20260823` | Pause-1-100M | Doc-PPL | COMPLETE, PPL=9.83125 | `artifacts/evaluation/terminal_doc_ppl_20260823/pause1/logs/slurm-pause1_terminal_doc_ppl_full_20260823-3445.out` | §3 Pause-1-100M |
| `pause2_terminal_doc_ppl_full_20260823` | Pause-2-100M | Doc-PPL | COMPLETE, PPL=9.92898（Slurm 3446） | `artifacts/evaluation/terminal_doc_ppl_20260823/pause2/logs/slurm-pause2_terminal_doc_ppl_full_20260823-3446.out` | §3 本轮 Doc-PPL 结果补充 |
| `terminal_500m_terminal_doc_ppl_full_20260823` | Terminal-500M | Doc-PPL | COMPLETE, PPL=2.82518（Slurm 3447） | `artifacts/evaluation/terminal_doc_ppl_20260823/terminal_500m/logs/slurm-terminal_500m_terminal_doc_ppl_full_20260823-3447.out` | §3 本轮 Doc-PPL 结果补充 |
| `terminal_1b_terminal_doc_ppl_full_20260823` | Terminal-1B | Doc-PPL | FAILED：未开始 eval；缺失 FineWeb-Edu Arrow shards（Slurm 3448） | `artifacts/evaluation/terminal_doc_ppl_20260823/terminal_1b/logs/slurm-terminal_1b_terminal_doc_ppl_full_20260823-3448.out` | §4 Terminal；不记录数值 |
| `treereg_terminal_doc_ppl_full_20260823` | TreeReg-layer6（作废） | Doc-PPL | FAILED：未开始 eval；训练 dataloader 错指 `terminal/train.npy` | `artifacts/evaluation/terminal_doc_ppl_20260823/treereg/logs/slurm-treereg_terminal_doc_ppl_full_20260823-3449.out` | 不记录数值 |
| `treereg_layer9_terminal_doc_ppl_full_20260823` | TreeReg-layer9（补充） | Doc-PPL | COMPLETE, PPL=12.36841（Slurm 3451） | `artifacts/evaluation/terminal_doc_ppl_20260823/treereg_layer9/logs/slurm-treereg_layer9_terminal_doc_ppl_full_20260823-3451.out` | §3 本轮 Doc-PPL 结果补充 |
| `eval_treereg_layer9_BLiMP` | TreeReg-layer9（补充） | BLiMP | COMPLETE, accuracy=66.68%（Slurm 3453） | `analysis-output/logs/eval_treereg_layer9_BLiMP_3453.out` | §3 TreeReg-layer9（补充） |
| `terminal_SG_seed6198` / `terminal_blimp_seed6198` | Terminal-100M | SG / BLiMP | COMPLETE, 69.80% / 70.77%（Slurm 3454 / 3455） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal_{SG,blimp}_seed6198/logs/` | §3 Terminal-100M |
| `terminal-500M_SG_seed6198` / `terminal-500M_blimp_seed6198` | Terminal-500M | SG / BLiMP | COMPLETE, 71.11% / 64.74%（Slurm 3456 / 3457） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal-500M_{SG,blimp}_seed6198/logs/` | §3 Terminal-500M |
| `terminal-1B_SG_seed6198` / `terminal-1B_blimp_seed6198` | Terminal-1B | SG / BLiMP | FAILED：缺失 FineWeb-Edu Arrow shards（Slurm 3458 / 3459） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal-1B_{SG,blimp}_seed6198/logs/` | §4 Terminal；不记录数值 |
| `pause1_SG_seed6198` / `pause1_blimp_seed6198` | Pause-1-100M | SG / BLiMP | COMPLETE, 76.29% / 72.52%（Slurm 3460 / 3461） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/pause1_{SG,blimp}_seed6198/logs/` | §3 Pause-1-100M |
| `pause2_SG_seed6198` / `pause2_blimp_seed6198` | Pause-2-100M | SG / BLiMP | COMPLETE, 75.85% / 73.15%（Slurm 3462 / 3463） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/pause2_{SG,blimp}_seed6198/logs/` | §3 Pause-2-100M |
| `eval_treereg_layer9_SG` | TreeReg-layer9（补充） | SG | COMPLETE, accuracy=67.65%（Slurm 3464） | `analysis-output/logs/eval_treereg_layer9_SG_3464.out` | §3 TreeReg-layer9（补充） |
| `treereg_layer9_xsum_finetune_seed6198` | TreeReg-layer9（补充） | XSum 微调 + test | CANCELLED（Slurm 3465，用户取消）；该单 seed 结果不记录，已由五 seed campaign 替代 | `artifacts/evaluation/treereg_layer9_xsum_boolq_20260823/runs/treereg_layer9_xsum_finetune_seed6198/{config.yaml,logs/}` | §3 TreeReg-layer9 五 seed XSum / BoolQ |
| `treereg_layer9_boolq_seed6198` | TreeReg-layer9（补充） | BoolQ 微调 + eval | CANCELLED（Slurm 3466，用户取消）；该单 seed 结果不记录，已由五 seed campaign 替代 | `artifacts/evaluation/treereg_layer9_xsum_boolq_20260823/runs/treereg_layer9_boolq_seed6198/{config.yaml,logs/}` | §3 TreeReg-layer9 五 seed XSum / BoolQ |
| `treereg_layer9_multiseed_20260824` | TreeReg-layer9（补充） | XSum / BoolQ 五 seed 微调 + 独立评测 | COMPLETE：XSum R-AVG=17.59 ± 0.10，BoolQ=63.54 ± 0.56；XSum train/eval=3493_[0-4]/3495_[0-4]，BoolQ retry train/eval=3518_[5-9]/3519_[5-9]；初始 BoolQ 3493_[5-9] OOM，未计入 | `artifacts/experiment/treereg_layer9_multiseed_20260824/runs/` | §3 TreeReg-layer9（补充）及五 seed 明细 |
| `treereg_layer9_auxloss_multiseed_20260825` | TreeReg-layer9（parse-aligned + auxiliary loss） | XSum / BoolQ 五 seed微调 + 独立评测 | COMPLETE：XSum R-AVG=20.88 ± 0.03，BoolQ=65.87 ± 0.31；train=3528_[0-9]，eval=3531_[0-9]，全部 exit=0；每 10 global optimizer batch 应用 layer-9 TreeReg loss | `artifacts/experiment/treereg_layer9_auxloss_multiseed_20260825/runs/` | §3 parse-aligned auxiliary-loss 五 seed 对照；与 legacy 差异不能仅归因于 loss |
| `pushdown_finetune_5seeds_20260825` | Pushdown-100M（gold unary spans） | XSum / BoolQ 五 seed 微调 + 独立评测 | COMPLETE：XSum R-AVG=15.87 ± 0.17，BoolQ=65.49 ± 0.55；train=3541_[0-9]；valid eval=3576_[1-9] + 3582_0；XSum prompt 使用 gold spans，仅生成后缀联合搜索 token + attachment；原 3576_0 端口冲突，未计入 | `artifacts/experiment/pushdown_finetune_5seeds_20260825/runs/` | §3 Pushdown gold-unary-span 五 seed 明细 |
| `treereg_layer9_terminal_docppl_20260825` | TreeReg-layer9 | 新 terminal Doc-PPL | RESULT VALID：PPL=12.37，完整 148836/148836 records；Slurm 3561 显示 FAILED(1)，原因是评测已结束后 grep 使用错误 metric key，非计算失败，脚本已修复 | `artifacts/evaluation/treereg_layer9_terminal_docppl_20260825/run.log` | §3 TreeReg-layer9（补充；legacy FT 无辅助 loss） |
| `tree_shuffle_terminal_doc_ppl_full_20260823` | Tree-Shuffle-100M | Doc-PPL | FAILED：未开始 eval；checkpoint 缺少 `optim.pt`（Slurm 3450） | `artifacts/evaluation/terminal_doc_ppl_20260823/tree_shuffle/logs/slurm-tree_shuffle_terminal_doc_ppl_full_20260823-3450.out` | §3 本轮 Doc-PPL 结果补充；`terminal_doc`，诊断见 tree_shuffle_terminality_20260823.md |
| `tree_shuffle_{docppl,SG,blimp}_seed6198` | Tree-Shuffle-100M | Doc-PPL / SG / BLiMP | Doc-PPL COMPLETE=50.91（3467）；SG=68.77%（3468）及 BLiMP 3469 为 legacy auto/beam，均已作废；有效 SG/BLiMP 见 3473/3474 | `artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/tree_shuffle_{docppl,SG,blimp}_seed6198/` | §3 Tree-Shuffle-100M（已回填有效结果） |
| `tree_shuffle_terminal_syntax_SG_seed6198` | Tree-Shuffle-100M | SG | COMPLETE，accuracy=61.59%（Slurm 3473）；terminal-only teacher-forced，替代 legacy 3468 | `artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_SG_seed6198/logs/slurm-tree_shuffle_SG_seed6198-3473.out` | §3 新版 model-only 结果 |
| `tree_shuffle_terminal_syntax_blimp_seed6198` | Tree-Shuffle-100M | BLiMP | COMPLETE，accuracy=71.68%（Slurm 3474）；terminal-only teacher-forced，替代 legacy 3469 | `artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_blimp_seed6198/logs/slurm-tree_shuffle_blimp_seed6198-3474.out` | §3 新版 model-only 结果 |
| `terminal-1B_{docppl,SG,blimp}_seed6198` | Terminal-1B（BBC checkpoint） | Doc-PPL / SG / BLiMP | COMPLETE：PPL=1.705、SG=67.41%、BLiMP=63.31%（3470/3471/3472）；model-only restore，已移除继承 FineWeb shards | `artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/terminal-1B_{docppl,SG,blimp}_seed6198/` | §3 新版 model-only 结果 |
| `pause_sep_100m_sist_20260828` | Pause-1/2-100M dedicated SEP | 从头预训练；terminal + SEP 50261；8 GPU | ACTIVE：Pause-1 job 988670 RUNNING；Pause-2 job 988671 PENDING (`afterok:988670`)；计算节点已通过 `FLASH_ATTN_OK 2.8.4` 与 `CONFIG_OK` | `artifacts/experiment/pause_sep_100m_sist_20260828/` | §2A checkpoint 级复核后的补充重训；完成前不填结果 |

每次完成后，在对应单元格填写数值，并在“重跑来源 / protocol”写成：

`job=<id>; checkpoint=<absolute path>; evaluator=<type/label>; dataset=<path>; protocol=<K/格式>; log=<path>; completed=<ISO date>`

若失败，填 `FAILED` 和错误摘要，但不以失败覆盖论文值。若改变 tokenizer、训练数据、
checkpoint step、评测集版本或 terminal/full scoring，必须新开 run id，不得覆盖原 run。
