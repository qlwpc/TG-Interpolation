# 论文实验复现登记表

更新：2026-08-31。本文是本仓库的唯一人工维护登记表，用于把论文中的
模型、可加载 checkpoint、论文报告值和本次重跑值放在同一张记录中。

> [!IMPORTANT]
> 仓库整理时先读 [`REPOSITORY_CLEANUP_MEMORY.md`](REPOSITORY_CLEANUP_MEMORY.md)。其中集中
> 记录已确认错误的**原因**、协议替代关系和未关闭风险。本表只保留有效主结果与定义
> 明确的协议对照；错误实现产生的数值和无结果的失败任务不再列入。

来源：

- 论文：`14901_A_Scaled_Up_Empirical_St.pdf`，Table 3/4/6/7。
- checkpoint：本机 `saved_models/` 的实际目录；均已检查目录存在。
- `artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/` 只在逐 run
  `TRAIN_DONE/EVAL_DONE`、日志和 checkpoint 身份均可核验时使用；其中复制式 Pause
  BoolQ 作为历史对照录入；旧 XSum v1 数值已删除，只在记忆文档保留错误原因。

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
| Tree-Shuffle（unmasked 历史对照） | 100M / BBC | LIN1shuf, causal | `saved_models/Tree_shuffle_pretrain/step49440-unsharded` | `tree_shuffle` | 可用；不是当前论文行的首选 checkpoint |
| Tree-Shuffle（mask；论文行恢复） | 100M / BBC | LIN1shuf, NT label mask | `saved_models/treeshufflemask_pretrain/step49440-unsharded` | `tree_shuffle_mask` | 可用；2026-08-30 五项完整复现与论文行一致 |
| TGNomask-Mix-TG | 100M / BBC | LIN2, mixed heads | `saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded` | `nomask_mix_tg` | 可用 |
| TGTree-Mix-TG | 100M / BBC | LIN2, mixed heads | `saved_models/tgtree_mix_tg_pretrain/step69817-unsharded` | `tree_mix_tg` | 可用 |
| Pause-1（复制式历史对照） | 100M / BBC | 复制真实 token，1 slot/token | `saved_models/pretrain_pause1_100M/step45487-unsharded` | `pause1` | 可用；`pause_token_id=null`，不是论文式 learned pause token |
| Pause-2（复制式历史对照） | 100M / BBC | 复制真实 token，2 slots/token | `saved_models/pretrain_pause2_100M/step52609-unsharded` | `pause2` | 可用；`pause_token_id=null`，不是论文式 learned pause token |
| Pause-1（论文式 dedicated SEP） | 100M / BBC | 1 learned SEP/token | `saved_models/pretrain_pause1_100M_SEP50261_steplaw/step45487-unsharded` | `pause1` | 可用；`pause_token_id=50261` |
| Pause-2（论文式 dedicated SEP） | 100M / BBC | 2 learned SEP/token | SIST `saved_models/pretrain_pause2_100M_SEP50261_steplaw/step54156-unsharded` | `pause2` | SIST 可用；`pause_token_id=50261`，sequence length 2049 |
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

现存两个同为 step 49440 的 Tree-Shuffle checkpoint。旧
`Tree_shuffle_pretrain` 是 unmasked `tree_shuffle` 历史对照；恢复出的
`treeshufflemask_pretrain` 是 `tree_shuffle_mask`。checkpoint/protocol 审计显示，masked
checkpoint 的完整 SG 为 75.83%（historical tree/auto）或 76.13%（terminal-only），
而 unmasked 对应 68.77% / 61.59%；同一 10 句 forward slice 的 terminal PPL 也为
12.24 vs 30.71。证据见
`artifacts/experiment/tree_shuffle_checkpoint_protocol_20260829/RESULTS.md`。2026-08-30
进一步以 masked checkpoint 完成五项正式复现，五项均与论文行接近，因此本登记表将
masked checkpoint 作为论文 Tree-Shuffle 行的恢复映射，旧 unmasked 结果只保留为历史
对照。基础 SG、BLiMP、Doc-PPL 使用 terminal 评测；XSum/BoolQ 微调使用
`tree_shuffle_mask`，不把两套 checkpoint 或 protocol 混写。

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
| Tree-Shuffle（unmasked 历史对照） | LIN1，collator 在线 shuffle NT | `tree_shuffle` | 24.706B / 49440 | 0.007000 | 244 / 28 | shuffle 不是独立 `.npy`；NT label 未屏蔽 |
| Tree-Shuffle（mask；论文行恢复） | LIN1，collator 在线 shuffle NT + NT label mask | `tree_shuffle_mask` | 24.706B / 49440 | 0.007000 | 244 / 28 | checkpoint=`treeshufflemask_pretrain`；仅 terminal label 计入 LM loss |
| TGNomask-Mix-TG | LIN2 | `mixing` | 32.029B / 69817 | 0.007600 | 224 / 28 | `mix_head_type=[tg:6,tgnomask:6]` |
| TGTree-Mix-TG | LIN2 | `mixing` | 32.029B / 69817 | 0.007600 | 224 / 28 | `mix_head_type=[tgtree:6,tg:6]` |
| Pause-1（复制式） | terminal 在线复制 1 slot/token | `pause1` | 20.122B / 45487 | 0.006585 | 216 / 8 | seq=2048，`pause_token_id=null`，实际重复组内 terminal token |
| Pause-2（复制式） | terminal 在线复制 2 slots/token | `pause2` | 30.183B / 52609 | 0.007595 | 280 / 15 | **seq=2049**，`pause_token_id=null`，实际重复 terminal token |
| Pause-1（论文式 SEP） | terminal 在线插入 1 SEP/token | `pause1` | 20.122B / 45487 | 0.006585 | 216 / 9（accum=3） | seq=2048，`pause_token_id=50261`；SIST job 988670 |
| Pause-2（论文式 SEP） | terminal 在线插入 2 SEP/token | `pause2` | 30.183B / 54156 | 0.007458 | 272 / 17（accum=2） | **seq=2049**，`pause_token_id=50261`；SIST job 988671 |

旧 Pause-1/2-100M 的 `pause_token_id=null` 很重要：其实现不是插入一个独立可学习的
共享 pause embedding，而是把每组最后一个真实 token 广播到 pause slot。论文把控制
描述为 learned latent token，因此本登记表将这些旧 checkpoint 统一称为
“复制式 Pause（repeat-token compute control）”；论文模型名称和值只对应后面的
dedicated-SEP 重训，不再与复制式 checkpoint 混用。

**Checkpoint 级复核（2026-08-28）。** 本机这两个目录各只有一个实际 final step：
`pretrain_pause1_100M/step45487-unsharded` 与
`pretrain_pause2_100M/step52609-unsharded`；`latest-unsharded` 只是指向对应 step 的软链接，
并不存在第二个隐藏的 SEP step。run-root 与 step 内的 YAML 均为
`pause_token_id: null`。更关键的是，直接用权重预测 pause 槽时，Pause-1/2 对“前一个
真实 token”的 top-1 命中率均为 100%，平均概率分别为 0.9844/0.9896；GPT-2 embedding
范围内候选特殊 token 50257--50263 的平均概率仅为约 1e-9--1e-6。对 held-out BBC
片段进行协议似然比较也一致：repeat 序列的 pause NLL 为 0.0063/0.0089，而固定
SEP-50263 序列为 28.674/12.126。因此，本机这两个 **100M 权重本身就是 repeat-token**，
不是“SEP 训练但 YAML 漏写”。真正采用独立 SEP 的 checkpoint 包括下文 FineWeb-Edu
1B 的 `pause1_1B_SEP` / `pause2_1B_SEP`（id 151673），以及本轮重新训练的 100M
SEP-50261 checkpoint；两类权重的身份和结果必须分开登记。

为补齐这一缺口，2026-08-28 已在 SIST `ShangHAI` 分区启动独立 SEP 的 100M
重训 campaign：`pause_sep_100m_sist_20260828`。两者都使用 terminal train stream、
seed 6198、`pause_token_id=50261`、FlashAttention-2 和严格按实际 pause-expanded D
重算的 Step Law。Pause-1 为 LR=0.006585、global/device/micro/accum=
216/27/9/3（job 988670，8×RTX 6000D，`COMPLETED/0:0`，耗时 06:49:49）；Pause-2
为 LR=0.007458、272/34/17/2（job 988671，`COMPLETED/0:0`，耗时 11:03:04）。
Pause-1/2 最终 checkpoint 分别为 step45487/step54156，均完成日志、SHA-256、配置和
反序列化验收；模型哈希分别以 `a5cd2c46...` / `1f494...` 开头。配置、公式展开、
提交脚本、job manifest 与迁移验收记录位于
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
| TGTree、TGNomask、TGNomask-Aug、三种 Tree 数据变体、Tree-Shuffle、复制式及 SEP Pause-1/2-100M | 无独立正式模板 | 否；以对应 checkpoint 的 `config.yaml` 为唯一核心参数源，重写绝对数据路径；Pause 还必须核对 `pause_token_id` |
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

本表中 20 个 scale-up 模型的 XSum/BoolQ 均值来自
`artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/STATUS_AND_RESULTS.md`，
按五个 fine-tuning seed 汇总为均值 ± 样本标准差。源文件中的 Pause-1/2 是
`pause_token_id=null` 的复制式模型，因此其**有效非 XSum 指标**只回填“复制式；历史
对照”行；旧 XSum v1 使用错误实现，数值已删除，原因见记忆文档 M-02。

| 模型 | 论文 XSum R-AVG | 论文 BoolQ | 论文 SG | 论文 BLiMP | 论文 Doc-PPL | 本次 XSum | 本次 BoolQ | 本次 SG | 本次 BLiMP | 本次 Doc-PPL | 重跑来源 / protocol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Terminal-100M | 21.99 | 67.40 | 69.98 | 70.77 | 10.19 | 22.02 ± 0.06 | 66.70 ± 0.36 | 69.80 | 70.77 | 9.88981 | `SG job=3454; BLiMP job=3455; checkpoint=saved_models/Terminal-lr005-bs144/step34115-unsharded; evaluator=terminal teacher-forced; dataset=full SG suite / full BLiMP (67×1000 pairs); BLiMP warnings=7 tied-or-non-finite pairs; logs=artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal_{SG,blimp}_seed6198/logs/; completed=2026-08-23. Doc-PPL: job=3444; evaluator=terminal_doc/terminal_doc_ppl; dataset=dataset/bbc-news/terminal/test.npy; protocol=K1,BOS-context,EOS-scored,denom=3284061; log=artifacts/evaluation/terminal_doc_ppl_20260823/terminal/logs/slurm-terminal_terminal_doc_ppl_full_20260823-3444.out` |
| Tree-100M | 22.82 | 67.86 | 81.64 | 80.90 | 10.17 | 22.84 ± 0.03 | 68.07 ± 0.35 | 81.40 | 80.95 | 10.01 | `SG job=3419; checkpoint=saved_models/Tree_test/step49440-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tree_beam300_len6_3419.out; completed=2026-08-21. Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27` |
| TGTree-100M | 22.69 | 68.38 | 79.40 | 81.48 | 10.36 | 23.03 ± 0.02 | 68.12 ± 0.62 | 79.47 | 81.46 | 10.21 | `SG job=3418; checkpoint=saved_models/TGtree/step69817-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgtree_beam300_len6_3418.out; completed=2026-08-21. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgtree_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-25` |
| TG-100M | 20.13 | 64.31 | 78.80 | 83.23 | 11.20 | 20.15 ± 0.04 | 64.50 ± 0.40 | 79.19 | 83.28 | 11.02 | `SG job=3413; checkpoint=saved_models/TG_test/step55457-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tg_beam300_len6_3413.out; completed=2026-08-21. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tg_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27` |
| TGNomask-100M | 21.60 | 67.25 | 78.47 | 81.77 | 10.14 | 21.91 ± 0.04 | 66.66 ± 0.45 | 78.62 | 81.78 | 9.971 | `SG job=3414; checkpoint=saved_models/nomask_test/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgnomask_beam300_len6_3414.out; completed=2026-08-21. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgnomask_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27` |
| TGNomask-Aug-100M | 22.04 | 68.81 | 78.44 | 83.64 | 10.24 | 22.12 ± 0.02 | 67.01 ± 0.28 | 78.68 | 83.63 | 10.08 | `SG job=3415; checkpoint=saved_models/TGnomask_aug_pretrain/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgnomask_aug_beam300_len6_3415.out; completed=2026-08-21. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgnomask_aug_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28 [10.08 identical on both flex_attention=true and false paths; flex=true run tgnomask_aug_docppl_seed6198 completed 2026-08-29 04:16 with doc_ppl=10.08]` |
| Tree-NoONT-100M | 22.29 | 68.13 | 85.37 | 81.00 | 10.08 | 22.41 ± 0.04 | 68.67 ± 0.32 | 85.38 | 81.00 | 9.947 | `SG job=3475; checkpoint=saved_models/tree_noont/step42440-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=tree_noont; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_noont_beam300_len6_3475.out; completed=2026-08-24. Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_noont_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-26` |
| Tree-Compress-100M | 22.63 | 68.04 | 78.18 | 81.44 | 10.09 | 22.70 ± 0.05 | 68.48 ± 0.49 | 78.27 | 81.40 | 9.909 | `SG job=3476; checkpoint=saved_models/tree_compress/step45965-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=tree_compress; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_compress_beam300_len6_3476.out; completed=2026-08-24. Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_compress_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-26` |
| Tree-TripleCNT-100M | 21.18 | 69.02 | 79.83 | 83.38 | 10.41 | 22.43 ± 0.04 | 68.09 ± 0.12 | 81.57 | 83.42 | 10.26 | `SG job=3479; checkpoint=saved_models/tree_triplecnt/step60045-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=tree_triplecnt; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tree_triplecnt_beam300_len6_3479.out; completed=2026-08-24. Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_triplecnt_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-26` |
| Tree-Shuffle-100M（unmasked 历史对照） | — | — | — | — | — | 21.36 ± 0.08 | 65.67 ± 0.23 | 61.59 | 71.68 | 50.91 | `SG job=3473; BLiMP job=3474; Doc-PPL job=3467; checkpoint=saved_models/Tree_shuffle_pretrain/step49440-unsharded; grammar=tree_shuffle (unmasked); evaluator=terminal-only teacher-forced / terminal_doc; dataset=full SG suite / full BLiMP / BBC terminal test; protocol=model-only restore, input_format=terminal,beam_search=false; logs=artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_{SG,blimp}_seed6198/logs/; completed=2026-08-23. 该行不再承载论文 Table 4 数值` |
| **Tree-Shuffle-100M（mask；论文行恢复）** | **21.76** | **67.77** | **76.96** | **67.67** | **13.69** | **21.56 ± 0.09** | **68.08 ± 0.33** | **76.56** | **67.77** | **13.41** | `job=989550_[0-4]; checkpoint=saved_models/treeshufflemask_pretrain/step49440-unsharded; base evaluator=terminal teacher-forced / terminal_doc; downstream grammar=tree_shuffle_mask; XSum=3ep,lr=6e-5; BoolQ=5ep,lr=3e-4; fp32 DDP 4×RTX6000D,global batch=40,microbatch=10; seeds=6198,13171,31723,42,2026; all array elements COMPLETED/0:0,10/10 TRAIN_DONE,13/13 EVAL_DONE; evidence=artifacts/experiment/treeshufflemask_terminal_multiseed_20260830/{REPORT.md,results.json,runs/,slurm/}; completed=2026-08-30` |
| TGNomask-Mix-TG-100M | 21.86 | 67.49 | 75.96 | 83.15 | 10.16 | 21.87 ± 0.04 | 66.13 ± 0.54 | 75.77 | 83.14 | 10.01 | `SG job=3416; checkpoint=saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU; log=analysis-output/sg_100m_rtx3090_20260820/logs/sg_100m_tgnomask_mix_tg_beam300_len6_3416.out; completed=2026-08-21. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/nomask_mix_tg_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27` |
| TGTree-Mix-TG-100M | 22.25 | 67.74 | 79.25 | 82.76 | 10.16 | 22.16 ± 0.07 | 67.91 ± 0.36 | 78.84 | 82.75 | 10.00 | `SG job=3478; checkpoint=saved_models/tgtree_mix_tg_pretrain/step69817-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,2 GPU, grammar=mixing; log=analysis-output/sg_100m_additional_rtx3090_20260823/logs/sg_100m_tgtree_mix_tg_beam300_len6_3478.out; completed=2026-08-24. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree_mix_tg_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28` |
| Pause-1-100M（复制式；历史对照） | — | — | — | — | — | — | 67.85 ± 0.38 | 76.29 | 72.52 | 9.83125 | `checkpoint=saved_models/pretrain_pause1_100M/step45487-unsharded; pause_token_id=null; protocol=repeat previous real token. SG job=3460; BLiMP job=3461; Doc-PPL job=3445. BoolQ five-seed=artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/runs/pause1_100m_boolq_seed*/. XSum v1 数值因错误实现已删除，见记忆文档 M-02` |
| Pause-2-100M（复制式；历史对照） | — | — | — | — | — | — | 68.51 ± 0.48 | 75.85 | 73.15 | 9.92898 | `checkpoint=saved_models/pretrain_pause2_100M/step52609-unsharded; pause_token_id=null; protocol=repeat previous real token. SG job=3462; BLiMP job=3463; Doc-PPL job=3446. BoolQ five-seed=artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/runs/pause2_100m_boolq_seed*/. XSum v1 数值因错误实现已删除，见记忆文档 M-02` |
| Pause-1-100M（论文式 dedicated SEP） | 19.86 | 66.06 | 78.47 | 69.99 | 10.64 | **22.38 ± 0.05** | **62.04 ± 0.19** | **75.11** | **70.85** | **9.771** | `pretrain job=988670; checkpoint=saved_models/pretrain_pause1_100M_SEP50261_steplaw/step45487-unsharded; grammar=pause1; pause_token_id=50261. SG/BLiMP/Doc-PPL jobs=3610/3611/3612. XSum v2 five-seed job=989161; full test, phase-constrained KV-cache generation, eval batch=1. BoolQ five-seed train retry/eval=3642_[5-9]/3614_[5-9], microbatch=1, global batch=40. evidence=artifacts/experiment/pause1_sep50261_full_eval_20260829/ and SIST pause1_sep50261_xsum_v2_sist_20260829/` |
| Pause-2-100M（论文式 dedicated SEP；补充） | — | — | — | — | — | **22.25 ± 0.06** | **65.70 ± 3.45** | **76.97** | **72.59** | **10.32** | `pretrain job=988671; checkpoint=SIST saved_models/pretrain_pause2_100M_SEP50261_steplaw/step54156-unsharded; grammar=pause2; pause_token_id=50261; seq=2049. SG/BLiMP/Doc-PPL jobs=989010/989011/989012. XSum v2 five-seed job=989161. BoolQ completion job=989304_[5-9], 5/5 TRAIN_DONE/EVAL_DONE, full validation, microbatch=1, global batch=40; evidence=SIST artifacts/experiment/pause2_sep50261_{full_eval,xsum_v2}_sist_20260829/` |
| Terminal-500M | 20.71 | 69.97 | 71.25 | 64.74 | 3.06 | 20.75 ± 0.06 | 69.69 ± 0.60 | 71.11 | 64.74 | 2.82518 | `SG job=3456; BLiMP job=3457; checkpoint=saved_models/terminal_500M/step34115-unsharded; evaluator=terminal teacher-forced; dataset=full SG suite / full BLiMP (67×1000 pairs); BLiMP warnings=7 tied-or-non-finite pairs; logs=artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal-500M_{SG,blimp}_seed6198/logs/; completed=2026-08-23. Doc-PPL: job=3447; evaluator=terminal_doc/terminal_doc_ppl; dataset=dataset/bbc-news/terminal/test.npy; protocol=K1,BOS-context,EOS-scored,denom=3284061; log=artifacts/evaluation/terminal_doc_ppl_20260823/terminal_500m/logs/slurm-terminal_500m_terminal_doc_ppl_full_20260823-3447.out` |
| Tree-500M | 22.05 | 71.90 | 63.26 | 76.07 | 3.32 | 22.25 ± 0.05 | 70.72 ± 0.35 | 63.13 | 76.09 | 3.182 | `SG job=3437; checkpoint=saved_models/Tree_500M/step49440-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,non-flex,2 GPU; log=analysis-output/sg_500m_nonflex_rtx3090_20260822/logs/sg_500m_tree_beam300_len6_nonflex_3437.out; completed=2026-08-22. Flex reference: A6000 job=38152, 62.89%. Doc-PPL: evaluator=txl_approx_doc; dataset=dataset/bbc-news/testppl_tree/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tree-500M_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-27` |
| TGTree-500M | 21.96 | 70.86 | 66.50 | 77.63 | 3.40 | 22.01 ± 0.08 | 70.63 ± 0.32 | 65.95 | 77.65 | 3.279 | `SG job=3438; checkpoint=saved_models/TGTree_500M/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,non-flex,2 GPU; log=analysis-output/sg_500m_nonflex_rtx3090_20260822/logs/sg_500m_tgtree_beam300_len6_nonflex_3438.out; completed=2026-08-22. Flex reference: A6000 job=38153, 65.59%. Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgtree-500M_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28` |
| TGNomask-Aug-500M | 21.63 | 71.04 | 57.73 | 75.99 | 3.32 | 21.77 ± 0.06 | 70.78 ± 0.74 | 57.58 | 75.97 | 3.188 | `SG job=3439; checkpoint=saved_models/TGnomaskaug_500M/step55853-unsharded; evaluator=word_sync beam search; dataset=full SG suite; protocol=beam=300,nc=max(term_len,5),pc=3,max_len=6×term_len,non-flex,2 GPU; log=analysis-output/sg_500m_nonflex_rtx3090_20260822/logs/sg_500m_tgnomask_aug_beam300_len6_nonflex_3439.out; completed=2026-08-22. 本表只登记 non-flex SG，暂无有效 flex 对照。Doc-PPL: evaluator=tg_approx_doc; dataset=dataset/bbc-news/testppl_tg/ (bos/eos-normalized, transferred 2026-08-25 from this host to local H20 cluster, md5-verified); protocol=SENT_SIZE=300, batch=60, steps=744180 over 44,650,800 records, seed=6198; log=artifacts/evaluation/docppl_structural_20260825/runs/tgnomask_aug-500M_docppl_seed6198/logs/launch.out (local H20 cluster, log not on this host); completed=2026-08-28` |
| Terminal-1B（BBC，补充） | — | — | — | — | — | 20.47 ± 0.06 | 70.64 ± 1.18 | 67.41 | 63.31 | 1.705 | `SG job=3471; BLiMP job=3472; Doc-PPL job=3470; checkpoint=saved_models/terminal_1B/step34115-unsharded; evaluator=terminal teacher-forced / terminal_doc; dataset=full SG suite / full BLiMP / BBC terminal test; protocol=model-only restore, reset optimizer/trainer state; logs=artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/terminal-1B_{docppl,SG,blimp}_seed6198/logs/; completed=2026-08-23` |
| Tree-1B-BBC（补充） | — | — | — | — | — | 21.42 ± 0.08 | 70.95 ± 0.37 | — | 73.37 | — | `checkpoint=saved_models/Tree_1B/step49440-unsharded; BLiMP=direct downstream over existing 300 parses; protocol and evidence in the direct300 subsection below; completed=2026-08-28` |
| TGTree-1B-BBC（Qwen，补充） | — | — | — | — | — | — | — | — | 80.98 | — | `checkpoint=saved_models/tgtree_1B/step143658-unsharded; tokenizer=dataset/TG_QWEN3_tokenizer.json; BLiMP=direct downstream over existing Qwen 300 parses; protocol and evidence in the direct300 subsection below; completed=2026-08-28` |
| TreeReg-layer9（补充；legacy FT 无辅助 loss） | — | — | — | — | — | 17.59 ± 0.10 | 63.54 ± 0.56 | 67.65 | 66.68 | 12.37 | `XSum/BoolQ five-seed mean ± sample-SD: training=3493_[0-4] / 3518_[5-9]; eval=3495_[0-4] / 3519_[5-9]; checkpoint=saved_models/treereg_layer9/step34354-unsharded; protocol=scaleup_nonfineweb_multiseed_20260815 settings, fp32 DDP 4 GPU, global batch=40, XSum 3ep LR=6e-5, BoolQ 5ep LR=3e-4, device microbatch=1. Although config declared treereg_alpha=1, downstream batches had no tree_spans / word boundaries / sentence ids, so train.py did not execute the auxiliary branch; it also retained TG-token input rather than parse-aligned terminal input. SG job=3464; BLiMP job=3453; evaluator=terminal teacher-forced, no test-time tree input; dataset=full SG suite / full BLiMP (67×1000 pairs); logs=analysis-output/logs/eval_treereg_layer9_{SG,BLiMP}_{3464,3453}.out; completed=2026-08-23. terminal_doc: complete 148836 records, PPL=12.37; log=artifacts/evaluation/treereg_layer9_terminal_docppl_20260825/run.log; protocol=K1,BOS-context,EOS-scored` |
| TreeReg-layer9（parse-aligned + auxiliary loss） | — | — | — | — | — | 20.88 ± 0.03 | 65.87 ± 0.31 | — | — | — | `Five-seed parse-aligned fine-tune: train=3528_[0-9], eval=3531_[0-9], all exit=0; checkpoint=saved_models/treereg_layer9/step34354-unsharded; XSum=3ep LR=6e-5, BoolQ=5ep LR=3e-4, fp32 DDP 4 GPU, device microbatch=1, global batch=40. Loader converts local NT-token trees with right binarization + unary collapse and supplies spans, word boundaries, sentence ids; TreeReg layer=9, heads=3, alpha=1, every 10 global optimizer batches. Evidence=artifacts/experiment/treereg_layer9_auxloss_multiseed_20260825/runs/. No post-finetune SG/BLiMP/Doc-PPL run.` |
| Pushdown-100M（sentence-local gold spans） | — | — | — | — | — | 21.06 ± 0.05 | 65.54 ± 0.29 | 74.86 | 74.97 | — | `Five-seed XSum/BoolQ fine-tune on the local H20 cluster; seeds=6198,13171,31723,42,2026; checkpoint=saved_models/pushdown_terminalonly/step34354-unsharded; global batch=40, amp_bf16; XSum=3ep LR=6e-5; BoolQ=5ep LR=3e-4. XSum source prompts use gold unary-collapsed spans, while every generated summary starts from an empty ROOT-free sentence stack; beam=6,max_reduce=null, EOS has no attachment action. BoolQ fine-tuning normalizes summed attachment loss by the global valid attachment-query count; evaluation is full validation, gold-span teacher-forced terminal-format MC scoring, batch=40. All five TRAIN_DONE/EVAL_DONE in both campaigns; evidence=/2024233198/TG-Interpolation/artifacts/experiment/pushdown_{xsum_sentence_local,boolq_corrected}_5seeds_20260829/{metrics.json,runs/}; completed=2026-08-29. SG: job=2512; evaluator=full 32-task SG suite; protocol=incremental Pushdown parse marginalization,beam=300,max_reduce=null; score=74.86%; log=analysis-output/syntax_eval_34354/logs/pushdown_sg_beam300_step34354_2512.out; completed=2026-08-14. BLiMP: job=2521; evaluator=full BLiMP (67×1000 pairs); protocol=gold300 supplied-parse marginalization; score=74.97%; warnings=8 tied-or-non-finite pairs; log=analysis-output/syntax_eval_34354/logs/pushdown_blimp_gold300_step34354_2521.out; completed=2026-08-14. Terminal-only ablations are SG=35.98% and BLiMP=66.43% and are not used in the main cells.` |
| Pushdown-100M（native top-K Doc-PPL） | — | — | — | — | — | — | — | — | — | **13.293598** | `checkpoint=saved_models/pushdown_terminalonly/step34354-unsharded; evaluator=native_document_ppl; dataset=BBC News full document test; protocol=up-to-300 native candidates/current sentence, truncated joint sum over token+attachment probabilities, candidate-0 prefix with persistent KV cache, BOS-context, EOS-scored,denom=3284061; result=/home/wangpch/TG-Interpolation/docppl_runs/pushdown_native_4gpu_45474/aggregate.json (local H20 cluster); finalizer=45580; completed=2026-08-29; paper value confirmed by the author on 2026-08-30` |

### Tree-Shuffle-Mask：terminal 基础评测与五 seed shuffle-tree-mask 微调（2026-08-30）

本轮固定 checkpoint 为
`saved_models/treeshufflemask_pretrain/step49440-unsharded`。SG 与 BLiMP 均为完整
terminal-only teacher-forced 评测；Doc-PPL 为 `terminal_doc_ppl`。为避免改写 SIST 上
哈希不一致且缺 sidecar 的共享 `terminal/test.npy`，Doc-PPL 使用 campaign 自带、哈希固定的
`data/terminal/{test.npy,test.csv.npy,test.idx.npy}`，仅在当前评测进程中改写 registry 路径。
XSum 与 BoolQ 从该预训练 checkpoint 独立微调，训练 grammar 均为
`tree_shuffle_mask`；XSum 使用 3 epochs、LR 6e-5，BoolQ 使用 5 epochs、LR 3e-4，
均为 fp32 DDP、4 GPU、global batch 40、device microbatch 10。

| seed | XSum R1 | XSum R2 | XSum RL | XSum R-AVG | BoolQ accuracy |
|---:|---:|---:|---:|---:|---:|
| 6198 | 30.29 | 10.13 | 23.90 | 21.44 | 68.20 |
| 13171 | 30.63 | 10.31 | 24.16 | 21.70 | 68.23 |
| 31723 | 30.44 | 10.20 | 23.97 | 21.54 | 68.23 |
| 42 | 30.41 | 10.24 | 24.00 | 21.55 | 67.49 |
| 2026 | 30.43 | 10.23 | 24.02 | 21.56 | 68.26 |
| **mean ± sample SD** | **30.440 ± 0.122** | **10.222 ± 0.065** | **24.010 ± 0.095** | **21.558 ± 0.093** | **68.082 ± 0.332** |

| terminal 基础指标 | 本次 | 论文 | 本次 − 论文 |
|---|---:|---:|---:|
| SG | 76.56 | 76.96 | −0.40 |
| BLiMP | 67.77 | 67.67 | +0.10 |
| Doc-PPL | 13.41 | 13.69 | −0.28 |
| XSum R-AVG | 21.558 ± 0.093 | 21.76 | −0.202 |
| BoolQ | 68.082 ± 0.332 | 67.77 | +0.312 |

Slurm array `989550_[0-4%2]` 五个元素均为 `COMPLETED/0:0`；共有 10/10
`TRAIN_DONE`、13/13 `EVAL_DONE`，collector 得到 13/13 个有限 canonical metric。
所有 `.out`/`.log` 未检出 Traceback、OOM、FAILED、AssertionError 或 RuntimeError。
因此结果状态为 **supported**：masked checkpoint 在五项指标上均复现论文行，且明显优于
旧 unmasked checkpoint 与论文值的对应关系。完整汇总和逐任务日志位于
`artifacts/experiment/treeshufflemask_terminal_multiseed_20260830/`。

### Pause：复制式历史对照与论文式 dedicated-SEP 重训（2026-08-29）

本文从本次更新起固定使用以下命名：`pretrain_pause{1,2}_100M` 是
`pause_token_id=null` 的**复制式 Pause**，pause slot 复制前一个真实 token；
`pretrain_pause{1,2}_100M_SEP50261_steplaw` 是 `pause_token_id=50261` 的
**论文式 dedicated-SEP Pause**。论文 Table 4 的 Pause-1 数值只放在后者所在行。
复制式 checkpoint 仅作为历史 compute-control，不再冒充论文方法。

论文式模型的基础评测均来自预训练 checkpoint 本身；SG 是完整 32-task suite，
BLiMP 是完整 67×1,000 pairs，Doc-PPL 是 148,836 句、4,966 文档、
3,284,061 个计分 terminal/EOS token 的 `terminal_doc_ppl`。XSum v2 从同一预训练
checkpoint 重新微调，修复了 context/pause-id 传递、pause 展开后的 summary label mask
以及 phase-constrained generation；旧 XSum finetune checkpoint 不再作为结果来源。

| 论文式模型 | SG | BLiMP | Doc-PPL | XSum R1 | XSum R2 | XSum RL | XSum R-AVG | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pause-1 SEP50261 | 75.11 | 70.85 | 9.771 | 31.276 ± 0.066 | 10.968 ± 0.022 | 24.892 ± 0.049 | **22.378 ± 0.046** | **62.036 ± 0.194** |
| Pause-2 SEP50261 | 76.97 | 72.59 | 10.32 | 31.124 ± 0.053 | 10.856 ± 0.065 | 24.768 ± 0.072 | **22.250 ± 0.061** | **65.700 ± 3.453** |

XSum 与 BoolQ 的 `±` 均为五个共享 fine-tuning seed 的样本标准差；单 seed 如下。
Pause-2 dedicated-SEP BoolQ 的旧 SIST 尝试 989023 只完成了 seed 6198/13171 的
训练，没有正式 validation；completion array `989304_[5-9]` 随后复用这两个完整
step-1175 checkpoint，并完成其 validation 以及其余三个 seed 的训练和 validation。
五个任务均为 `COMPLETED/0:0`，且有 5/5 `TRAIN_DONE/EVAL_DONE`。高低 seed 的
config diff 只有 seed、run name 与保存路径，因此保留实际较大的跨 seed 方差，
不借用复制式 Pause-2 的 68.51 ± 0.48。

| seed | Pause-1 SEP XSum R-AVG | Pause-2 SEP XSum R-AVG | Pause-1 SEP BoolQ | Pause-2 SEP BoolQ |
|---:|---:|---:|---:|---:|
| 6198 | 22.30 | 22.30 | 61.80 | 61.71 |
| 13171 | 22.39 | 22.28 | 62.23 | 62.17 |
| 31723 | 22.39 | 22.20 | 62.02 | 68.53 |
| 42 | 22.42 | 22.17 | 61.90 | 67.65 |
| 2026 | 22.39 | 22.30 | 62.23 | 68.44 |
| mean ± sample SD | 22.378 ± 0.046 | 22.250 ± 0.061 | 62.036 ± 0.194 | 65.700 ± 3.453 |

XSum job `989161_[0-4]` 五个元素全部 `COMPLETED/0:0`；每个元素在同一 seed 上
串行运行 Pause-1/Pause-2，完整评测为 `eval_subset_num_batches=-1`、每 rank
2,834 eval steps。证据分别位于 SIST
`artifacts/experiment/pause{1,2}_sep50261_xsum_v2_sist_20260829/runs/`。
Pause-1 BoolQ 五个 `TRAIN_DONE/EVAL_DONE` 位于
`artifacts/experiment/pause1_sep50261_full_eval_20260829/runs/`。
Pause-2 BoolQ completion job `989304_[5-9]` 的五个完整 validation 日志与标记位于
SIST `artifacts/experiment/pause2_sep50261_full_eval_sist_20260829/runs/`；日志未检出
Traceback、OOM、NaN 或 fatal error。

复制式历史对照的有效非 XSum 结果为：Pause-1 SG/BLiMP/Doc-PPL/BoolQ =
76.29/72.52/9.83125/67.85 ± 0.38；Pause-2 =
75.85/73.15/9.92898/68.51 ± 0.48。旧 XSum v1 使用错误的 label mask、pause
convention 和生成路径，数值已从仓库正式文档删除；根因见记忆文档 M-02。

### BLiMP existing-300-parse direct downstream（本地 H20，2026-08-27—29）

本节结果是对已有 300 棵 parse tree 的 direct downstream 打分，**不是 beam search**。
统一协议为 full BLiMP（67 tasks × 1,000 minimal pairs）、model-only restore、
`beam_search=false`、`samples_per_sent=300`、`tree_eval_type=default`；每个句子的
300 个 joint tree losses 通过 `logsumexp(-CE)` 边缘化，再比较 grammatical / ungrammatical
成员。Tree-family 使用 `dataset/BLiMP/tree300/blimp_tree_300.npy`，TG/LIN2-family 使用
`blimp_tg_300.npy`，Qwen TGTree-1B 使用 `blimp_tree_300_qwen.npy`。DataLoader 为
`num_workers=4,prefetch_factor=2,pin_memory=true,persistent_workers=true`；所有已完成运行均有
7 个 tied-or-non-finite minimal pairs。日志位于本地 H20 集群
`/2024233198/TG-Interpolation/analysis-output/blimp_direct300_20260827/logs/`
（不在本 3090B 主机）。

| 模型 | 本次 BLiMP | eval batch / GPU | steps | 完成时间（UTC） |
|---|---:|---:|---:|---|
| Tree-100M | 80.95 | 1200 / 1 | 33,500 | 2026-08-29 06:31 |
| TGTree-100M | 81.46 | 1000 / 1 | 40,200 | 2026-08-27 16:43 |
| TG-100M | 83.28 | 1200 / 1 | 33,500 | 2026-08-28 00:18 |
| TGNomask-100M | 81.78 | 1000 / 1 | 40,200 | 2026-08-28 07:54 |
| TGNomask-Aug-100M | 83.63 | 1000 / 1 | 40,200 | 2026-08-28 15:30 |
| Tree-NoONT-100M | 81.00 | 1200 / 1 | 33,500 | 2026-08-29 03:19 |
| Tree-Compress-100M | 81.40 | 1200 / 2 | 16,750/rank | 2026-08-29 07:48 |
| TGNomask-Mix-TG-100M | 83.14 | 1000 / 1 | 40,200 | 2026-08-28 23:14 |
| TGTree-Mix-TG-100M | 82.75 | 1000 / 1 | 40,200 | 2026-08-29 06:57 |
| Tree-500M | 76.09 | 1200 / 1 | 33,500 | 2026-08-29 02:15 |
| TGTree-500M | 77.65 | 1000 / 1 | 40,200 | 2026-08-29 05:21 |
| TGNomask-Aug-500M | 75.97 | 1000 / 1 | 40,200 | 2026-08-28 09:51 |
| Tree-1B-BBC | 73.37 | 1200 / 1 | 33,500 | 2026-08-28 12:05 |
| TGTree-1B-BBC（Qwen） | 80.98 | 300 / 1 | 134,000 | 2026-08-28 23:20 |
| Tree-TripleCNT-100M | 83.42 | 1000 / 2 | 20,100/rank | 2026-08-29 09:42 |

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

XSum 的训练/评测链为 `3493_[0-4] → 3495_[0-4]`；BoolQ 为
`3518_[5-9] → 3519_[5-9]`，device microbatch=1、global batch=40。十个训练与十个
独立评测均以退出码 0 完成；逐 seed 日志在
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

### Pushdown：sentence-local gold-span 五 seed XSum / corrected BoolQ（2026-08-29）

本组从 `saved_models/pushdown_terminalonly/step34354-unsharded` 独立微调。训练加载器
将 parsed TG 树转换为 terminal token stream，并对 right-binarized tree 做 unary
collapse；所得 gold spans 同时指导 Pushdown attention/stack tape 和 attachment-head
监督。XSum 评测直接加载 source prompt 的 gold parse spans，prompt 部分不做 latent
beam parse；每个新生成的 summary sentence 从空的 ROOT-free stack 开始，prompt spans
只保留为 attention history，不能成为 summary 的 attachment target。生成联合搜索 token
与 attachment，设置为 `beam=6,max_reduce=null`，EOS 不执行 attachment action。

| seed | XSum ROUGE-1 | XSum ROUGE-2 | XSum ROUGE-L | XSum R-AVG |
|---:|---:|---:|---:|---:|
| 6198 | 29.30 | 10.15 | 23.52 | 20.99 |
| 13171 | 29.43 | 10.18 | 23.59 | 21.07 |
| 31723 | 29.49 | 10.27 | 23.61 | 21.12 |
| 42 | 29.45 | 10.16 | 23.63 | 21.08 |
| 2026 | 29.36 | 10.17 | 23.60 | 21.05 |
| mean ± sample SD | 29.41 ± 0.08 | 10.19 ± 0.05 | 23.59 ± 0.04 | 21.06 ± 0.05 |

五个 seed 的 `TRAIN_DONE`、`EVAL_DONE` 和总完成标记均存在，日志未发现 traceback、
OOM、NaN、NCCL error 或 fatal error。批量候选打分与严格 top-6 upper-bound 提前终止
在同一 checkpoint、同一 8 条 XSum 输入上与优化前生成文本逐字节一致，目标回归测试
33/33 通过。正式证据位于本地 H20 cluster：
`/2024233198/TG-Interpolation/artifacts/experiment/pushdown_xsum_sentence_local_5seeds_20260829/`
（`metrics.json` 及逐 seed `runs/` 日志）。旧的 root-containing-stack XSum 记录无效，
已从本文的结果表、逐 seed 明细和任务账本中移除。

BoolQ 的修复重跑使用同一预训练 checkpoint 和五个 seed，训练 5 epochs、LR=3e-4、
global batch=40、amp_bf16。修复点是将 summed attachment loss 除以跨 batch/rank 的
有效 attachment-query 总数，不再错误地除以 LM label-token 数。评测覆盖完整 BoolQ
validation，使用 gold-span teacher-forced terminal-format multiple-choice scoring，
device eval batch=40。

| seed | BoolQ accuracy |
|---:|---:|
| 6198 | 65.32 |
| 13171 | 65.96 |
| 31723 | 65.72 |
| 42 | 65.44 |
| 2026 | 65.26 |
| mean ± sample SD | 65.54 ± 0.29 |

修复重跑的五个 `TRAIN_DONE`、五个 `EVAL_DONE` 和总完成标记均存在；checkpoint/model
哈希与代码哈希复核通过，训练和评测日志未发现 traceback、CUDA OOM、NCCL error、NaN、
fatal error 或 RuntimeError。正式证据位于本地 H20 cluster：
`/2024233198/TG-Interpolation/artifacts/experiment/pushdown_boolq_corrected_5seeds_20260829/`。
2026-08-27 的旧 BoolQ 数值来自错误 attachment-loss 分母，已删除且不再参与比较。

### Pushdown：SG / BLiMP 结构感知评测（本机 RTX 3090B，2026-08-11—14）

本组直接评测预训练 checkpoint
`saved_models/pushdown_terminalonly/step34354-unsharded`（seed 6198），没有下游微调。
SG 主结果对完整 32 项 suite 做增量 Pushdown parse marginalization；BLiMP 主结果覆盖
67 tasks × 1,000 minimal pairs，并对每句提供的 300 棵 parse 做 gold-structure
marginalization。两份主结果日志均成功加载 checkpoint、跑完整评测并以
`Training complete` 结束。

| 评测 | 协议 | accuracy | job | 完成时间 | 证据 |
|---|---|---:|---:|---|---|
| SG | beam300；`pushdown_beam_size=300`，`max_reduce=null` | **74.86%** | 2512 | 2026-08-14 | `analysis-output/syntax_eval_34354/logs/pushdown_sg_beam300_step34354_2512.out` |
| BLiMP | gold300；67×1,000 pairs，8 个 tied/non-finite pairs | **74.97%** | 2521 | 2026-08-14 | `analysis-output/syntax_eval_34354/logs/pushdown_blimp_gold300_step34354_2521.out` |
| SG（对照） | terminal-only teacher-forced；无 gold parse/stack tape | 35.98% | 2386 | 2026-08-11 | `analysis-output/syntax_eval_34354/logs/pushdown_sg_terminal_step34354_2386.out` |
| BLiMP（对照） | terminal-only teacher-forced；无 gold parse/stack tape | 66.43% | 2385 | 2026-08-11 | `analysis-output/syntax_eval_34354/logs/pushdown_blimp_terminal_step34354_2385.out` |

主表填写结构感知的 SG beam300 与 BLiMP gold300；terminal-only 两行只作为协议消融，
不能与主结果混用。机器可读汇总及完整配置分别见
`analysis-output/syntax_eval_34354/results.json` 和同目录 `configs/`。

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

注：TGNomask-Aug-100M 的 10.08 在 flex_attention=true/false 两条路径下逐位一致
（flex=true 复核任务 `tgnomask_aug_docppl_seed6198` 于 2026-08-29 04:16 完成，
doc_ppl=10.08），确认该值为编译路径无关的模型固有结果。

### Pushdown native top-K 完整 Document-PPL（2026-08-29，本地 H20 cluster）

正式聚合文件为
`/home/wangpch/TG-Interpolation/docppl_runs/pushdown_native_4gpu_45474/aggregate.json`
（位于本地 H20 cluster，不在本机 3090B）。严格 finalizer `45580` 检查了文档 ID
恰为 `0..4965`、每句候选上限一致、字段有限且计数匹配，然后原子写入该文件。
最终覆盖 4,966 篇文档、148,836 句、3,284,061 个计分 terminal/EOS，候选上限
`K=300`；实际有效 candidate slots / model forwards 均为 38,581,363，候选压缩率为 1.0。

| 口径 | log likelihood | PPL | 论文使用方式 |
|---|---:|---:|---|
| joint truncated candidate sum：每句 `logsumexp(token+attachment)` | -8,496,793.6809 | **13.293598** | **是；论文 Table 4 的 13.29。按论文 Eq. 4 对候选 latent structures 求和，不除以 `K_s`；作者于 2026-08-30 确认该口径** |
| joint uniform average：每句 `logsumexp(token+attachment)-log K_s` | -9,258,028.3308 | 16.761411 | 否；作为对候选集合施加均匀先验后的敏感性结果保留 |
| token-only uniform average：由同一 `sum(log K_s)` 严格换算 | -7,881,783.4321 | 11.023301 | 否；token-only 诊断，不是论文 joint PPL |
| token-only truncated sum：每句 `logsumexp(token)` | -7,120,548.7821 | 8.742660 | 否；token-only 诊断，不是论文 joint PPL |

其中 `sum_s log K_s = 761,234.6499`；因此 token-only uniform-average 值不是新跑结果，
而是由已保存的 token truncated-sum likelihood 加上完全相同的候选归一化项精确得到：
`exp((7,120,548.7821 + 761,234.6499) / 3,284,061) = 11.023301`。

评测协议固定如下：同一篇 document 中，第 `s` 句之前的每一句一律使用其
`candidate-0` 构造前缀；该前缀选择规则对 Pushdown、GPST 与其他 tree 模型的
document-level 比较保持一致。本次 Pushdown 实现保留真实 KV cache；GPST/R2D2 当前
尚未暴露等价的持久 chart-KV cache，因此只能确认 candidate-0 前缀语义一致，不能把
两者的缓存实现表述为一致。当前句则对全部有效 native candidates 评分；
joint NLL 为 token NLL 与 attachment/action NLL 之和。首句没有历史句前缀，后续记录
去掉重复 BOS；文档上下文上限为 2,048，超长时只按完整句边界滑窗，不截断当前句。

严格“历史句 candidate-0，且当前句也只取 candidate-0”的 token PPL **尚未计算**。
现有逐文档输出仅保存当前句各候选 token NLL 的 `logsumexp`，未保存
`scores.token_nll[0]`，所以不能从 `aggregate.json` 反推。原始 native mmap 中仍有
tokens、candidate-0 spans 与 document boundaries；后续可新增 `--candidate-slot 0`
重跑，candidate forwards 将由 38,581,363 降至 148,836（约 259 倍），且不需要
beam/top-K generation 或 attachment scoring。现有 `--token-only` 仍会评分全部候选，
不能替代这个严格实验。

运行效率异常的根因是工作量和长文档分布不均，而不是单张 GPU 本身变慢。原始按文档
数均分时，shard 0 承担约 34.54% 估算工作量，且包含 832、694、545、541、498 句的
极长文档；旧实现还为每批候选构造完整 `B×L×L` depth tape，在 `L≈2048` 时 batch
被压到约 4，并因完整句滑窗反复重建 candidate-0 cache。修复后只构造 cached query
需要的矩形 last-q depth rows、只计算对应 attachment rows，将 attention cap 提至
256M，并按估算成本重切 `[571,695)`、`[695,898)`、`[898,1103)`、`[1103,1241)`。
四个有效 worker 均正常完成；续跑任务 `45576`、`45577` 以及两个直接 GPU worker
与既有 `1241..4965` 结果由 `45580` 严格聚合。

数值一致性检查包括：last-q depth rows 与完整矩阵 CPU/CUDA 精确一致；attachment
query slice 与旧公式一致；candidate-0 KV-cache 对 full-prefix 的 attachment、joint、
token NLL 最大绝对误差分别为 `4.05e-6`、`2.31e-5`、`1.91e-5`，在 `3e-5` 容差内
全部通过。对长文档 doc 568 前 100 句，16M/64M/256M 三种 batch cap 的 joint PPL
均约 18.12132、token legacy PPL 均约 12.29568，说明加速没有改变评分语义。

作为尺度诊断，GPST 在完整 doc 0（37 句、927 terminals）的外部候选测试为：固定
candidate-0 前缀时 joint PPL=127.06、token-only PPL=28.36；将前缀改为论文 Eq. 7
的 greedy joint-MAP 后为 128.02 / 27.82。前缀策略只改变 joint PPL 约 0.75%，因此
GPST 的异常高值主要来自 tree-action likelihood，而不是 candidate-0 前缀或长句适配。
该单文档诊断不与 Pushdown 全测试集的 13.2936 直接作模型优劣比较。

### 本轮 Doc-PPL 结果补充（2026-08-23）

下表是同一轮完整 BBC News test 集评测的最终 Slurm 状态。成功任务均完成
148,836 个 sentence step，使用 `terminal_doc`：K=1、仅 BOS 作上下文、EOS 计分；
`denom=3,284,061`。Pause 模型另使用 document-phase pause 并屏蔽 pause target。

| 模型 | 最终状态 | 本次 Doc-PPL | job / 日志 |
|---|---|---:|---|
| Terminal-100M | COMPLETE | 9.88981 | `3444`; `artifacts/evaluation/terminal_doc_ppl_20260823/terminal/logs/slurm-terminal_terminal_doc_ppl_full_20260823-3444.out` |
| Pause-1-100M（复制式） | COMPLETE | 9.83125 | `3445`; `artifacts/evaluation/terminal_doc_ppl_20260823/pause1/logs/slurm-pause1_terminal_doc_ppl_full_20260823-3445.out` |
| Pause-2-100M（复制式） | COMPLETE | 9.92898 | `3446`; `artifacts/evaluation/terminal_doc_ppl_20260823/pause2/logs/slurm-pause2_terminal_doc_ppl_full_20260823-3446.out` |
| Pause-1-100M（论文式 SEP50261） | COMPLETE | 9.771 | `3612`; `artifacts/experiment/pause1_sep50261_full_eval_20260829/base/pause1_sep50261_docppl/eval.log` |
| Pause-2-100M（论文式 SEP50261） | COMPLETE | 10.32 | `989012`; SIST `artifacts/experiment/pause2_sep50261_full_eval_sist_20260829/base/pause2_sep50261_sist_docppl/eval.log` |
| Terminal-500M | COMPLETE | 2.82518 | `3447`; `artifacts/evaluation/terminal_doc_ppl_20260823/terminal_500m/logs/slurm-terminal_500m_terminal_doc_ppl_full_20260823-3447.out` |
| TreeReg-layer9 | COMPLETE | 12.36841 | `3451`; `artifacts/evaluation/terminal_doc_ppl_20260823/treereg_layer9/logs/slurm-treereg_layer9_terminal_doc_ppl_full_20260823-3451.out` |

### 新版 model-only 评测结果（2026-08-23）

修复后的配置从 checkpoint model config 建立，并设置
`reset_optimizer_state=true`、`reset_trainer_state=true`，因此以下结果均不读取
`optim.pt`/`train.pt`。旧 unmasked Tree-Shuffle 的 Doc-PPL 使用 `terminal_doc`（terminal
`test.npy`、BOS 上下文、EOS 计分、148,836 句）。其 SG/BLiMP 首轮任务曾误走
auto/beam；正式结果统一采用 terminal-only teacher-forced protocol。

| 模型 | Doc-PPL | SG | BLiMP | evidence |
|---|---:|---:|---:|---|
| Tree-Shuffle-100M（unmasked 历史对照） | 50.91（job 3467，COMPLETE） | 61.59%（job 3473，COMPLETE，terminal-only） | 71.68%（job 3474，COMPLETE，terminal-only） | `artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_{SG,blimp}_seed6198/logs/` |
| Terminal-1B（BBC） | 1.705（job 3470，COMPLETE） | 67.41%（job 3471，COMPLETE） | 63.31%（job 3472，COMPLETE） | `artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/terminal-1B_{docppl,SG,blimp}_seed6198/logs/` |

旧 unmasked Tree-Shuffle 新任务的运行时 grammar type 已明确设为 `terminal`，并设置
`input_format=terminal`、`structure_mode=terminal`、`tree_eval_type=terminal`、
`beam_search=false`；权重仍来自 `Tree_shuffle_pretrain/step49440-unsharded`。
因此 SG/BLiMP 不再使用 beam search 或 gold300。新任务 SG job 3473 已完成，
`avg=0.6159`（61.59%）；BLiMP job 3474 已完成，`overall/overall=0.7168`
（71.68%）。

Terminal-1B Doc-PPL 日志达到 `eval_step=148836`；SG/BLiMP 分别记录
`avg=0.6741` 与 `overall/overall=0.6331`。旧 unmasked Tree-Shuffle 的 SG/BLiMP 此前在
完整 suite 上运行；现已分别由 3473/3474 完成并回填至 §3 总表。

## 4. 1B FineWeb-Edu：论文数据与本次重跑

论文 Table 6 是 11-task OLMES；这里只登记用户要求的 BoolQ，并保留 AVG 作为
定位信息。SG、BLiMP、Doc-PPL、XSum 未在该表以同一设置报告，不填论文值。

| 论文模型 / 评测协议 | 论文 BoolQ | 论文 11-task AVG | 本次 BoolQ | 本次 XSum | 本次 SG | 本次 BLiMP | 本次 Doc-PPL | checkpoint / 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Terminal | 59.11 | 53.23 | — | — | — | — | — | `saved_models/A800_models/terminal_1B/step94299-unsharded`；BBC model-only 结果只列 §3，不回填本行 |
| Tree | 64.83 | 53.93 | — | — | — | — | — | `saved_models/A800_models/tree_1B/step137217-unsharded` |
| TGTree | 68.20 | 54.08 | — | — | — | — | — | `saved_models/A800_models/tgtree_1B/step143658-unsharded` |
| Treeterm（Tree eval） | 64.83 | 54.72 | — | — | — | — | — | 与上一行 Tree 共用 FineWeb checkpoint；terminal-format scoring，无独立权重 |
| TGTreeterm（TGTree eval） | 68.20 | 55.92 | — | — | — | — | — | 与上一行 TGTree 共用 FineWeb checkpoint；terminal-format scoring，无独立权重 |
| Pause-1 | 63.91 | 54.95 | — | — | — | — | — | `saved_models/A800_models/pause1_1B_SEP/step116061-unsharded` |
| Pause-2 | 61.62 | 55.68 | — | — | — | — | — | `saved_models/A800_models/pause2_1B_SEP/step141380-unsharded` |

## 5. 本轮任务登记与填写格式

| run id | 模型 | 任务 | 状态 | 证据路径 | 结果填写位置 |
|---|---|---|---|---|---|
| `terminal_terminal_doc_ppl_full_20260823` | Terminal-100M | Doc-PPL | COMPLETE, PPL=9.88981 | `artifacts/evaluation/terminal_doc_ppl_20260823/terminal/logs/slurm-terminal_terminal_doc_ppl_full_20260823-3444.out` | §3 Terminal-100M |
| `pause1_terminal_doc_ppl_full_20260823` | Pause-1-100M（复制式） | Doc-PPL | COMPLETE, PPL=9.83125 | `artifacts/evaluation/terminal_doc_ppl_20260823/pause1/logs/slurm-pause1_terminal_doc_ppl_full_20260823-3445.out` | §3 复制式历史对照 |
| `pause2_terminal_doc_ppl_full_20260823` | Pause-2-100M（复制式） | Doc-PPL | COMPLETE, PPL=9.92898（Slurm 3446） | `artifacts/evaluation/terminal_doc_ppl_20260823/pause2/logs/slurm-pause2_terminal_doc_ppl_full_20260823-3446.out` | §3 复制式历史对照 |
| `terminal_500m_terminal_doc_ppl_full_20260823` | Terminal-500M | Doc-PPL | COMPLETE, PPL=2.82518（Slurm 3447） | `artifacts/evaluation/terminal_doc_ppl_20260823/terminal_500m/logs/slurm-terminal_500m_terminal_doc_ppl_full_20260823-3447.out` | §3 本轮 Doc-PPL 结果补充 |
| `treereg_layer9_terminal_doc_ppl_full_20260823` | TreeReg-layer9（补充） | Doc-PPL | COMPLETE, PPL=12.36841（Slurm 3451） | `artifacts/evaluation/terminal_doc_ppl_20260823/treereg_layer9/logs/slurm-treereg_layer9_terminal_doc_ppl_full_20260823-3451.out` | §3 本轮 Doc-PPL 结果补充 |
| `eval_treereg_layer9_BLiMP` | TreeReg-layer9（补充） | BLiMP | COMPLETE, accuracy=66.68%（Slurm 3453） | `analysis-output/logs/eval_treereg_layer9_BLiMP_3453.out` | §3 TreeReg-layer9（补充） |
| `terminal_SG_seed6198` / `terminal_blimp_seed6198` | Terminal-100M | SG / BLiMP | COMPLETE, 69.80% / 70.77%（Slurm 3454 / 3455） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal_{SG,blimp}_seed6198/logs/` | §3 Terminal-100M |
| `terminal-500M_SG_seed6198` / `terminal-500M_blimp_seed6198` | Terminal-500M | SG / BLiMP | COMPLETE, 71.11% / 64.74%（Slurm 3456 / 3457） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/terminal-500M_{SG,blimp}_seed6198/logs/` | §3 Terminal-500M |
| `pause1_SG_seed6198` / `pause1_blimp_seed6198` | Pause-1-100M（复制式） | SG / BLiMP | COMPLETE, 76.29% / 72.52%（Slurm 3460 / 3461） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/pause1_{SG,blimp}_seed6198/logs/` | §3 复制式历史对照 |
| `pause2_SG_seed6198` / `pause2_blimp_seed6198` | Pause-2-100M（复制式） | SG / BLiMP | COMPLETE, 75.85% / 73.15%（Slurm 3462 / 3463） | `artifacts/evaluation/terminal_pause_syntax_20260823/runs/pause2_{SG,blimp}_seed6198/logs/` | §3 复制式历史对照 |
| `eval_treereg_layer9_SG` | TreeReg-layer9（补充） | SG | COMPLETE, accuracy=67.65%（Slurm 3464） | `analysis-output/logs/eval_treereg_layer9_SG_3464.out` | §3 TreeReg-layer9（补充） |
| `treereg_layer9_multiseed_20260824` | TreeReg-layer9（补充） | XSum / BoolQ 五 seed 微调 + 独立评测 | COMPLETE：XSum R-AVG=17.59 ± 0.10，BoolQ=63.54 ± 0.56；XSum train/eval=3493_[0-4]/3495_[0-4]，BoolQ train/eval=3518_[5-9]/3519_[5-9] | `artifacts/experiment/treereg_layer9_multiseed_20260824/runs/` | §3 TreeReg-layer9（补充）及五 seed 明细 |
| `treereg_layer9_auxloss_multiseed_20260825` | TreeReg-layer9（parse-aligned + auxiliary loss） | XSum / BoolQ 五 seed微调 + 独立评测 | COMPLETE：XSum R-AVG=20.88 ± 0.03，BoolQ=65.87 ± 0.31；train=3528_[0-9]，eval=3531_[0-9]，全部 exit=0；每 10 global optimizer batch 应用 layer-9 TreeReg loss | `artifacts/experiment/treereg_layer9_auxloss_multiseed_20260825/runs/` | §3 parse-aligned auxiliary-loss 五 seed 对照；与 legacy 差异不能仅归因于 loss |
| `pushdown_xsum_sentence_local_5seeds_20260829` | Pushdown-100M（sentence-local gold spans） | XSum 五 seed微调 + 完整评测 | COMPLETE：R-AVG=21.06 ± 0.05；seeds=6198,13171,31723,42,2026；source prompt 使用 gold spans；summary 从空的 ROOT-free sentence stack 开始；beam=6,max_reduce=null；五个 TRAIN_DONE/EVAL_DONE 和总完成标记齐全 | `/2024233198/TG-Interpolation/artifacts/experiment/pushdown_xsum_sentence_local_5seeds_20260829/`（本地 H20 cluster） | §3 Pushdown sentence-local gold-span 五 seed XSum |
| `pushdown_boolq_corrected_5seeds_20260829` | Pushdown-100M（corrected attachment loss） | BoolQ 五 seed微调 + 完整 validation 评测 | COMPLETE：accuracy=65.54 ± 0.29；seeds=6198,13171,31723,42,2026；5ep，LR=3e-4，global batch=40；attachment loss 以 global valid attachment-query count 归一化；gold-span teacher-forced terminal-format MC scoring；五个 TRAIN_DONE/EVAL_DONE 和总完成标记齐全 | `/2024233198/TG-Interpolation/artifacts/experiment/pushdown_boolq_corrected_5seeds_20260829/`（本地 H20 cluster） | §3 Pushdown sentence-local gold-span XSum / corrected BoolQ |
| `syntax_eval_34354/pushdown_{sg_beam300,blimp_gold300}` | Pushdown-100M | SG / BLiMP 结构感知完整评测 | COMPLETE：SG beam300=74.86%（job 2512）；BLiMP gold300=74.97%（job 2521，67×1,000 pairs，8 ties）；terminal-only 对照分别为 35.98% / 66.43%，不填主表 | `analysis-output/syntax_eval_34354/{results.json,logs/}` | §3 Pushdown SG / BLiMP 结构感知评测 |
| `pushdown_native_4gpu_45474` | Pushdown-100M（native top-K） | BBC News 完整 Document-PPL | COMPLETE：4,966 documents / 148,836 sentences / 3,284,061 terminals；论文使用 truncated joint-sum PPL=13.293598；uniform-average sensitivity=16.761411；token-only diagnostics=8.742660/11.023301；finalizer 45580 exit=0 | `/home/wangpch/TG-Interpolation/docppl_runs/pushdown_native_4gpu_45474/aggregate.json`（本地 H20 cluster） | §3 Pushdown native top-K 完整 Document-PPL |
| `treereg_layer9_terminal_docppl_20260825` | TreeReg-layer9 | 新 terminal Doc-PPL | COMPLETE：PPL=12.37，完整 148836/148836 records | `artifacts/evaluation/treereg_layer9_terminal_docppl_20260825/run.log` | §3 TreeReg-layer9（补充；legacy FT 无辅助 loss） |
| `tree_shuffle_docppl_seed6198` | Tree-Shuffle-100M（unmasked 历史对照） | Doc-PPL | COMPLETE=50.91（3467） | `artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/tree_shuffle_docppl_seed6198/` | §3 unmasked 历史对照 |
| `tree_shuffle_terminal_syntax_SG_seed6198` | Tree-Shuffle-100M（unmasked 历史对照） | SG | COMPLETE，accuracy=61.59%（Slurm 3473）；terminal-only teacher-forced | `artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_SG_seed6198/logs/slurm-tree_shuffle_SG_seed6198-3473.out` | §3 model-only 结果 |
| `tree_shuffle_terminal_syntax_blimp_seed6198` | Tree-Shuffle-100M（unmasked 历史对照） | BLiMP | COMPLETE，accuracy=71.68%（Slurm 3474）；terminal-only teacher-forced | `artifacts/evaluation/tree_shuffle_terminal_syntax_20260823/runs/tree_shuffle_blimp_seed6198/logs/slurm-tree_shuffle_blimp_seed6198-3474.out` | §3 model-only 结果 |
| `terminal-1B_{docppl,SG,blimp}_seed6198` | Terminal-1B（BBC checkpoint） | Doc-PPL / SG / BLiMP | COMPLETE：PPL=1.705、SG=67.41%、BLiMP=63.31%（3470/3471/3472）；model-only restore，已移除继承 FineWeb shards | `artifacts/evaluation/tree_shuffle_terminal1b_eval_20260823/runs/terminal-1B_{docppl,SG,blimp}_seed6198/` | §3 新版 model-only 结果 |
| `pause_sep_100m_sist_20260828` | Pause-1/2-100M（论文式 dedicated SEP） | 从头预训练；terminal + SEP 50261；8 GPU | COMPLETE：Pause-1 job 988670、Pause-2 job 988671 均 `COMPLETED/0:0`；final step=45487/54156，checkpoint 和哈希已校验 | `artifacts/experiment/pause_sep_100m_sist_20260828/` | §2A 论文式 dedicated-SEP checkpoint 身份 |
| `pause1_sep50261_full_eval_20260829` | Pause-1-100M（论文式 dedicated SEP） | SG / BLiMP / Doc-PPL / BoolQ 五 seed | COMPLETE：SG=75.11%，BLiMP=70.85%，Doc-PPL=9.771，BoolQ=62.036 ± 0.194；base jobs=3610/3611/3612，BoolQ train/eval=3642_[5-9]/3614_[5-9] | `artifacts/experiment/pause1_sep50261_full_eval_20260829/` | §3 Pause dedicated-SEP 总表与明细 |
| `pause2_sep50261_full_eval_sist_20260829` | Pause-2-100M（论文式 dedicated SEP） | SG / BLiMP / Doc-PPL / BoolQ | COMPLETE：SG=76.97%、BLiMP=72.59%、Doc-PPL=10.32（jobs 989010/989011/989012）；BoolQ=65.700 ± 3.453，completion array `989304_[5-9]` 全部 `COMPLETED/0:0` 且 5/5 TRAIN_DONE/EVAL_DONE | SIST `artifacts/experiment/pause2_sep50261_full_eval_sist_20260829/` | §3 Pause dedicated-SEP 总表与明细 |
| `pause_xsum_v2_sist_20260829` | Pause-1/2-100M（论文式 dedicated SEP） | XSum v2 五 seed重微调 + 完整 test | COMPLETE：job `989161_[0-4]` 全部 `COMPLETED/0:0`；Pause-1 R-AVG=22.378 ± 0.046，Pause-2=22.250 ± 0.061；10/10 TRAIN_DONE/EVAL_DONE/v2 contracts | SIST `artifacts/experiment/pause{1,2}_sep50261_xsum_v2_sist_20260829/` | §3 Pause dedicated-SEP 总表与明细；替代所有旧 XSum v1 结果 |
| `treeshufflemask_terminal_multiseed_20260830` | Tree-Shuffle-100M（mask；论文行恢复） | terminal SG / BLiMP / Doc-PPL；XSum / BoolQ 五 seed `tree_shuffle_mask` 微调与完整评测 | COMPLETE：SG=76.56%、BLiMP=67.77%、Doc-PPL=13.41、XSum R-AVG=21.558 ± 0.093、BoolQ=68.082 ± 0.332；Slurm `989550_[0-4]` 全部 `COMPLETED/0:0`；10/10 TRAIN_DONE、13/13 EVAL_DONE、collector 13/13 finite | SIST 与本地 `artifacts/experiment/treeshufflemask_terminal_multiseed_20260830/` | §2 checkpoint 映射；§3 主表与逐 seed 明细 |
| `gpst_binary_pushdown_document_ppl_20260830` | Pushdown-100M，`pushdown_terminalonly/step34354-unsharded` | direct GPST strict-binary CKY top-K；真实预训练 fixed-word-atom BPE 表示；candidate-0 document history；v1 `stack_legal` | COMPLETE：4,966 documents / 148,836 sentences / 3,284,061 terminals / 37,227,054 candidates；joint LL=-9,124,537.0987，PPL=16.093751；candidate-0 token PPL=16.050817；8 分片重放与旧 repaired split 相差仅 6.8e-7 PPL。真实 checkpoint scorer 对 dense reference 的 joint NLL 最大误差为 1.38e-5（full）/5.16e-5（KV） | `artifacts/experiment/gpst_binary_pushdown_document_ppl_20260830/nan_diagnosis/repaired_full_single/aggregate.json`；`artifacts/analysis/gpst_binary_pushdown_scorer_equivalence_20260830/result.json` | checkpoint 训练管线保留 multi-BPE preterminal；该 direct adapter 与训练树表示一致，但 candidate support 和概率口径仍不能替换论文 n-ary-v1 13.293598 |
| `nary_spliced_right_binary_pushdown_ppl_200sent_20260831` | 同一 Pushdown checkpoint | native n-ary support 在 BPE splice 后整体 right-CNF 的 200-sentence 拓扑消融 | DIAGNOSTIC：v1 PPL=13.716485，candidate-0 token PPL=14.285710；该表示不是 checkpoint-training representation | `docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md` “Preserved alternative-protocol diagnostics”；`artifacts/analysis/nary_right_binary_candidate_audit_20260830.json` | 保留为不同 BPE 拓扑协议的局部结果，不外推全语料 |
| `pushdown_word_atom_nary_v1_20260831` | 同一 Pushdown checkpoint | native n-ary proposal → 词级 right-CNF + fixed right-recursive multi-BPE word atoms；v1 `stack_legal` | COMPLETE：LL=-9,127,769.3338，PPL=16.10959864，candidate-0 token PPL=16.05088377；4,966 docs / 148,836 sentences / 3,284,061 terminals / 33,007,452 candidates；forwards exact；nonfinite retries=22（均由确定性缩批恢复），OOM retries=1,585；topology audit 3789 COMPLETE，33,007,316/33,007,452 位于 direct support，零 support-disjoint 句，candidate-0 match=148,834/148,836；array=3783，merge=3786 | `artifacts/experiment/nary_right_binary_pushdown_document_ppl_20260830/nary_word_atom_full_v1/aggregate.json`；`artifacts/analysis/nary_word_atom_right_binary_candidate_audit_20260831.json` | 与 direct-v1 16.093751 接近，确认 proposal 轴不是 13→16 主因 |
| `pushdown_word_atom_nary_v2_20260831` | 同一 Pushdown checkpoint | 与上一行相同的真实训练树表示；attachment 使用训练 CE 的 `sentence_causal` v2 | COMPLETE：joint LL=-9,160,975.3534，PPL=16.27331339，candidate-0 token PPL=16.05087503；4,966 docs / 148,836 sentences / 3,284,061 terminals / 33,007,452 candidates；forwards exact；nonfinite retries=23（均由确定性缩批恢复），OOM retries=1,467；4,966/4,966 文档满足 v2 likelihood <= v1；full array=3784，strict merge=3787 | `artifacts/experiment/nary_right_binary_pushdown_document_ppl_20260830/nary_word_atom_full_v2/aggregate.json` | 训练-objective likelihood；不能代替 Table-4 v1 |
| `gpst_direct_strict_binary_v2_20260831` | 同一 Pushdown checkpoint | direct strict-binary GPST fixed-word-atom support；训练 CE `sentence_causal` v2 | COMPLETE：joint LL=-9,157,799.3091，PPL=16.25758293，candidate-0 token PPL=16.05086226；4,966 docs / 148,836 sentences / 3,284,061 terminals / 37,227,054 candidates；forwards exact；nonfinite retries=38（均由确定性缩批恢复），OOM retries=1,434；8 shards 与 4,966 个逐文档文件独立求和一致；full array=3785，strict merge=3788 | `artifacts/experiment/nary_right_binary_pushdown_document_ppl_20260830/gpst_direct_full_v2/aggregate.json` | 当前最直接的 strict-binary + training-representation + training-objective top-K 评测；仍是 truncated external-candidate sum，不是全树精确边缘化 |

每次完成后，在对应单元格填写数值，并在“重跑来源 / protocol”写成：

`job=<id>; checkpoint=<absolute path>; evaluator=<type/label>; dataset=<path>; protocol=<K/格式>; log=<path>; completed=<ISO date>`

失败任务不写入本结果表；只把无数值的根因写入 `REPOSITORY_CLEANUP_MEMORY.md`。若改变
tokenizer、训练数据、checkpoint step、评测集版本或 terminal/full scoring，必须新开
run id，不得覆盖原 run。
