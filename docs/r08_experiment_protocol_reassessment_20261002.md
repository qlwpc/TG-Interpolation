# R-08：新旧 BBC 实验标准、路径与结果差异核定

核定日期：2026-10-02。实验完成状态依据截至 2026-09-29 的原始回执；本次核对本地证据、
配置、路径与数值，不重新评测、不查询远端、不修改论文。本报告是固定日期的差异说明，
用于后续论文实验数据修订和复现方案选择；逐模型正式数据继续由原结果 JSON 维护。

**必须分开使用三组证据：旧论文结果、旧权重的 clean 重测、新 dedup 500M 重训。**
后两组都采用完整 clean test，但只有第三组重新训练。旧论文到 clean 重测同时改变测试集和
结构评分协议；clean 重测到新 500M 又改变训练数据量、LR、batch、步数及部分模型实现。
这些差异支持描述性比较，不能单独量化“去污染收益”或参数规模的因果效果。

## 1. R-08 的重新核定标准与实验身份

本次 R-08 的完成条件是：明确每组数据与 checkpoint；说明候选、历史、分母和覆盖标准；
给出可追溯的路径及结果差异；排除已知无效结果；界定后续修稿能使用的证据和复现缺口。
**R-08 文档核定完成不表示论文已修订、全量重训已公开复现，或未完成的资产交付已关闭。**

| 组别 | 本报告含义 | 训练与权重 | BBC DocPPL 测试条件 | 结果权威来源 |
|---|---|---|---|---|
| A：冻结稿旧实验 | 当前冻结稿 `tab:exp1-results` 的 19 行；不是更早投稿 PDF 的表值 | 历史 BBC 训练；旧 100M / 四个 500M checkpoint | 旧 4,966 篇；结构模型历史记录为 candidate-0 | [冻结表 JSON](../reproducibility/evidence/paper_bbc_table.json) |
| B：旧模型 clean 重测 | 对既有权重更换测试与评分协议；有效汇总共 22 模型，其中 19 行对应 A | 与 A 对应行保持同一权重；三项 grammar 修正仅改配置副本 | clean 5,000 篇；有效候选 joint sum、model-best | [有效 clean JSON](../reproducibility/evidence/old_checkpoint_clean.json) |
| C：新 dedup 500M | 十个从零训练的新模型及完整下游评测 | dedup train、clean dev/test、重新计算训练参数；单一预训练 seed | 同一 clean 5,000 篇；有效候选 joint sum、model-best | [十模型结果页](bbc_dedup_500m_evaluation_results_20260925.md)及三批原结果副本 |

A → B 不是“相同实验仅更新分数”；B → C 不是“旧模型继续重测”。A 与 C 共有的 500M
模型名仅为 Terminal、Tree、TGTree。**TGNomask-Aug-500M 没有 C 组对应项**；新 TGNomask
及 TGNomask-Mix-TG 都不能代替它。B 的 TreeReg 与 BBC 1B 两项附加结果不属于 A 的主表；
BBC 1B 也不等于 FineWeb-Edu 1B。本报告不重核 FineWeb-Edu 的 11-task 实验。

## 2. 数据标准：训练集、测试集与污染证据

### 2.1 不能用文件名或文档数量判定版本

| 数据 | 文档数 | 句数 | terminal/EOS 计分 token | 核定用途 |
|---|---:|---:|---:|---|
| 历史 train 的 terminal 数组 | 15,187,326 | — | 训练数组含 special tokens，共 10,061,025,584 tokens | A 及 B 权重原训练经历 |
| 历史 split 索引 test | 5,025 | — | — | 原索引；不能代替下面的 DocPPL 对齐版 |
| 历史 DocPPL test | 4,966 | 148,836 | 3,284,061 | A |
| reserved-clean-v1 dev | 1,000 | — | — | C 的训练验证 |
| reserved-clean-v1 完整 test | 5,000 | 129,085 | 3,068,713 | B 与 C 的独立 DocPPL |
| dedup train | 1,466,132 | — | 见 §3 各表示的训练 D | C |

clean terminal 基础数组有 3,073,713 tokens；扣除每篇一个仅作上下文的 BOS 后，计分数
为 3,068,713，EOS 计分。`test_docppl_1000` 与 smoke 都不能代替完整 5,000 篇。
5,025 → 4,966 还涉及组装范围、边界和正文版本变化，不应概括为只删掉 59 篇。

C 的冻结训练 manifest 记载：使用文件名年份 `<2023` 的 89 个分片，首尾为
`CC-MAIN-2013-20` / `CC-MAIN-2022-49`，使用新的解析/编码产物（源 receipt 协议为
`explicit-bos-eos-ptb-mapping-v1`）。这是 dedup 数据池的固定组装，**不能直接描述成在旧
10.061B token 数组中删除重复项后保留其他条件不变**。本次没有证实新数组是旧数组的逐字节子集。
普通历史 89 分片 `assemble` 也不自动生成这份 dedup train。

### 2.2 污染结论的证据范围

对指定历史 train 的完整 token exact 扫描发现：A 的 4,966 篇中 **4,752 篇有完整训练副本
（95.6907%）**，对应 91,394 个训练文档实例、91,631 个匹配对。故 A 的旧 BBC DocPPL
不能作为独立 held-out 泛化证据；这不自动证明 SG、BLiMP、XSum、BoolQ 全部受到同类污染。
定义与训练哈希见 [historical overlap](../reproducibility/evidence/historical_overlap.json)。

clean 筛选相对于固定历史 train 使用 exact、5-gram near 与 13-gram local reuse 规则。
完整 test 的独立复核中，normalized exact 和所检查的 near 阈值（0.8 / 0.7 / 0.6）命中为 0；
但 2,290 篇存在共享 13-gram，未屏蔽模板时 41 篇局部覆盖 ≥30%。按冻结模板屏蔽与参考集
规则处理后的该阈值命中为 0，故应写“通过指定排除规则”，不能写“零文本重合”。见
[clean overlap](../reproducibility/evidence/clean_overlap.json)与[完整规则](bbc_data_provenance.md#clean-audit)。

**新 dedup train 与 clean dev/test 的独立完整重叠审计未在本次重做。** 配置、传输 hash
与评测身份通过不代替这项审计，也不支持将 C 宣称为经过一切污染检验的零重合数据。

## 3. 预训练标准：同为 500M，不代表训练预算相同

新旧共有骨架为 16 层、hidden size 1408、16 heads、MLP ratio 8、RoPE/RMSNorm/SwiGLU，
seed 6198、AdamW、cosine、warmup 2000、训练 1 epoch。C 的非 embedding 参数量为
507,896,576；总参数量为 578,747,136。“500M”沿用非 embedding 规模口径。
epoch 相同不代表 token 数、更新次数或训练计算量相同。

### 3.1 A/B 的旧四模型训练配置

下表取[历史配置 manifest](../train_configs/paper_pretraining_manifest.json)与固定 YAML；
D 的 Tree/TG 数量按原登记保留近似值，不冒充本次完整扫描。

| 模型 | 训练表示 / D | 峰值 LR | global batch | microbatch | checkpoint step | attention |
|---|---|---:|---:|---:|---:|---|
| Terminal | terminal / 10.061B | 0.001308 | 144 | 16 | 34115 | causal Flash |
| Tree | LIN1 / 24.706B | 0.001723 | 244 | 16 | 49440 | causal Flash |
| TGTree | LIN2 / 32.029B | 0.001866 | 280 | 16 | 55853 | causal Flash |
| TGNomask-Aug | LIN2 / 32.029B | 0.001866 | 280 | 12 | 55853 | Nomask-Aug / Flex |

旧 TGTree 的原配置 grammar key 为 `tree`，但输入为 LIN2/TG；不能因此把它当作 Tree
权重。运行时须按模型身份和输入格式核对。旧四模型配置为 DDP/BF16；来源配置的
global/device batch 比值为 Terminal/Tree=4、TGTree/TGNomask-Aug=8，不能一律描述为
旧模型均用 4 卡；这些配置字段也不替代实际训练资源回执。C 的原回执明确为 2 卡。
B 没有重新训练这些权重。

### 3.2 C 的十模型训练配置

所有模型在 scvi5090 使用 2×RTX 5090、DDP/BF16；最终 microbatch 均为 4。
LR/global batch 按新的表示长度 D 重算，完整精度以[冻结配置](../reproducibility/configs/dedup500m/manifest.json)
为准。以下 LR 仅为展示舍入。

| 模型 | 训练 D（含结构或在线 SEP） | 峰值 LR | global batch | 序列长度 | 完成 step | attention |
|---|---:|---:|---:|---:|---:|---|
| Terminal | 800,848,405 | 0.000601208621 | 34 | 2048 | 11501 | causal Flash |
| Tree | 1,963,388,153 | 0.000791750134 | 58 | 2048 | 16529 | causal Flash |
| TGTree | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | causal Flash |
| TG | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| TGNomask | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| TGTree-Mix-TG | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| TGNomask-Mix-TG | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| Tree-NoONT | 1,382,118,279 | 0.000710857635 | 46 | 2048 | 14670 | causal Flash |
| Pause-1 | 1,601,696,810 | 0.000743774711 | 52 | 2048 | 15039 | causal Flash |
| Pause-2 | 2,402,545,215 | 0.000842367155 | 64 | 2049 | 18321 | causal Flash |

Pause 的底层训练文件仍是 800,848,405-token terminal 数组，D 分别按在线 SEP 展开为
2 倍 / 3 倍；不能把这些 SEP 当作新正文。TGNomask-Mix-TG 为 8 TG + 8 TGNomask heads；
TGTree-Mix-TG 为 8 TGTree + 8 TG heads；顺序见原配置。

固定 m=4 的七个模型（Terminal/Tree/TGTree/TG/TGNomask/TGNomask-Mix-TG/Tree-NoONT）
没有最大 microbatch 搜索。Pause-1/2 与 TGTree-Mix-TG 有搜索及确认；后者以修复重提的
`retry-20260921`、Job 249348 为正式完成版本，原 Job 248318 在正式训练前失败。
每卡 batch 不必整除 4，例如 Terminal 为 4×4+1，TG 系列为 4×8+1；以真实累积记录为准。
不得把硬件调节写成所有模型均完成相同 probe。

C 每 3000 步的训练 `lm` evaluator 在各自表示上计算 loss。Terminal / Tree / TGTree 的
最终 clean-test **LM PPL** 为 14.36448 / 3.02325 / 2.35796；Pause-1/2 为 3.82742 / 2.46185，
其中包含 SEP targets。**这些值不是 §6 的 DocPPL，也不能跨表示排名**。新增结果页中
14.028763 / 13.863770 / 13.946850 才是前三模型的独立 clean DocPPL。

## 4. DocPPL 评分标准：测试集与算法同时变化

| 核对项 | A：旧结果的已登记合同 | B/C：clean 合同 |
|---|---|---|
| 正文与分母 | 4,966 篇、3,284,061 scored tokens | 5,000 篇、3,068,713 scored tokens |
| Terminal / Pause | 单路径 terminal/EOS；Pause 插入位置不计分 | 同类单路径；dedicated SEP、document-global phase |
| Tree/TG 本句概率 | 历史 300 proposal/slotted truncated sum；旧批次完整候选唯一性证据未在本次补齐 | 仅唯一有效候选 joint probability 求和；填充槽不计质量；不除以 K |
| 后续句子的历史 | 旧 Tree/TG 实测登记为 candidate-0；不能按正文改贴为 model-best | 本句全部有效候选评分后取 joint NLL 最小项；并列取最早项 |
| 数值与 cache | 以当次冻结配置/代码为准，不能套用当前实现 | DocPPL FP32/SDPA、关闭 TF32；句内共同 padding 长度，仅提交胜出真实长度；换文档清空 cache |
| 终验 | 历史完成记录与协议各自保留 | 全文档、句、token、finite NLL、候选/历史计数及 fingerprint；不能只看退出码 |

clean Tree/TG 有 38,725,500 个物理槽、**37,524,534 个有效候选**：122,338 句有 300 个，
6,747 句有 122 个。TG 是对应 Tree closing token 原位重复的转换，候选轴一一对应。
Parser proposal score 负责生成与排序，不作为 LM joint sum 的额外权重。

对结构模型，当前句评分为 `-log(sum(exp(-joint_NLL_k)))`，全语料 NLL 相加后除以正文
terminal/EOS 总数，再取指数。joint 的计分项由该模型 label mask 定义；分母不计结构
或 Pause 位置。不是逐文档 PPL 的平均，也不是除以候选数量的 uniform mixture。
截断只保证**在固定所选历史下**对当前句概率取下界；这里没有边缘化全部前文树历史，
不能称为精确文档 marginal。定义见 [Evaluation §4](../Evaluation.md#4-当前-document-ppl-合同)。

Tree-Shuffle 使用 masked checkpoint 的 terminal-only 协议。B 的 Pushdown 使用 native
n-ary / v1 `stack_legal` / model-best，独立有效候选共 34,991,281；其 token-only、uniform
mixture、binary 或 v2 都不属于本报告的主值。C 不含 Tree-Shuffle 或 Pushdown。
这两类结果只能注明协议并列，不纳入同一受控 Tree/TG 排名。

冻结稿正文 `sec:doc-ppl` 写 evaluated-model argmax，附录 `sec:ppl-protocol-details` 写
parser-selected history，Pushdown 写 candidate-0；它们与 A 的实测登记及 B/C 的现行合同
不能混用。后续修稿需要一并修改方法、附录和表注，本次只记录差异。

## 5. 路径核定：数据源、部署路径、权重与结果分开

所有相对路径以仓库根为起点。下面的本地/远端路径用于来源追踪，**不是公开下载入口**。
改部署路径不能改变数据身份；相同文件名不能证明同一实验。

### 5.1 数据与 campaign 路径

| 用途 | 实验源路径 / 运行路径 | 核定说明 |
|---|---|---|
| A/B 原训练 | `dataset/bbc-news/{terminal,tree,tg}/train.npy`；旧 YAML 使用 `/dev/shm/dataset/bbc-news/` | 历史训练数据，不是 dedup train |
| A DocPPL | `dataset/bbc-news/terminal/`、`testppl_tree/`、`testppl_tg/` | 旧 4,966 篇；结构旧运行登记在 `artifacts/evaluation/docppl_structural_20260825/`，该 campaign 完整目录不在当前本机 |
| B/C clean DocPPL | `dataset/bbc-news-reserved-clean-v1/evaluation_terminal/`、`testppl/{tree300,tg300}/` | terminal 输入及句/文档索引从 canonical 派生；结构入口不是训练用单树 `tree/test.npy` |
| C train 真源 | `dataset/bbc-news-parsed-dedup/{terminal,tree,tg,tree_noont}/train.npy` | 对应冻结 train manifest |
| C 5090 部署 | `/data/run01/scvi351/TG-Interpolation/dataset/bbc-news-dedup-clean-v1/{format}/` | train 映射到 parsed-dedup，dev/test 映射到 reserved-clean；不是第四套数据 |
| C NoONT dev/test | `dataset/bbc-news-parsed-dedup/tree_noont/{dev,test}.npy` | manifest 证明源为 reserved-clean Tree 删除 ONT；目录名不代表使用旧 test |
| B 原评分 campaign | `artifacts/evaluation/reserved_docppl_sist_20260907/` | SIST 运行根为 `/inspurfs/group/tukw/wangpch/docppl_reserved_20260907/` |
| B grammar 修正 | 上述目录的 `grammar_fix_20260909/` | 三模型使用正确 grammar 和原权重；原错误值永久排除 |
| C 5090 训练 | `validation/pretrain_5090_dedup_20260915/`、`pretrain_5090_dedup_structural_20260919/`、`pretrain_5090_dedup_pause_mix_20260920/` | 远端前缀为 `/data/run01/scvi351/TG-Interpolation/validation/`；Mix 使用 retry 子目录 |
| C 评测 | `artifacts/evaluation/bbc_dedup_500m_{local_20260919,pause_noont_20260922,sist_20260922}/` | 前两批 RTX3090B，第三批 SIST；公开副本位于 `reproducibility/results/dedup500m/` |

C 的 SIST 部署根为 `/inspurfs/group/tukw/wangpch/bbc_dedup_500m_eval_20260922/repo/`。
四模型句法/微调采用 BF16/SDPA、DDP full replicas，评测时关闭 typed/Flex 内核；DocPPL
仍为 FP32/SDPA。这与 5090 预训练的 typed/Flash backend 有区别，不能把执行地或 backend
当作不同训练数据，也不声称两轮 scorer 二进制完全相同。
保存的 Tree 评分 contract 中，B 的 scorer hash 为
`c1c545738423d3301cf3503725612d5cb117188a8a6c806f7cd5e13d231fe1ac`，C 为
`7dcfee0eef1cd4f059204a9aca75bdfc553ed5306a5f2766553c43cfb56a86be`。
history/aggregation/precision/attention 等协议字段一致不代表评分源码字节一致；复现时须选对应快照。

### 5.2 checkpoint 路径与模型缺项

A/B 旧 500M 路径分别为：

| 模型 | 旧 checkpoint（仓库相对路径） | 固定配置 |
|---|---|---|
| Terminal | `saved_models/terminal_500M/step34115-unsharded` | [terminal](../train_configs/paper_sources/bbc_500m_terminal.yaml) |
| Tree | `saved_models/Tree_500M/step49440-unsharded` | [tree](../train_configs/paper_sources/bbc_500m_tree.yaml) |
| TGTree | `saved_models/TGTree_500M/step55853-unsharded` | [tgtree](../train_configs/paper_sources/bbc_500m_tgtree.yaml) |
| TGNomask-Aug | `saved_models/TGnomaskaug_500M/step55853-unsharded` | [tgnomask_aug](../train_configs/paper_sources/bbc_500m_tgnomask_aug.yaml) |

C 统一本地根为 `saved_models/bbc_dedup_500m_scvi5090/`：

| 模型 key | 根目录下最终 checkpoint | 原评测批次 |
|---|---|---|
| `terminal` | `terminal/step11501-unsharded` | `bbc_dedup_500m_local_20260919` |
| `tree` | `tree/step16529-unsharded` | `bbc_dedup_500m_local_20260919` |
| `tgtree` | `tgtree/step18825-unsharded` | `bbc_dedup_500m_local_20260919` |
| `tg` | `tg/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tgnomask` | `tgnomask/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tgtree_mix_tg` | `tgtree_mix_tg/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tgnomask_mix_tg` | `tgnomask_mix_tg/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tree_noont` | `tree_noont/step14670-unsharded` | `bbc_dedup_500m_pause_noont_20260922` |
| `pause1` | `pause1/step15039-unsharded` | `bbc_dedup_500m_pause_noont_20260922` |
| `pause2` | `pause2/step18321-unsharded` | `bbc_dedup_500m_pause_noont_20260922` |

`bbc_dedup_500m_scvi5090_20260919` / `..._20260920` 是指向统一目录的兼容链接；不能
算成多组独立权重。C 组 Tree DocPPL 的两个 contract 因路径迁移产生不同 fingerprint，
原严格收集已核对权重、数据、评分器与参数一致；这一特例不授权将其他不一致合同混合。
本次确认旧四模型的权重路径存在、新十模型 30 个 model/config/train 文件路径与大小符合
manifest，并重新计算十份小型 config 的 hash；**没有重新计算大权重 hash**。
新本地副本没有回收 `optim.pt`，可用于评测，不能据此承诺完整 AdamW 续训。

### 5.3 数据身份锚点（SHA-256）

以下从原 manifest/预检记录对照得出；本次没有重新扫描大数组。完整候选数组及索引身份
由各自 manifest 和最终回执绑定。

| 文件 / 内容 | SHA-256 |
|---|---|
| 新 train：terminal | `b589676a1f3d5b485de17ee7715acf5d3d8f6d03a40188199bfa7c563345b427` |
| 新 train：tree | `792ae397857fb7b9ff9266ba147e11edb7e27bccd5fa780287e93b67d23b660a` |
| 新 train：tg | `370df2b8bc49be2b6cbd21d20cdbe3bc751c8dbe1c5a19f07d2b47f12c016970` |
| 新 train：tree_noont | `55bc4ff70abf87e9ad495c6e865cbeba3ab09a0a1f0d05432ca2e5c5c0d3a7d9` |
| clean `terminal/test.npy` | `fd95d40241cb13fbc84c7bee01f645847c520d1bd46eb0697d8f40569bd9c5c0` |
| clean `canonical/test/manifest.json` | `8995b8e031d69c821ade432678b14b587aae5e2f2ef7914594e3b5ae6f866fe9` |
| clean `tree300/tree_300.npy` | `25241ac512ea9eb419bc1debcd299a6dc623206b82b13f275ccb99160489f403` |
| clean Tree/TG `valid_counts.npy` | `67efabdae9e3d0a41a918de08f3b7290b7cdb6529d2a7f09c0c503fde9f7cc6e` |

B 与 C 本地/SIST 输入记录共同覆盖的 9 个 clean 基础文件 hash 一致；C 本地与 SIST 的
19 个 clean 基础/候选/索引文件记录也一致。B 的候选审计、评分 contract 和 C 的候选
manifest 指向同一冻结 clean 候选产物；B 的基础输入清单本身不含候选数组 SHA，不能把
这 9 项对照夸大为本次重新核验 B 全部候选大文件。

BBC runtime tokenizer hash 为 `94a26c15b2edbe45b08c17e20a2f5ad9485a268a7b3ac477a5bf9290ef7f0d46`；
部分 train manifest / [公开参考 JSON](../reproducibility/tokenizer-reference.json) 的 byte hash 为
`7fc243801e0d0551948380e59031976ed1eed89bd0efe867fd8a70cfadc19cff`。
原协议已检查完整 JSON 等价，应写“字节不同、JSON 等价”，不能写 byte hash 完全相同。
本次也对本机 runtime tokenizer 重新计算小文件 hash，并确认它与公开参考的完整 JSON 相等。

## 6. 结果差异与可用结论

下列是核定日固定快照。A 使用冻结稿展示精度，B/C 展示六位小数；不得将 A 的舍入值
当作原日志精确值。`—` 表示该组没有对应模型结果，不表示零、失败或等待中的任务。

### 6.1 100M：A 与 B 的 DocPPL

| 模型 | A：旧 test | B：旧权重 clean | 解读范围 |
|---|---:|---:|---|
| Terminal | 9.89 | 15.369225 | 单路径 |
| Tree | 10.01 | 14.736097 | CRF 有效候选 |
| TGTree | 10.21 | 14.709097 | CRF 有效候选 |
| TG | 11.02 | 16.405236 | CRF 有效候选 |
| TGNomask | 9.97 | 14.658209 | CRF 有效候选 |
| TGNomask-Aug | 10.08 | 14.700370 | CRF 有效候选 |
| Tree-NoONT | 9.95 | 15.061134 | 正确 grammar 修正值 |
| Tree-Compress | 9.91 | 14.730862 | 正确 grammar 修正值 |
| Tree-TripleCNT | 10.26 | 14.801293 | 正确 grammar 修正值 |
| Tree-Shuffle | 13.41 | 17.852117 | 独立协议，仅作背景 |
| TGNomask-Mix-TG | 10.01 | 14.572762 | CRF 有效候选 |
| TGTree-Mix-TG | 10.00 | 14.481718 | CRF 有效候选 |
| Pause-1 | 9.77 | 14.916121 | 单路径 |
| Pause-2 | 10.32 | 15.116468 | 单路径 |
| Pushdown | 13.29 | 19.316305 | 独立协议，仅作背景 |

100M 的 Tree 相对 Terminal 从 A 的高 0.12 变为 B 的低 0.633128；B 的两个 Mix 值低于
Tree/TGTree。因果线性化变体中，A 的 Compress 最低，B 的 TGTree 最低；NoONT 相对 Tree
从 A 的略低变为 B 的高 0.325036。旧 NoONT/Compress/TripleCNT 的错误 grammar clean
结果没有进入此表，也不能用于前后排名。

### 6.2 500M：A、B、C 必须分列

| 模型 | A：旧 test | B：旧权重 clean | C：dedup 新权重 clean | C−B（同名模型） |
|---|---:|---:|---:|---:|
| Terminal | 2.83 | 17.302676 | 14.028763 | -3.273913 |
| Tree | 3.18 | 16.001136 | 13.863770 | -2.137366 |
| TGTree | 3.28 | 15.894648 | 13.946850 | -1.947798 |
| TGNomask-Aug | 3.19 | 15.483865 | — | — |
| TG | — | — | 15.465373 | — |
| TGNomask | — | — | 13.995203 | — |
| TGTree-Mix-TG | — | — | 13.784714 | — |
| TGNomask-Mix-TG | — | — | 13.972621 | — |
| Tree-NoONT | — | — | 13.938351 | — |
| Pause-1 | — | — | 14.019490 | — |
| Pause-2 | — | — | 14.030113 | — |

A 的旧四模型中 Terminal 最低；B 的旧四模型中 TGNomask-Aug 最低。C 的十模型中
TGTree-Mix-TG 数值最低，Tree 次之；若仅看 A/C 共有的三个模型，C 的 Tree 最低。
TGTree-Mix-TG 相对 Tree 低 0.079055，TGNomask-Mix-TG 相对 TGNomask 低 0.022583。
这些接近值没有显著性检验，不应写成稳定排序或普遍收益。

三项 C−B 为在同一 clean 数据和同类评分合同下的跨 checkpoint 描述性差值，不能解释为
单独的 dedup 效果：训练数据、表示生成、D、LR、batch、步数和运行源码/backend 都可能参与。
B 的旧 500M 比对应旧 100M 的 clean PPL 高；C 的 500M 又不能与历史 100M 拼成同一
受控 scaling 曲线。要核定新的 scaling 结论，需要匹配新训练/评测合同的 100M 对照。

### 6.3 共有三个 500M 模型的下游变化

下表写 A → C。SG/BLiMP 为百分数；XSum 为 R-AVG 百分数；XSum/BoolQ 的 ± 为五个
**微调** seed 的样本标准差。B 只重测 DocPPL，没有生成一组新的 SG/BLiMP/XSum/BoolQ。

| 模型 | SG ↑ | BLiMP ↑ | XSum ↑ | BoolQ ↑ |
|---|---|---|---|---|
| Terminal | 71.11 → 80.8848 | 64.74 → 72.3149 | 20.75 ± 0.06 → 23.1149 ± 0.0735 | 69.69 ± 0.60 → 70.5382 ± 0.5065 |
| Tree | 63.13 → 83.4316 | 76.09 → 82.7552 | 22.25 ± 0.05 → 23.5731 ± 0.0649 | 70.72 ± 0.35 → 70.0489 ± 0.5797 |
| TGTree | 65.95 → 80.6760 | 77.65 → 83.7209 | 22.01 ± 0.08 → 23.5886 ± 0.0503 | 70.63 ± 0.32 → 68.4281 ± 0.2609 |

新十模型的完整下游值、逐 seed 与子项以[现有结果页](bbc_dedup_500m_evaluation_results_20260925.md)
为准。C 内 TGTree 的 XSum/BLiMP 数值最高、Tree-NoONT 的 SG 最高、Terminal 的 BoolQ
最高，不能从 DocPPL 推出所有任务同步改善。尤其共有模型的 BoolQ 并非全部上升。

共有三模型的旧微调配置与新协议均使用 XSum 3 epoch / LR 6e-5、BoolQ 5 epoch / LR 3e-4、
global batch 40、五 seed（42/2026/6198/13171/31723）；XSum test=11,333、BoolQ
validation=3,270。旧 campaign 与 C 记录的 XSum train/test、BoolQ train/validation 和
runtime tokenizer hash 一致。SG 为 32 项六类、BLiMP 为 67×1,000 pairs；结构模型保留
beam300 / supplied K=300 的任务区别。这些共同口径提高比较的可解释性，但新旧微调/评测
运行组织、microbatch 与源码身份不同，本次未逐样本重跑证明数值等价。

2026-10-02 补充核对 SG 的实际约束：Tree/TG 计分分支 sub-beam=300、terminal
fast-track=30，总线性化长度上限为 `max(6L,10)`。虽然入口传入 `nc=max(L,5)`、
`pc=3`，消费 `Stop_Add_NT` 并屏蔽 opening-NT logits 的代码已注释，这两个 NT 数量
限制不生效。不得从配置或传参推断已启用限制；论文 SG 附录相应描述与代码一致，见
[Evaluation 第 5 节](../Evaluation.md#5-sg-与-blimp)。此更正不改变已登记评测数值。

## 7. 完成状态、证据可信度与未解决项

| 证据 | 本次核定 | 可以声明的范围 |
|---|---|---|
| A 冻结稿与 JSON | 源码 SHA 与提取 manifest 一致，19 行冻结表 | 数值转录可信；旧数据污染与方法文字不一致仍限制解释 |
| B 有效 22 模型汇总 | 与原结果/grammar 修正结果一致；逐行完整覆盖字段；NLL 重算 PPL 一致；权重 hash 对齐原 provenance | 可使用已登记有效值并注明评分协议；不是本次重新运行 22 模型 |
| B 原总终验 | 原 `final_status.json` 仍是 `failed`；`identity_validation.json` 明确只认证身份，不认证 full coverage | 不能声称“旧 22 模型原总终验成功”；缺成功总回执的边界保留 |
| B 三项 grammar 修正 | 2026-09-11、Job 999568，scoped final receipt 为 `complete` | 只认证 NoONT/Compress/TripleCNT 三个修正模型，不能扩大为全部 22 模型重新终验 |
| C 三批评测 | 66/66、66/66、88/88，共 220/220；三份严格汇总与终验均 `complete` | 十个模型各 22/22 任务完成，非排队、probe 或部分评测 |
| C 评测后身份核对 | 2026-09-29 原回执记录输入 77/77/72 文件、checkpoint 9/9/12 文件 hash | SIST 这次终验使用本地同步镜像；不是重新扫描远端 SIST 存储 |
| 公开复现 | 代码、冻结配置与选定结果已在仓库；大数据/权重/逐文档日志下载入口待补 | 有可审阅协议与身份摘要；未完成外部干净环境完整重训 |

C 的 22 项是每模型 2 个句法任务、10 个微调及其评测任务、10 个 DocPPL 文档分片；
不是 22 个指标或 22 个训练 seed。三批结果 JSON 的 `evidence_validation` 写
`no fresh checkpoint rehash` 指的是**收集器阶段**；其后 08:55 UTC 的单独 final receipt
完成了记录中的文件重算。两阶段职责不同，不能只读早期字段就认定最后没有终验。
公开 [final status](../reproducibility/results/dedup500m/final_status.json)保留此范围。

B 原总回执失败与三项修正成功没有被改写或掩盖。本报告采用的 22 行有效结果有各自的
覆盖与身份记录；若后续修稿要求“全部旧模型通过同一成功总终验”的更强表述，应另行完成
相应认证，不能用本报告的算术/来源核对代替逐文档终验。

历史 91 篇结构变化经 2026-10-02 复核，仍缺当次 parser 权重、配置与候选分数；
R-03 已按用户决定以“原因未知、停止追查”关闭，历史解释限制保留，见
[数据溯源](bbc_data_provenance.md#provenance)。其余证据缺口包括：历史训练数组精确字节
重建未证实；新 train/clean 独立完整重叠审计未补；大数组、权重和
完整日志公开获取位置未补。这些缺口不等于 C 的 220 项评测尚未完成。

## 8. 对后续论文修订与复现计划的使用规则

### 8.1 后续修稿需同步处理的对应关系

| 旧论述 / 表格位置 | 本报告支持的处理依据 |
|---|---|
| BBC DocPPL 主表 | 选择 B 或 C 时明确训练/测试来源；保留旧数据身份追溯。C 的十行不能假装仍是旧四行；新增/缺失模型须明示 |
| Tree 带来少量 LM cost、100M Pause-1 最低 | 在 B 下该排序改变；需按所选新表重写，不能只替换数字保留旧结论 |
| 混合注意力无收益 | B 的 100M 和 C 的 500M 存在较低 DocPPL；改为分任务、分训练条件的观察，不推导全面或显著收益 |
| BBC 规模变大，域内 PPL 改善同时句法下降 | A 的旧 PPL 有训练重叠；B/C 不能支持同一未限定推理。新 scaling 主张需要匹配数据与协议的对照 |
| DocPPL 正文、附录与表注 | 统一 valid K、joint sum、model-best、terminal/EOS 分母；保留 Pushdown/Shuffle 协议区别；不再给 A 改贴现行合同 |
| SG/BLiMP/XSum/BoolQ 与效率结论 | B 没有重测这些任务；C 的共有行可以明确替换为新训练的独立测量。旧效率、FineWeb 或其他权重的结果不能随 BBC PPL 一起重标 |

五 seed 仅覆盖微调随机性；DocPPL、SG、BLiMP 是单次评测。未做显著性检验，不能把
微调 SD 当成 DocPPL 置信区间，也不能将单一预训练 seed 写成训练稳定性证据。

### 8.2 复现先选目标，再生成新运行

| 目标 | 需要固定的标准与入口 | 当前限制 |
|---|---|---|
| 重现 A 的历史数值 | 明确选择历史 train/test、旧权重与当次 scorer/candidate-0；[历史配置生成器](pretraining_reproduction.md)仅负责训练配方 | 当前 model-best evaluator 不等于历史 scorer；部分运行现场、parser 身份和历史字节仍缺证据 |
| 重现 B 的 clean 重测 | 原旧权重 + reserved-clean-v1 完整 test + 正确 grammar；[clean 评测协议](bbc_reserved_docppl_sist_20260907.md)、独立输出、严格合并与终验 | 外部权重/数据/逐文档证据获取待补；需要区分逐行结果与 scoped 总回执 |
| 重现 C 的新十模型 | 新 dedup/clean 身份 + 十模型冻结配置 + [预训练工作流](pretraining_workflow.md) + 同类任务协议 | 公共 dedup CLI 覆盖八种配方但集合不等于本批十模型；Pause-1/2 与 TGTree-Mix-TG 要按对应冻结配方生成新运行 |
| 分离数据与评分变化的贡献 | 固定同一 checkpoint，分别切换 test 与 history/candidate 规则，建立有明确合同的配对对照 | 本报告没有运行这些对照；已有 A/B 差值不能替代 |
| 建立新的 scaling 或 dedup 因果论证 | 匹配训练来源、表示、token/更新预算与评分合同，明确 LR/batch 是否控制；增加对应规模或训练 seed 对照 | 属于后续实验设计，不是本次报告验收条件，也未提交任务 |

新复现必须创建新的 run/save/result 目录，保存数据/tokenizer/config/checkpoint/source
身份及实际环境/扩展；不得直接重放旧 validation/campaign 路径或覆盖既有结果。
R-08 本次收尾仅完成区分与防误导说明；论文修订和可能的新复现实验由后续任务承接。

## 9. 本次核对证据与验证

核对起点 commit 为 `d3df67d08bfdb677fdf0c14fa17ea498d31dc58e`。论文源码 hash 为
`5b4e17f515b8dbbc0638a3b6e6176c71a77dad1842e81753381eeab558935481`，与冻结表来源一致；
本次没有更改该文件。证据内容身份见[提取 manifest](../reproducibility/evidence/manifest.json)、
[新配置 manifest](../reproducibility/configs/dedup500m/manifest.json)及[新结果终验](../reproducibility/results/dedup500m/final_status.json)。

本地补充核对来源（不作为公开下载链接）：

- `EXPERIMENT_REPRODUCTION_RECORD.md`、`REPOSITORY_CLEANUP_MEMORY.md` 的 R-08/R-11 与原实验登记。
- `artifacts/evaluation/reserved_docppl_sist_20260907/{results,provenance,input_validation,identity_validation,final_status}.json`；
  `grammar_fix_20260909/{results,final_status}.json` 及三个评分 contract。
- 三份 C campaign 的 `protocol.json`、`tasks.json`、`input_validation.json`、`results.json`、
  `final_status.json`；训练三批 `validation/pretrain_5090_dedup*/` 的 protocol/receipt/data manifest。
- `dataset/bbc-news-parsed-dedup/{format}/train.manifest.json`、NoONT 转换 manifest、clean top-300 manifest。
- `artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/run_manifest.json` 与共有三模型的微调配置。

本次程序核对通过：5 份公开 evidence 文件及 8 个本地提取来源的 hash，29 份历史固定配置，
22 个 B 模型 NLL/PPL 与原来源（含 3 项修正），10 份 C 配置 hash 与 30 个 checkpoint 文件
路径/大小，三批公开/原结果及终验 hash，220 项计数、五 seed 均值/样本 SD、full-evaluation
标记，9 个 B/C clean 基础身份、19 个 C 本地/SIST clean 身份、4 个 train manifest 身份。
另外读取 4 个新 train 和 3 个 clean 基础 test 的 NPY header，长度/dtype 与 manifest 一致；
runtime tokenizer 小文件 hash 与完整 JSON 等价检查通过。五张对照表的 49 行从同一
JSON/配置提取，并核对展示舍入与差值。

文档交付检查使用 `python scripts/check_public_release.py` 和 `git diff --check`。
上述验证是本地来源、配置、算术和文档检查，未重新 hash 大权重/大数组、重跑 GPU、执行
全量重叠审计或完成外部干净环境复现。
