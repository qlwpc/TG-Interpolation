# 预训练流程与语料构建：论文 ↔ 仓库对照报告

> [!NOTE]
> 本文已按 2026-08-31 仓库状态回写：已完成项直接反映在 §1--§3，不再保留会误导的
> 初始缺口记录。集中纠错索引见
> [`../REPOSITORY_CLEANUP_MEMORY.md`](../REPOSITORY_CLEANUP_MEMORY.md) M-11。

生成日期：2026-08-23；自动化状态更新至 2026-08-28。依据：论文 `14901_A_Scaled_Up_Empirical_St.pdf`（Table 2/3/8、§5.1、附录 A），
仓库代码核查（datatools/、olmo/、train_configs/、saved_models/*/config.yaml、tests/）。
本文只覆盖**预训练流程与语料构建**；评测协议与数值登记见 `EXPERIMENT_REPRODUCTION_RECORD.md`。

## 1. 端到端流水线总览

```
FineWeb(BBC 子集) ──get-bbc-fineweb.py──▶ 预览/拉取
      │
      ├─ BBC 分支: benepar_parse.py（benepar_en3_large, spacy 分句, 断点续跑按行数）
      │    └─ Slurm 入口 parse_bbc.sh
      ├─ FineWeb-Edu 分支: parse_input.py + worker.py + manage_parse.py
      │    （benepar_en3, 多进程流水线, filelock 任务队列 task_status.json）
      │    └─ Slurm 入口 parse_edu.sh（246 组 arrow 分片正则, 覆盖 984 shard）
      ▼
<split>.txt（每文档一行, 括号树文本）
      │ convert_TG_and_tokenize.py（joblib 并行, 按已存在文件跳过）
      ▼
dataset/<corpus>/{tree,terminal,tg}/<split>.npy   ← 一维 uint16 token 流（无结构数组）
      │ build_pretrain_data assemble / assemble_streams.py（两遍 mmap，统一生成 train/dev/test 与索引）
      │ sample_testset.py（seed=42, ~10% 文档抽测, 写 test_*_index.json）
      ▼
train/dev/test.npy + *_sent_index.npy + *_doc_index.npy
```

## 2. 论文 ↔ 仓库逐项对照

### 2.1 语料与解析

| # | 论文声称 | 仓库证据 | 判定 |
|---|---|---|---|
| 1 | BBC News 从 FineWeb 过滤 | `get-bbc-fineweb.py:3` 拉取 `permutans/fineweb-bbc-news` | ✅ |
| 2 | Benepar 解析（news-domain 模型） | BBC=`benepar_en3_large`（`benepar_parse.py:152`）；FineWeb-Edu=`benepar_en3`（`worker.py:169,237`） | ⚠️ 两分支模型不一致 |
| 3 | BBC ~10B terminal tokens | `terminal/train.npy` 20.12 GB ÷ 2 B = 10.06 B tokens | ✅ 实测吻合 |
| 4 | LIN1 ≈ 2.4× token 膨胀 | `tree/train.npy` 49.41 GB = 2.46× | ✅ 实测吻合 |
| 5 | FineWeb-Edu-100BT（1B 用） | `parse_edu.sh` 246 组正则；`saved_models/A800_models/{tree,tgtree,pause1,pause2}_1B*/step*/config.yaml` 各保存 246 组 `fineweb-edu-v2` shard patterns | ✅（本机无 Edu .npy；checkpoint config + 链路佐证） |
| 6 | 保留 NT token 的 tokenizer | `get_TG_tokenizer.py:45-48` 注册 26 开 + 26 闭 NT（id 50268–50319） | ✅ |
| 7 | GPT-2 或 Qwen3 tokenizer | BBC=GPT-2 JSON；1B 默认=Qwen3 JSON（`olmo/config.py:882`） | ✅ |
| 8 | Pause-1/Pause-2 | 在线插入（`memmap_dataset.py:263`→`pause_input_ids`），支持任意 p/q | ✅（比论文更泛） |
| 9 | LIN1-shuf 打乱 | 在线：`collator.py:65-68` 调 `random_shuffle_tree` | ✅ |
| 10 | LIN2（tg） | 预生成 `convert_treenpy_to_TG`（.so；Python 副本 `tokenize_BLiMP_tg.py:61-74`；流式版 `parse_data/tree_to_tg.py:168-212`） | ✅ |
| 11 | LIN3/noONT/merge | `make_tree_variant.py` 已实现 mmap 流式生成并通过真实 dev 验证；三个全量 train 变体尚未在本机生成 | 🟠 实现已验证，数据资产待生成 |
| 12 | 变体差异承载方式 | noont/compress/triplecnt 训练时 grammar type 均为 `tree`/`tgtree`，差异全由 `.npy` 目录承载 | ⚠️ 数据承载变体 |

### 2.2 训练配置

| # | 论文声称 | 仓库证据 | 判定 |
|---|---|---|---|
| 1 | 100M = 768·12L·12H | `train_configs/terminal.yaml:12-14`（run_name 却叫 "OLMo-300M"） | ✅（命名混乱） |
| 2 | 500M = 1408·16L·16H | `tree-500M.yaml:11-14`（mlp_ratio 8） | ✅ |
| 3 | 1B = 2048·16L·16H | `terminal-1B.yaml:11-14`（mlp_ratio 8, project=TG-finewebedu） | ✅ |
| 4 | 1 epoch | 多数 `max_duration: 1ep`；TG.yaml=固定 21637 步；多个配置 `stop_at` 提前截断 | ⚠️ 有例外 |
| 5 | AdamW β=(0.9,0.95) ε=1e-8 wd=0.1 | `olmo/optim.py:989-1001` + YAML 覆写 | ✅ |
| 6 | grad clip 1.0 | 优化器 step 内全局固定裁剪 | `olmo/optim.py:229-240` ✅ |
| 7 | cosine, 2000 warmup → 1e-5 保持 | `CosWithWarmup`（`olmo/optim.py:694-712`） | ✅ |
| 8 | Step Law 定 lr/B | `scripts/step_law.py` + 15 个测试；三个 YAML 的 lr 与公式精确吻合，Tree 截断运行另有说明 | ✅ 已实现并核验 |
| 9 | causal→FA2, 结构→Flex | flash+flex 双开，按 batch 是否带 bias 隐式路由 | `olmo/model.py:573-655` ✅ |
| 10 | seq len 2048 | 所有 YAML 2048 + 训练循环强断言 | `olmo/train.py:2033-2035` ✅ |
| 11 | 论文变体可形成 campaign | `paper_pretraining_manifest.json` 登记 27 个 run，`prepare_paper_pretraining.py` 从 checkpoint config 清洗生成；部分原始 YAML 仍不完整 | 🟠 campaign 可生成，原始模板不齐 |

## 3. 当前未关闭项（按影响排序）

1. **三个 Tree 变体的全量 train 数据尚未生成**：生成器与真实 dev 已验证，但
   noONT/compress/triplecnt 的全量资产仍需显式执行并登记哈希，不能仅凭目录名声称可训练。
2. **Benepar 模型是明确的历史协议差异**：BBC=`benepar_en3_large`，FineWeb-Edu=`benepar_en3`；
   复现应保留这一区别，不能静默统一。
3. **部分原始 `train_configs` YAML 不齐**：当前 campaign 可由 manifest 和 checkpoint
   config 清洗生成，但这不等于原始手写模板已经补全。
4. **`tg_mask` 权威源码缺席**：`olmo/data/tg_mask.py` 不在仓库（只有编译 `.so`），
   `random_shuffle_tree`/`convert_treenpy_to_*` 的权威实现仍不可审阅。
5. **硬编码个人环境**：若干旧数据、rsync、集群和 `/dev/shm` 路径仍需参数化后再发布。
6. **死/名不符实脚本**：`save_datasets.py`、`native_binary.py`、cartesian、reset_error、
   split_testwork 等尚未归档或移除。
7. **1B config 路径可能被后续评测污染**：FineWeb-Edu Terminal checkpoint 当前
   `data.paths` 指向 HellaSwag，根目录 TGTree 副本指向 `fewshot_sources.py`；训练语料须以
   论文、一 epoch处理量和 `saved_models/A800_models/` 内其余 246-shard config 交叉核验。

## 4. 磁盘数据布局（实测）

```
dataset/
├── bbc-news/
│   ├── TG_GPT2_tokenizer.json         # added_tokens 64 个, id 50256–50319
│   │                                  # 开 NT 50268–50293, 闭 NT 50294–50319
│   ├── terminal/{train,dev,test}.npy  # train 20.12 GB ≈ 10.06 B tokens
│   ├── tree -> ../tree                # symlink
│   ├── tg/                            # 本机缺 train.npy
│   ├── tree_triplecnt/                # 空
│   ├── parse_aligned/…                # 评测/辅助: input_ids+spans+span_counts+chunk_index
│   └── testppl_aligned/{terminal,tree,tg}/  # [BOS,…,EOS] 统一边界评测流
├── tree/train.npy                     # 49.41 GB ≈ 24.7 B tokens（= 2.46×）
└── TG_QWEN3_tokenizer.json
corpus/bbc-{terminal,tree}.lazy        # GPST lazy memmap
```

§2.1 #3/#4 的 token 数与膨胀比由文件大小实测得出（uint16 = 2 字节/token）。

## 5. 测试健康（本轮修复记录）

- `tests/test_bracket_mapping.py`：2 个用例原联网下载 gpt2 tokenizer（登录节点无网络必失败），
  改为内存构造 `WordLevel` tokenizer（新增 `_offline_hf_tokenizer()` helper）。
- `tests/test_config.py`：`test_ispause_exact_pause_crashes` 为过期测试——裸 `"pause"` 崩溃 bug 已在
  `olmo/data/util.py:408-412` 修复（按 `(1,1)` 解析），测试改写为 `test_ispause_bare_pause_defaults_to_one`。
- 修复后：`test_config + test_bracket_mapping + test_data_utils + test_memmap_formats` 35 用例全绿（2.7s 离线）；
  `test_parse_align + test_gen_tgppl_fromtree` 34 passed / 2 条件 skip（缺再生成的 Pushdown dev 数据）；
  `test_prepare_syntactic_baselines` 真实数据位级一致性 6/6 通过。

## 5b. 解析/Tokenize 管线审计（2026-08-25 补充）

| 组件 | 结论 | 关键证据 |
|---|---|---|
| `convert_TG_and_tokenize.py` | ✅ 正常 | 合成冒烟三路不变量全对（terminal=去全部 NT、tg=闭 NT×2、-LRB-→(、EOS）；已修 demo print/默认路径/resume 三目录/空输入 4 处 |
| `benepar_parse.py`（BBC） | ✅ 代码正常；数据源被禁 | benepar 模型本机缓存（~/nltk_data/models）；`permutans/fineweb-bbc-news` 在 hf-mirror 上 API 元数据 200 但文件 resolve GET/HEAD 均 403（模型类文件正常）→ 仓库级封锁，按用户指示不修。已加 `--data_files` 本地 parquet 模式 + `--max-docs` |
| `parse_input.py`/`worker.py`（Edu） | ✅ **CPU 端到端冒烟通过** | text→spacy 分句→t5 预算→benepar_en3（无 CUDA 自动回退 CPU）→chart 解码→writer 重排→输出合法 PTB 括号树，且可被 `convert_TG_format` 往返消费（/tmp/smoke_worker/tmp_parse.txt） |
| 依赖自举 `setup_parse_deps.py` | ✅ 闭环 | 实测自动装 spacy en_core_web_md（GitHub）+ t5-small（hf-mirror 回退；hub 不可达时自动切换）；已接入三个脚本入口（--skip-deps 可跳） |

冒烟中发现并修复的环境坑（均已写入脚本）：
1. t5-small 无 tokenizer.json → benepar Retokenizer 走 slow→fast 转换 → protobuf≥4 与 sentencepiece 旧 _pb2 不兼容；
   解法：三个脚本顶部 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`（parse_input.py/worker.py/benepar_parse.py）。
2. `parse_input.py` 模块导入即加载 t5-small → 离线 import 失败；改为 `_get_split_tokenizer()` 懒加载。
3. spawn 多进程的驱动脚本必须落盘为真实 .py（stdin 管道喂代码会让子进程 `runpy` 找不到 `<stdin>`）。
4. `benepar_parse.py` `batch_buffer.write` 引用全局 `pbar` → 改 `self.pbar`。

新增测试：`tests/test_parse_pipeline_smoke.py`（12 用例：解析↔tokenize 契约、自举模块、离线导入、长句切分）。

## 6. 已完成修复与后续动作

1. ✅ **已实现（2026-08-24）**：Step-Law 脚本化 → `scripts/step_law.py`（测试 `tests/test_step_law.py`，15 用例）。
   审计结论：terminal.yaml / terminal-500M.yaml / tree-500M.yaml 的 lr 与公式**6 位小数精确吻合**
   （N=非嵌入参数量 70,900,224 / 507,896,576，D=实际线性化 token 流大小）；B 预测 145.7→144、
   243.4→244（取整）。**例外**：tree.yaml 的 lr=0.003608 对应 D=2.835e9（=stop_at 19233 步 × 72 × 2048），
   即按截断运行推导；论文所用 checkpoint（step49440=7.29B token）超出该设计 D 2.6 倍。
   三个已验证 YAML 已加来源注释。注意：论文 "~100M" 实为 70.9M 非嵌入 + 38.6M 嵌入 ≈ 109M；
   本仓库 SwiGLU gate/up 各为 hidden/2（每层 MLP = 1.5·d·hidden）。
2. ✅ **已实现（2026-08-24）**：变体生成脚本 →
   `datatools/parse_pretrain_data/make_tree_variant.py`（测试
   `tests/test_make_tree_variant.py`，15 用例）。noont/compress/triplecnt 三变体，mmap 两遍流式、
   chunk 边界对齐闭 NT run、NT 范围从 tokenizer JSON 推导（不依赖 tg_mask .so）、可选多进程。
   真实 dev.npy 验证：8,082,906 → noont 5,687,613 / compress 6,773,896 / triplecnt 12,873,492 tokens，
   token 级抽查正确。dev 输出已写入 `dataset/bbc-news/tree_{noont,compress,triplecnt}/dev.npy`。
   **train.npy 未生成**（约需 155G 磁盘：≈34.8G+41.4G+78.7G，当前 /home 余 914G）——待用户确认后运行：
   `python -m datatools.parse_pretrain_data.make_tree_variant --input dataset/tree/train.npy --output-dir dataset/bbc-news/tree_noont --variant noont --tokenizer dataset/bbc-news/TG_GPT2_tokenizer.json`（compress/triplecnt 同理）。
3. ✅ **已实现（2026-08-28）**：`assemble_streams.py` 使用两遍 mmap、标准 NumPy header
   和固定 shard/split 索引生成三路 train/dev/test；统一入口为
   `python -m datatools.parse_pretrain_data.build_pretrain_data assemble --corpus bbc`，不再依赖人工 `cat`。
4. ✅ **已显式登记（2026-08-28）**：BBC=`benepar_en3_large`、FineWeb-Edu=`benepar_en3`
   作为历史数据协议差异写入 README 与 `docs/pretraining_reproduction.md`，复现时不静默统一。
5. ✅ **已实现（2026-08-28）**：`train_configs/paper_pretraining_manifest.json` 覆盖 27 个已登记
   run；`scripts/prepare_paper_pretraining.py` 从对应 checkpoint config 清洗生成全部论文模型
   campaign，并逐项校验 §2A 中的架构、LR、batch、dtype、pause 与 mixing 协议。
6. 找回/重建 `olmo/data/tg_mask.py`（对照 `old_tg_mask.py` 或 .so），恢复可审阅性。
7. 死脚本清理：save_datasets / native_binary / cartesian / reset_error / split_testwork 归档或删除。
8. ✅ **已核验（2026-08-28）**：FineWeb-Edu 1B 的 Terminal/Tree/TGTree/Pause-1/Pause-2
   checkpoint 均位于 `saved_models/A800_models/`；已在复现记录 §2/§2A 登记无歧义
   corpus-qualified model key、实际 LR/batch/step、Qwen3 `uint32` 和 pause id。
