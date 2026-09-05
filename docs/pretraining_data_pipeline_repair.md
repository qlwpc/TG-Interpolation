# 预训练数据 pipeline 修复记录（2026-09-05）

操作入口与完整命令见 [`pretraining_reproduction.md`](pretraining_reproduction.md)。本记录只说明
此次代码修复与验证，不代表已重跑完整语料、预训练或论文指标。

## 已提供输入（文件解析结果）

| 输入 | config 数 | 文档数 | SHA-256 |
| --- | ---: | ---: | --- |
| `dataset/bbc-news/dev_index.json` | 89 | 4,980 | `a220f75c9cdb71721f8d7e373cebbb6505fbdcb7db708ea529c92f7036f9e583` |
| `dataset/bbc-news/test_index.json` | 89 | 5,025 | `8994db665aa1fc0962941df62ad895bd6ba1806d37dedf56b9b6e73d55c8ced0` |

两份原文件未改写；列表无重复、无 dev/test 交叉，保留原有非排序顺序。94 个 BBC config 中
最后五个 2023 config 不在留出索引内，新流程将其全部纳入 train。这是明确的构建规则，
不据此声称它等同于历史训练流的字节内容。

`.gitignore` 仅放行这两份 JSON，其他原始数据、tokenizer、NumPy 大文件仍不进入发布源码。

## 修复范围

| 阶段 | 原问题 | 当前行为 |
| --- | --- | --- |
| plan / split | 默认重新抽样，且自定义路径没有进入生成命令 | 默认固定发布索引，保留全部路径参数及 shell 引号；新 split 必须显式另存 |
| download | 网络中断可留下正式文件名的残缺 parquet | 临时文件、长度检查、成功后替换；失败保留原文件 |
| parse | 标准 Benepar 的 `Tree` 被当 top-k 列表取 `[0]`，`zip` 可截断结果 | 区分两种 API，检查树数与完整 terminals |
| parse / tokenize 衔接 | 部分解析与完整输出无法区分，FineWeb 忽略 `--max-docs` | 两套 parser 写完成记录；partial 或哈希变化不能进入统一 tokenization |
| FineWeb parse | Arrow 加载依赖目录枚举顺序 | 按文件名排序；无匹配立即报错 |
| tokenization | 全 shard 拼接占用内存、坏行被跳过、仅凭文件存在续跑 | 逐文档写盘；错误定位到行并失败；三路 NPY + SHA-256 完成记录 |
| assembly | 丢失索引列表顺序；写完后才检查总文档数 | 写前检查全部输入/目标、逐 shard 对齐、索引范围与交叉；保留选择顺序 |
| validation | 只核对 shard 数量和 dtype，未检查最终组装流 | 精确文件集合、tokenizer 布局、BOS/EOS、token 范围、完成记录/哈希；独立 assembled/all scope |

tokenization 的完成记录最后发布；assembly 的各数组分别原子替换，全部成功才发布组装
manifest。它们不是整套语料的文件系统事务。发生中断时半成品不自动视为成功，核对重建
目录后显式 `--overwrite` 恢复。不要并发写同一输出目录。

## 验证结果（本次实际执行）

环境：现有 `LLM` Python 3.10，NumPy 2.2.6，tokenizers 0.22.2；没有安装新依赖或下载语料。

```bash
python -m pytest -q \
  tests/test_pretraining_data_integrity.py \
  tests/test_parse_pipeline_smoke.py \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_paper_pause_protocol.py \
  tests/test_bracket_mapping.py \
  tests/test_make_tree_variant.py
```

结果：**118 passed**。包括：

- 小型解析文本 → 三路 tokenization → 按非排序索引组装 → 最终哈希验证；
- 真实本机 GPT-2/Qwen3 tokenizer + 编译 `SentencepieceVocab` 的 uint16/uint32 三路写盘；
- 坏行、空行、未结束行、输出损坏、缺少完成记录、源文件/tokenizer 变化的拒绝和恢复；
- 逐 shard 错位但总文档数相同、重复/交叉/越界索引、晚出现的已有目标文件均在写前拒绝；
- 标准/top-k Benepar 返回值、空 holdout、解析 partial 完成记录、下载截断保护；
- 既有 Pause 协议、预训练配置与 tree variant 回归。

在独立源码副本中，仅复制数据工具、BBC config 清单、两份 split JSON 与新增完整性测试，
不复制本机 tokenizer、语料或编译扩展：**37 passed, 1 skipped**。唯一跳过项是实际
tokenizer/编译扩展集成测试，其余合成 pipeline 测试离线通过。

两套 `plan`、`validate-splits --corpus bbc`、Python 编译检查及本次修改的 `git diff --check`
均通过。测试警告为环境的 NVML 不可用及 Google API 对 Python 3.10 的后续支持提醒，
未影响 CPU 数据测试。

## 尚未验证的边界

没有重新下载 BBC/FineWeb、跑全量 Benepar、构建完整 token 流、启动 GPU 预训练或覆盖
已有数据。修复时缺少完整 raw/parsed/tokenized 中间目录，不能做逐 shard 历史数据对账。
因此发布前仍需一次隔离目录中的全量构建与语料指纹验收；Hub revision、Arrow cache 版本、
解析器依赖和权重身份也应进一步固定。

旧 `gen_final_train.py` 的追加索引副作用不作为新协议。5,025 文档 test 与 4,966 文档
DocPPL 的边界不能自动互换，参见 [`tree300_vs_test_boundary_report.md`](tree300_vs_test_boundary_report.md)。
`validate --scope all` 只验三路 shard/组装流，不承诺派生基线监督或整套论文指标正确。

本次采用 science 技能的输入/验证/主张分离原则记录证据；当前环境未提供其
`artifact.science`/`bash_exec` 接口，故使用本地执行、测试源码与本记录，不生成虚构证据图。
