# FlexAttention 本地路径、短序列正确性与 SDPA 性能审计（2026-09-01）

> **Implementation update:** 本报告提出的任务/长度路由已经在原 torch
> 2.7.1 环境实现并通过 GPU 回归。最终阈值、配置变化和验证证据见
> `2026-09-01-attention-backend-router-implementation.md`。下文保留的是实现
> 前的审计证据和决策依据。

## 结论

不能确认“本地路径已经完美修复”。本轮用真实 BBC TG 掩码发现了此前合成
reproducer 没覆盖到的反例：在 RTX 3090、BF16、`B=4, H=12, D=64,
N=127` 下，TG 和 tgnomask 的 FlexAttention 前向有限，但 Q/K/V 反传产生
非有限值。生产环境 torch 2.7.1 和隔离升级环境 torch 2.9.1 都失败；升级
PyTorch 并没有修复这个真实任务形状。

已确认的可靠修复是把 Flex 计算补齐到 128：真实 TG/tgnomask `N=127 ->
128` 的输出、Q/K/V 梯度均有限，并与未补齐的 SDPA 参考高度一致。不过该
修复目前由 `OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE=128` 显式启用，只覆盖
已经导出这个变量的启动脚本，不是所有本地入口的默认行为。

性能疑问同样成立。Flex 的 `torch.compile` 只是生成融合内核，不保证比
SDPA 快；在本机 3090 上，普通因果注意力从 `N=16` 到 `N=2048` 都比 SDPA
慢。真实 TG/tgnomask 在短序列和一次性评测中也更适合 SDPA；Flex 只在长
序列、结构化掩码、重复运行足以摊销编译时显示优势。

## 运行时到底走哪条路径

`model.flex_attention: true` 不等于一次前向一定执行 Flex。当前优先级是：

1. 有文档长度时走 `flash_attn_varlen_func`；
2. 没有 additive/BlockMask 且 FlashAttention 可用时走 `flash_attn_func`；
3. 有 `BlockMask` 时走独立 `torch.compile(flex_attention)` 的 Flex；
4. 其余情况走 PyTorch SDPA。

因此 YAML 中的 `compile: null` 只是不编译 OLMo block，并不会关闭 Flex
自身的 `torch.compile`。

| 任务/语法 | 配置现状 | 实际路由与判断 |
|---|---|---|
| terminal、tree、tgtree、pause、treereg | 部分旧配置仍写 `flex_attention: true` | 这些类型不生成 TG bias；通常实际走 FlashAttention，缺少 flash-attn 时走 SDPA。`terminal-500M.yaml`、`tree-500M.yaml`、`tgtree-500M.yaml` 中的 true 是误导性的死标志 |
| TG | `train_configs/TG.yaml` 和评测配置为 true | 有真实 TG bias，实际构造 BlockMask 并走 Flex |
| tgnomask | 普通评测为 true，XSum 配置有 false | true 时实际走 Flex；false 时用同一语义的 dense additive mask 走 SDPA，不能替换成纯 causal SDPA |
| mixing | 当前 false | 使用逐头结构化 bias，但当前走 SDPA；尚未做真实 mixing A/B，不建议仅凭合成 mask 打开 Flex |
| Pushdown 预训练/解析 PPL | `flex_attention: true` 且 `pushdown_use_flex: true` | 用 Flex `score_mod` 融合深度 bias；这是与普通 TG 不同的专用路径 |
| Pushdown BoolQ | 每指标配置已设 `pushdown_use_flex: false` | 短任务走 fallback，策略合理 |
| KV-cache 生成 | 代码明确不构造固定长度 BlockMask | 回退到 Flash/SDPA；Flex 标志不代表 decoding 使用 Flex |

## 正确性证据

| 环境/用例 | 结果 | 证据 |
|---|---|---|
| torch 2.7.1，真实 TG `N=127` | 前向有限，Flex Q/K/V 梯度非有限 | job 3907 |
| torch 2.7.1，真实 tgnomask `N=127` | 前向有限，Flex Q/K/V 梯度非有限 | job 3908 |
| torch 2.9.1，以上两个真实 `N=127` 用例 | 两者仍失败 | jobs 3913、3914 |
| torch 2.7.1，真实 TG/tgnomask `N=128` | 前向、反向、SDPA parity 均通过 | jobs 3915、3916 |
| torch 2.7.1，真实 TG/tgnomask `N=512/1024/2048` | 全部前向、反向通过 | jobs 3909–3912、3917–3918 |
| torch 2.7.1，真实 `N=127` 补齐到 128 | 两种 mask 均通过；梯度 cosine 至少 0.999994，relative L2 至多 0.00346 | job 3919 |
| 合成上游 `H_BlockMask=8, N=127` | torch 2.7.1/2.9.1 均通过 | jobs 3879、3880 |
| Pushdown 完整小模型 `N=16` | torch 2.7.1/2.9.1 均通过 | jobs 3885、3886 |

合成用例与真实用例并不矛盾：失败依赖掩码结构/广播形状，不能从一个通过
的短序列形状推广到所有任务。此前“当前环境已经通过相关短序列路径”的
报告结论已被本轮真实 mask 结果推翻，原报告已加更正说明。

“完美修复”仍缺少这些覆盖：真实 mixing 的逐头 mask、更多 batch/head/head
dimension 组合、不同 GPU 架构、所有动态长度 bucket、完整 checkpoint 的
逐任务端到端 A/B，以及仍未得到结论的独立 GQA 上游形状。Pushdown 的
`N=16` 通过只能证明 Pushdown 专用 score-mod 路径，不能替 TG 路径背书。

静态审计还发现一个尚未修复、也未被本轮任务触发的组合分支：TG bias 与
`attention_mask` 同时存在时，`TG_mask_with_pad` 使用 Python `and`，并写成
`attention_mask[kv_idx]` 而不是按 batch 的逐元素索引。相邻 Pushdown 分支
已经明确说明这种 Python 数据依赖控制流会被 vmap 拒绝。与此同时，现有
pad-to-128 workaround 会直接拒绝非空 `attention_mask`。因此这个组合路径
不能计入“已修复”；在实际启用前应改为 batch-aware 的位运算并单独回归。

## Flex 与 SDPA 基准

固定环境为 RTX 3090、torch 2.7.1+cu126、Triton 3.3.1、BF16，注意力形状
`B=4, H=12, D=64`。表中的时间是 12 层 attention-only 估算：SDPA 为
12 个稳定态 kernel；Flex 为 12 个稳定态 kernel，加一次由所有层复用的
稳定态 BlockMask 构建。它不包含 QKV/输出投影、MLP、通信和数据加载，不能
直接当作完整训练 step 的加速比。

SDPA 输入预先转成 BF16，这比当前 OLMo 每层从 float bias 转换更有利于
SDPA；Flex 端则完整计入一次 BlockMask 构建。因此这是偏保守、偏向 SDPA
的算子比较，适合否定短序列 Flex 优势；长序列仍需要完整任务 A/B 才能决定
最终吞吐。

### 真实 TG/tgnomask：训练前向+反向

| Mask | N | SDPA 12 层 (ms) | Flex 12 层+mask (ms) | Flex/SDPA | 判断 |
|---|---:|---:|---:|---:|---|
| TG | 128 | 3.03 | 9.33 | 3.08x | SDPA 明显更快；127 的精确 Flex 还会坏梯度 |
| TG | 512 | 5.76 | 9.44 | 1.64x | SDPA 更快 |
| TG | 1024 | 18.44 | 16.81 | 0.91x | Flex 仅快约 9%，很容易被冷启动抵消 |
| TG | 2048 | 66.23 | 46.85 | 0.71x | Flex attention 部分快约 29% |
| tgnomask | 128 | 3.05 | 9.41 | 3.08x | SDPA 明显更快 |
| tgnomask | 512 | 5.78 | 9.75 | 1.69x | SDPA 更快 |
| tgnomask | 1024 | 18.42 | 16.78 | 0.91x | Flex 仅快约 9% |
| tgnomask | 2048 | 66.22 | 44.05 | 0.67x | Flex attention 部分快约 33% |

### 真实 TG/tgnomask：只前向评测

| Mask | N | SDPA 12 层 (ms) | Flex 12 层+mask (ms) | Flex/SDPA | 判断 |
|---|---:|---:|---:|---:|---|
| TG | 128 | 0.27 | 4.54 | 16.7x | SDPA |
| TG | 512 | 1.31 | 4.68 | 3.58x | SDPA |
| TG | 1024 | 4.10 | 6.47 | 1.58x | SDPA |
| TG | 2048 | 14.56 | 12.84 | 0.88x | Flex 稳定态仅快约 12% |
| tgnomask | 128 | 0.27 | 4.55 | 16.7x | SDPA |
| tgnomask | 512 | 1.31 | 4.67 | 3.56x | SDPA |
| tgnomask | 1024 | 4.11 | 6.48 | 1.58x | SDPA |
| tgnomask | 2048 | 14.62 | 12.83 | 0.88x | Flex 稳定态仅快约 12% |

真实结构化用例的 Flex compile+cold 时间在 `N>=512` 时约为前向 4.1 秒、
训练前反向合计 9.9–10.0 秒。理想情况下，`N=2048` 训练也需要约 500 个
同形状 attention stack 才能摊销这部分差额；`N=1024` 的 9% 小优势需要
数千次。动态长度产生新编译 bucket 时，摊销条件更差。

### 普通 causal 控制

| N | SDPA 12 层训练 (ms) | Flex 12 层+mask (ms) | Flex/SDPA |
|---:|---:|---:|---:|
| 16 | 2.87 | 7.96 | 2.78x |
| 128 | 2.87 | 7.55 | 2.63x |
| 512 | 2.93 | 7.50 | 2.56x |
| 1024 | 7.83 | 11.73 | 1.50x |
| 2048 | 23.17 | 32.45 | 1.40x |

这里 Flex 连 PyTorch SDPA 都没有赢；生产环境还有可用的第三方
FlashAttention，所以普通 causal 任务没有理由为了 `compile` 改走 Flex。

## 逐任务决策

1. **普通 causal 任务**：保持 FlashAttention/SDPA；把三个 500M 配置中
   误导性的 `flex_attention: true` 清理为 false，但这只是配置卫生，不会
   改变当前实际 kernel 路由。
2. **TG/tgnomask 短训练与下游评测**：`N<=512` 优先 SDPA。`N<128` 禁止
   精确长度 Flex 反传；如果必须保持 Flex，就必须 pad-to-128。
3. **TG/tgnomask 长训练**：固定或少量 bucket 的 `N=2048` 保留 Flex，并
   强制 pad-to-128；`N=1024` 只有约 9% 的 attention-only 优势，应以完整
   step A/B 决定，而不是因配置中出现 `compile` 就启用。
4. **TG/tgnomask 评测**：`N<=1024` 用 SDPA。`N=2048` 只有在同一 shape
   重复足够多次、编译缓存可复用时才考虑 Flex；BoolQ/XSum 这类短而动态的
   一次性任务默认 SDPA 更合理。
5. **Pushdown 长训练/解析 PPL**：保留 Flex。其 `score_mod` 表达深度 bias，
   当前 SDPA fallback 会物化 `(B,H,N,N)` dense bias，并已有约 100x 慢或
   OOM 的项目证据，不能套用普通 TG 的短序列结论。
6. **Pushdown 短下游/KV-cache**：BoolQ 每指标配置已经关闭
   `pushdown_use_flex`；KV-cache 路径本来就回退。继续按任务做 A/B，不应
   仅看顶层 `flex_attention: true`。
7. **mixing**：维持 false，等待真实逐头 mask 的端到端 A/B。

本轮没有直接批量改任务 YAML。建议先实现一个显式的任务/长度路由策略：
短 TG/tgnomask 用同语义 additive-mask SDPA；长训练 Flex；任何进入
TG/tgnomask Flex 反传且长度不整除 128 的输入自动补齐。Pushdown 保持其
独立策略，不应直接套用 TG padding。这样同时解决已复现的正确性问题和短
任务效率问题。同时应先修复并测试 TG bias + `attention_mask` 的组合
mask_mod；不能用全局切换某一个 kernel 来掩盖分支差异。

## 证据文件

- 基准程序：`benchmark_flex_vs_sdpa.py`；Slurm 入口：
  `benchmark_flex_vs_sdpa.sbatch`
- 真实 pad-to-128 validator：`verify_real_tg_pad128.py`；结果：
  `results/real-tg-pad128_3919.json`
- 真实短序列失败：
  `results/flex-vs-sdpa-tg-n127-real-v2-train_3907.json`、
  `results/flex-vs-sdpa-tgnomask-n127-real-v2-train_3908.json`
- torch 2.9.1 升级复测：
  `results/flex-vs-sdpa-tg-n127-real-torch291-train_3913.json`、
  `results/flex-vs-sdpa-tgnomask-n127-real-torch291-train_3914.json`
- 128 边界：jobs 3915–3916；真实长序列：jobs 3909–3912、3917–3918
- 历史完整 pad 验证与 100-step 训练：
  `artifacts/experiment/flexattention_illegal_access_diagnostic_20260820/README.md`
- Pushdown 3090 性能记录：
  `artifacts/experiment/pushdown-gpu-opt-20260802/RESULT.md`
