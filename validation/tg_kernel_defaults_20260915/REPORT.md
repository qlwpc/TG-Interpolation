# TG 系列 C++ 默认输入与内核整理

## 结果

已启用支持配置的默认 C++ CPU 元信息路径：`tg`、`tgnomask`、`tgnomaskaug` 和包含 `tg/tgnomask/tgnomaskaug/tgtree` 的混合头。最终单卡 GPU 回归 **442 项通过，0 失败、0 跳过**。Slurm 作业 4605、4607 均 `COMPLETED / 0:0`，资源已释放。

## 默认设置与边界

```yaml
model:
  tg_typed_attention: null
  attention_dropout: 0.0
data:
  tg_layout_backend: native
  generate_attention_mask: false
  generate_doc_lengths: false
```

- `null` 自动选择支持的 fresh-segment sequential MHA（head dimension 16–128）；`true` 严格要求启用，`false` 显式关闭。
- C++ 构建默认要求成功；显式 `auto` 可警告后回退 Python，`python` 是参考实现。
- CPU worker 构建一次，GPU 上跨层和 microbatch 复用。纯 TGnomask/aug 只构建 O(BN) 前缀/compose 元信息，不分配 TG 的 tile schedule。
- aug 普通 query 看到此前所有位置（包含重复 closing、padding key）；compose 和 padding query 保持原有树栈/自身语义。与原始 `KProximal_TG_attention_bias(..., is_aug=True)` 对照验证。
- `tgproximal` 等尚无对应紧凑内核的模式、其他架构、额外 masks、dropout、ALiBi、finetuning 保留原后端；KV cache/文档增量路径保留原有有状态 mask。CPU 模型执行使用等价 dense mask 的 SDPA。
- 这是输入管线及 attention 正确性验证，未进行预训练，也未测量本次改动的完整训练吞吐。

## 代码组织

- `layouts.py` 合并 TG/TGnomask 公共解析参考、layout、storage 打包、IPC、pinning、transfer 和 microbatch slice。
- `tg_layout.cpp` 按公共解析、TG schedule、NumPy 导出分工；`_tg_layout_native.py` 管理 CPU 编译缓存。
- `kernels.py` 合并原 `tg_kernels.py` 和 `tgnomask_kernels.py`，共享 prefix/compose、TG 区间及梯度拼接代码；21 个原内核函数/类的 AST 保持一致，见 [核对结果](kernel_ast_parity.json)。
- `tg_attention.py` 统一校验和 autograd 入口；`tgnomask.py` 保留公开导入兼容。
- 数据集和 train/eval collator 共用 `use_typed_tg`，各自使用对应 DataConfig；所有新路径统一传 `tg_layout`，旧 `tgnomask_layout` 参数仍可用。
- 已更新相关基准/调参脚本的当前源码导入路径；历史源码快照和已有测量结果保留。

## 验证回执

环境：3090B / `rtx3090b`，单张 RTX 3090 24576 MiB，驱动 535.230.02；Python 3.10.0、PyTorch 2.7.1+cu126、Triton 3.3.1、pybind11 2.13.6。Slurm `normal` 分区、`normal` QoS、账号 `wangpch`，每作业 1 GPU、8 CPU，申请 24 GiB 内存。

| 作业 | 验证 | 结果 | 作业耗时 |
| --- | --- | --- | --- |
| 4605 | 完整 layout、输入管线、TG/TGnomask/aug、混合头、模型、默认路由与相关配置回归 | 438 passed | 3 分 23 秒 |
| 4607 | N=2048、D=64、FP16/BF16、TGnomask/aug 优化前缀的输出与 Q/K/V 梯度 | 4 passed | 19 秒 |

主要覆盖：原 C++ dense mask 和 label mask 一致性；native/Python 元信息逐字段一致性；截断树、内部 pad、重复 closes、singleton、宽节点；非连续 strides、梯度累积、retained graph；整模型 logits/loss/参数梯度；默认 dataset → CPU worker → pinning → 异步 H2D → GPU 模型前反向；默认路径实际调用专用算子且不构造 GPU dense mask；eval 独立配置、显式关闭和不支持配置回退。

初期作业 4601、4603 在完整套件的 fork worker 中 abort。定位并修复了整理时引入的递归局部函数引用环：它延迟 CUDA tensor 释放，fork 后 CPU worker 的 GC 会碰到遗留 CUDA tensor。重建改为无引用环的方法递归，并增加禁用 cyclic GC 后 storage 立即释放的 4 项回归。修复后的 4605 包含并通过此前失败的完整测试顺序。失败日志作为排查记录保留。

最终测试保持原有数值误差要求；长序列输出/梯度相对 L2 误差上限为 FP16 0.002、BF16 0.012，并检查逐元素误差及有限性。原内核测试的 FP32 阈值和模型梯度阈值未放宽。

证据：[完整日志](verify_4605.out)、[完整 JUnit](verify_4605.xml)、[长序列日志](long_4607.out)、[长序列 JUnit](long_4607.xml)、[测试汇总](test_summary.json)、[源码校验和](source_sha256.json)。执行脚本：[完整](verify.sbatch)、[长序列](long.sbatch)。详细使用方式见 [输入管线文档](../../docs/tg_input_pipeline.md)。
