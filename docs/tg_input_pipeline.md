# TG / TGnomask(aug) / mix-TG 输入管线

`tg_typed_attention` 的 layout 在 `DataCollator` 内构建。`num_workers > 0` 时，解析 token、构建 lifetime/CSR/tile schedule、打包 tensor 都在 DataLoader CPU worker 中执行；每个 local batch 构建一次，传到 GPU 后供所有层和 microbatch 共用。此路径适用于 `tg`、`tgnomask`、`tgnomaskaug`，以及由 `tg/tgnomask/tgnomaskaug/tgtree` 构成的 `mixing`。所有布局都沿用原始 C++ dense mask 的可见性和 label mask 语义。

## 配置

在已经确认的数据、模型和 global batch 配置上叠加以下字段。新预训练仍须执行 [预训练工作流](pretraining_workflow.md)。这些字段本身不是完整训练配置。

```yaml
model:
  tg_typed_attention: null  # 默认自动启用支持的配置；true 严格要求启用；false 关闭
  attention_dropout: 0.0
  # transformer_grammar_type: tg / tgnomask / tgnomaskaug / mixing

data:
  generate_attention_mask: false
  generate_doc_lengths: false
  tg_layout_backend: native  # 默认，要求 C++ 构建成功
  num_workers: 1
  persistent_workers: true
  prefetch_factor: 2
  pin_memory: true
  cuda_prefetch: false
```

`tg_typed_attention: null`（默认）在 fresh-segment sequential MHA、head dimension 16–128、无 attention dropout/ALiBi/额外 attention mask 或 document mask 的上述语法中启用。有限窗口 `tgproximal`、其他架构及 finetuning 保留原路径；显式 `true` 遇到不支持的配置会报错。训练与 MemMap 验证集使用同一个路由函数，并分别读取自己的 DataConfig。带 KV cache 的文档/增量评估仍使用原有有状态 mask 生成器。

纯 `tgtree` 继续使用原 causal 后端。CPU 上执行模型时通过 SDPA 重建等价 mask；CUDA 上直接使用元信息和 Triton 内核。显式 `model.tg_typed_attention: false` 可回到原始 dense mask / SDPA / Flex 路径。独立调用模型时通过 `tg_layout=` 传入 CPU 构建并转移到对应设备的 layout；旧的 `tgnomask_layout=` 参数保留为兼容别名。

每个 DDP rank 启动自己的 workers；CPU 预算应覆盖全部 ranks、workers、主训练线程及 pin-memory 线程。worker 数应通过目标机器实测调整，不能按整台主机 CPU 数给每个 rank 配置。本机 B4/B16 实测推荐每 rank 1 worker、prefetch factor 2；CPU 供给变慢时再测 2 workers。`num_workers=0` 仍可运行；若同时开启 `cuda_prefetch`，后台预取线程也执行 CPU 取数/构建。

实测结果见 [报告](../validation/tg_input_pipeline_20260915/REPORT.md)。在这组小型紧凑 layout 上，专用 CUDA 预取线程没有稳定收益，推荐关闭；实现保留 `cuda_prefetch: true`，供其他传输负载比较。`pin_memory: true` 时 Trainer 的直接传输也使用 `non_blocking=True`。

## 构建和传输

- `tg_layout_backend: native` 是默认值，要求原生 C++ CPU 构建成功。显式 `auto` 允许编译不可用时警告并回退 Python；`python` 是语义参考及消融入口。三个公开 builder 均支持这些设置。
- CPU 扩展只依赖 pybind11、Python headers、C++17 编译器，不依赖 CUDA/Torch C++ ABI。训练 collator 初始化时预加载，避免把首次编译算进 worker 的稳态耗时。缓存位于临时目录中的 `olmo-tg-<uid>/<source+ABI hash>`，并以文件锁保护并发编译。
- 每个 TG layout 的 23 个 tensor、每个 TGnomask/aug layout 的 15 个 tensor，分别打包到 int32 和 bool 两块 storage。worker IPC、pinning、H2D 按 storage 执行，microbatch 保留字段 view。混合头共享 TG 和非 aug 前缀元信息；包含 aug 头时另构建一次 aug layout，不按层或头重复解析。
- DataLoader pin-memory 线程提供 pinned batch。额外的有界线程按原顺序提交 H2D 到专用 CUDA stream，最多排队两个 GPU batch；消费 stream 等待 event，并 `record_stream` 保护 allocator 生命周期。CPU 源保持到拷贝结束；提前退出、异常和 epoch 切换都会关闭线程。
- `IterableDataset` 的 epoch、start_index、max_examples 放在共享 CPU 状态中，使 persistent workers 能看到恢复位置和新 epoch。训练必须先关闭上一轮预取，再修改这些状态，Trainer 已按此顺序接入。
- `OLMO_STEP_PROFILE=1` 的 `data_ms` 现在包含实际 `next(batch)` 等待。启用预取后异步 H2D 可以重叠，此指标不能解释为完整 H2D 耗时。

CPU 构建与 GPU 计算通过多进程 workers 重叠。直接传输发生在消费 stream，独立 stream 预取需要显式启用 `cuda_prefetch`。

PyTorch 官方说明了 [DataLoader 的多进程、pinning 和 persistent workers 参数](https://docs.pytorch.org/docs/main/data.html)，以及 [pinned memory / non_blocking 传输的条件](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。只开启 `non_blocking=True` 并不保证计算与传输重叠；本实现还使用独立 stream、event 和有界预取。

## 代码组织

| 文件 | 职责 |
| --- | --- |
| `olmo/attention_kernels/layouts.py` | 公共 layout、打包/传输/IPC、统一 Python 参考解析与 builder |
| `olmo/attention_kernels/tg_layout.cpp` | 公共 C++ 解析、按需构建 TG lifetime/tile schedule、NumPy 导出 |
| `olmo/attention_kernels/_tg_layout_native.py` | CPU 扩展的惰性编译和并发缓存 |
| `olmo/attention_kernels/tg_attention.py` | 校验、公开 attention 入口、mixed autograd |
| `olmo/attention_kernels/kernels.py` | 合并后的 prefix、compose、TG interval 和梯度拼接内核 |
| `olmo/attention_kernels/tgnomask.py` | 旧公开导入路径的兼容转发 |

推荐从 `olmo.attention_kernels` 导入公开 API。旧 `tg_attention` / `tgnomask` 公开导入路径保持可用；内部 `tg_kernels` / `tgnomask_kernels` 已合并为 `kernels`。

TGnomask 普通 query 的前缀排除重复 closing 和 pad key；aug 的普通 query 前缀包括此前所有位置。两者的 compose query 仍只读被弹出的树栈节点和自身，padding query 只读自身。TGnomask/aug 不构建 TG 的 tile schedule，元信息存储为 O(BN)。

## 验证与边界

性能协议、作业日志、源码快照和测试结果保存在 [本轮验证目录](../validation/tg_input_pipeline_20260915/)。基准包含真实 MemMapDataset 取数和 12 次 attention 前反向，分别测试 local batch 4、16（microbatch 4），不包含投影、MLP、optimizer、DDP 通信，不能作为完整训练吞吐结论。首批启动与稳态分开计时。

树退化时 tile union schedule 仍可能为二次规模，worker/prefetch 会增加相应的在途内存。此次优化不改变其表示复杂度。`cuda_prefetch` 默认关闭以保持已有配置行为；原生 CPU builder 和紧凑 storage 随 typed TG 路径启用。

默认路由与本次整理的测试回执见 [验证记录](../validation/tg_kernel_defaults_20260915/REPORT.md)。
