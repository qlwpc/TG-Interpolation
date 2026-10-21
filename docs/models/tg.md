# TG / TGnomask(aug) / mix-TG Input Pipeline

The layout for `tg_typed_attention` is built inside the `DataCollator`. When `num_workers > 0`, token parsing, construction of the lifetime/CSR/tile schedule, and tensor packing all execute in the DataLoader CPU workers; the layout is built once per local batch, and after being transferred to the GPU it is shared by all layers and microbatches. This path applies to `tg`, `tgnomask`, `tgnomaskaug`, and to `mixing` composed of `tg/tgnomask/tgnomaskaug/tgtree`. All layouts follow the visibility and label mask semantics of the original C++ dense mask.

## Configuration

Layer the following fields on top of the data, model, and global batch configuration that has already been confirmed. New pretraining must still execute the [pretraining workflow](../guides/pretraining.md). These fields by themselves are not a complete training configuration.

```yaml
model:
  tg_typed_attention: null  # Default: automatically enable supported configurations; true strictly requires enabling; false disables
  attention_dropout: 0.0
  # transformer_grammar_type: tg / tgnomask / tgnomaskaug / mixing

data:
  generate_attention_mask: false
  generate_doc_lengths: false
  tg_layout_backend: native  # Default; requires a successful C++ build
  num_workers: 1
  persistent_workers: true
  prefetch_factor: 2
  pin_memory: true
  cuda_prefetch: false
```

`tg_typed_attention: null` (the default) enables the above grammars when they use fresh-segment sequential MHA, head dimensions 16–128, and no attention dropout/ALiBi/extra attention mask or document mask. The finite-window `tgproximal`, other architectures, and finetuning retain the original path; an explicit `true` raises an error on an unsupported configuration. Training and the MemMap validation set use the same routing function and each read their own DataConfig. Document/incremental evaluation with KV cache still uses the original stateful mask generator.

Pure `tgtree` continues to use the original causal backend. When the model runs on CPU, an equivalent mask is reconstructed via SDPA; on CUDA, metadata and Triton kernels are used directly. An explicit `model.tg_typed_attention: false` returns to the original dense mask / SDPA / Flex path. When calling the model standalone, pass the layout built on CPU and transferred to the corresponding device via `tg_layout=`; the legacy `tgnomask_layout=` parameter is retained as a compatibility alias.

Each DDP rank starts its own workers; the CPU budget should cover all ranks, workers, the main training thread, and pin-memory threads. Tune worker count on the target machine rather than assigning the entire host CPU count to each rank. The original local B4/B16 measurements suggested 1 worker per rank with prefetch factor 2, with 2 workers as a candidate when CPU supply is slower. This is a tuning starting point, not a measurement on the reader's hardware. `num_workers=0` still works; if `cuda_prefetch` is also enabled, the background prefetch thread also performs CPU data fetching/construction.

The dedicated CUDA prefetch thread is currently disabled by default; the implementation retains `cuda_prefetch: true` for comparison with other transfer loads. With `pin_memory: true`, the Trainer's direct transfer also uses `non_blocking=True`.

## Pretraining Launch Conditions

For the registered 500M TG/TGNomask/Mix recipes, `tg_typed_attention=true`, Flash/Flex both false,
`generate_attention_mask` / `generate_doc_lengths` are false for train and both evals,
and Mix explicitly uses 8 TG + 8 TGNomask heads. You cannot change only the training loader while keeping the eval default doc lengths.
Tree-NoONT is a separate causal model; TGNomask and aug semantics cannot be interchanged.
See [TG routing in the pretraining workflow](../guides/pretraining.md#tg-routing) for the complete configuration and condition table.

Worker transfer of layouts depends on real multi-process tensor IPC. Single-process collator/kernel tests do not verify
Unix socket path length or IPC. Before starting Python, set a
short, dedicated TMPDIR, first run [IPC precheck](../../scripts/pretraining/check_ipc.py), then do a training smoke with real data,
the same world size/microbatch/autocast. The native builder cache follows TMPDIR;
if setup and training are on different nodes/temp directories, training still requires a compiler and cannot assume the setup cache is automatically visible.


## Construction and Transfer

- `tg_layout_backend: native` is the default and requires a successful native C++ CPU build. An explicit `auto` allows warning and falling back to Python when compilation is unavailable; `python` is the semantic reference and ablation entry point. All three public builders support these settings.
- The CPU extension depends only on pybind11, Python headers, and a C++17 compiler; it does not depend on CUDA/Torch C++ ABI. It is preloaded at training collator initialization to avoid counting the first compilation in worker steady-state time. The cache is located at `olmo-tg-<uid>/<source+ABI hash>` in the temp directory and uses a file lock to protect concurrent compilation.
- The 23 tensors of each TG layout and the 15 tensors of each TGnomask/aug layout are packed into two storages, int32 and bool, respectively. Worker IPC, pinning, and H2D are performed per storage; microbatches retain field views. Mixed heads share TG and non-aug prefix metadata; when aug heads are present, the aug layout is built once more and is not re-parsed per layer or head.
- The DataLoader pin-memory thread provides the pinned batch. An additional bounded thread submits H2D to a dedicated CUDA stream in original order, queuing at most two GPU batches; the consumer stream waits on events and uses `record_stream` to protect allocator lifetimes. The CPU source is retained until the copy completes; early exit, exceptions, and epoch switches all shut down the thread.
- `IterableDataset`'s epoch, start_index, and max_examples are placed in shared CPU state so that persistent workers can see the resume position and new epoch. Training must shut down the previous round of prefetch before modifying these states; the Trainer is already wired in this order.
- With `OLMO_STEP_PROFILE=1`, `data_ms` now includes the actual `next(batch)` wait. After enabling prefetch, asynchronous H2D can overlap, so this metric cannot be interpreted as the complete H2D duration.

CPU construction and GPU computation overlap via multi-process workers. Direct transfer occurs on the consumer stream; independent-stream prefetch requires explicitly enabling `cuda_prefetch`.

PyTorch officially documents [DataLoader's multiprocessing, pinning, and persistent workers parameters](https://docs.pytorch.org/docs/main/data.html), as well as [the conditions for pinned memory / non_blocking transfer](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html). Enabling only `non_blocking=True` does not guarantee overlap of computation and transfer; this implementation also uses an independent stream, events, and bounded prefetch.

## Code Organization

| File | Responsibility |
| --- | --- |
| `olmo/attention_kernels/layouts.py` | Public layout, packing/transfer/IPC, unified Python reference parsing and builder |
| `olmo/attention_kernels/tg_layout.cpp` | Public C++ parsing, on-demand construction of TG lifetime/tile schedule, NumPy export |
| `olmo/attention_kernels/_tg_layout_native.py` | Lazy compilation and concurrent cache for the CPU extension |
| `olmo/attention_kernels/tg_attention.py` | Validation, public attention entry point, mixed autograd |
| `olmo/attention_kernels/kernels.py` | Merged prefix, compose, TG interval, and gradient concatenation kernels |
| `olmo/attention_kernels/tgnomask.py` | Compatibility forwarding for the legacy public import path |

It is recommended to import the public API from `olmo.attention_kernels`. The legacy public import paths `tg_attention` / `tgnomask` remain available; the internal `tg_kernels` / `tgnomask_kernels` have been merged into `kernels`.

For normal queries in TGnomask, the prefix excludes duplicate closing and pad keys; for aug, the normal query prefix includes all previous positions. For both, the compose query still reads only the popped tree-stack nodes and itself, and the padding query reads only itself. TGnomask/aug do not build TG's tile schedule, and metadata is stored as O(BN).

## Validation and Boundaries

Standalone layout/attention benchmarks cannot replace full training throughput; they should cover the real MemMapDataset, all
model layers, optimizer, and DDP, and distinguish initialization from steady state. When the tree degenerates, the tile union schedule may still
be quadratic in scale; workers/prefetch increase in-flight memory. The CPU builder and compact storage do not change this complexity.
