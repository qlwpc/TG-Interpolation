# GPST Local Implementation Plan

## Goal
Implement the GPST training algorithm (Hu et al., ACL 2024) locally in this repo, in **both** a supervised (gold-tree) and an unsupervised (hard-EM, parser-induced) variant. Faithful to the paper + reference repo `ant-research/StructuredLM_RTDT`, adapted to this repo's conventions and existing data (BBC tree-stream corpus + GPT-2 tokenizer).

This is a **self-contained new module** under `olmo/gpst/` — it does **not** modify the existing OLMo training stack (`olmo/model.py`, `olmo/train.py`, TG attention bias). It reuses the repo's tokenizer and parse data only.

## Algorithm recap (from paper + ref repo)

Two models trained jointly:
1. **Composition model** — pruned deep inside-outside encoder:
   - `TransformerParser` (top-down): scores each split point; merge order = reverse of descending scores (unsupervised) OR derived from a gold tree (supervised).
   - Pruned CKY chart (`window_size=2`) built by a C++ `TableManager`. Inside pass composes `i_ij = Σ_k â_k * f_α(i_ik, i_kj)`; split score `a_ij = softmax(φ_α)`. Outside pass computes `o_ij` top-down.
   - Auto-encoding loss `L_ae = -log p(token | outside repr)`.
2. **Generative model** — type layers (shallow, predict COMP/GEN action) + token layers (deep, predict next token). Input = internal span reprs `i_ij` as **surrogates**, gathered in post-order.
   - `L_ar = 0.5*gpt_loss + action_loss + parser_loss`.
   - **Gradient stop**: stop grad from `L_ar` → `a_ij` (softmax split weights) to kill left-branching bias. Implemented via a custom `WeightedSumFunc` autograd with a module-level flag toggled by the trainer.
   - `L = L_ae* + L_ar` where `L_ae* = L_ae + L_h (height penalty) + L_p (parser NLL)`.

Hard-EM training: E-step induces tree + reprs (no grad to parser params for struct loss path; grad flows for the ae-path), M-step updates all params on `L_ae* + L_ar`.

## Architecture decision: port the C++ extension

The pruned CKY chart + `prepare_generation` (producing post-order surrogate indices, position ids, action targets, split targets) is **~1100 lines of intricate C++** with subtle pointer-linked-list pruning logic. Re-implementing in pure Python would be slow and error-prone, and the algorithm's correctness depends on exact index bookkeeping.

**Decision:** Port the C++ extension (`cpp_extension/{py_backend.cpp,py_backend.h,binding.cpp}`) and `data_structure/tensor_cache.py` + `data_structure/py_backend.py` verbatim, build with `torch.utils.cpp_extension.CppExtension` (CPU-only — no CUDA ops, all GPU work is in PyTorch). This matches `setup.py` in the ref repo. Verified locally: `g++` present, `torch.utils.cpp_extension` imports OK.

The Python model code (`model/*.py`) ports with minimal changes (replace `from model.X` → `from olmo.gpst.model.X`, fix the HF-GPT2 import to use this repo's tokenizer path).

## File layout (new, self-contained)

```
olmo/gpst/
  __init__.py
  config.py              # GPSTConfig dataclass (r2d2 + gpt2 sub-configs), OmegaConf
  cpp_extension/
    py_backend.cpp        # ported verbatim
    py_backend.h          # ported verbatim
    binding.cpp           # ported verbatim
    setup.py              # build_ext --inplace → produces cppbackend.so
  data_structure/
    __init__.py
    tensor_cache.py       # ported verbatim
    py_backend.py         # ported verbatim (CPPChartTableManager wrapper)
  model/
    __init__.py
    r2d2_base.py          # ported
    r2d2_common.py        # ported
    r2d2_insideoutside.py # ported (InsideOutsideModule)
    fast_parser.py        # ported (TransformerParser)
    topdown_parser.py     # ported (BasicParser base)
    tree_encoder.py       # ported (InsideEncoder/OutsideEncoder/TreeEncoderLayer)
    weighted_sum_func.py  # ported (grad-stop autograd)
    gpt2_flash_attn.py    # ported (GPT2Model/GPT2LMHeadModel; SDPA fallback if flash-attn missing)
    generative_r2d2_fast.py # ported (FastGenerativeR2D2)
    model_factory.py      # ported
    modeling_outputs.py   # ported (R2D2GenOutput dataclass)
  reader/
    __init__.py
    lazy_loader.py        # ported (memmap corpus reader)
    dataset.py            # ported (GPT2Dataset) + new GoldTreeDataset
    data_collator.py      # ported (DefaultCollator) + gold-tree collator
  trainer/
    __init__.py
    ddp_trainer.py        # ported trainer (the hard-EM loop with WeightedSumFunc flag toggling)
  data/
    en_config/r2d2_256_4_1.json   # ported config
    gpt2-small/config.json        # ported config
scripts/gpst/
  preprocess_corpus.py    # build .lazy memmap from BBC terminal stream
  gold_tree_to_merge_orders.py  # NEW: convert repo's tree-stream → merge_orders for supervised
  pretrain_gpst_small_supervised.sh
  pretrain_gpst_small_unsupervised.sh
tests/test_gpst_*.py      # TDD tests (see below)
```

## Supervised vs Unsupervised — the single switch

Both share the **entire** model + trainer. The only difference is the **source of merge orders** fed to `CPPChartTableManager`:

- **Unsupervised**: `TransformerParser` predicts split scores each step → `BasicParser.parse()` returns `split_indices` (merge orders) → fed to `TableManager`. Parser params trained by `L_p` (parser NLL on induced tree) + the E-step induces the tree. This is the paper's main model.

- **Supervised (gold-tree)**: merge orders come **directly from a gold constituency parse** (the repo's `dataset/bbc-news/tree/*.npy`), converted to the same `merge_orders` array format via `gold_tree_to_merge_orders.py` (a binarized gold tree → left-to-right merge sequence). The `TransformerParser` is **still present** but trained only to *predict* the gold split order (`L_p` with gold targets), and is not used to induce the tree at train time. This is the "gold trees" baseline referenced in §4.2.3. The composition + generative models then train on the gold tree exactly as in the M-step.

This switch is a single config flag `gpst.supervised: bool` + a different dataset/collator that supplies `merge_orders` in the batch. The `InsideOutsideModule.forward` already accepts external `split_indices` — in supervised mode we pass the gold-derived ones and skip the parser's `parse()` call for induction (but keep `parser_loss` against gold `split_points`).

## Implementation phases (TDD)

### Phase 0 — Build the C++ extension (foundation)
- Port `cpp_extension/*` + `setup.py` into `olmo/gpst/cpp_extension/`.
- `python olmo/gpst/cpp_extension/setup.py build_ext --inplace` → `cppbackend.cpython-*.so`.
- **Test (RED→GREEN):** `tests/test_gpst_cppbackend.py` — construct a `TableManager` on a 2-sentence batch with known merge orders, call `construct_inside_groups` + `prepare_generation`, assert the returned `ldr_cache_ids`/`tgt_ids`/`split_targets` match a hand-computed post-order traversal for a 3-token sentence `A B C` merged as `((A B) C)`.

### Phase 1 — Composition model (port + unit test)
- Port `tensor_cache.py`, `py_backend.py`, `r2d2_base.py`, `r2d2_common.py`, `tree_encoder.py`, `fast_parser.py`, `topdown_parser.py`, `r2d2_insideoutside.py`.
- **Test:** `tests/test_gpst_inside_outside.py` — random init, 1 sentence of 5 tokens, run `InsideOutsideModule.forward` on CPU, assert shapes of `ldr_repr`, `ctx.scores`, `split_targets`; assert `L_ae` is finite and `parser_loss` is finite.

### Phase 2 — Generative model + factory (port + unit test)
- Port `gpt2_flash_attn.py` (SDPA fallback if flash-attn import fails), `modeling_outputs.py`, `generative_r2d2_fast.py`, `model_factory.py`.
- **Test:** `tests/test_gpst_generative.py` — build `r2d2-gen-fast` model on CPU, forward a tiny batch, assert `struct_loss`/`non_struct_loss` finite, `action_logits` shape `(B, L, 2)`.

### Phase 3 — Reader (unsupervised corpus + supervised gold-tree)
- Port `lazy_loader.py`, `dataset.py`, `data_collator.py`.
- **NEW `GoldTreeDataset`**: reads `dataset/bbc-news/tree/*.npy`, uses `olmo.data.parse_align.parse_tree_block` + `binarize_tree` + `tree_spans` to produce per-sentence terminal ids + gold merge orders (post-order binary-merge sequence). Reuses existing tested `parse_align` utils — no new parsing logic.
- **NEW collator**: like `generative_r2d2_collate_fn` but injects `merge_orders` from the gold tree (supervised) — for unsupervised, `merge_orders` is None and the parser fills it.
- **Test:** `tests/test_gpst_reader.py` — assert `GoldTreeDataset` for a hand-coded tree-stream block yields terminal ids matching `terminal/*.npy` and a merge-order sequence that reconstructs the binarized gold tree.

### Phase 4 — Trainer (the hard-EM loop)
- Port `ddp_trainer_nosync.py` → `ddp_trainer.py`. Key: the two-backward-pass pattern with `WeightedSumFunc.a_ij_require_grad` toggling (struct loss with grad→a_ij True; non-struct with False). Keep DDP `no_sync` + manual all_reduce (verify torch 2.7 compatibility).
- Adapt to torch 2.7.1: replace legacy `torch.cuda.amp.autocast`/`GradScaler` with `torch.amp.autocast('cuda')`/`torch.amp.GradScaler('cuda')`.
- **Test:** `tests/test_gpst_trainer.py` — 2-step train on CPU (tiny model), assert loss decreases and no NaN; assert `WeightedSumFunc.a_ij_require_grad` toggling works (grad on `a_ij` is None after non-struct backward).

### Phase 5 — End-to-end smoke + scripts
- `scripts/gpst/pretrain_gpst_small_unsupervised.sh` and `..._supervised.sh` (torchrun, 1-8 GPU, BBC corpus).
- Integration test: `tests/test_gpst_e2e_cpu.py` — full forward+backward on CPU, 1 batch, both modes.

## Configs
- `olmo/gpst/data/en_config/r2d2_256_4_1.json` (small, ported verbatim).
- `olmo/gpst/data/gpt2-small/config.json` (ported; `action_layer_num: 3`).
- A top-level `train_configs/gpst_small.yaml` wiring `GPSTConfig` for OmegaConf.

## Verification / acceptance
1. C++ extension builds and `test_gpst_cppbackend` passes.
2. Both supervised + unsupervised run a forward+backward on CPU without error (unit tests green).
3. A 50-step smoke pretrain on BBC (1 GPU if available, else CPU-tiny) produces decreasing total loss and logs a parsed tree.
4. Coverage ≥ 80% on new `olmo/gpst/` modules (per repo testing rules).

## Decisions (confirmed with user)
- **C++ TableManager + TensorCache + WeightedSumFunc autograd**: verbatim port (recommended path).
- **Flash-attn**: optional. Findings from reading the ref repo: both "flash_attn" files (`model/gpt2_flash_attn.py`, `model/Llama_flash_attn.py`) actually use `F.scaled_dot_product_attention` (SDPA), **not** the `flash_attn` package. So no `flash_attn` dependency exists — the "flash-attn if available else SDPA" requirement is satisfied naturally by SDPA, which on CUDA auto-dispatches to the flash/efficient kernel. No config flag needed; keep the file names but document that SDPA is the path.
- **Scope**: full implementation, all phases (0–5) in this session.

## Findings during planning (inform the port)
- **`fast_parser.py` has a latent bug**: it references `ModelArgs` and `Transformer` (defined in `model/Llama_flash_attn.py`) without importing them — NameError on first use. Fix during port: add `from olmo.gpst.model.llama_transformer import ModelArgs, Transformer` (port the Llama-style RoPE+SDPA transformer into `llama_transformer.py`, dropping the `with sdp_kernel()` context that disables math backend on CPU — that context forces flash/mem-efficient backends which don't exist on CPU, breaking CPU tests; replace with a plain SDPA call).
- **`gpt2_flash_attn.py`**: HF GPT-2 with SDPA attention (`_attn` already uses `F.scaled_dot_product_attention`). Port verbatim; it works on CPU. The only deps are `transformers` (present, 4.57.3) — but the file uses `transformers.pytorch_utils.Conv1D` and `SequenceSummary` which still exist in 4.57. Verify import in Phase 2 test.
- **torch 2.7 API drift**: `torch.cuda.amp.autocast`/`GradScaler` deprecated → `torch.amp.autocast('cuda')`/`torch.amp.GradScaler('cuda')`. `DDP.no_sync` unchanged. Apply in Phase 4.
- **C++ build**: `CppExtension` (not CUDA) → `g++` only, no nvcc. One-shot build.
- **Supervised merge-order semantics**: `TableManager` expects `merge_orders` as `(N, L-1)` per-sentence split-position sequences (the order in which adjacent cells merge, left-to-right index of the merged gap). For gold trees: use `olmo.data.parse_align.binarize_tree` → `tree_spans` → derive the post-order merge sequence. Validate against a hand case (`((A B) C)` → merge order `[0, 1]`) in Phase 3 test.
- **Reusing repo data**: `dataset/bbc-news/tree/{train,dev,test}.npy` (uint16 tree-stream, BOS=50257, NT brackets 50258–50316, leaves=GPT-2 ids) + `olmo/data/parse_align.py` (already tested: `parse_tree_block`, `binarize_tree`, `tree_spans`) feed the supervised reader. Unsupervised reader uses `dataset/bbc-news/terminal/*.npy` (plain token streams).

## Status: COMPLETE (all phases, 2026-07-24)

All 5 phases done; 16 tests pass (`tests/test_gpst_*.py`), CPU + both modes.
- Phase 0: C++ `cppbackend` builds (`python olmo/gpst/cpp_extension/setup.py build_ext --inplace`); test asserts post-order gen indices.
- Phase 1: composition model (inside-outside) forward+backward on CPU.
- Phase 2: full generative model (FastGenerativeR2D2, 145M params) forward+backward + hard-EM two-backward with grad-stop.
- Phase 3: GoldTreeDataset converts repo's BBC tree-stream → gold merge orders (reusing olmo.data.parse_align); verified on real corpus.
- Phase 4: compact hard-EM trainer (`olmo/gpst/trainer/trainer.py`), torch 2.7 AMP, both modes.
- Phase 5: launcher `scripts/gpst/run_gpst.py` + 2 sbatch scripts + e2e tests (both modes, 3 steps, produces model.bin).

Key port adaptations (deviations from verbatim ref):
- `gpt2_flash_attn.py`: replaced the 1200-line vendored HF GPT-2 (broken on transformers 4.57 — `SequenceSummary` removed) with a thin HF-GPT2Model wrapper using SDPA. Satisfies "flash-attn if available else SDPA" via PyTorch's SDPA auto-dispatch.
- `llama_transformer.py`: fixed latent `ModelArgs`/`Transformer` NameError in fast_parser; dropped `with sdp_kernel():` context (forced flash/mem-efficient backends, breaks CPU).
- `r2d2_insideoutside.py` + `py_backend.py`: added `.clone()` on all `torch::from_blob` view tensors — on GPU `.to('cuda')` copies (hides the dangling-view bug), on CPU `.to('cpu')` is a no-op so the views would dangle after CPPChartTableManager destruction. The clone fixes make it work on both.
- Supervised mode: new `merge_orders` kwarg on `InsideOutsideModule.forward`/`FastGenerativeR2D2.forward`; `GoldTreeDataset`/`GoldTreeCollator` supply gold merge orders; parser still trained (supervised NLL) but does not induce the tree.

Note: pytest-cov / `coverage` tooling conflicts with torch.distributed Meta-kernel registration in this env (collection-time RuntimeError) — coverage could not be measured, but 16 functional tests pass.
