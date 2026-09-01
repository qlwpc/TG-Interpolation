# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TG-Interpolation is a research codebase for training transformer LMs with **tree-grammar (TG) structured attention biases**. It extends the [OLMo](https://github.com/allenai/OLMo) training framework (AI2) to support multiple grammar-based attention patterns: `terminal`, `tg`, `tree`, `tgproximal`, `tgnomask`, `tgheight`, `tree_shuffle`, and `pause{num}`. The core TG attention bias is computed by a compiled C++ extension (`olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so`); its source and CMake build are under `olmo/data/tgmasking/`.

## Environment

Create the `LLM` Conda environment from `environment.yml`. The file already
pins PyTorch, Triton, FlashAttention and their CUDA 12.6 runtime libraries; do
not install a second PyTorch/CUDA stack over it.

```bash
conda env create -f environment.yml
conda activate LLM
```

The PyTorch wheel includes CUDA runtime libraries but not the CUDA compiler.
Running the existing FlashAttention wheel does not need `nvcc`; rebuilding
FlashAttention requires a compatible external CUDA Toolkit and `CUDA_HOME`.

Build the two CPU-only C++ extensions with the target Conda Python. CMake 3.29.6
is installed through Spack on this host:

```bash
spack load cmake@3.29.6
cmake -S olmo/data/tgmasking -B olmo/data/tgmasking/build -G Ninja \
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build olmo/data/tgmasking/build --parallel 2
cp olmo/data/tgmasking/build/tg_mask*.so olmo/data/

MAX_JOBS=2 python olmo/gpst/cpp_extension/setup.py build_ext --inplace
```

Set `PYTHONPATH` to include the repo root (scripts do this in sbatch files):
```bash
export PYTHONPATH="${PYTHONPATH}:${HOME}/TG-Interpolation"
```

## Key commands

**Generate a new training config YAML:**
```bash
python scripts/init_config.py path/to/output.yaml \
    --run_name=my-run \
    --model.transformer_grammar_type=tree \
    ...
```

**Launch training (single node, 8 GPUs):**
```bash
torchrun --master_port=10826 --nproc-per-node=8 \
    scripts/train.py config.yaml \
    --run_name=my-run \
    --save_folder=/path/to/saved_models/my-run \
    --save_overwrite
```

**Evaluation only** (load a checkpoint and evaluate without training):
```bash
torchrun --nproc-per-node=N scripts/train.py config.yaml \
    --load_path=/path/to/checkpoint \
    --eval_on_load --eval_return --dry_run
```

**SLURM training**: See `run_scripts/` for sbatch templates (e.g., `sbatch_TGnomask_mix_tree_pretrain.sh`).
sbatch should submit with compulsory parameters "-c 1" and "--mems=1M"

## Architecture

### Core library (`olmo/`)

| Module | Purpose |
|---|---|
| `config.py` | All dataclass configs: `TrainConfig`, `ModelConfig`, `DataConfig`, `OptimizerConfig`, `SchedulerConfig`, `TGConfig`, etc. Uses OmegaConf with YAML. |
| `model.py` | `OLMo` - GPT-style transformer with pluggable attention bias. `OLMoBlock` handles attention + FFN. Supports flash-attn, flex-attention, RoPE, ALiBi, and document-level masking. |
| `train.py` | `Trainer` class - the main training loop (forward/backward, checkpointing, eval, W&B logging). Manages FSDP/DDP distributed strategies. |
| `scripts/train.py` | Entry point. Parses YAML config, sets up distributed environment, builds model/optimizer/dataloader, instantiates `Trainer`. |
| `data/__init__.py` | `build_train_dataloader`, `build_eval_dataloader`, `get_TG_generate_bias_func` - factory for TG attention bias objects. |
| `data/memmap_dataset.py` | Memory-mapped dataset from `.npy` token files. |
| `data/iterable_dataset.py` | Sharded iterable wrapper for distributed training. |
| `data/collator.py` | `DataCollator` - pads and assembles batches, applies TG attention bias masks. |
| `eval/__init__.py` | `build_evaluators` - constructs evaluators for LM perplexity, downstream tasks, TG document/sentence perplexity, syntactic generalization, and ROUGE. |
| `eval/downstream.py` | Task definitions for ICL, SG, BLiMP, XSum, etc. |
| `beam_search.py` | Self-contained beam search with word-sync strategies for TG-constrained generation. |
| `transformers_model.py` | Wrapper for HuggingFace models (used when `modelname` is set). |
| `tokenizer.py` | GPT-2 compatible tokenizer with SentencePiece vocabulary. |
| `optim.py` | Optimizer/scheduler builders (LION, AdamW, various LR schedules). |
| `checkpoint.py` | Sharded/unsharded checkpoint save/restore via FSDP/DDP. |

### TG attention bias types (`transformer_grammar_type` in `ModelConfig`)

The grammar type determines how attention is masked. The mapping is in `get_TG_generate_bias_func` (`olmo/data/__init__.py`):

- **`terminal`** - Standard causal attention (no TG masking)
- **`tg`** - Full TG attention bias via `TG_attention_bias` (C extension)
- **`tgproximal`** / **`tgnomask`** - Proximal TG bias via `KProximal_TG_attention_bias`
- **`tgtree`** / **`tree`** - Causal mask only (used with tree-structured token sequences)
- **`pause{num}`** - Interleaved pause tokens for filler-based TG
- **`mixing`** - Mixed head types via `HeadMixingBias` (different attention patterns per attention head)

### Data flow

1. Raw data is preprocessed by scripts in `datatools/` into `.npy` memmap files of token IDs
2. `MemMapDataset` loads token sequences from `.npy` files with chunking
3. `DataCollator` pads batches, generates document lengths, and computes TG attention biases
4. During training, `Trainer.train_batch()` splits batches into micro-batches, runs forward/backward with gradient accumulation
5. The TG bias is passed as `attention_bias` in the batch dict and injected into each `OLMoBlock.forward()`

### Configuration hierarchy

`TrainConfig` is the root config containing all sub-configs: `ModelConfig`, `DataConfig`, `OptimizerConfig`, `SchedulerConfig`, `TokenizerConfig`, evaluators list, etc. Configs are YAML files loaded by OmegaConf. Training YAML configs live in `train_configs/`.

### Evaluation pipeline

`Trainer.eval()` dispatches to specialized eval step methods based on `EvaluatorType`:
- `EvaluatorType.lm` → `eval_step()` (standard LM loss)
- `EvaluatorType.tg_doc` → `TG_doc_eval_step()` (document-level TG perplexity with KV cache management)
- `EvaluatorType.rouge` → `summarization_eval_step()` (beam search generation + ROUGE)
- Syntactic generalization → `SG_eval_step()` (word-sync beam search with TG constraints)

### Beam search with TG

`OLMo.word_sync_beam_search()` performs word-level synchronous beam search that calls `generate_TG_bias` at each generation step to maintain TG-consistent attention patterns. This is used for SG evaluation and summarization with TG models.

## Saved models and checkpoints

- Checkpoints saved under `saved_models/{run_name}/step{N}` (sharded) or `step{N}-unsharded`
- `save_interval`, `save_interval_unsharded`, `save_interval_ephemeral` control checkpoint frequency
- Load with `--load_path=...` or `--try_load_latest_save` to auto-resume from latest checkpoint
- `run_folder/` contains experiment output; `train_logs/` has training logs
