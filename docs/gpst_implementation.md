# GPST implementation record

Status: implemented. The original implementation campaign completed on 2026-07-24; this
document records the durable architecture and verification boundaries, not an active plan.

## Scope

`olmo/gpst/` is a self-contained implementation of Generative Pretrained Structured
Transformers (Hu et al., ACL 2024). It supports both unsupervised hard-EM training with
parser-induced trees and supervised training with gold constituency trees. It reuses the
repository's tokenizers and parsed BBC data without changing the main OLMo training stack.

The model combines:

- a pruned inside-outside composition model backed by a CPU C++ chart manager;
- a parser that supplies merge orders in unsupervised mode;
- type and token Transformer stacks that consume post-order surrogate representations;
- a two-pass training loop that controls whether the autoregressive objective propagates
  through split weights.

## Training modes

Both modes share the model and trainer. Their only structural difference is the source of
the merge order passed to the chart manager:

- **Unsupervised:** `TransformerParser` predicts split scores and induces merge orders.
- **Supervised:** `GoldTreeDataset` converts the repository's constituency trees into gold
  merge orders; the parser is trained to predict them but does not choose the training tree.

The implementation retains the GPT-2-compatible backbone and also supports the optional
OLMo block stack in `olmo/gpst/model/olmo_stack.py`. The OLMo-backed path uses learned
position embeddings so that externally supplied, tree-ordered position IDs preserve the
semantics expected by GPST.

## Source map

| Path | Responsibility |
| --- | --- |
| `olmo/gpst/cpp_extension/` | pruned-CKY chart backend and Python binding |
| `olmo/gpst/data_structure/` | tensor cache and chart-manager wrapper |
| `olmo/gpst/model/` | parser, inside-outside encoder, generative model, and backbones |
| `olmo/gpst/reader/` | lazy, text, and gold-tree datasets plus collators |
| `olmo/gpst/trainer/` | local and distributed hard-EM training loops |
| `olmo/gpst/eval/` | GPST-native evaluation code |
| `scripts/gpst/` | preprocessing, training, diagnosis, and evaluation entry points |
| `tests/test_gpst_*.py` | C++ backend, model, reader, trainer, launcher, and evaluation tests |

## Important implementation adaptations

- The C++ chart backend uses cloned tensors for `torch::from_blob` views, preventing
  dangling CPU views after the chart manager is destroyed.
- The local GPT-2 wrapper relies on PyTorch SDPA instead of a vendored Transformers copy.
- The Llama-style parser Transformer uses plain SDPA so CPU tests retain a valid math
  backend.
- `WeightedSumFunc.a_ij_require_grad` is toggled by the trainer to preserve the intended
  two-backward-pass gradient boundary.

## Verification

Build the CPU extension before running the complete GPST test group:

```bash
python olmo/gpst/cpp_extension/setup.py build_ext --inplace
PYTHONPATH=. python -m pytest -q tests/test_gpst_*.py
```

The original campaign verified CPU forward/backward execution for both training modes and
end-to-end smoke training. Treat later evaluation protocols and results as separate records;
the current entry points are listed in [the documentation index](README.md).
