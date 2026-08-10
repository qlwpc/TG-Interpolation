"""GPST — Generative Pretrained Structured Transformers (Hu et al., ACL 2024).

Self-contained reimplementation of the paper's training algorithm, supporting
both an unsupervised (hard-EM, parser-induced tree) and a supervised
(gold-tree) variant. See docs/PLAN_gpst.md for the design.

Subpackages:
- ``cpp_extension``  — compiled C++ pruned-CKY chart backend (``cppbackend``).
- ``data_structure`` — TensorCache + CPPChartTableManager wrapper.
- ``model``          — composition (inside-outside) + generative models.
- ``reader``         — datasets + collators.
- ``trainer``        — the hard-EM training loop.
"""
