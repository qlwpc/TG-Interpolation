"""Pretraining corpus construction (raw text → parse → tokenize → variants).

Use ``python -m datatools.parse_pretrain_data.build_pretrain_data plan`` as the
stable entry point.  The older component scripts remain available for resuming
individual historical jobs, while the stage runner supplies canonical paths,
task ordering, stream assembly, validation, and provenance manifests.
"""
