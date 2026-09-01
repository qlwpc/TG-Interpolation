# Document-level PPL validation and run record (2026-08-23)

> [!NOTE]
> 本文只保留 2026-08-23 已完成且协议仍可解释的 terminal 结果、正确性检查和性能观察。
> 排队状态、失败任务和后来被推翻的 1B 判断已删除。当前完整数值以
> [`../EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md) 为准，
> Pushdown 当前协议以
> [`../docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](../docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md)
> 为准。

## Protocol and data contract

All terminal and pause evaluations use the terminal projection at
`dataset/bbc-news/terminal/{test.npy,test_sent_index.npy,test_doc_index.npy}`.
It contains 148,836 sentence records, 4,966 documents, and 3,284,061 scored
terminal-or-EOS tokens.  Each document starts with BOS; BOS is context only and
every ordinary terminal plus EOS is scored.  Evaluation is single-path and
batch size one so that KV cache is continuous inside a document and resets at a
document boundary.

Native GPST and Pushdown evaluations use
`dataset/bbc-news/testppl/native_model_topk_300_v2`.  The 28 mmap shards have
the same terminal sentences and document boundaries as the terminal corpus.
They contain 148,836 sentences and 4,966 documents.  GPST consumes strict
binary CKY candidates directly; Pushdown consumes the shared n-ary unlabeled
candidates.  `valid_count` rather than physical padded rows defines the
candidate mixture, so short sentences are never treated as having fabricated
300-way support.

## Correctness checks

| Check | Result |
|---|---|
| Focused regression suite (`test_gpst_document_ppl.py`, `test_pushdown_document_ppl.py`) | 6 passed, 1 skipped |
| Native GPST batch tensors vs prior `GoldTreeCollator` | bitwise identical |
| Native Pushdown terminals, document/sentence IDs, spans, targets, and legal actions vs prior composition path | identical |
| Real RTX model score equivalence, 32 candidates (GPST joint; Pushdown attachment, joint, token) | maximum absolute difference `0.0` |

The equivalence result establishes that the native mmap path changes data
loading and batch construction only, not the evaluated likelihood.

## Completed terminal-projection results

Both runs evaluated all 148,836 sentences and completed normally.

| Model | Checkpoint | Slurm job | Document PPL |
|---|---|---:|---:|
| Terminal 100M | `saved_models/Terminal-lr005-bs144/step34115-unsharded` | 3444 | 9.88981 |
| Pause-1 100M | `saved_models/pretrain_pause1_100M/step45487-unsharded` | 3445 | 9.83125 |

## Native evaluator timing observations

| Path / workload | Observation |
|---|---|
| GPST, 8-sentence reference batch | 15.99 s |
| GPST, native fast batch (same scores) | 15.81 s |
| Pushdown, reference batch size 32 | 45.35 s |
| Pushdown, native packed batch size 64 | 32.36 s |
| GPST profiler, batch 64 | Python preparation 0.000197 s; self CPU 178.829 ms; CUDA 18.577 ms |
| Pushdown profiler, batch 32 | Python preparation 0.000001 s; self CPU 51.617 ms; CUDA 10.684 ms |

The GPST improvement is intentionally reported as no material throughput
change: its remaining bottleneck is R2D2's C++ chart computation, not Python
tree parsing/collation.  Pushdown's packed native batch removes repeated
candidate terminal/prefix construction and host-to-device transfers; the
observed end-to-end comparison is about 1.40x faster, though it also uses a
larger safe batch and should not be interpreted as an isolated microbenchmark.
