# 数据工具入口

| 工作 | 入口 |
|---|---|
| 新 BBC dedup + clean 预训练配置 | [配置生成器](../scripts/prepare_bbc_dedup_500m.py)；训练数据用 dedup train，评估数据用 clean dev/test |
| BBC / FineWeb-Edu 下载、解析、组装 | `python -m datatools.parse_pretrain_data.build_pretrain_data`；[数据构建细则](../docs/pretraining_reproduction.md) |
| clean dev/test 筛选与冻结规则 | [reserved_clean/](reserved_clean/README.md) |
| DocPPL 候选生产、边界对齐、历史重建 | [parse_test_docppl_data/](parse_test_docppl_data/README.md) |

顶层保留了一些历史路径的兼容 wrapper，例如 `native_binary.py` 与
`gen_tgppl_fromtree.py`；它们仍被现有测试/调用方使用，不按文件名认定为废代码。

旧 testppl 解析/编码、split 采样/组装、手工解析队列和 `process_bbc.py` 原型已停用，
不保留旧路径转发。旧实现可通过 Git 历史查阅；本地 `history/datatools/` 如存在，
仅用于只读留档，不是运行依赖，也不随本次代码发布。
当前组装由 `assemble_streams.py` 执行，拒绝五个预留分片进入 train；
按 `build_pretrain_data` 的明确 stage/task-index 调度，不读归档的 todo/doing。

当前入口与旧工具的替代关系：

| 旧工具 | 当前入口 |
|---|---|
| `gen_final_train.py`、`sample_testset.py` | `build_pretrain_data` 的 split/assemble 阶段；新 clean 筛选用 `reserved_clean` |
| `parse_testppl*.py`、`tokenize_testppl.py`、`parse_test.sh` | `parse_test_docppl_data/generate_labeled_topk.py`、`generate_native_topk.py`；历史字节重建用 `reproduce_bbc_test.py` |
| 旧解析队列、手工启动脚本、reset 和状态 | `build_pretrain_data parse --corpus ... --task-index ...` |
| `process_bbc.py` | `parse_pretrain_data/make_tree_variant.py` |

一次性调试/打印脚本、旧 profiler 模板和输出已删除；正式回归测试保留在 `tests/`。

XSum/SuperGLUE sidecar、BLiMP 老格式再生、个人路径工具仍有未确认的用途；
`rsync.sh` 仍被当前配置工具引用，`save_datasets.py` 是 MMLU-Redux 导出工具。
这些文件继续保留。
