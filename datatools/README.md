# 数据工具入口

| 工作 | 入口 |
|---|---|
| 新 BBC dedup + clean 预训练配置 | [配置生成器](../scripts/prepare_bbc_dedup_500m.py)；训练数据用 dedup train，评估数据用 clean dev/test |
| BBC / FineWeb-Edu 下载、解析、组装 | `python -m datatools.parse_pretrain_data.build_pretrain_data`；[数据构建细则](../docs/pretraining_reproduction.md) |
| clean dev/test 筛选与冻结规则 | [reserved_clean/](reserved_clean/README.md) |
| DocPPL 候选生产、边界对齐、历史重建 | [parse_test_docppl_data/](parse_test_docppl_data/README.md) |

顶层保留了一些历史路径的兼容 wrapper，例如 `native_binary.py` 与
`gen_tgppl_fromtree.py`；它们仍被现有测试/调用方使用，不按文件名认定为废代码。

当前训练组装由 `assemble_streams.py` 执行，拒绝五个预留分片进入 train；
按 `build_pretrain_data` 的 stage/task-index 调度。停用源码仅本地留档，不是运行依赖。

当前入口与旧工具的替代关系：

| 旧工具 | 当前入口 |
|---|---|
| `gen_final_train.py`、`sample_testset.py` | `build_pretrain_data` 的 split/assemble 阶段；新 clean 筛选用 `reserved_clean` |
| `parse_testppl*.py`、`tokenize_testppl.py`、`parse_test.sh` | `parse_test_docppl_data/generate_labeled_topk.py`、`generate_native_topk.py`；历史字节重建用 `reproduce_bbc_test.py` |
| 旧解析队列、手工启动脚本、reset 和状态 | `build_pretrain_data parse --corpus ... --task-index ...` |
| `process_bbc.py` | `parse_pretrain_data/make_tree_variant.py` |

下游数据采用直接发布已处理文件的方式，按 [Evaluation](../Evaluation.md) 的目录加载。
现有 sidecar/旧格式兼容工具仍有调用方；`rsync.sh` 被配置工具引用，`save_datasets.py`
导出 MMLU-Redux。它们与上表正式预训练数据入口分开使用。
