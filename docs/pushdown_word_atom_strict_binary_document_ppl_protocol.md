# Pushdown fixed-word-atom：结构、概率与历史对照

当前历史选择为 [model-best](document_ppl_model_best_history.md)，输入/转换细则见
[binary support 合同](gpst_binary_pushdown_document_ppl_protocol.md)。历史 candidate-0 和当前 clean/model-best 为不同协议。

## 训练表示与候选空间

`saved_models/pushdown_terminalonly/step34354-unsharded` 的输入为
`dataset/bbc-news/parse_aligned/train_pushdown_unary_terminals`。
`preprocessing.json`、`scripts/convert_treereg_to_pushdown_terminals.py` 与
`olmo/data/parse_align.py` 一致表明：多 BPE 单词先保留固定右递归词内子树，词级树再
unary collapse + deterministic right-CNF。转换仅删除 singleton spans，不把词内 BPE 拼进父节点。
生成器把此合同记录为 `fixed-per-word-right-recursive-v1`。

| 候选轴 | 结构 |
|---|---|
| `gpst-strict-binary` | 词级 strict-binary CKY top-K，再展开固定 word atoms |
| `pushdown-nary-word-atom-right-binary` | native n-ary top-K 在词级 right-CNF，再展开相同 word atoms |
| native n-ary 原始评测 | 不补齐词内/人工 binary reductions；不是该 checkpoint 的 training-representation likelihood |

n-ary 转换用 `word_starts` 把 constituent 映射到词区间，词级 `(l,s,r)` 转为
`(start[l],end[s],end[r])`；每个多 BPE 词 `[a,b]` 添加 `(g,g,b)`（g=a..b−1），
按 split gap 规范化并稳定去重。同拓扑只计一次。
实现为 `right_binarize_native_nary_spans_with_word_atoms` / `NativeNaryWordAtomRightBinarizedPushdownCorpus`。

## 概率

`sentence_ll = logsumexp(token_ll + attachment_ll)`，不除以候选数、不乘 proposal 权重。
v1 `stack_legal` 在 stack 合法位置归一化；v2 `sentence_causal` 在完整句内因果位置归一化，
对应训练 attachment CE 的 support。固定路径 v2 NLL ≥ v1；二者须分别登记。
均用实际计分 terminal/EOS 数作 PPL 分母；历史策略是另一个独立协议轴。

## 与论文结果的关系

论文的 native n-ary v1 测量没有补齐全部词内/人工 binary reductions；fixed-word-atom
binary 测量计入这些 attachment 因子并改变 depth 轨迹。不能仅凭两者数值不同断定实现
错误，也不能将 binary/v2 对照改贴为论文主值。旧数据/candidate-0 的逐案对照留在本地。
有效 clean 结果及论文边界见[论文结果说明](paper_results.md)。
