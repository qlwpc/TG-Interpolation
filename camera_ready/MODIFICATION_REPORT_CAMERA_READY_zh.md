# EMNLP 2026 Camera-ready 修复、叙述对齐与差异报告

生成日期：2026-08-30
对比基线：`review_version/paper.tex` → `paper.tex`
本轮执行边界：修复全部 P0；P1 仅修复 P1-6；另按作者新增指示对齐 scoring-format 叙述；其余 P1 保持原样
作者确认：Table 4 中的 Doc-PPL 数值为准，即 Tree-Shuffle = 13.41、Pushdown = 13.29

## 0. 结论

**在本轮约定范围内，P0 已全部关闭，P1-6 已修复，scoring-format 的主文与附录口径也已统一，可以进入 camera-ready 最终元数据与上传检查。**

本轮没有修改其他 P1。特别是摘要中的下述强表述仍保持原样：

> SLMs with standard causal attention match or exceed structural-mask variants across downstream and syntactic benchmarks.

这符合作者“先不修复”的明确要求，但从行文与证据匹配角度看，它仍是当前最值得后续斟酌的一处表述。本文已被 EMNLP main 录取，camera-ready 阶段更应优先保证口径一致、证据可追溯和不过度扩张结论；本轮修复遵循了这一原则。

## 1. 本轮修改范围

| 类别 | 状态 | 说明 |
|---|---|---|
| P0-1 至 P0-5 | 已处理 | PPL 数字与协议、过时 availability 描述、final 版式、最终 PDF 同步 |
| P1-6 | 已处理 | Validation-8/10 的任务、split、样本量、演示样本、统计协议与结果表 |
| Scoring-format 叙述对齐 | 已处理 | 统一 Section 4、主文 sensitivity 段和 Appendix E 的指标层级，不改实验数字 |
| P1-1 至 P1-5 | 未修改 | 包括摘要主张、机制性语言、hybrid/scaling/统计措辞 |
| P1-7 至 P1-8 | 未修改 | 包括部分 provenance、术语、tokenizer、作者信息等建议 |

P0/P1-6 修复完成后，本次追加改动仅涉及主文 `Sensitivity to scoring format` 段和 Appendix E 的两个收束段，共三个自然段；实验数字、表格和结论方向均未改变。

## 2. P0 修复结果

### P0-1：Tree-Shuffle Doc-PPL 数字与协议

已完成：

- Table 4 保留作者确认的 **13.41**，并加 `†` 标记。
- 正文旧值 **50.91** 已改为 **13.41**。
- 正文明确说明该值来自 terminal-only protocol，并将其排除在与 CRF-marginalized PPL 的直接比较之外。
- 附录新增模型特定 PPL 协议：Tree-Shuffle 使用 reproduced terminal-only scoring。
- SG 与 BLiMP 方法同步说明 Tree-Shuffle 使用 terminal-format scoring，而非 word-sync / CRF tree marginalization。

该处理保留了表格正确数值，同时避免把不同评分口径包装成严格同口径比较。

### P0-2：Pushdown Doc-PPL 13.29 的口径统一

按作者确认，Table 4 保留 **13.29**，并加 `*` 标记。论文与复现记录现统一为：

- 当前句：对最多 300 个 native attachment candidates 的 joint token–attachment probability 做 truncated sum；
- 历史句：使用 candidate 0 构造前缀；
- 完整值：13.293598，论文四舍五入为 13.29；
- uniform-average 16.761411 保留为敏感性结果，不作为 Table 4 主值；
- 因 Pushdown latent structure 与 CRF proposal tree 的构造不同，论文将其标为 contextual comparison。

同时修正了新增协议段中的概率方向：truncation 下界化 current-sentence probability，因此上界化对应 PPL；不再写成“upper-bounds marginalization”。

`EXPERIMENT_REPRODUCTION_RECORD.md` 已同步这一作者确认的正式口径，避免论文、实现记录和表格再次分叉。

### P0-3：清除正文与表格冲突的旧状态

已完成：

- 删除“Pushdown SG / BLiMP unavailable”的旧说法，改为报告 74.86 / 74.97，并限定为 model-specific inference 下的 contextual comparison。
- 删除“Tree-TripleCNT BLiMP unavailable”的旧说法，正文报告 83.42。

### P0-4：ACL final 版式与页数

- 源文件使用 `\usepackage[final]{acl}`。
- 最终 PDF 无页码。
- Conclusion 完整结束在第 9 页。
- Limitations 从第 10 页开始，位于 Conclusion 之后、References 之前。ACL 官方格式说明明确指出 Limitations 不计入正文页数限制，因此这一分页满足 long-paper 的 9 页正文要求。

### P0-5：最终构建同步

已将经过核验的 `build/paper.pdf` 同步为交付路径 `camera_ready/paper.pdf`，两者逐字节一致，不再存在根目录旧 PDF 与源码不一致的问题。

## 3. P1-6 修复结果：验证集定义与可复现性

### 3.1 Validation-8

正文和附录现明确给出：

- 任务：WinoGrande、HellaSwag、OpenBookQA、CommonsenseQA、SocialIQA、ARC-Easy、ARC-Challenge、PIQA；
- 数据：各任务 validation split，共 **17,691** 个 scored examples；
- demonstrations：固定、来自 training data，与 scored examples 无 normalized-text overlap；
- 聚合：8 个任务的 unweighted macro average；
- 统计：10,000 次 paired bootstrap，seed 2026。

还补充解释了 flip 数字的守恒关系：未计入 flip-to-correct / flip-to-wrong 的变化属于 wrong-to-different-wrong option switch，因此不改变准确率。

Validation-8 的主要结果保持不变：

| Model | Full | Terminal | Δ = Terminal − Full |
|---|---:|---:|---:|
| Tree | 51.52 | 53.68 | +2.16，95% CI [1.35, 2.99] |
| TGTree | 54.08 | 54.70 | +0.62，95% CI [−0.09, 1.35] |

### 3.2 Separate Validation-10

附录新增独立小节与 Table 15，对正文中的 ten-task validation 结果建立完整落点：

- 在 Validation-8 基础上加入 BoolQ 与 held-out MMLU validation；
- 总计 **22,207** 个 scored examples；
- MMLU 覆盖 57 个 subjects，每个 subject 选 5 个固定 demonstrations 并从 scoring 中移除，剩余 **1,246** 个 examples；
- MMLU 先按 subject macro-average；其他任务按 example micro-average；headline score 为 10 个任务的 unweighted macro-average；
- 10,000 次 paired bootstrap，seed 2026。

| Model | Macro accuracy | Δ vs. Terminal (95% CI) |
|---|---:|---:|
| Terminal | 54.43 | reference |
| Tree | 54.75 | +0.32 [−0.67, 1.30] |
| TGTree | **57.35** | +2.92 [1.95, 3.84] |
| Pause-1 | 56.47 | +2.05 [1.15, 2.93] |
| Pause-2 | 57.09 | +2.66 [1.82, 3.49] |

TGTree − Pause-2 为 **+0.26**，95% CI **[−0.68, 1.20]**。这一结果支持当前正文的谨慎结论：二者 aggregate performance 接近、task-level strengths 互补，而不是 TGTree 全面占优。

### 3.3 Scoring-format 叙述对齐

原稿 Section 4 已把 terminal scoring 定义为 primary content-focused metric，但后文一度称其为 content-focused alternative，形成叙述层级不一致。本次已统一为：

- **Terminal scoring：**主文使用的 primary content-focused metric；
- **Full scoring：**用于检查纳入 parser-selected structural continuation 后结果是否变化的 complementary sensitivity analysis。

主文 sensitivity 段现在先重申这一层级，再报告 ranking-change rate 和宏平均准确率变化。Appendix E 同步采用同一口径，并明确两种 score 回答相关但不同的问题。

对齐没有掩盖不利或不确定证据：Tree 的 +2.16 points、95% CI [1.35, 2.99] 支持主指标选择；TGTree 的 +0.62 points、95% CI [−0.09, 1.35] 仍表述为 statistically indistinguishable change；两个 TGTree 任务偏向 full scoring 的事实也予以保留。因此，修改提升了全文一致性，但没有把 sensitivity result 改写为“terminal 在所有模型和任务上都更好”。

## 4. 行文与科学叙事评价

本轮新增文字的质量总体适合 camera-ready：

- PPL 部分把“数值正确”与“协议可比”分开处理，没有因为保留 13.41 / 13.29 就暗示所有模型严格同口径。
- Validation 部分将数据组成、样本量、demonstration 去重、聚合规则、bootstrap 次数和 seed 放入附录，正文只保留结论所需信息，主文负担较小。
- Validation-10 表直接支撑正文中的 +0.26 与置信区间，不再出现无法追溯的孤立数字。
- “remaining ranking changes” 的补充消除了 flip percentage 看似不守恒的问题。
- Scoring-format 段现在与 Section 4 使用同一指标层级，同时保留 TGTree 置信区间跨零和两个任务偏向 full scoring 的限制，语气更适合 camera-ready。
- 新增措辞以 sensitivity、contextual comparison、competitive performance 为主，没有把方法差异写成未经验证的机制结论。

从整篇 camera-ready 的风险排序看：

1. **最高剩余行文风险仍是摘要与 contribution 中 causal-vs-structural 的强主张。** 本轮按要求未改；若提交前只允许再改一类文字，这一处优先级最高。
2. 已录取稿不宜在最后阶段再引入大规模新结果或改变核心结论。当前新增 Validation-10 的作用是为已有句子补证据链，而不是扩张主张，风险可控。
3. PPL 脚注和附录现在足以让读者识别 protocol difference；正文不应再把 Tree-Shuffle / Pushdown 纳入“所有 PPL 严格可比”的排序性结论。
4. 上传前仍需人工核对 START/ACLPUB 中的 title、author order、affiliations、corresponding author、copyright 与 supplemental-material metadata；这些外部元数据无法仅靠本地 PDF 验证。

## 5. 按要求保留、未修改的问题

以下内容仍是建议，不是本轮遗漏。本次新增授权仅覆盖 scoring-format 叙述对齐，并未扩大到其他 P1：

- P1-1：摘要语法、causal-vs-structural 强主张、`architectural modifications` 的范围。
- P1-2：structural mask 的机制性解释和绝对化措辞。
- P1-3：hybrid variants “yield no benefit” 未区分下游与 BLiMP。
- P1-4：500M scaling 文字与 BoolQ 均值排序。
- P1-5：single-pretraining-seed 条件下的统计措辞。
- P1-7：部分 1B SG / BLiMP provenance 的持久化索引。
- P1-8：`best trade-off`、LDD 范围、sample/proposal 术语、tokenizer、作者块等精确性建议。

其中摘要强主张是作者特别指定暂不修复的项目，已确认最终源码保持原句。

## 6. PDF 与源码质量检查

### 主稿 `paper.pdf`

- 16 页，A4（595.276 × 841.89 pt）。
- Conclusion 完整结束于第 9 页；Limitations 从第 10 页开始。
- final 模式，无可见页码。
- 所有字体嵌入。
- 无 LaTeX error、fatal error、overfull box。
- 无未定义 bibliography citation 或正文 cross-reference。
- Table 4、PPL 协议段、Validation-8/10、Table 15、主文 scoring-format 段与 Appendix E 收束段均已视觉检查，无裁切、重叠或不可读内容。
- 仍有预先存在的 `Hfootnote.1` PDF destination warning，来自作者脚注链接；它属于未授权修改的 P1-8 作者块范围，因此本轮未处理，不影响正文引用解析或 PDF 显示。

### Diff PDF `paper-review-to-camera-diff.pdf`

- 19 页，A4，所有字体嵌入。
- 红色删除、蓝色新增；结构变化较大的 table 保持原子化，避免 latexdiff 破坏 `booktabs`。
- 对 pause-token 长公式做了仅限 diff 展示文件的分行处理，消除了 overfull box；不改变论文源码。
- 已重点检查首页、评分方法、Table 4、PPL 协议、Discussion、主文 scoring-format 删改、Appendix E、Validation-8/10 和末页新表。

## 7. 交付物与校验值

下表给出 SHA-256 前 16 位，便于快速核对；完整值可用 `sha256sum` 获取。

| 文件 | SHA-256 前 16 位 |
|---|---|
| `review_version/paper.tex` | `e288194110d56552` |
| `paper.tex` | `6c04cc5df8800d86` |
| `paper.pdf` | `10b09c470de58f46` |
| `paper-review-to-camera-diff.pdf` | `3d2994072baa63df` |

## 8. 官方格式依据

- [ACLPUB Formatting Guidelines](https://acl-org.github.io/ACLPUB/formatting.html)：long paper 正文页数、Limitations 不计页数、A4、字体嵌入和 final 无页码。
- [ACLPUB Final Version Checklist](https://acl-org.github.io/ACLPUB/final-version.html)：作者信息、版权、最终 PDF 与提交前检查。
- [EMNLP 2026 Main Conference Papers](https://2026.emnlp.org/calls/main_conference_papers/)：EMNLP 2026 主会论文与 camera-ready 信息。

## 9. 最终建议

**本轮范围内建议 Go。** 当前主稿、diff 与复现记录已对齐；提交前剩余动作应以外部元数据和上传文件核对为主。若作者决定开放一次额外文字修改，应优先重审摘要和 contribution 中 causal-vs-structural 的结论强度；否则维持当前冻结版本，不再加入会改变主要证据结构的新实验。
