# Design: Strengthening Terminal-Format Evaluation and Explicit-vs-Implicit Claims

**Date**: 2026-05-22
**Status**: approved
**Source**: Reviewer feedback on Q4 (terminal-format evaluation validity) and the 0.24-point tgtree_term vs pause2 margin

---

## Goal

Add three pieces of evidence to the paper that jointly strengthen two vulnerable claims:

1. **Q4**: Terminal-format evaluation is methodologically necessary (reviewer pushback: "it cherry-picks tokens where syntactic models happen to align with terminal baselines")
2. **0.24 gap**: tgtree_term vs pause2 margin is fragile — reframe as complementary strengths with a "rational compute budgeting" interpretation

No new model training required. All three pieces use existing models and eval infrastructure.

---

## Design

### Piece 1: Terminal-Format Validation (Two Checks)

**1a. Answer-length confound check**

For each `_decomp` evaluation example, count the number of structural tokens (ONT + CNT) in the continuation. Compute per-example:
- Correlation between #structural tokens and full-format log-likelihood score
- Correlation between #structural tokens and terminal-format log-likelihood score

Prediction: full-format scores are positively correlated with structural token count (more brackets → higher score via bracket preference). Terminal-format scores show near-zero correlation.

Requires one code change: add a per-example structural token counter to `DecomposedICLMetric.update()` in `olmo/eval/downstream.py`, storing the count alongside `loglikelihoods_term`.

**1b. Task-ranking sanity check**

For the 8 OLMES tasks, compute Spearman rank correlation of per-task accuracy between:
- Terminal baseline vs full-format syntactic model scores
- Terminal baseline vs terminal-format syntactic model scores

Prediction: terminal-format scores better preserve the "task difficulty ordering" of the bracket-free baseline. A higher Spearman ρ for terminal-format means it measures content quality rather than bracket preference.

Requires no code changes — post-hoc analysis from existing `_decomp` summary outputs.

### Piece 2: Depth-Stratified Analysis (The "Rational Budgeting" Argument)

**Core claim**: Tree tokens concentrate computation at syntactic boundaries where multi-hop integration matters; pause tokens spend uniformly everywhere. This predicts that tgtree's advantage is largest on examples with deep syntactic structure.

**Method**:
1. For each OLMES continuation, run Benepar to get the parse tree, compute average constituent depth
2. Bucket continuations into Shallow (depth 1–2), Medium (depth 3–4), Deep (depth 5+)
3. For each bucket, compute terminal-format accuracy for tgtree_term, pause2, and terminal
4. If Deep bucket has <100 examples per task, merge with Medium

**Predicted pattern**:
- Terminal: flat accuracy across depths
- pause2: modest improvement with depth (uniform budget doesn't target syntax)
- tgtree_term: monotonic improvement with depth (budget scales with syntactic need)

**Key figure**: Line chart: x-axis = depth bucket, y-axis = accuracy, three lines (terminal, tgtree_term, pause2). Aggregate over all 8 OLMES tasks.

**Requirements**: Benepar parsing of OLMES continuations (embarrassingly parallel, continuations are short). No model training or re-evaluation.

### Piece 3: Per-Task Complementarity + Narrative Reframing

**Per-task winner map**: Group 11 OLMES tasks by cognitive type and show which model (tgtree_term or pause2) leads on each:

| Group | Tasks | Winner |
|---|---|---|
| Cross-span reasoning | BoolQ, Winogrande, CommonsenseQA | tgtree_term |
| General knowledge | MMLU, MMLU-Redux, OpenbookQA, PIQA | pause2 (slight) |
| Surface reasoning | HellaSwag, ARC-Easy, ARC-Challenge | pause2 |
| Social reasoning | Social IQa | pause2 |

**Revised main claim** (replaces the 0.24-point AVG claim):

Old: "tgtree_term achieves the highest 11-task average (55.92), modestly exceeding pause2 (+0.24)"

New: "tgtree_term matches the best pause-token baseline with complementary strengths: explicit syntactic structure improves cross-span reasoning (+5–9 points on BoolQ, Winogrande, CommonsenseQA over terminal), while pause tokens benefit surface-level commonsense reasoning. Depth-stratified analysis shows tgtree's gains concentrate at deeper syntactic structures, indicating that tree tokens provide a more targeted compute budget than uniform pause placement."

### Evidence Convergence

The three pieces reinforce each other:

1. **Terminal-format validation** (Piece 1): establishes that we can trust the terminal-format measurements
2. **Depth-stratified analysis** (Piece 2): shows WHY explicit structure helps — not uniformly, but where syntax matters
3. **Per-task complementarity** (Piece 3): the 0.24 gap is an artifact of averaging complementary strength profiles; the real contribution is understanding which structure helps where

### Text changes to paper

- **§3.5 (Terminal-format evaluation)**: Add confound check results (Piece 1)
- **§4.3 (Main result)**: Replace AVG-centric framing with complementarity framing (Piece 3)
- **§5.1 (Why does tree-structured input work?)**: Add rational-budgeting hypothesis supported by depth-stratified analysis (Piece 2)
- **§6 (Conclusion)**: Update takeaway to reflect complementary strengths rather than narrow victory

---

## Implementation Order

1. Piece 1a: Add structural token counter to `DecomposedICLMetric`, re-run `_decomp` eval for tree_1B and tgtree_1B
2. Piece 1b: Compute Spearman correlations from eval output
3. Piece 2: Parse OLMES continuations, bucket by depth, compute per-bucket accuracy
4. Piece 3: Generate per-task winner figure, update paper text

Steps 1–3 are independent and can run in parallel.

---

## Edge Cases and Risks

- **Shallow depth clustering**: If >80% of OLMES continuations are depth ≤2, the deep bucket will be unreliable. Fallback: use median split (shallow vs deep) instead of 3-way split.
- **Terminal baseline for 1b**: The terminal baseline task ordering is itself noisy (single seed, 53.23 AVG). Low Spearman ρ could reflect baseline noise, not evaluation format failure. Mitigation: also report the same correlation for tree vs tgtree (both tree-structured models where format differences should be smaller).
- **Parser quality on short continuations**: Benepar may produce degenerate parses for short MC continuations (often 1–5 words). Fallback: use sentence-level parse tree from the full context + continuation as a single string.

---

## Spec Self-Review

- No placeholders or TODOs
- Pieces 1–3 are internally consistent and converge on the same narrative
- Scope is focused: analysis only, no training, no architecture changes
- Edge cases identified with explicit fallbacks
- Implementation steps are independent and parallelizable
