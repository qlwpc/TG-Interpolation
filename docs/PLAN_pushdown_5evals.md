# Plan: Wire pushdown inference into BLiMP / boolq / XSUM, then run all 5 evals

## Context

The pushdown checkpoint (`saved_models/pushdown/step33862-unsharded`) is inference-dependent
on `tree_spans`: without them, the trained `_pushdown_attention` depth bias degenerates
(PPL 78923 vs 1035 with gold spans). Only `SG_eval_step` is wired to the fix
(`OLMo.pushdown_beam_search`, train.py:1133). The other three downstream tasks run the
plain teacher-forced / `generate()` path with `tree_spans=None`, so they crash or collapse:

| Task | Eval step | Last result (Jul 14–16) | Root cause |
|---|---|---|---|
| lmppl/docppl | `eval_step` (type=lm) | PPL=8.394 OK | works — `parse_tree_paths` supplies gold `tree_spans` via collator |
| SG | `SG_eval_step` -> `pushdown_beam_search` | job 1235 CANCELLED mid-run | wired; needs re-run to completion |
| BLiMP | `eval_step` (teacher-forced, no spans) | crash exit=1 | no pushdown inference path |
| boolq | `eval_step` (ICL, no spans) | acc=0.3804 (<chance) | depth bias degenerates |
| XSUM | `summarization_eval_step` -> `generate()` (no spans) | R-AVG=0.0108, "It is, and it is..." | depth bias degenerates during generation |

`pushdown_beam_search(eval_input_ids, beam_size=20, max_reduce=4, bos_id, tag, use_attachment_head)`
(olmo/model.py:2772) marginalizes over shift-reduce parses of a *given* terminal
sequence: `tag=None` -> surprisal `-log p(x)` (logsumexp over beams); `tag=list` -> sum of
per-token CE at tagged positions for the best beam. It is the only working pushdown
inference primitive and the foundation for all three fixes below.

## Goal

Make all 5 tasks produce valid numbers for pushdown, then submit them via the existing
driver `run_folder/eval_pushdown_treereg.sh pushdown` (one SLURM job per metric, GPUs in
parallel). lmppl needs no code change (re-run only).

---

## Phase 1 — BLiMP: marginalized surprisal via existing beam infra

BLiMP already has an opt-in beam path: config sets `beam_search: true` -> dispatches to
`BLiMP_beam_eval_step` (train.py:1245), which currently calls `word_sync_beam_search`
(wrong for pushdown — inserts NT tokens the model never learned). `BLiMPMetric.update_beam`
(downstream.py:2006) accepts a per-sentence log-prob and scatters it; `compute()` is reused
unchanged.

**Change** — add a pushdown branch in `BLiMP_beam_eval_step` (mirrors the SG branch at
train.py:1133):

```python
gt = self.cfg.model.transformer_grammar_type
if gt == "pushdown":
    sent_d = move_to_device(sent, self.device)
    with self._summon_params_ctx():
        surprisal = self.dist_model.module.pushdown_beam_search(
            eval_input_ids=sent_d["input_ids"][0],
            beam_size=20, max_reduce=4,
            bos_id=self.cfg.model.eos_token_id,
            tag=None,                       # marginalized -log p(x)
            use_attachment_head=self.cfg.model.pushdown_use_attachment_head_inference,
        )
    log_likelihood = -surprisal             # log p(x)
else:
    ... existing word_sync_beam_search ...
log_likelihoods = torch.tensor([log_likelihood], device=self.device)
evaluator.eval_metric.update_beam(batch, log_likelihoods)
```

**Config** — reuse `train_configs/eval_per_metric/pushdown_BLiMP_beam.yaml` (already has
`beam_search: true`); add `pushdown_beam_size: 20` / `pushdown_max_reduce: 4` fields to
`EvaluatorConfig` (config.py:815) with defaults, read in the branch.

**Test (TDD)** — extend `tests/test_pushdown_beam_search_cpu.py`:
- `test_blimp_branch_logp`: load ckpt on CPU, run the pushdown BLiMP branch on one BLiMP
  sentence, assert `log_likelihood` finite and `log p(good) > log p(bad)` for one known
  minimal pair (sanity). Assert it does NOT call `word_sync_beam_search`.

---

## Phase 2 — boolq: best-beam parse -> teacher-forced forward -> ICLMetric

boolq is `type: downstream` -> default `eval_step` -> `ICLMetric.update(batch, ce_loss,
logits)` (downstream.py:130). Reusing it unchanged keeps the metric/compute path identical.

**Approach** — a new `pushdown_icl_eval_step` that, per example: (1) runs
`pushdown_beam_search` over `ctx+cont` to get the **best beam's closed spans** (point
estimate of the parse), (2) runs one forward with those `tree_spans` to get logits, (3)
slices the continuation logits and calls `ICLMetric.update` with a 1-row batch.

**Why point estimate, not marginalized**: boolq is 2-way classification; the best parse's
depth bias is enough to recover above-chance accuracy, and it reuses `ICLMetric.update`
verbatim. Full `log p(cont|ctx)` marginalization would need a `past_input`/ctx-forced mode
in `pushdown_beam_search` (run twice, subtract) — deferred unless boolq stays below chance.

**Change** in `olmo/train.py` dispatch (after the BLiMP-beam branch, before the rouge branch):
```python
elif evaluator.type == EvaluatorType.downstream and \
     self.cfg.model.transformer_grammar_type == "pushdown" and \
     evaluator.label != "BLiMP":
    self.pushdown_icl_eval_step(eval_batch, evaluator)
```

`pushdown_icl_eval_step` per example:
1. `seq = ctx + cont` (from `input_ids[idx]`, `ctx_len`, `cont_len`).
2. **Extend `pushdown_beam_search`** with `return_spans: bool = False` -> also returns the
   best beam's `closed` list as an `(M,3)` tensor.
3. `out = model.forward(input_ids=seq.unsqueeze(0), attention_mask=..., tree_spans=ts)`.
4. Slice `logits[idx][ctx_len-1 : ctx_len+cont_len-1]`, gather cont tokens -> per-token
   log-prob -> `log_likelihood`. Build a 1-row `batch` dict and call
   `evaluator.update_metrics(batch, ce_loss, logits)` reusing `ICLMetric.update`.

**Config** — `pushdown_boolq.yaml` already exists (`type: downstream`); the new dispatch
routes it automatically. No config change.

**Test** — `tests/test_pushdown_icl_eval_cpu.py`:
- Load ckpt (CPU, flex disabled). Take one boolq example.
- Assert: (a) best-beam `closed` spans non-empty, (b) forward with those spans gives
  lower continuation CE than `tree_spans=None` (depth path activates), (c) `ICLMetric`
  accepts the 1-row update without error.

---

## Phase 3 — XSUM: `pushdown_generate` (shift-reduce generation beam search)

`summarization_eval_step` (train.py:1369) pushdown branch calls
`self.dist_model.module.generate()` — plain autoregressive, no `tree_spans` -> degenerate.
Need a generating variant of `pushdown_beam_search` that SHIFTs the model's own sampled
tokens while tracking spans, so the depth bias stays active.

**Change** — new method `OLMo.pushdown_generate(input_ids, max_steps, beam_size=6,
max_reduce=4, bos_id, use_attachment_head)` in `olmo/model.py`. Structure mirrors
`pushdown_beam_search`:

- Initial beam: BOS, empty closed/stack.
- For each step (until EOS or `max_steps`):
  1. Expand each beam by reduce-prefixes (0..max_reduce) -> candidate states (same
     `apply_reduces` logic).
  2. One batched forward with each candidate's `tree_spans` (collated `(N,M,3)` -1-padded),
     `last_logits_only=True`.
  3. `log_softmax` over vocab -> top-`beam_size` next tokens per candidate.
  4. Score = parent logprob + shift log-prob (+ attachment log-prob if enabled).
  5. Prune to `beam_size`; finished beams (emitted EOS) park to a finished list.
- Return best finished beam's `input_ids` (terminals only), or best unfinished by length.

**Wire** into `summarization_eval_step` pushdown branch (train.py:1383):
```python
if gt == "pushdown":
    batch = move_to_device(batch, self.device)
    preds = self.dist_model.module.pushdown_generate(
        batch["input_ids"],
        max_steps=evaluator.eval_loader.dataset.MAX_SUMMARY_LENGTH,
        beam_size=6, max_reduce=4,
        bos_id=self.cfg.model.eos_token_id,
        use_attachment_head=self.cfg.model.pushdown_use_attachment_head_inference,
    )
    predictions = preds.unsqueeze(0)  # (1, T)
```

**Config** — `pushdown_XSUM.yaml` already exists; no change.

**Test** — `tests/test_pushdown_generate_cpu.py`:
- Load ckpt (CPU). Run `pushdown_generate` on one XSUM prompt, `max_steps=40`.
- Assert: (a) finite length output, (b) NOT the degenerate "It is , and it is ..."
  repetition (token-diversity threshold), (c) forward with tracked spans gives lower CE
  than `tree_spans=None`.

---

## Phase 4 — Run all 5 evals (SLURM, local rtx3090b)

After all unit tests pass on CPU:

1. **Smoke (1 GPU, ~5 min each)**: submit each metric with `subset_num_batches: 4` first
   to confirm no GPU crash before the full run. Check `analysis-output/logs/`.
2. **Full run**: `bash run_folder/eval_pushdown_treereg.sh pushdown` submits 5 jobs
   (docppl/SG/BLiMP/boolq/XSUM), one GPU each, parallel. Each writes to
   `analysis-output/eval_pushdown_{metric}/`.
3. **Collect results**: tail logs for the final metric line:
   - docppl: `eval/TG-ppl-validation-test/Perplexity=...`
   - SG: `syntactic_generalization` score dict
   - BLiMP: `eval/downstream/blimp_acc=...`
   - boolq: `eval/downstream/boolq_acc__=...`
   - XSUM: `rouge1/rouge2/rougeL/R-AVG=...`
4. Update memory `pushdown-span-beam-search-inference.md`: mark boolq/BLiMP/XSUM wired,
   record final numbers.

---

## Files changed

| File | Change |
|---|---|
| `olmo/model.py` | add `pushdown_generate`; extend `pushdown_beam_search` with `return_spans` |
| `olmo/train.py` | pushdown branch in `BLiMP_beam_eval_step`; new `pushdown_icl_eval_step` + dispatch; pushdown branch in `summarization_eval_step` |
| `olmo/config.py` | add `pushdown_beam_size`, `pushdown_max_reduce` to `EvaluatorConfig` (optional, with defaults) |
| `tests/test_pushdown_beam_search_cpu.py` | extend with BLiMP-branch test |
| `tests/test_pushdown_icl_eval_cpu.py` | new |
| `tests/test_pushdown_generate_cpu.py` | new |

No change to: `pushdown_docppl.yaml`, `pushdown_SG.yaml`, `pushdown_boolq.yaml`,
`pushdown_XSUM.yaml`, `pushdown_BLiMP_beam.yaml` (verify `beam_search: true` only).

## Risk / notes

- **boolq point-estimate**: a single best-beam parse may underperform full marginalization.
  If boolq acc stays below chance after the fix, the fallback is extending
  `pushdown_beam_search` with a `past_input` (ctx-forced) mode and computing
  `log p(cont|ctx) = log p(ctx,cont) - log p(ctx)` via two beam searches. Deferred unless needed.
- **Cost**: pushdown beam search is ~1 batched forward/token. boolq ICL ctx can be long
  (~500 tok); full boolq (~1k examples x 500 tok x beam 20 x 5 reduce) may take several
  hours on one 3090 — acceptable for a one-off eval.
- **XSUM generation**: beam-6 generation with span tracking is ~6x the cost of plain
  `generate()`. `MAX_SUMMARY_LENGTH` bounds it.
- Login shell has no CUDA (per memory); all GPU runs go through `sbatch`. CPU unit tests
  run locally with `flex_attention=False` override (pattern from existing test).
