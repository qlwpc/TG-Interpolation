# Evaluation: Data, Model Identity, and Scoring Protocols

This document describes models, tasks, data, and configuration paths. For valid results and paper interpretation, see
[paper results](../results/paper.md) and [additional experiments](../results/bbc_dedup_500m.md);
for available materials, see the [public assets inventory](../../reproducibility/README.md). Historical values must not be relabeled as the current protocol.

> [!IMPORTANT]
> "Current" in this document refers to the currently auditable evaluator/data/config contract; it does not mean that all models share the same
> probabilities, candidate sets, or denominators. The Document-PPL of Terminal/Pause, Tree/TG, Tree-Shuffle, and Pushdown
> comprises four protocol classes; the correct way to report them is side by side with the protocol noted, not to rewrite them as a single metric.

## 1. First select a protocol by model family

| Model family | Document-PPL | SG | BLiMP | XSum / BoolQ or OLMES |
|---|---|---|---|---|
| BBC Terminal | `terminal_doc`, single terminal path | terminal teacher-forced | terminal, K=1 | BBC five-seed task-specific finetune |
| BBC dedicated-SEP Pause | `terminal_doc`; document-global pause phase, pause target excluded from denominator | pause-expanded teacher-forced | pause-expanded terminal, K=1 | Pause v2 campaign only; XSum phase-constrained KV-cache generation |
| Tree/TG and their linearized variants | `tg_doc`; at most 300 valid CRF proposals, truncated sum | word-synchronous DFS beam, beam=300 | after reading existing 300 parses, `logsumexp` over joint tree loss | BBC five seeds; retain the model's own tree/TG representation |
| Tree-Shuffle (masked paper row) | terminal-only, K=1 | terminal-only teacher-forced | terminal-only, K=1 | checkpoint keeps `tree_shuffle_mask`; must not be swapped for an unmasked checkpoint |
| TreeReg-layer9 | `terminal_doc` | terminal teacher-forced | terminal, K=1 | out-of-paper architecture comparison; downstream must distinguish legacy FT from parse-aligned auxiliary-loss FT |
| Pushdown | native attachment candidates; clean registered protocol is n-ary v1 `stack_legal` / model-best | incremental attachment beam=300 | supplied gold300 attachment candidates | XSum uses source gold spans + ROOT-free summary stack; BoolQ uses corrected gold-span scoring |
| FineWeb-Edu 1B | do not substitute with the BBC Doc-PPL entry point | current SG framework + Qwen3 tags | current BLiMP framework + Qwen3 top-300; K selected by grammar | Qwen3 tokenizer; OLMES completion/cloze, multi-shot, primary=`tree_eval_type: terminal` |

`Treeterm` and `TGTreeterm` are terminal-score evaluation names for Tree/TGTree checkpoints, not independent
weights. The 100M legacy `pause_token_id=null` weights are a repeat-token compute control, and are not
dedicated-SEP Pause. The paper row for Tree-Shuffle uses
`saved_models/treeshufflemask_pretrain/step49440-unsharded`;
`saved_models/Tree_shuffle_pretrain/step49440-unsharded` is retained only as an unmasked historical comparison.

BBC 100M `pause1` / `pause2` public aliases are now bound to the SEP50261 checkpoint; historical weights are explicitly selected
via `pause1-repeat` / `pause2-repeat`. The general config generator checks the checkpoint's original pause ID,
and refuses to rewrite model identity via override; the paper's XSum/BoolQ must go through the Pause v2 campaign.
Pretraining parameters and full commands are in [`docs/models/pause.md`](../models/pause.md).

## 2. Data entry points

All paths are relative to the repository root. Existence of a file only means the asset is present on this machine; before entering the results registry it must still be bound to hashes, split,
checkpoint, and run id.

### BBC Document-PPL: first select the data version

The current clean full evaluation is described in the [reserved-clean run protocol](../protocols/clean_document_ppl.md).
The root is `dataset/bbc-news-reserved-clean-v1/`:

| Representation | Path under root | Contract |
|---|---|---|
| Terminal / Pause / TreeReg / Tree-Shuffle terminal | `evaluation_terminal/{test.npy,test_sent_index.npy,test_doc_index.npy}` | text comes from `terminal/test.npy`, indices derived from canonical boundaries by input audit |
| Tree / TG | `testppl/tree300/`, `testppl/tg300/` | at most 300 valid candidates per sentence; scored by `valid_counts.npy`, padding is not counted in probability mass |
| GPST / Pushdown native | `native_model_topk_300_v2/test/` | independent candidate spaces and valid counts for the two models; candidate IDs cannot be shared |

The full test set is **5,000 documents, 129,085 sentences, 3,068,713 scored terminal/EOS tokens**.
NoONT/Compress, after reading the valid tree300 candidates and converting them, must also deduplicate per sentence by the full token sequence.
The raw `valid_counts.npy` is not equal to the projected unique K; both running and merging must be bound to the full projection audit.
The base arrays including BOS contain 3,073,713 tokens. The fixed test-1000 subset is registered separately and must not be passed off as the full test.

**Historical data are only for reproducing old results that are explicitly selected**: `dataset/bbc-news/terminal/`,
`testppl_tree/`, `testppl_tg/` and `testppl/native_model_topk_300_v2/` correspond to the old 4,966 documents,
148,836 sentences, and 3,284,061 scored tokens; they are not the clean split. The old index test of 5,025 documents
is yet another version, see [version provenance](../protocols/bbc_data.md#versions).

### Other evaluation data

| Task | Current data entry point | Structure that must be checked |
|---|---|---|
| SG | `evaluation/SG/tokenized/*.json`; for Qwen3 use `evaluation/SG/tokenized/qwen3/*.json` | currently 32 items, 6 categories; `nn-nv-rpl` is not counted |
| BLiMP | `dataset/BLiMP/tree300/blimp_{terminal,tree_300,tg_300,tree_300_qwen}.npy` | full suite=67 tasks × 1,000 pairs; terminal K=1, structural models K=300 |
| XSum | `dataset/Xsum/` | filtered train is determined by `save_ids.json`; full test=11,333; source, summary, gold JSONL must form a complete set |
| BoolQ | `dataset/SuperGLUE/BoolQ/` | train=9,427, validation=3,270; parsed passage/question sidecars must be the same version as the JSONL |
| FineWeb-Edu OLMES | `olmo_data/oe_eval_tasks/`, `olmo_data/hf_datasets/` and task registry | tokenizer=`dataset/TG_QWEN3_tokenizer.json`; must not fall back to the BBC GPT-2 tokenizer |

Downstream data is delivered by directly open-sourcing already-processed files (including required parse-tree sidecars), and loaded from the above directories;
no separate data-construction pipeline is required. MultiRC/ReCoRD keep their existing stubs, are not planned for implementation,
and are not listed as supported downstream evaluation tasks.

The BBC GPT-2 evaluator uses `dataset/bbc-news/TG_GPT2_tokenizer.json`. FineWeb-Edu/Qwen3
models use `dataset/TG_QWEN3_tokenizer.json`, and require the corresponding Qwen3 SG/BLiMP data. Any configuration containing both
a FineWeb checkpoint and the BBC tokenizer, or a BBC checkpoint and Qwen3 task sidecars, should
fail before submission, rather than letting the evaluator guess automatically.

## 3. Configuration layering and entry points

### 3.1 Four-layer contract

1. **checkpoint config**: only responsible for the trained architecture, grammar, tokenizer identity, context
   length, and pause id; `data.paths` in the checkpoint that were later overridden by evaluation are not evidence of the training source.
2. **protocol config**: explicitly selects evaluator label/type, data, `structure_mode`,
   `tree_eval_type`, number of candidates, denominator, and task split.
3. **runtime config**: device, DDP/FSDP, microbatch, workers, and output paths. Changing runtime must not
   change layer-2 probability semantics.
4. **result record**: records checkpoint, config, data/hash, seed, job/run id, logs, completion
   status, and primary protocol/diagnostic identity.

test-only runs must be rebuilt from the checkpoint model config and set
`eval_on_load=true`, `eval_no_save=true`, `max_duration=0`,
`reset_optimizer_state=true`, `reset_trainer_state=true`, `try_load_latest_save=false`.
Do not inherit the checkpoint's training shards, old evaluator, `stop_at`, or optimizer/trainer state.

### 3.2 Executable entry points

| Scope | Preferred entry point | Boundary |
|---|---|---|
| SG/BLiMP for conventional BBC models, historical Doc-PPL, and ordinary XSum/BoolQ campaigns | `scripts/init_cfg_and_sbatch.py` | BBC keys bind GPT-2; `terminal-1B` / `tree-1B` remain BBC-compatible aliases |
| FineWeb-Edu 1B SG/BLiMP | `scripts/init_cfg_and_sbatch.py`, `*-fwedu-1B` keys | binds Qwen3 tokenizer / tags / BLiMP arrays; launch method see §5.1; do not reuse BBC's DocPPL/finetuning recipe |
| full dedicated-SEP Pause evaluation | `scripts/pause_eval_campaign.py`, `scripts/pause_eval/README.md` | XSum must be pipeline v2 with eval batch=1; old v1 checkpoints/markers cannot be reused |
| TreeReg/Pushdown SG vs BLiMP protocol comparison | `scripts/make_syntax_eval_configs.py` | `structure_mode` must be explicit; Pushdown primary values are SG `beam`, BLiMP `gold` |
| reserved-clean full Document-PPL | `scripts/evaluate_reserved_document_ppl.py`; Pushdown uses standalone evaluator | execute per the input audit, independent configs, and strict finalizer in the [run protocol](../protocols/clean_document_ppl.md) |
| Pushdown current Document-PPL | `scripts/evaluate_pushdown_document_ppl.py` and `docs/protocols/pushdown_ppl.md` | paper primary values, v1/v2, direct/n-ary, and topology diagnostic are registered separately |
| masked Tree-Shuffle | `scripts/init_cfg_and_sbatch.py`, `tree_shuffle_mask` key | the three base items use runtime `terminal`; XSum/BoolQ keep checkpoint grammar=`tree_shuffle_mask` |
| FineWeb-Edu 11-task OLMES | `evaluation/eval_configs/a800_bootstrap_{terminal,tree,pause}.yaml` | uses Qwen3 tokenizer; Tree/TGTree primary score must be `tree_eval_type: terminal` |
| FineWeb terminal/full validation | `scripts/campaigns/make_terminal_format_validation_configs.py` | Validation-8 decomposition and Validation-10 must not be passed off as 11-task test results |

`evaluation/eval_configs/` and `train_configs/eval_per_metric/` still contain historical, smoke, and topical YAMLs.
Configs whose filenames contain `smoke`, `profile`, `decomp`, or old absolute paths cannot be promoted to primary protocol merely by existing.
Prefer generating new run configs from audited checkpoints; save frozen campaign configs together with the manifest.

### 3.3 `structure_mode` is the current structural protocol switch

`EvaluatorConfig.structure_mode` takes the following values:

| Value | Semantics |
|---|---|
| `auto` | retain the model-family default; only suitable for ordinary Tree/TG paths already explicitly covered by this table |
| `terminal` | terminal teacher-forced; no parse/stack tape provided |
| `gold` | currently used for Pushdown BLiMP; consumes supplied 300 parses/spans |
| `beam` | latent structure inferred incrementally by the model; Pushdown and Tree/TG use their respective searchers |

Legacy `beam_search: true` is compatibility-only; new configs that also write a conflicting `structure_mode` will error. Paper primary protocols
must not rely on implicit `auto` to select the Tree-Shuffle or Pushdown branch.

## 4. Current Document-PPL contract

### 4.1 Terminal, Pause, TreeReg, and Tree-Shuffle terminal projection

The evaluator is `label=terminal_doc_ppl, type=terminal_doc`, batch=1, worker=0. Each document is provided
with BOS only at the start; BOS is not scored, ordinary terminal and EOS are scored. The denominator is determined by the selected data version:
clean full is 3,068,713, the old 4,966-document version is 3,284,061. The DataLoader distributes complete documents to ranks, and each sentence must be evaluated exactly once; the metric merges
global NLL and counts, and rejects partial evaluation.

Within the same document, KV cache continues across sentences. The first token of the next sentence must be scored from the previous sentence's frozen final distribution, and must not be overwritten by the distribution after submitting the current sentence's cache.
When context length is exceeded, only the historical cache is truncated.

Pause uses a document-global phase; it continues across sentences and resets across documents. Insertion positions are determined by the checkpoint's
`pause_token_id`: `null` means repeat-token control, 50261/151673 mean dedicated SEP.
Insertion positions are excluded via `label_mask`, so numerator and denominator remain aligned with terminal projection.

### 4.2 Tree/TG CRF-300 truncated marginal

The evaluator is `type=tg_doc`, Tree label=`txl_approx_doc`, TG label=`tg_approx_doc`, the current
physical slot cap is 300; the clean entry applies `logsumexp(-NLL)` only over the joint tree NLL of valid candidates,
and the total is then divided by the terminal/EOS denominator, not by K.

The current implementation scores all valid candidates for the sentence, then selects the candidate with the smallest total NLL to commit as history for subsequent documents,
updating it together with KV cache, next-token distribution, and mask state; ties take the earliest candidate.
See [model-best history policy](../protocols/document_history.md). Old candidate-0 results
retain their original protocol identity and cannot be relabeled by code updates.

### 4.3 Pushdown native top-K

The current standalone Pushdown uses `prefix_policy="model_best"`, selecting history based on
joint token + attachment NLL under the current protocol; token-only diagnostics select by token NLL.
The registered reserved-clean run uses native n-ary / v1 `stack_legal`, see
[run protocol](../protocols/clean_document_ppl.md). Each sentence's joint probability of the unique valid structure
is truncated-summed, without subtracting `log K_s`.

fixed-word-atom n-ary/direct strict-binary, v2 `sentence_causal`, uniform-average,
token-only, and BPE-spliced topology are different protocols or diagnostics. Definitions and historical scope of early measurements are in
[Pushdown protocol](../protocols/pushdown_ppl.md).
The old data/candidate-0 measurements retained there cannot be merged with current clean/model-best results; continuation and merging must check
`prefix_policy`, data hash, checkpoint, and full document coverage. Old paper values see [paper results](../results/paper.md).

## 5. SG and BLiMP

The revised manuscript flags BLiMP candidate-uniqueness checks for NoONT-100M,
Compress-100M and NoONT-500M as pending. Their DocPPL projection-deduplication
reevaluation is complete; see the [completed results](../results/paper.md#completed-noontcompress-docppl-reevaluation).
The DocPPL receipt does not certify BLiMP candidate handling.

SG uses 32 items and 6-category formulas, judging by target-region surprisal and aggregating the category average.
Terminal/Pause/Tree-Shuffle/TreeReg use teacher-forced logits; Tree/TG use
word-synchronous DFS beam=300, with sub-beam=300 and terminal fast-track=30 at scoring,
`max_length=max(6*term_len,10)`; the Pushdown primary protocol uses attachment beam=300.

The Tree/TG SG entry still passes `nc=max(term_len,5)`, `pc=3`, but the NT count limits of these two parameters are
**not enabled**: although `olmo/model.py::word_sync_beam_search` constructs `Stop_Add_NT`, the two lines of code that use it to mask
opening-NT logits are commented out, so the total number or consecutive count of opening NTs is not limited.
The parameter values passed in cannot be registered as effective search constraints; the above total linearization length cap still applies.
The call path and consumption sites were audited on 2026-10-02, and Git history confirms that the masking was commented out at `c6c8f07` (2025-10-16); the camera_ready SG appendix statement "no additional NT count limit" is consistent with this.

The BLiMP full suite is 67×1,000 minimal pairs. Terminal/Pause/Tree-Shuffle/TreeReg use K=1 per sentence;
Tree/TG read existing 300 parses and apply truncated `logsumexp` over joint tree likelihood; the Pushdown
primary protocol converts supplied gold300 trees into terminal-coordinate spans/action candidates and then marginalizes.
terminal-only or inferred-beam results are protocol diagnostics and cannot replace the registered gold300
primary values. Any number of tied/non-finite pairs should also be registered with the results.

### 5.1 Current launch method for FineWeb-Edu 1B SG/BLiMP

FineWeb-Edu 1B SG/BLiMP uses the current code test framework and Qwen3 tokenizer.
At the repository root, after creating and activating the `LLM` environment per [environment.yml](../../environment.yml), run:

```bash
conda activate LLM
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/init_cfg_and_sbatch.py \
  --device RTX3090 \
  --models terminal-fwedu-1B tree-fwedu-1B tgtree-fwedu-1B \
           pause1-fwedu-1B pause2-fwedu-1B \
  --tasks SG blimp --seeds 6198 \
  --campaign-dir artifacts/evaluation/fineweb_1b_syntax \
  --submit
```

Removing `--submit` only generates configs and launch scripts. Each task defaults to two-card DDP; `--device` selects the target cluster
configuration. Output is `runs/<model>_<task>_seed6198/config.yaml`, a `.sh` in the same directory, and a top-level
`jobs.json`; checkpoints use the corresponding final weights under `saved_models/A800_models/`.

The generator verifies the checkpoint's vocab size, EOS/PAD, and Pause SEP identity, and binds
`dataset/TG_QWEN3_tokenizer.json` (Pause SEP=151673). SG reads aligned tags from
`evaluation/SG/tokenized/qwen3/`; BLiMP reads uint32 data from
`dataset/BLiMP/tree300/blimp_tree_300_qwen.npy`, with Terminal/Pause taking
candidate 0 per sentence and converting to terminal, and Tree/TGTree using 300 candidates and corresponding representation conversion.
Explicit terminal/beam modes also convert from Qwen arrays and do not read GPT-2's `blimp_terminal.npy`.
The generator also writes a small `config.eval-only.npy`, only for the Trainer to construct an unused training loader;
keep it in the same directory as the YAML. Formal SG/BLiMP does not score this placeholder array and does not depend on FineWeb training shards.

BBC models use the same generator, BBC model keys, and GPT-2 data. FineWeb's 11-task OLMES
continues to use the dedicated configs in §6.2; SG/BLiMP are not merged into its average.

## 6. XSum, BoolQ, and FineWeb-Edu OLMES

### 6.1 BBC XSum / BoolQ

Ordinary BBC checkpoints are independently finetuned after resetting optimizer/trainer from pretrained weights. XSum is 3 epochs,
LR `6e-5`, global batch 40; the full 11,333-example test reports ROUGE-1/2/L and their mean R-AVG.
BoolQ is 5 epochs, LR `3e-4`, global batch 40; the full validation reports accuracy. Paper tables use
seeds `42, 2026, 6198, 13171, 31723` with mean ± sample SD.

Pause uses v2 only: the training supervision mask is aligned with the expanded summary, generation maintains the pause phase,
uses KV cache with eval batch=1. Any old artifact lacking a matching `training_contract.json` or
`xsum_pipeline_version: 2` cannot be reused.

The registered primary Pushdown XSum run uses source gold spans, ROOT-free summary stack, beam=6,
`max_reduce=null`; prompt spans serve only as attention history and do not become summary attachment targets.
Pushdown BoolQ uses the corrected attachment-loss denominator and gold-span teacher-forced terminal-format
MC scoring. BoolQ from the old incorrect denominator and root-containing/max-reduce=4 XSum do not enter the main table.

### 6.2 FineWeb-Edu OLMES

The 11 tasks use completion/cloze, not classification that displays option letters. HellaSwag, WinoGrande, MMLU,
MMLU-Redux, OpenBookQA are 5-shot; BoolQ, ARC-Easy, ARC-Challenge, PIQA, SocialIQA,
CommonsenseQA are 3-shot. Each textual component uses Benepar 1-best; 300 proposals are not used.

The primary metric for Tree/TGTree is the sum of continuation terminal-token log-probs
(`tree_eval_type: terminal`); full score is only for sensitivity analysis. The 11-task test,
Validation-8 decomposition, and Validation-10 are three different data contracts and must not be substituted for one another.
Completed 11-task per-example evidence and bootstrap summaries are located at
`analysis-output/bootstrap/`; public summaries must be bound to checkpoint and completion evidence; the retrieval location for the complete per-sample artifacts is still to be filled in.

## 7. Distributed and integrity gates

- The evaluator uses `DistributedEvalSampler` with no padding duplication and no tail truncation.
  Document-PPL distributes by complete document across ranks; K>1 BLiMP distributes by complete sentence group across ranks.
- Only `eval_subset_num_batches=-1` is a full run. smoke/partial metrics must not fill the main table; the terminal
  Document-PPL metric actively rejects runs with incomplete record counts.
- Current custom generation and some structural searches do not support calling module methods directly on FSDP shards. Formal downstream evaluation
  continues to use DDP/full replica until real multi-GPU integration tests close the risks in
  [`docs/protocols/distributed_evaluation.md`](../protocols/distributed_evaluation.md).
- A model-only checkpoint may lack `optim.pt`/`train.pt`; as long as both kinds of state restore are explicitly disabled,
  this is not evidence of failure.
- Before output enters the results page, finite metrics, complete sample counts, checkpoint/config/data identity,
  run/job id, and log completion markers must be checked. Primary protocol and alternative-protocol diagnostic go on separate rows.

## 8. Known paper discrepancies and protocol boundaries

### 8.1 Historical DocPPL results vs current protocol

Current Tree/TG and standalone Pushdown already use model-best; old candidate-0 results and new-data results must be kept separate.
Residual discrepancies in the paper body/appendix and the impact of re-measurement on conclusions see [data notes](../protocols/bbc_data.md#paper-impact).
New/old standards, experimental paths, result differences, and basis for subsequent paper data revisions see
[R-08 determination report](../results/protocol_comparison.md). This round only completed document determination;
the paper was not modified; subsequent revisions must align the main table, methods, appendix, and model scope simultaneously, and cannot relabel old values with the new protocol.

<a id="pushdown-paper-discrepancies"></a>

### 8.2 Pushdown: known discrepancies between the frozen paper and actual runs

The following explains the gaps between the frozen paper and the implementation based on audited run records;
it does not change historical metrics, nor does it automatically treat the current implementation as the source code at the time of the historical runs.
Paper version and numeric transcription see [paper results](../results/paper.md).

| Item | Frozen paper description | Registered primary run / code behavior | Explanation and evidence |
|---|---|---|---|
| XSum reduce limit | Downstream appendix, line 708: `at most 4 consecutive reductions` | primary run is `beam=6,max_reduce=null`; source parser spans serve only as attention history, summary uses an empty, sentence-local ROOT-free attachment stack | clear difference in parameter description. `beam=6` is consistent, no additional consecutive reduce limit. From audited records of sentence-local five-seed runs; the local campaign's `run_eval.sh` also explicitly sets `null` |
| BoolQ search method | Same paragraph: `candidate scoring with beam size 20` | primary run uses supplied-span teacher-forced terminal-format multiple-choice scoring; beam=20 fallback only when `tree_spans` is missing | clear difference in scoring path. From audited records of corrected BoolQ runs, and `Trainer.pushdown_icl_eval_step` |
| BLiMP probability definition | BLiMP appendix, lines 749, 753: explains tree marginalization as `sum_k p(x,y_k)`, and states that Pushdown marginalizes over 300 candidate trees | current gold300 path applies `logsumexp` only over token log-likelihood under the given tree, without adding attachment log-probability; nearby historical commit `5ee0e8f` has the same implementation | current implementation cannot be interpreted as marginalizing over joint token–attachment probability. Historical job 2521's config and 74.97% log have been audited, but its exact run source identity has not been confirmed; old scores cannot be relabeled as joint marginal without verification or re-measurement |
| Inference structure source | Setup, line 346: attachment parser predictions produce stack-depth bias at inference | BoolQ, BLiMP, and XSum source use provided parse trees; SG and XSum summary use model-searched attachment structure | statement scope is too broad; reading must distinguish supplied structure from inferred structure by task |
| Meaning of `gold` | Downstream appendix, line 708: `gold spans / gold parse` | trees come from a parser, belonging to silver trees / supplied parser spans | terminology ambiguity; does not represent human gold annotation. Setup already states that parse trees are silver |

The primary XSum/BoolQ runs are independently finetuned from `pushdown_terminalonly/step34354-unsharded`.
SG/BLiMP use the same pretrained weights, with SG attachment beam=300. Complete historical run logs remain local
evidence; asset status see [assets inventory](../../reproducibility/README.md#assets). In particular, the exact run source identity of historical BLiMP has not been confirmed, and current code tests do not substitute for the full evaluation at that time.

Scoring implementation evidence is the `model_forward`, `eval`,
`pushdown_icl_eval_step` of [Trainer](../../olmo/train.py), and the `update/compute` of [BLiMPMetric](../../olmo/eval/downstream.py).
The BLiMP gold path does not compute attachment logits in eval mode; the metric only receives LM logits.
This round executed CPU checks in the `LLM` Conda environment related to Pushdown beam probabilities, provided prompt spans, ROOT-free stack,
attachment-loss denominator, and downstream data conversion; 11 passed; these checks do not substitute for a full historical rerun.

**Applicability boundary of DocPPL:** The paper's Pushdown 13.29 matches the historical candidate-0
protocol in appendix line 758, and is not in itself a contradiction between value and protocol. It cannot be relabeled as a current clean/model-best result.
The model-selected history in paper body line 366 and the appendix's
parser-selected history for generic CRF tree models have another internal wording difference; Pushdown is listed separately in the appendix as a candidate-0
exception. The current protocol see [model-best contract](../protocols/document_history.md);
historical and clean measurements each retain their original data, checkpoint, candidate set, and history-selection policy identity.

## 9. Evaluation correctness checks

The following boundaries are retained as evaluation correctness requirements; the troubleshooting process is archived locally only.

| Check item | Current requirement |
|---|---|
| Pause BoolQ answer mask | first define the raw answer boundary as `len(full_query)-len(continuation)`, then expand the mask with Pause; each option must have valid scoring tokens |
| Pause PPL denominator | use only terminal/EOS NLL and counts, excluding inserted Pause positions |
| Cross-sentence cache | before submitting the current sentence, retain the previous sentence's end prediction distribution; reset at document boundaries, and carry forward history limited by the length budget across sentences |
| Structural variants | config explicitly binds grammar, data, and tokenizer; unknown types must not implicitly fall back |


Pause XSum label expansion, generation phase, and checkpoint compatibility requirements see
[Pause v2 protocol](../models/pause.md).
Model-only recovery and distributed limits see §7; native continuation, full-document write-out, and strict merging see
[recovery contract](document_ppl_recovery.md).
