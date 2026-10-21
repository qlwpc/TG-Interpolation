# BBC experiment comparison: historical, clean reevaluation and dedup training (R-08)

Verification date: 2026-10-02. Experimental completion status is based on original receipts through 2026-09-29; this check verifies local evidence, configs, paths, and values, and does not re-evaluate, query remote systems, or modify the paper. This report is a fixed-date description of differences, intended for subsequent revision of paper experimental data and selection of reproduction plans; per-model formal data continues to be maintained by the original result JSON.

**Three sets of evidence must be used separately: old paper results, clean re-test of old weights, and new dedup 500M retraining.**
The latter two groups both use the full clean test, but only the third group retrains. From old paper to clean re-test, both the test set and the structural scoring protocol change; from clean re-test to new 500M, the training data volume, LR, batch, steps, and some model implementations also change. These differences support descriptive comparison and cannot alone quantify the "decontamination benefit" or causal effects of parameter scale.

## 1. R-08's Re-verification Criteria and Experimental Identity

The completion conditions for this R-08 are: clearly identify the data and checkpoint for each group; explain candidates, history, denominators, and coverage criteria; give traceable paths and result differences; exclude known invalid results; define the evidence and reproduction gaps that can be used for subsequent manuscript revision.
**R-08 document verification completion does not mean the paper has been revised, full retraining has been publicly reproduced, or undelivered incomplete assets have been closed.**

| Group | Meaning in this report | Training and weights | BBC DocPPL test conditions | Authoritative source of results |
|---|---|---|---|---|
| A: Frozen-manuscript old experiments | The 19 rows of the current frozen manuscript `tab:exp1-results`; not the table values of the earlier submitted PDF | Historical BBC training; old 100M / four 500M checkpoints | Old 4,966 documents; structural model historical records are candidate-0 | [Frozen table JSON](../../reproducibility/evidence/paper_bbc_table.json) |
| B: Clean re-test of old models | Replace the test and scoring protocol for existing weights; 22 valid models in total, of which 19 rows correspond to A | Keep the same weights as the corresponding rows in A; the three grammar fixes only change config copies | Clean 5,000 documents; valid candidates joint sum, model-best | [Valid clean JSON](../../reproducibility/evidence/old_checkpoint_clean.json) |
| C: New dedup 500M | Ten newly trained-from-scratch models and complete downstream evaluation | dedup train, clean dev/test, recomputed training parameters; single pretraining seed | Same clean 5,000 documents; valid candidates joint sum, model-best | [Ten-model results page](bbc_dedup_500m.md) and three batches of original result copies |

A → B is not "the same experiment with only updated scores"; B → C is not "continued re-testing of old models." The only 500M model names shared by A and C are Terminal, Tree, and TGTree. **TGNomask-Aug-500M has no corresponding entry in group C**; neither the new TGNomask nor TGNomask-Mix-TG can substitute for it. B's two additional results, TreeReg and BBC 1B, are not part of A's main table; BBC 1B is also not equal to FineWeb-Edu 1B. This report does not re-verify FineWeb-Edu's 11-task experiments.

## 2. Data Standards: Training Set, Test Set, and Contamination Evidence

### 2.1 Version cannot be determined from file names or document counts

| Data | Document count | Sentence count | terminal/EOS scored tokens | Verified use |
|---|---:|---:|---:|---|
| terminal array of historical train | 15,187,326 | — | Training array contains special tokens, 10,061,025,584 tokens in total | Original training experience of A and B weights |
| Historical split index test | 5,025 | — | — | Original index; cannot substitute for the DocPPL-aligned version below |
| Historical DocPPL test | 4,966 | 148,836 | 3,284,061 | A |
| reserved-clean-v1 dev | 1,000 | — | — | Training validation for C |
| reserved-clean-v1 full test | 5,000 | 129,085 | 3,068,713 | Independent DocPPL for B and C |
| dedup train | 1,466,132 | — | See training D of each representation in §3 | C |

The clean terminal base array has 3,073,713 tokens; after deducting one context-only BOS per document, the scored count is 3,068,713, with EOS scored. Neither `test_docppl_1000` nor smoke can substitute for the full 5,000 documents.
5,025 → 4,966 also involves changes in assembly scope, boundaries, and body-text version, and should not be summarized as merely deleting 59 documents.

C's frozen training manifest records: 89 shards with file-name year `<2023` were used, from `CC-MAIN-2013-20` to `CC-MAIN-2022-49`, using new parsing/encoding artifacts (source receipt protocol is `explicit-bos-eos-ptb-mapping-v1`). This is a fixed assembly of the dedup data pool and **cannot be directly described as deleting duplicates from the old 10.061B-token array while keeping all other conditions unchanged**. This check did not confirm that the new array is a byte-for-byte subset of the old array. The ordinary historical 89-shard `assemble` also does not automatically generate this dedup train.

### 2.2 Scope of evidence for contamination conclusions

A complete token exact scan of the specified historical train found: among A's 4,966 documents, **4,752 have complete training copies (95.6907%)**, corresponding to 91,394 training document instances and 91,631 matched pairs. Therefore A's old BBC DocPPL cannot serve as independent held-out generalization evidence; this does not automatically prove that SG, BLiMP, XSum, and BoolQ are all subject to the same kind of contamination.
Definitions and training hashes are in [historical overlap](../../reproducibility/evidence/historical_overlap.json).

Clean filtering uses exact, 5-gram near, and 13-gram local reuse rules relative to the fixed historical train. In independent re-checking of the full test, normalized exact and the checked near thresholds (0.8 / 0.7 / 0.6) had 0 hits; however, 2,290 documents share 13-grams, and when templates are not masked, 41 documents have local coverage ≥30%. After processing according to the frozen template masking and reference-set rules, this threshold has 0 hits, so one should write "passes the specified exclusion rules" and not "zero text overlap." See [clean overlap](../../reproducibility/evidence/clean_overlap.json) and [full rules](../protocols/bbc_data.md#clean-audit).

**An independent complete overlap audit of the new dedup train against clean dev/test was not redone in this check.** Config, transfer hash, and evaluation identity checks do not substitute for this audit, nor do they support claiming that C is zero-overlap data that has passed all contamination checks.

## 3. Pretraining Standards: Both 500M Does Not Mean the Same Training Budget

The shared backbone of old and new is 16 layers, hidden size 1408, 16 heads, MLP ratio 8, RoPE/RMSNorm/SwiGLU, seed 6198, AdamW, cosine, warmup 2000, training for 1 epoch. C's non-embedding parameter count is 507,896,576; total parameter count is 578,747,136. "500M" follows the non-embedding size convention. The same epoch does not mean the same token count, update count, or training compute.

### 3.1 Old four-model training config for A/B

The table below takes the [historical config manifest](../../train_configs/paper_pretraining_manifest.json) and fixed YAML; the Tree/TG counts for D retain the approximate values from the original registration and do not purport to be a complete scan from this check.

| Model | Training representation / D | Peak LR | Global batch | Microbatch | Checkpoint step | Attention |
|---|---|---:|---:|---:|---:|---|
| Terminal | terminal / 10.061B | 0.001308 | 144 | 16 | 34115 | causal Flash |
| Tree | LIN1 / 24.706B | 0.001723 | 244 | 16 | 49440 | causal Flash |
| TGTree | LIN2 / 32.029B | 0.001866 | 280 | 16 | 55853 | causal Flash |
| TGNomask-Aug | LIN2 / 32.029B | 0.001866 | 280 | 12 | 55853 | Nomask-Aug / Flex |

The original config grammar key of the old TGTree is `tree`, but the input is LIN2/TG; it therefore cannot be treated as Tree weights. Runtime checking must verify model identity and input format. The old four-model config uses DDP/BF16; the global/device batch ratios from the source configs are Terminal/Tree=4 and TGTree/TGNomask-Aug=8, and cannot be uniformly described as the old models all using 4 GPUs; these config fields also do not substitute for the actual training-resource receipts. C's original receipt explicitly states 2 GPUs.
B did not retrain these weights.

### 3.2 Ten-model training config for C

All models use 2×RTX 5090 with DDP/BF16 on scvi5090; the final microbatch is 4 for all. LR/global batch are recomputed based on the new representation length D; full precision follows the [frozen configs](../../reproducibility/configs/dedup500m/manifest.json). The LRs below are only rounded for display.

| Model | Training D (including structural or online SEP) | Peak LR | Global batch | Sequence length | Completed step | Attention |
|---|---:|---:|---:|---:|---:|---|
| Terminal | 800,848,405 | 0.000601208621 | 34 | 2048 | 11501 | causal Flash |
| Tree | 1,963,388,153 | 0.000791750134 | 58 | 2048 | 16529 | causal Flash |
| TGTree | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | causal Flash |
| TG | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| TGNomask | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| TGTree-Mix-TG | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| TGNomask-Mix-TG | 2,544,658,027 | 0.000857360531 | 66 | 2048 | 18825 | typed |
| Tree-NoONT | 1,382,118,279 | 0.000710857635 | 46 | 2048 | 14670 | causal Flash |
| Pause-1 | 1,601,696,810 | 0.000743774711 | 52 | 2048 | 15039 | causal Flash |
| Pause-2 | 2,402,545,215 | 0.000842367155 | 64 | 2049 | 18321 | causal Flash |

The underlying training files of Pause are still the 800,848,405-token terminal array, with D expanded by 2× / 3× for online SEP respectively; these SEPs cannot be treated as new body text. TGNomask-Mix-TG has 8 TG + 8 TGNomask heads; TGTree-Mix-TG has 8 TGTree + 8 TG heads; the order is in the original configs.

The seven models with fixed m=4 (Terminal/Tree/TGTree/TG/TGNomask/TGNomask-Mix-TG/Tree-NoONT) did not undergo maximum microbatch search. Pause-1/2 and TGTree-Mix-TG have search and confirmation; the latter uses the fixed-and-resubmitted `retry-20260921`, Job 249348, as the formal completed version, and the original Job 248318 failed before formal training. Per-GPU batch need not divide 4 exactly, e.g., Terminal is 4×4+1 and TG series is 4×8+1; the actual accumulation records prevail. Hardware adjustments must not be written as all models having completed the same probe.

C's training `lm` evaluator every 3000 steps computes loss on its respective representation. The final clean-test **LM PPL** for Terminal / Tree / TGTree is 14.36448 / 3.02325 / 2.35796; for Pause-1/2 it is 3.82742 / 2.46185, including SEP targets. **These values are not the DocPPL in §6 and cannot be ranked across representations**. In the new results page, 14.028763 / 13.863770 / 13.946850 are the independent clean DocPPL of the first three models.

## 4. DocPPL Scoring Standards: Test Set and Algorithm Change Simultaneously

| Checked item | A: Registered contract of old results | B/C: clean contract |
|---|---|---|
| Body text and denominator | 4,966 documents, 3,284,061 scored tokens | 5,000 documents, 3,068,713 scored tokens |
| Terminal / Pause | Single-path terminal/EOS; Pause insertion positions are not scored | Same kind of single path; dedicated SEP, document-global phase |
| Tree/TG per-sentence probability | Historical 300 proposal/slotted truncated sum; complete candidate uniqueness evidence for old batches not filled in this check | Sum only over unique valid candidates' joint probabilities; padding slots do not contribute mass; not divided by K |
| Historical subsequent sentences | Old Tree/TG actual records are candidate-0; cannot be relabeled as model-best according to the body text | After scoring all valid candidates for the sentence, take the item with the minimum joint NLL; on ties take the earliest |
| Values and cache | Follow the config/code frozen at that time; current implementation cannot be applied | DocPPL FP32/SDPA, TF32 disabled; common padding length within a sentence, commit only the winning real length; clear cache on new document |
| Final validation | Historical completion records and protocol are each retained | All documents, sentences, tokens, finite NLL, candidate/history counts, and fingerprint; cannot rely only on exit code |

Clean Tree/TG has 38,725,500 physical slots and **37,524,534 valid candidates**: 122,338 sentences have 300, and 6,747 sentences have 122. TG is a conversion that duplicates the corresponding Tree closing token in place, and the candidate axes correspond one-to-one. The parser proposal score is responsible for generation and sorting, and is not used as an additional weight in the LM joint sum.

For structural models, the current sentence score is `-log(sum(exp(-joint_NLL_k)))`, and the NLL of the full corpus is summed, divided by the total number of body-text terminal/EOS tokens, and then exponentiated. The scored items of joint are defined by the model's label mask; the denominator does not count structural or Pause positions. It is not the average of per-document PPL, nor a uniform mixture divided by the number of candidates.
Truncation only guarantees a lower bound on the current sentence probability **under the fixed selected history**; here the full tree history of previous context is not marginalized, so it cannot be called an exact document marginal. The definition is in [Evaluation §4](../guides/evaluation.md#4-current-document-ppl-contract).

Tree-Shuffle uses the terminal-only protocol of the masked checkpoint. B's Pushdown uses native n-ary / v1 `stack_legal` / model-best, with 34,991,281 independent valid candidates in total; its token-only, uniform mixture, binary, or v2 are not the main values of this report. C does not include Tree-Shuffle or Pushdown.
These two types of results can only be noted as protocol-distinct in parallel and are not included in the same controlled Tree/TG ranking.

The frozen manuscript body `sec:doc-ppl` says evaluated-model argmax, appendix `sec:ppl-protocol-details` says parser-selected history, and Pushdown says candidate-0; these cannot be mixed with A's actual records and B/C's current contracts. Subsequent manuscript revision needs to modify the method, appendix, and table notes together; this check only records the differences.

## 5. Path Verification: Data Sources, Deployment Paths, Weights, and Results Kept Separate

All relative paths start from the repository root. The local/remote paths below are for source tracing and **are not public download entry points**. Changing deployment paths cannot change data identity; identical file names cannot prove the same experiment.

### 5.1 Data and campaign paths

| Use | Experiment source path / run path | Verification notes |
|---|---|---|
| A/B original training | `dataset/bbc-news/{terminal,tree,tg}/train.npy`; old YAML uses `/dev/shm/dataset/bbc-news/` | Historical training data, not dedup train |
| A DocPPL | `dataset/bbc-news/terminal/`, `testppl_tree/`, `testppl_tg/` | Old 4,966 documents; old structural run registered under `artifacts/evaluation/docppl_structural_20260825/`, and the complete directory of this campaign is not on this local machine |
| B/C clean DocPPL | `dataset/bbc-news-reserved-clean-v1/evaluation_terminal/`, `testppl/{tree300,tg300}/` | Terminal input and sentence/document indices derived from canonical; structural entry is not the single-tree `tree/test.npy` used for training |
| C train true source | `dataset/bbc-news-parsed-dedup/{terminal,tree,tg,tree_noont}/train.npy` | Corresponds to the frozen train manifest |
| C 5090 deployment | `/data/run01/scvi351/TG-Interpolation/dataset/bbc-news-dedup-clean-v1/{format}/` | train maps to parsed-dedup, dev/test maps to reserved-clean; not a fourth dataset |
| C NoONT dev/test | `dataset/bbc-news-parsed-dedup/tree_noont/{dev,test}.npy` | manifest proves source is reserved-clean Tree with ONT removed; directory name does not mean the old test is used |
| B original scoring campaign | `artifacts/evaluation/reserved_docppl_sist_20260907/` | SIST run root is `/inspurfs/group/tukw/wangpch/docppl_reserved_20260907/` |
| B grammar fixes | `grammar_fix_20260909/` under the above directory | Three models use correct grammar and original weights; original erroneous values permanently excluded |
| C 5090 training | `validation/pretrain_5090_dedup_20260915/`, `pretrain_5090_dedup_structural_20260919/`, `pretrain_5090_dedup_pause_mix_20260920/` | Remote prefix is `/data/run01/scvi351/TG-Interpolation/validation/`; Mix uses retry subdirectory |
| C evaluation | `artifacts/evaluation/bbc_dedup_500m_{local_20260919,pause_noont_20260922,sist_20260922}/` | First two batches RTX3090B, third batch SIST; public copies are in `reproducibility/results/dedup500m/` |

C's SIST deployment root is `/inspurfs/group/tukw/wangpch/bbc_dedup_500m_eval_20260922/repo/`.
The four models' syntax/fine-tuning use BF16/SDPA, DDP full replicas, with typed/Flex kernels disabled during evaluation; DocPPL remains FP32/SDPA. This differs from the typed/Flash backend of 5090 pretraining; the execution location or backend must not be treated as different training data, nor is it claimed that the two rounds of scorer binaries are completely identical.
Among the saved Tree scoring contracts, B's scorer hash is `c1c545738423d3301cf3503725612d5cb117188a8a6c806f7cd5e13d231fe1ac`, and C's is `7dcfee0eef1cd4f059204a9aca75bdfc553ed5306a5f2766553c43cfb56a86be`.
Agreement in protocol fields such as history/aggregation/precision/attention does not mean the scoring source code is byte-identical; reproduction must select the corresponding snapshot.

### 5.2 Checkpoint paths and missing models

The old 500M paths for A/B are respectively:

| Model | Old checkpoint (repository-relative path) | Fixed config |
|---|---|---|
| Terminal | `saved_models/terminal_500M/step34115-unsharded` | [terminal](../../train_configs/paper_sources/bbc_500m_terminal.yaml) |
| Tree | `saved_models/Tree_500M/step49440-unsharded` | [tree](../../train_configs/paper_sources/bbc_500m_tree.yaml) |
| TGTree | `saved_models/TGTree_500M/step55853-unsharded` | [tgtree](../../train_configs/paper_sources/bbc_500m_tgtree.yaml) |
| TGNomask-Aug | `saved_models/TGnomaskaug_500M/step55853-unsharded` | [tgnomask_aug](../../train_configs/paper_sources/bbc_500m_tgnomask_aug.yaml) |

C's unified local root is `saved_models/bbc_dedup_500m_scvi5090/`:

| Model key | Final checkpoint under root | Original evaluation batch |
|---|---|---|
| `terminal` | `terminal/step11501-unsharded` | `bbc_dedup_500m_local_20260919` |
| `tree` | `tree/step16529-unsharded` | `bbc_dedup_500m_local_20260919` |
| `tgtree` | `tgtree/step18825-unsharded` | `bbc_dedup_500m_local_20260919` |
| `tg` | `tg/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tgnomask` | `tgnomask/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tgtree_mix_tg` | `tgtree_mix_tg/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tgnomask_mix_tg` | `tgnomask_mix_tg/step18825-unsharded` | `bbc_dedup_500m_sist_20260922` |
| `tree_noont` | `tree_noont/step14670-unsharded` | `bbc_dedup_500m_pause_noont_20260922` |
| `pause1` | `pause1/step15039-unsharded` | `bbc_dedup_500m_pause_noont_20260922` |
| `pause2` | `pause2/step18321-unsharded` | `bbc_dedup_500m_pause_noont_20260922` |

`bbc_dedup_500m_scvi5090_20260919` / `..._20260920` are compatibility links pointing to the unified directory; they must not count as multiple independent sets of weights. The two contracts of group C's Tree DocPPL produced different fingerprints due to path migration; the original strict collection has verified that weights, data, scorer, and parameters are consistent; this special case does not authorize mixing other inconsistent contracts.
This check confirmed that the weight paths of the old four models exist, that the paths and sizes of the 30 model/config/train files of the new ten models match the manifest, and recomputed the hashes of ten small configs; **large weight hashes were not recomputed**.
The new local copy did not recover `optim.pt`; it can be used for evaluation, but cannot be relied upon to promise complete AdamW continued training.

### 5.3 Data identity anchors (SHA-256)

The following are from comparison with original manifests/precheck records; large arrays were not rescanned in this check. The complete candidate arrays and index identities are bound by their respective manifests and final receipts.

| File / content | SHA-256 |
|---|---|
| New train: terminal | `b589676a1f3d5b485de17ee7715acf5d3d8f6d03a40188199bfa7c563345b427` |
| New train: tree | `792ae397857fb7b9ff9266ba147e11edb7e27bccd5fa780287e93b67d23b660a` |
| New train: tg | `370df2b8bc49be2b6cbd21d20cdbe3bc751c8dbe1c5a19f07d2b47f12c016970` |
| New train: tree_noont | `55bc4ff70abf87e9ad495c6e865cbeba3ab09a0a1f0d05432ca2e5c5c0d3a7d9` |
| clean `terminal/test.npy` | `fd95d40241cb13fbc84c7bee01f645847c520d1bd46eb0697d8f40569bd9c5c0` |
| clean `canonical/test/manifest.json` | `8995b8e031d69c821ade432678b14b587aae5e2f2ef7914594e3b5ae6f866fe9` |
| clean `tree300/tree_300.npy` | `25241ac512ea9eb419bc1debcd299a6dc623206b82b13f275ccb99160489f403` |
| clean Tree/TG `valid_counts.npy` | `67efabdae9e3d0a41a918de08f3b7290b7cdb6529d2a7f09c0c503fde9f7cc6e` |

The hashes of the 9 clean base files jointly covered by B and C local/SIST input records are consistent; the records of the 19 clean base/candidate/index files of C local and SIST are also consistent. B's candidate audit and scoring contract and C's candidate manifest point to the same frozen clean candidate artifact; B's base input list itself does not include candidate array SHAs, so this check cannot exaggerate the 9-item comparison into a re-verification of all of B's large candidate files.

The BBC runtime tokenizer hash is `94a26c15b2edbe45b08c17e20a2f5ad9485a268a7b3ac477a5bf9290ef7f0d46`; the byte hash of some train manifests / the [public reference JSON](../../reproducibility/tokenizer-reference.json) is `7fc243801e0d0551948380e59031976ed1eed89bd0efe867fd8a70cfadc19cff`.
The original protocol has checked full JSON equivalence, so one should write "bytes differ, JSON equivalent" and not "byte hash completely identical." This check also recomputed the small-file hash of the local runtime tokenizer and confirmed that its full JSON is equal to the public reference.

## 6. Result Differences and Usable Conclusions

2026-10-12 review note: the table below retains the verification-date snapshot. On 2026-10-07 it was confirmed that B's NoONT/Compress and C's NoONT DocPPL lack post-projection deduplication; these three old values must not be used for current conclusions, and the differences and rankings involving them are also invalid. The historical completion receipts in §7 cannot certify newly added requirements; see [current protocol](../protocols/clean_document_ppl.md). The public original JSONs retain historical identity and are not rewritten as new results.

The following is a fixed snapshot from the verification date. A uses the frozen-manuscript display precision, and B/C display six decimal places; A's rounded values must not be treated as exact original log values. `—` means that group has no corresponding model result, not zero, failure, or a pending task.

### 6.1 100M: DocPPL of A and B

| Model | A: Old test | B: Old weights clean | Interpretation scope |
|---|---:|---:|---|
| Terminal | 9.89 | 15.369225 | Single path |
| Tree | 10.01 | 14.736097 | CRF valid candidates |
| TGTree | 10.21 | 14.709097 | CRF valid candidates |
| TG | 11.02 | 16.405236 | CRF valid candidates |
| TGNomask | 9.97 | 14.658209 | CRF valid candidates |
| TGNomask-Aug | 10.08 | 14.700370 | CRF valid candidates |
| Tree-NoONT | 9.95 | 15.061134 | Historical grammar-fix value; lacks projection deduplication, withdrawn |
| Tree-Compress | 9.91 | 14.730862 | Historical grammar-fix value; lacks projection deduplication, withdrawn |
| Tree-TripleCNT | 10.26 | 14.801293 | Correct grammar-fix value |
| Tree-Shuffle | 13.41 | 17.852117 | Independent protocol, background only |
| TGNomask-Mix-TG | 10.01 | 14.572762 | CRF valid candidates |
| TGTree-Mix-TG | 10.00 | 14.481718 | CRF valid candidates |
| Pause-1 | 9.77 | 14.916121 | Single path |
| Pause-2 | 10.32 | 15.116468 | Single path |
| Pushdown | 13.29 | 19.316305 | Independent protocol, background only |

For 100M, Tree relative to Terminal changes from A's higher by 0.12 to B's lower by 0.633128; in B, the two Mix values are lower than Tree/TGTree. Among causal linearization variants, A's Compress is lowest and B's TGTree is lowest; NoONT relative to Tree changes from slightly lower in A to higher by 0.325036 in B. The erroneous-grammar clean results of old NoONT/Compress/TripleCNT did not enter this table and cannot be used for before/after ranking.

### 6.2 500M: A, B, and C must be listed separately

| Model | A: Old test | B: Old weights clean | C: dedup new weights clean | C−B (same-name models) |
|---|---:|---:|---:|---:|
| Terminal | 2.83 | 17.302676 | 14.028763 | -3.273913 |
| Tree | 3.18 | 16.001136 | 13.863770 | -2.137366 |
| TGTree | 3.28 | 15.894648 | 13.946850 | -1.947798 |
| TGNomask-Aug | 3.19 | 15.483865 | — | — |
| TG | — | — | 15.465373 | — |
| TGNomask | — | — | 13.995203 | — |
| TGTree-Mix-TG | — | — | 13.784714 | — |
| TGNomask-Mix-TG | — | — | 13.972621 | — |
| Tree-NoONT | — | — | 13.938351 | — |
| Pause-1 | — | — | 14.019490 | — |
| Pause-2 | — | — | 14.030113 | — |

Among A's old four models, Terminal is lowest; among B's old four models, TGNomask-Aug is lowest. Among C's ten models, TGTree-Mix-TG has the lowest value, followed by Tree; if looking only at the three models shared by A/C, C's Tree is lowest. TGTree-Mix-TG is lower than Tree by 0.079055, and TGNomask-Mix-TG is lower than TGNomask by 0.022583.
These close values have no significance test and should not be written as stable rankings or universal benefits.

The three C−B items are cross-checkpoint descriptive differences under the same clean data and the same kind of scoring contract and cannot be interpreted as separate dedup effects: training data, representation generation, D, LR, batch, steps, and runtime source/backend may all participate.
B's old 500M has higher clean PPL than the corresponding old 100M; C's 500M also cannot be combined with historical 100M into the same controlled scaling curve. To verify a new scaling conclusion, a 100M control matching the new training/evaluation contract is needed.

### 6.3 Downstream changes of the three shared 500M models

The table below writes A → C. SG/BLiMP are percentages; XSum is R-AVG percentage; the ± for XSum/BoolQ is the sample standard deviation over five **fine-tuning** seeds. B only re-tested DocPPL and did not generate a new set of SG/BLiMP/XSum/BoolQ.

| Model | SG ↑ | BLiMP ↑ | XSum ↑ | BoolQ ↑ |
|---|---|---|---|---|
| Terminal | 71.11 → 80.8848 | 64.74 → 72.3149 | 20.75 ± 0.06 → 23.1149 ± 0.0735 | 69.69 ± 0.60 → 70.5382 ± 0.5065 |
| Tree | 63.13 → 83.4316 | 76.09 → 82.7552 | 22.25 ± 0.05 → 23.5731 ± 0.0649 | 70.72 ± 0.35 → 70.0489 ± 0.5797 |
| TGTree | 65.95 → 80.6760 | 77.65 → 83.7209 | 22.01 ± 0.08 → 23.5886 ± 0.0503 | 70.63 ± 0.32 → 68.4281 ± 0.2609 |

The complete downstream values, per-seed values, and sub-items of the new ten models follow the [existing results page](bbc_dedup_500m.md). Within C, TGTree has the highest XSum/BLiMP values, Tree-NoONT has the highest SG, and Terminal has the highest BoolQ; one cannot infer from DocPPL that all tasks improve synchronously. In particular, BoolQ does not rise for all shared models.

The old fine-tuning configs of the three shared models and the new protocol both use XSum 3 epoch / LR 6e-5, BoolQ 5 epoch / LR 3e-4, global batch 40, five seeds (42/2026/6198/13171/31723); XSum test=11,333, BoolQ validation=3,270. The old campaign and C records agree on XSum train/test, BoolQ train/validation, and runtime tokenizer hash. SG has 32 items in six categories, and BLiMP has 67×1,000 pairs; structural models retain the task distinction of beam300 / supplied K=300. These shared conventions improve the interpretability of comparison, but the old and new fine-tuning/evaluation run organization, microbatch, and source identity differ, and this check did not rerun per-sample to prove numerical equivalence.

2026-10-02 supplementary check of SG's actual constraints: Tree/TG scoring branch sub-beam=300, terminal fast-track=30, and the total linearized length upper bound is `max(6L,10)`. Although the entry passes `nc=max(L,5)`, `pc=3`, the code that consumes `Stop_Add_NT` and masks opening-NT logits is commented out, so these two NT count limits do not take effect. One must not infer from configs or arguments that the limits are enabled; the paper's SG appendix description is consistent with the code, see [Evaluation Section 5](../guides/evaluation.md#5-sg-and-blimp). This correction does not change the registered evaluation values.

<a id="completion-status"></a>
## 7. Completion Status, Evidence Credibility, and Unresolved Items

This section registers receipts under the protocol at that time. The new projection deduplication requirement for NoONT/Compress is in the §6 review note; "22 valid models" and "three batches completed" cannot be interpreted as the affected DocPPL having passed the current protocol.

| Evidence | This verification | Scope that can be claimed |
|---|---|---|
| A frozen manuscript and JSON | Source SHA consistent with extraction manifest, 19-row frozen table | Numerical transcription is credible; old data contamination and inconsistency between method text and actual remain limiting interpretation |
| B valid 22-model summary | Consistent with original results/grammar-fix results; per-row complete coverage of fields; NLL recomputation PPL consistent; weight hashes aligned with original provenance | Registered valid values may be used with scoring protocol noted; not a rerun of the 22 models in this check |
| B original overall final validation | Original `final_status.json` remains `failed`; `identity_validation.json` explicitly certifies identity only, not full coverage | Cannot claim "the original overall final validation of the old 22 models succeeded"; the boundary of missing successful overall receipt remains |
| B three grammar fixes | 2026-09-11, Job 999568, scoped final receipt is `complete` | Certifies only the three fixed models NoONT/Compress/TripleCNT, and cannot be expanded to re-final-validation of all 22 models |
| C three batches of evaluation | 66/66, 66/66, 88/88, total 220/220; three strict summaries and final validations all `complete` | Each of the ten models has 22/22 tasks completed, not queued, probe, or partial evaluation |
| C post-evaluation identity check | 2026-09-29 original receipt records input 77/77/72 files, checkpoint 9/9/12 file hashes | This final validation of SIST used a local synchronized mirror; it is not a rescan of remote SIST storage |
| Public reproduction | Code, frozen configs, and selected results are already in the repository; download entry points for large data/weights/per-document logs are pending | Reviewable protocols and identity summaries exist; complete external clean-environment retraining is not done |

C's 22 items are, per model, 2 syntax tasks, 10 fine-tuning and their evaluation tasks, and 10 DocPPL document shards; they are not 22 metrics or 22 training seeds. The `evidence_validation` in the three batches of result JSON writes `no fresh checkpoint rehash`, which refers to the **collector stage**; the separate final receipt at 08:55 UTC afterward completed the recorded file recomputation. The two stages have different responsibilities; reading only the early fields cannot conclude that there was no final validation at the end.
The public [final status](../../reproducibility/results/dedup500m/final_status.json) retains this scope.

The failure of B's original overall receipt and the success of the three fixes were not rewritten or concealed. The 22 valid result rows used in this report each have their own coverage and identity records; if subsequent manuscript revision requires the stronger statement "all old models passed the same successful overall final validation," the corresponding certification should be completed separately, and the arithmetic/source check in this report cannot substitute for per-document final validation.

The historical 91-document structural change was reviewed on 2026-10-02 and still lacks the parser weights, configs, and candidate scores of that time; R-03 has been closed per user decision as "cause unknown, stop investigating," and the historical interpretation limitations remain, see [data provenance](../protocols/bbc_data.md#provenance). Remaining evidence gaps include: exact-byte reconstruction of the historical training array is unproven; an independent complete overlap audit of new train/clean is not filled; public access locations for large arrays, weights, and complete logs are not filled. These gaps do not mean C's 220 evaluations are incomplete.

## 8. Usage Rules for Subsequent Paper Revision and Reproduction Plans

### 8.1 Correspondences to handle synchronously in subsequent manuscript revision

| Old argument / table location | Basis for handling supported by this report |
|---|---|
| BBC DocPPL main table | When choosing B or C, clearly state training/test source; retain old data identity traceability. C's ten rows cannot pretend to still be the old four rows; added/missing models must be stated explicitly |
| Tree brings small LM cost, 100M Pause-1 lowest | Under B this ordering changes; rewrite according to the selected new table, and do not keep old conclusions by replacing numbers only |
| Mixed attention brings no benefit | B's 100M and C's 500M have lower DocPPL; change to observations by task and training condition, and do not derive comprehensive or significant benefits |
| As BBC scale increases, in-domain PPL improves while syntax declines | A's old PPL has training overlap; B/C cannot support the same unqualified inference. New scaling claims require controls matching data and protocol |
| DocPPL body, appendix, and table notes | Unify valid K, joint sum, model-best, terminal/EOS denominator; retain Pushdown/Shuffle protocol distinctions; no longer relabel A with the current contract |
| SG/BLiMP/XSum/BoolQ and efficiency conclusions | B did not re-test these tasks; C's shared rows can explicitly replace with independent measurements from new training. Results on old efficiency, FineWeb, or other weights cannot be relabeled together with BBC PPL |

Five seeds cover only fine-tuning randomness; DocPPL, SG, and BLiMP are single evaluations. No significance test was done; fine-tuning SD cannot be treated as a DocPPL confidence interval, nor can a single pretraining seed be written as evidence of training stability.

### 8.2 For reproduction, choose the target first, then generate a new run

| Target | Standards and entry points that need to be fixed | Current limitations |
|---|---|---|
| Reproduce A's historical values | Explicitly choose historical train/test, old weights, and the scorer/candidate-0 of that time; [historical config generator](../guides/historical_pretraining.md) is only responsible for the training recipe | The current model-best evaluator is not equal to the historical scorer; some run environments, parser identity, and historical bytes still lack evidence |
| Reproduce B's clean re-test | Original old weights + reserved-clean-v1 full test + correct grammar; [clean evaluation protocol](../protocols/clean_document_ppl.md), independent output, strict merging, and final validation | Access to external weights/data/per-document evidence is pending; need to distinguish per-row results from scoped overall receipts |
| Reproduce C's new ten models | New dedup/clean identity + ten-model frozen configs + [pretraining workflow](../guides/pretraining.md) + same-kind task protocols | The public dedup CLI covers eight recipes but the set is not equal to this batch of ten models; Pause-1/2 and TGTree-Mix-TG need new runs generated according to the corresponding frozen recipes |
| Separate the contributions of data and scoring changes | Fix the same checkpoint, switch test and history/candidate rules separately, and establish paired controls with explicit contracts | This report did not run these controls; existing A/B differences cannot substitute |
| Establish new causal arguments for scaling or dedup | Match training source, representation, token/update budget, and scoring contract, and clarify whether LR/batch are controlled; add corresponding-scale or training-seed controls | This is subsequent experimental design, not an acceptance condition of this report, and no task has been submitted |

New reproduction must create new run/save/result directories, save data/tokenizer/config/checkpoint/source identity and actual environment/extensions; old validation/campaign paths must not be replayed directly or existing results overwritten.
This R-08 wrap-up only completes the differentiation and anti-misleading explanation; paper revision and possible new reproduction experiments are handled by subsequent tasks.

## 9. Evidence and Verification of This Check

The starting commit of the check is `d3df67d08bfdb677fdf0c14fa17ea498d31dc58e`. The paper source hash is `5b4e17f515b8dbbc0638a3b6e6176c71a77dad1842e81753381eeab558935481`, consistent with the frozen table source; this file was not changed in this check. Evidence content identity is in the [extraction manifest](../../reproducibility/evidence/manifest.json), [new config manifest](../../reproducibility/configs/dedup500m/manifest.json), and [new results final validation](../../reproducibility/results/dedup500m/final_status.json).

Local supplementary check sources (not public download links):

- R-08/R-11 and original experiment registration in `EXPERIMENT_REPRODUCTION_RECORD.md`, `REPOSITORY_CLEANUP_MEMORY.md`.
- `artifacts/evaluation/reserved_docppl_sist_20260907/{results,provenance,input_validation,identity_validation,final_status}.json`; `grammar_fix_20260909/{results,final_status}.json` and three scoring contracts.
- `protocol.json`, `tasks.json`, `input_validation.json`, `results.json`, `final_status.json` of the three C campaigns; protocol/receipt/data manifest of the three training batches `validation/pretrain_5090_dedup*/`.
- `dataset/bbc-news-parsed-dedup/{format}/train.manifest.json`, NoONT conversion manifest, clean top-300 manifest.
- `artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/run_manifest.json` and fine-tuning configs of the three shared models.

Programmatic checks passed in this check: hashes of 5 public evidence files and 8 local extraction sources, 29 historical fixed configs, 22 B model NLL/PPL against original sources (including 3 fixes), 10 C config hashes and 30 checkpoint file paths/sizes, three batches of public/original results and final-validation hashes, 220 item counts, five-seed means/sample SD, full-evaluation markers, 9 B/C clean base identities, 19 C local/SIST clean identities, 4 train manifest identities. In addition, NPY headers of 4 new train and 3 clean base test files were read, with length/dtype consistent with manifest; runtime tokenizer small-file hash and full JSON equivalence check passed. The 49 rows of the five comparison tables were extracted from the same JSON/configs, and display rounding and differences were verified.

Document delivery checks use `python scripts/check_public_release.py` and `git diff --check`.
The above verification consists of local source, config, arithmetic, and document checks; large weights/large arrays were not re-hashed, GPUs were not rerun, full overlap audit was not executed, and external clean-environment reproduction was not completed.
