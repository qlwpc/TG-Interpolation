# FineWeb-Edu data identity and known issues

This page describes the locally processed `fineweb-edu-v2` corpus and subsequent
October 2026 repairs. It is not an audit of the entire upstream FineWeb-Edu
release, and these repaired arrays and the filtered 20B subset must not be
substituted for the historical paper inputs without declaring a new experiment.
The paper configuration identities remain in the
[historical manifest](../../train_configs/paper_pretraining_manifest.json).

**FWE-014 remains confirmed and unrepaired as of 2026-10-15:** literal `ADJ`
constituent markers occur in both the repaired source and the 20B training subset.
Passing projection and registered-label checks does not certify that all
constituents were encoded correctly. No full scan of either corpus, or targeted
scan of the two validation splits, has established the prevalence of this issue.

## Corpus identity

The historical source registration is `HuggingFaceFW/fineweb-edu:sample-100BT`.
The local name `v2` is a processing-directory name, not an upstream revision.
There are 246 paired terminal/Tree shards. Corpus-wide provenance has not been
verified against upstream document IDs and revisions. The source arrays lack
separate URL, education-score and original-document-ID sidecars.

| Snapshot | Terminal tokens | Tree tokens | Boundary convention |
|---|---:|---:|---|
| Before the October repair | 98,880,709,238 | 233,810,041,706 | Historical EOS-only streams; internal EOS may occur |
| After bracket/internal-EOS repair, 2026-10-13 | 98,708,787,845 | 233,638,120,313 | 93,584,610 record-ending segments; no initial BOS |

These files have `.npy` names but are **headerless little-endian uint32**.
Read them with `np.memmap(path, dtype='<u4', mode='r')`, not `np.load`.
EOS is 151643. The Qwen tokenizer SHA-256 is
`827837c463999c3e647b1eeb1b7f056744c56ddf374247e9cf0bab9eb8afe9cb`.
Newly built standard NPY files have a different storage contract; inspect the
manifest and header instead of inferring the format from the suffix.

Historical local locations are provenance references, not public downloads:
`/public/home/wangpch/SIST-assets/datasets/fineweb-edu-v2` and
`artifacts/analysis/fineweb_edu_scope_20261012/`. Selected receipt fields and their
source hashes are published in the [audit evidence](../../reproducibility/evidence/fineweb_data_audit.json).
Full arrays, tokenizer, reversible deltas and raw audit logs still need an
external release location; see [asset availability](../../reproducibility/README.md#assets).

## Sampled quality audit

The 2026-10-12 audit examined 7,872 EOS-delimited samples across 246 shards,
containing 36,464,416 source Qwen tokens. Sampling was stratified by token
position, so longer texts were more likely to be selected. Counts below are
sample observations, not estimates of full-corpus document prevalence.
The original formula filter classified 7,733 as keep, 101 as drop and 38 as review.

| ID | Finding | Evidence and status |
|---|---|---|
| FWE-001 | PTB bracket escapes remained inside words | 2,502 samples contained 34,578 markers; 2,176 had embedded markers. Ordinary BPE encoded the escapes. The shared converter and all 246 source pairs were repaired; see the [repair protocol](fineweb_bracket_repair.md). |
| FWE-002 | Structural AddedToken word-boundary restrictions | The frozen tokenizer encoded the closing marker in `A<NP)>` as ordinary text. The current converter inserts registered structural IDs explicitly. Negative decoded-text checks alone do not certify historical alignment. |
| FWE-003 | Mojibake | 13 sampled hits, all originally kept. Examples include `WÃ¼ster` and damaged quotation marks. The corruption stage is unknown; no full-document automatic recoding was applied. |
| FWE-004 | Uncorrected extraction or layout disorder | Four explicit extraction notices, all kept. Sample `104:69034908` interleaves sentences and splits words. Two-column extraction is a plausible explanation, not a verified source-PDF diagnosis. |
| FWE-005 | Repetition | Samples `065:270803770` and `179:92001306` were exact text/token duplicates of 19,150 tokens each. Another 26 samples had long lines repeated at least three times, including legitimate tables and references. |
| FWE-006 | Web controls, citations and comments in body text | 12 web-control hits, 12 HTML-tag clues, 2 entity clues and 11 wiki/citation clues. At least one medical article includes user comments and advertising-like content. These signals also have legitimate uses. |
| FWE-007 | Tables, lists and article collections | 610 samples had dense pipe separators; 587 were kept. Sample `194:181311765` contains multiple health-news excerpts in 125,110 tokens. An EOS segment need not be a single coherent article. |
| FWE-008 | Parser spacing and historical quotation marks | 7,033 samples had split contractions/possessives; all 7,872 had punctuation spacing; 85 had double-backtick/double-apostrophe clues. Bracket repair does not detokenize these forms. |
| FWE-009 | Formula-filter false negative | `197:290371261` contains `u(x ) = ln(x )` but passes `formula-syntax-v5`. The later conservative sample subset and 20B collection exclude this known case; the original filter output remains historical evidence. |
| FWE-010 | Storage and provenance limits | Raw uint32, EOS-only boundaries and missing upstream metadata prevent direct use of standard NPY readers or token offsets as document provenance. Terminal offsets cannot be applied to Tree. |
| FWE-011 | Internal EOS confused with a record terminator | A literal reserved string in a complete source record encoded as EOS inside a tree. One example was traced to an upstream Parquet record. The later repair deleted 11 internal EOS IDs in both streams; ending EOS IDs were retained. |
| FWE-012 | Historical input counts differ from surviving parsed lines | Groups 000/001/003 have logged input counts 396,000/388,000/394,000 versus parsed counts 313,301/307,451/393,408. Historical filtering, completion and order are not fully authenticated; the difference alone does not prove data loss. |
| FWE-013 | PTB escapes inside lexical AddedTokens | Shard 192 contains IDs 151674/151675 for `<-LRB->`/`<-RRB->`. The repair handles exactly the six bracket forms, preserving angle brackets, and rejects other unrecognized cases. The record and full shard passed recovery checks. |
| FWE-014 | Unnormalized ADJ labels leak into body text | 47 samples across 43 shards contain 54 opening/closing pairs. Confirmed in the current source and 20B train; unrepaired. See below. |

Absence of U+FFFD, selected control characters or registered literal tree markers
in the original sampled scan does not certify full-corpus quality. The marker
pattern omitted `ADJ`, so its earlier zero count is not evidence of no label leakage.

<a id="adj-label-leak"></a>
## ADJ label leakage: confirmed, unrepaired

The complete 7,872-record sample file has SHA-256
`fbf9d8f6d80f008035a6194e0041a561e433199628c50d594a0ccf02a15eb88f`.
The 47 hits had historical decisions keep=43, drop=3 and review=1.
The 5,012-record intermediate snapshot was not used for those counts.

For sample `139:385374804`, the reversible delta maps the old terminal interval
`[385374804, 385376135)` to `[384706045, 384707376)`. The current Tree interval is
`[909166181, 909169158)`. The terminal record SHA-256 remains
`86f8a9c714f75b04cf376e818097cd2cac75cad88d17dbafbfd7a04104098e32`.
The same complete record occurs at index 3033 (zero-based), source ordinal 362814,
in `train/139-0086.npz`; both arrays match the current source bytes exactly.

The tokenizer has no registered IDs for `<(ADJ>` or `<ADJ)>`; they encode as
`[28706, 1808, 41, 29]` and `[27, 1808, 41, 16018]`. Correct ADJP opening/closing
IDs are 151680/151706. Ordinary BPE therefore preserves the leaked labels in both
terminal text and the Tree projection, allowing projection equality and
registered typed-label balance to pass. Collection did not re-encode accepted
records or exclude this defect.

The earlier commit `8c24868644f185c78a60cf6d6120928e468878e1` repaired related
BBC test-candidate leakage; it does not establish a FineWeb pretraining repair.
The current [top-K generator](../../datatools/parse_test_docppl_data/labeled_topk.py),
[BBC clean builder](../../datatools/reserved_clean/build.py) and optional
[historical BBC normalization](../../datatools/parse_test_docppl_data/reproduce_bbc_test.py)
contain ADJ-to-ADJP handling. A future FineWeb repair must normalize constituent
labels and verify leaves and both encodings, rather than replace ordinary `ADJ`
words globally. This is a local representation/encoding defect, not evidence
that upstream article text contained these labels.

## Current tools and limits

The shared [PTB normalization function](../../datatools/ptb_brackets.py) handles
standalone and embedded escapes without adding spaces or changing tree labels.
New tokenization uses `explicit-bos-eos-atomic-tree-ptb-v2` and records the
BBC/PTB or Qwen/native policy. Old receipts are not silently accepted as v2.
This source fix does not retroactively repair every historical array.

Use the [read-only quality audit](../../scripts/campaigns/fineweb_slice_quality_audit.py),
[cleaning diagnostics](fineweb_edu_cleaning.md), [paired repair](fineweb_bracket_repair.md)
and [20B collection protocol](fineweb_clean_20b.md) for their respective scopes.
The October diagnostics and subset construction do not establish training gains,
zero contamination, or reproduction of the paper's FineWeb model results.
