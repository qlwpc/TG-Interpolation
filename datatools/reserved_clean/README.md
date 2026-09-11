# BBC reserved clean test builder

Builds a new corpus in its own directory. Every input scan is read-only and
checks frozen file fingerprints. An interrupted stage has no `complete.json`;
downstream stages refuse to consume it. Use a new output directory for reruns.

The data design is in [the proposal](../../docs/bbc_reserved_clean_test_plan.md).
The implementation run is tracked under
`artifacts/bbc_reserved_clean_20260906/run_status.json`; remote scan logs and
receipts live at
`RTX3090:/home/wangpch/TG-Interpolation/artifacts/reserved_clean_build_20260906/`.

## Stages

1. `python -m datatools.reserved_clean.scan_raw`: extracts 66,628 reserved rows,
   scans the frozen 94 files, byte-verifies raw cache hits and exact candidate
   hits, exports unique historical normalized bodies. Its independent count
   gates are 66,523 reserved strict contents and 16,501 exact candidates.
2. `python -m datatools.reserved_clean.scan_arrays`: scans the existing remote
   historical train directly, records exact hits and unique decoded bodies.
   It does not depend on a reconstructed train membership mapping.
3. `python -m datatools.reserved_clean.filter prepare`: materializes the 16,501
   candidate records, all their source positions, and raw-stage exclusions.
4. `python -m datatools.reserved_clean.filter match`: compiles `match.cpp`,
   streams references across eight exact string inverted-index workers, unions
   their coverage bitmaps (never sums percentages), and then
   checks candidate-to-candidate relationships. Remote references and local
   old dev/test references are audited separately; their coverage bitmaps are
   unioned by the builder, not added as percentages.
5. `python -m datatools.reserved_clean.build build`: propagates contamination
   through content components, checks source/encoding quality, partitions by
   shard and terminal length, and exports aligned Terminal/Tree/TG arrays.
6. `python -m datatools.reserved_clean.validate`: verifies every sentence's
   real parser-word/BPE adapter and scans the final exported dev+test against
   the actual train again. This includes any ADJ→ADJP normalization effects.

`run_after_scan` waits for the two remote scan receipts and runs steps 3–4.
It writes `driver_status.json`, including failures, and does not hide a failed
scan behind an endless wait. The scan process IDs are explicit arguments.

## Frozen matching rules

- Body normalization: NFKC, case folding, registered PTB escapes, whitespace
  and punctuation token spacing. Numbers and punctuation remain. Only this
  audit representation is normalized; scoring tokens retain source text.
- Near duplicates: word/punctuation-token 5-gram Jaccard ≥ 0.8, or shorter
  text containment ≥ 0.8 with at least 50 normalized tokens on both sides and
  at least 20 shared distinct 5-grams. This prevents a five-word footer from
  vetoing a long article through a tiny denominator.
- Local reuse: union the candidate word positions covered by shared 13-grams
  across **all** references; quarantine at ≥30% coverage. High frequency
  candidate 13-grams (DF ≥ max(100, ceil(N/100))) are masked as templates for
  this local criterion only. Exact and near checks still include them.
- Internal groups include exact, near, and ≥30% pairwise local reuse when
  both sides have at least 50 normalized tokens. Components exceeding 100
  candidates are quarantined for review. Each accepted group gets one fixed
  representative; no component can straddle dev/test/reserve.
- Documents shorter than 10 normalized tokens are quarantined because this
  protocol cannot reliably audit their overlap. Empty parse nodes can be
  removed without removing leaves; ADJ maps to ADJP. Other unknown constituent
  labels, embedded special tokens, or leaf/terminal disagreement are rejected.

All shared 5/13-grams are looked up with full string equality; no approximate
retrieval, high-frequency cap on near retrieval, or hash-only match is used.
The regression suite compares near decisions to an independent exhaustive
oracle and covers disjoint partial matches, template masks, internal self
matches, cross-block array boundaries, and contamination propagation.

## Local continuation after remote completion

Copy the remote `candidates/`, `matching/`, and the array/raw completion
receipts and `arrays/strict_matches.jsonl` into the local run directory.
Downloaded receipt-bearing folders must retain all their files so hashes can
be checked. The remote corpus itself need not be copied back.

```bash
python -m datatools.reserved_clean.build local-references \
  --inputs dataset/bbc-news/terminal/dev.npy dataset/bbc-news/terminal/test.npy \
  --output artifacts/bbc_reserved_clean_20260906/local_references

python -m datatools.reserved_clean.filter match \
  --candidates artifacts/bbc_reserved_clean_20260906/candidates \
  --references artifacts/bbc_reserved_clean_20260906/local_references/bodies.jsonl.gz \
  --output artifacts/bbc_reserved_clean_20260906/local_matching

python -m datatools.reserved_clean.build build \
  --candidates artifacts/bbc_reserved_clean_20260906/candidates \
  --matches artifacts/bbc_reserved_clean_20260906/matching artifacts/bbc_reserved_clean_20260906/local_matching \
  --array-matches artifacts/bbc_reserved_clean_20260906/arrays/strict_matches.jsonl \
  --local-references artifacts/bbc_reserved_clean_20260906/local_references/bodies.jsonl.gz \
  --output dataset/bbc-news-reserved-clean-v1

python -m datatools.reserved_clean.validate \
  --dataset dataset/bbc-news-reserved-clean-v1 \
  --expected-train-sha256 8f900f2acd1b19109dab522d74a0d1af35171838fbf547ee569a205d1649ceb1
```

The train array named above has the same full file SHA-256 locally and on
RTX3090. Automatic review initially rejected corpus transfers; the user then
explicitly authorized them ("我允许传递语料"). The workflow keeps local old
dev/test auditing local and downloads the candidate/evidence artifacts.

## Native candidate generation

`generate_native_topk.py` accepts `canonical-tree-sentences-v1` input with one
archived tree per sentence. The adapter emits model-specific binary/n-ary
top-300 candidates; it never treats repetitions of the archived tree as 300
distinct parses. It takes corpus sizes from the audited inputs, checks input
fingerprints, and rejects nonzero document offsets and alignment exceptions.

```bash
python datatools/parse_test_docppl_data/generate_native_topk.py audit \
  --ppl-dir dataset/bbc-news-reserved-clean-v1/canonical/dev \
  --test-tree dataset/bbc-news-reserved-clean-v1/tree/dev.npy \
  --output dataset/bbc-news-reserved-clean-v1/native_model_topk_300_v2/dev
```

Use the same explicit paths for `generate` and `finalize`, adding the desired
shard count and device. Generate/inspect a dev smoke subset first. The base
array manifest explicitly says top-K is pending until production completes;
it must not be used as a completion receipt for GPU candidates. Labeled
Tree/TG top-300 proposal generation is a separate production stage.

The frozen run additionally publishes native GPST/Pushdown candidates for
`docppl/dev_smoke` (100 documents) and `docppl/test_1000` (1,000 documents).
Consult `docppl/manifest.json` and each subset's `validation.json` for actual
completion; the immutable base manifest's top-K status refers to full splits.
`document_map.json` maps each subset document back to the full split.

After finalization, `audit_native` loads every sentence through the real
`NativeModelTopKCorpus` reader. It checks shared scoring tokens and word
boundaries against canonical inputs and the frozen split, verifies logical
candidate uniqueness, GPST gap permutations, Pushdown constituent boundaries,
finite ranked logical scores, and negative-infinite padding scores. It saves
hashes of every candidate output file in a separate validation receipt.

```bash
python -m datatools.reserved_clean.audit_native \
  --dataset dataset/bbc-news-reserved-clean-v1 --split test \
  --selection artifacts/bbc_reserved_clean_20260906/test_docppl_selection.json \
  --canonical artifacts/bbc_reserved_clean_20260906/test_docppl_canonical \
  --native artifacts/bbc_reserved_clean_20260906/test_docppl_native \
  --tokenizer dataset/bbc-news/TG_GPT2_tokenizer.json \
  --output artifacts/bbc_reserved_clean_20260906/test_docppl_native_audit.json
```

## Training exclusion

`assemble_streams.py` defaults to `bbc_train_shards.txt` (89 historical
shards). The `assemble()` API fails before writes if any reserved 2023 shard
is supplied, including via an explicit 94-shard list. The full 94-shard list
remains available for parsing and provenance scans.

## Evidence limits

The certified scope is the frozen historical BBC arrays actually audited.
Model config path inventories are evidence of intended inputs, not proof of
checkpoint training bytes. Mixed-corpus, continuation and unrelated training
sources remain uncertified unless separately audited. Clean under these
declared lexical rules does not mean zero semantic overlap. No model scores
are used for filtering or selection.
