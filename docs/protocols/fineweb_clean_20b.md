# Filtered FineWeb-Edu 20B paired collection

This October 2026 collection is a subsequent data experiment. It is not a
replacement for the paper's historical FineWeb-Edu training corpus.
**[FWE-014](fineweb_edu_data.md#adj-label-leak) remains confirmed and unrepaired**
in the source and this training split. The two validation splits have not been
specifically scanned for that defect. Projection and registered typed-label
validation do not detect unknown labels encoded as ordinary text.

## Recorded collection

Collection and validation completed on 2026-10-14. These are existing receipt
values, not a new execution during documentation preparation. The
[public evidence](../../reproducibility/evidence/fineweb_data_audit.json) includes
selected receipt and validation fields, source hashes, and the later defect status.

| Split | Body tokens, excluding EOS | Tree tokens, including structure and EOS | Source record segments | Compressed GiB |
|---|---:|---:|---:|---:|
| train | 20,000,000,517 | 47,432,346,354 | 19,299,617 | 83.2187 |
| validation_a | 10,002,504 | 23,656,428 | 9,512 | 0.0417 |
| validation_b | 10,005,600 | 23,763,725 | 9,693 | 0.0418 |

The complete paired output contains 4,916 NPZ files, totaling 89,445,064,869 bytes
(83.3022 GiB). The terminal training arrays also contain 19,299,617 ending EOS IDs,
for a total length of 20,019,300,134 tokens. Expanding both representations of all
splits requires 270,076,375,800 bytes; that full expansion has not been performed.

The scan covered 58/246 source shards, 22,717,154 EOS-delimited records and
23,865,987,142 body tokens. Rules excluded 325,947 records and 744,375,733 body
tokens. Reason counts overlap: 145,741 confirmed formulas, 55,579 formula-review
cases, 39,487 additional function equalities, 49,078 web-control cases and 51,413
mojibake cases. Exact fingerprints found 2,053,798 duplicates of selected content.
Other unselected records include candidates encountered after a split quota was
filled. These counts do not describe the full 100BT corpus.

The historical output location is
`/public/home/wangpch/SIST-assets/datasets/fineweb-edu-clean-20b-20261013`.
Local provenance is `artifacts/analysis/fineweb_clean_20b_20261013/`, with
`receipt.json`, `verification.json`, `validation_only_verification.json` and
`final_job_accounting.txt`. These are provenance references, not public download
links. Full arrays and original logs still await external release.

## Source, filtering and splits

The source is the [repaired paired corpus](fineweb_bracket_repair.md), using the
same Qwen tokenizer with SHA-256
`827837c463999c3e647b1eeb1b7f056744c56ddf374247e9cf0bab9eb8afe9cb`.
Records retain their source token bytes, EOS-only boundaries, no initial BOS,
parser spacing and quotation marks. No GPST tokenizer or new parser is used.
Targets are 20,000,000,000 body tokens for train and at least 10,000,000 for each
validation split; selecting whole records can slightly exceed a target.

The [filter](../../scripts/campaigns/fineweb_collection_filter.py) version is
`paired-collection-quality-v2`. It extends `formula-syntax-v5` with function
equalities such as `u(x)=ln(x)` and excludes both confirmed and uncertain formula
cases. Whole-record rules also cover selected cookie controls/subscription
prompts, explicit OCR/extraction notices, mojibake, residual HTML shells and
dominant repeated long lines. Length limits are 32 to 262,144 body tokens.
Topics such as medicine, finance, mathematics or code are not blanket exclusions.
These rules do not certify that every retained record is formula-free or noise-free.

Decoded text is used only for selection. The deduplication key is
`SHA256(NFC + collapsed whitespace)`; this normalization does not rewrite training
arrays. Exact keys are deduplicated globally among selected records. The first
eight hash bytes modulo 2048 assign buckets 0/1 to validation_a/validation_b and
all others to train. Identical keys cannot cross splits. Source order uses seed
20261013; within-shard order is stable. Collection stops when all three targets
are filled. No MinHash, semantic deduplication or complete 100BT scan is claimed.

## Storage, validation and rerunning

Each `train/*.npz` or `validation_*/*.npz` contains full little-endian uint32
`terminal` and `tree` arrays plus `records`. Both representations are physically
stored. Records include original intervals in both streams, ordinals, token/text
hashes and independent offsets within the compressed arrays.

Selection verifies full terminal-source hashes. Export verifies both source
hashes, per-record terminal hashes, complete Tree projection, typed labels,
EOS boundaries and exact compressed-array/coordinate readback. The final audit
checks every archive hash, all coordinates, global exact deduplication and split
assignment, with an additional complete paired archive readback per included
source shard. A supplementary check covers the three validation-only source shards.
These checks passed within their declared scope; they do not certify ADJ-label
normalization or broader data quality.

The [collector](../../scripts/campaigns/fineweb_paired_collection.py) exposes
`inventory`, `index` and `export`. It is a campaign-specific tool: inventory
requires exactly 246 repaired pairs, `acceptance_final.json`, per-shard repair
journals and the matching tokenizer. Use `--help` to inspect its required inputs;
it is not a general downloader. `protocol.json` binds source, code, tokenizer,
repair receipt, rules, seed and budgets. Changed identities prevent silent reuse.
CPU work is run inside a Slurm batch allocation, as required by these tools.

Storage defaults cap compressed arrays at 95 GiB and reserve at least 15 GiB
before writes and publication. Historical free space after delivery was
28,201,451,520 bytes (26.2646 GiB). Check actual filesystem and quota capacity for
a new run; historical free-space figures are not current resource guarantees.

## Reading and training input conversion

```python
import numpy as np

with np.load('train/068-0000.npz', allow_pickle=False) as shard:
    terminal = shard['terminal']
    tree = shard['tree']
    records = shard['records']
```

The existing MemMapDataset cannot consume compressed NPZ as raw uint32. Within a
Slurm batch allocation, run the [unpacker](../../scripts/campaigns/fineweb_collection_unpack.py)
from the repository root; the destination must be new and its parent must exist:

```bash
python scripts/campaigns/fineweb_collection_unpack.py \
  --collection /path/to/fineweb-edu-clean-20b-20261013 \
  --output /path/with/sufficient/space/fineweb-clean-raw \
  --split train --format both
```

The tool checks required capacity plus 15 GiB reserve, archive/token hashes and
full output readback. It writes headerless `<u4` files with the historical `.npy`
suffix. Read those outputs with `np.memmap(dtype='<u4')`, not `np.load`.
Compressed source files remain intact. Format conversion alone does not resolve
the known ADJ issue or establish readiness for a paper reproduction.
