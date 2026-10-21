# Paired FineWeb bracket and internal-EOS repair

This is a subsequent data-repair protocol for the local FineWeb-Edu arrays,
not the historical paper preprocessing recipe. The 2026-10-13 acceptance covered
PTB brackets and internal EOS only. **[ADJ label leakage](fineweb_edu_data.md#adj-label-leak)
remains confirmed and unrepaired**; registered-label checks cannot detect all
markers encoded as ordinary text.

## Input and transformation

The [repair program](../../scripts/campaigns/fineweb_bracket_repair.py) uses
`fineweb-paired-native-brackets-inner-eos-delta-v2`. Its inputs are 246 matching
terminal/Tree files, stored as headerless little-endian uint32 with no initial BOS.
The frozen tokenizer SHA-256 is
`827837c463999c3e647b1eeb1b7f056744c56ddf374247e9cf0bab9eb8afe9cb`.

The shared [mapping](../../datatools/ptb_brackets.py) restores
`-LRB-/-RRB-`, `-LSB-/-RSB-` and `-LCB-/-RCB-` to `()`, `[]` and `{}`,
including forms embedded inside words. Only affected lexical spans are re-encoded.
Registered structural IDs and other unaffected IDs retain their original bytes.
Exactly six angle-bracket-wrapped lexical AddedToken forms are also supported:
for example, `<-LRB->` becomes `<(>`, preserving the angle brackets. Other
AddedTokens containing unsupported escapes cause validation to fail.

EOS ID 151643 at positive tree depth is deleted from both streams; EOS at an
empty stack remains the record terminator. An EOS-only deletion does not re-encode
neighbors or insert whitespace. Thus the documented example `CDCnews` + internal
EOS + `A number` becomes `CDCnewsA number`. This is an explicit change to the
text representation. The procedure does not filter formulas, deduplicate,
delete documents, detokenize or reparse trees.

## Validation and commit

For every shard, the program requires:

1. Exact input terminal/Tree projection and correctly paired typed labels, allowing
   positive-depth EOS only as an item scheduled for deletion.
2. Complete UTF-8 lexical spans and decoded output exactly equal to the specified
   bracket restoration and internal-EOS deletion.
3. Exact staged projection, unchanged structural-ID sequence and record-ending
   count, preserved unaffected AddedTokens, and no remaining internal EOS or
   ordinary PTB escapes.
4. Matching staged-file readback hashes and reverse reconstruction of both
   original files to their complete SHA-256 hashes using permanent deltas.

Without `--commit`, the tool stages and validates. With `--commit`, it journals
replacement and retains temporary original-file hard links until both new hashes
have been verified. Permanent compressed deltas and receipts remain.

**Two file paths cannot be replaced in a single POSIX atomic operation.** Pause
readers of this dataset during repair. After interruption between replacements,
the same command resumes from the journal and original/new hashes. An interrupted
generation preserves its record and rebuilds derived staging files. Unexpected
source changes or corrupt backups stop the operation.

Parallel runners need immutable, disjoint shard assignments and effective locks
on the actual shared filesystem. The historical run found that locks on its
campaign mount did not exclude other nodes; source-mount locks did. Unique
host/PID/UUID temporary JSON names avoid concurrent publication collisions.
Capacity checks include staging, permanent backup caps and reserved free space.
The GPFS quota integration is site-specific and requires an authenticated quota
snapshot when live queries fail; it is not a portable storage-capacity guarantee.

## Recovery and coordinates

The campaign retains `source_plan_original.json`, `backups/NNN/patch.bin.gz`,
`backups/NNN/journal.json`, source snapshots and progress records. The source
copy of progress is `NATIVE_BRACKET_REPAIR_PROGRESS.json`. A partial receipt is
not full-corpus completion. The `FWPTB001` reversible delta format stores separate
positions for both streams and original/replacement lexical bytes.

Run commands from the repository root with the campaign's matching code,
tokenizer and receipts. These are templates for local assets, not download paths:

```bash
python scripts/campaigns/fineweb_bracket_repair.py \
  --data-root /path/to/fineweb-edu-v2 \
  --campaign /path/to/repair-campaign \
  --tokenizer /path/to/frozen-tokenizer.json \
  --restore-shard 0 --restore-output /path/to/new-restored-directory
```

To translate old document boundaries, use `--translate-shard 0
--translate-format terminal --old-boundary START --old-boundary END` with the
same data-root, campaign and tokenizer arguments. The result is bound to the new
file hash. Boundaries inside re-encoded spans are ambiguous and are rejected.
Historical sample coordinates must remain identified as pre-repair coordinates.

## Historical acceptance and provenance

At 2026-10-13T22:42:56+08:00, 246/246 pairs were committed. All 492 current source
arrays and 246 permanent delta files had complete SHA-256 independently recomputed.
Those hashes authenticate the earlier per-shard projection, typed-label, residual
and reverse-reconstruction checks; the final audit did not independently repeat
the entire tree scan. Four final continuation jobs completed successfully after
quota-query failures had left an earlier attempt incomplete.

| Quantity | Accepted snapshot |
|---|---:|
| Bracket replacements | 59,497,383 |
| Internal EOS IDs removed from each stream | 11 |
| Permanent compressed delta bytes | 1,648,859,387 |
| Record-ending segments | 93,584,610 |
| Terminal tokens after repair | 98,708,787,845 |
| Tree tokens after repair | 233,638,120,313 |

Selected fields and original receipt hashes are in the
[public audit evidence](../../reproducibility/evidence/fineweb_data_audit.json).
Full journals, deltas, code snapshots and logs remain external assets. Their local
provenance root is `artifacts/analysis/fineweb_bracket_repair_20261013/`; it is not
included in the public source tree. The documented regression suite grew to 38
checks during the historical campaign; that count is not a claim of a new run.

One internal-EOS example was traced through a complete parsed record to upstream
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`,
`sample/100BT/000_00004.parquet`, row 375,314 (zero-based). Its file SHA-256 is
`2f3f66cd93f0cf4919725e6dbcd1c240314a2866c9527870231e6148061ca73f`.
The upstream `text` already contained a literal reserved EOS string. Matching used
the full normalized terminal text, not guessed historical row offsets. The
[Parquet search utility](../../scripts/campaigns/fineweb_parquet_text_search.py)
supports this provenance work. The recovered single-record Arrow export is not
the missing historical 984-shard cache. WARC HTML was not inspected, and this one
match does not authenticate provenance for the entire processed corpus.
