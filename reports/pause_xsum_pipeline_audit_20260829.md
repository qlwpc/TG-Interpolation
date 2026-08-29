# Pause XSum pipeline v2 audit and SIST restart

Date: 2026-08-29 (Asia/Shanghai)

## Outcome

The abnormally low Pause XSum results were caused by implementation mismatches,
not by an incomplete local-to-SIST sync. The affected checkpoints must be
re-finetuned: changing only the decoder cannot repair the training labels or
pause convention used by the old finetune runs.

The corrected Pause1/Pause2 campaign completed on SIST as Slurm array
`989161_[0-4]` (five seed elements, `%2`; each element ran both models
serially). All five elements finished with `COMPLETED/0:0`, and all ten model
runs produced `TRAIN_DONE`, `EVAL_DONE`, and a version-2 training contract.

Five-seed full-test results (mean ± sample SD, percentage scale) are:

- Pause1 SEP50261: R1 `31.276 ± 0.066`, R2 `10.968 ± 0.022`,
  RL `24.892 ± 0.049`, R-AVG `22.378 ± 0.046`.
- Pause2 SEP50261: R1 `31.124 ± 0.053`, R2 `10.856 ± 0.065`,
  RL `24.768 ± 0.072`, R-AVG `22.250 ± 0.061`.

## Root causes

1. The finetune loader did not pass the checkpoint `max_sequence_length` and
   `pause_token_id=50261` to XSum. Training could therefore use the default
   context length and legacy repeated-token pauses while evaluation used SEP.
2. XSum constructed its label mask after pause expansion but used the raw
   summary length. Pause1 therefore supervised approximately the last half of
   the expanded stream and Pause2 approximately the last third, instead of the
   expanded summary positions.
3. Pause inference reused tree-grammar `word_sync_beam_search`. SEP was not
   phase constrained, extraction restarted from the wrong local phase, and the
   expanded-step budget shortened Pause2 summaries disproportionately.
4. Old `TRAIN_DONE` markers could silently reuse an affected XSum checkpoint.

## Corrections

- Expand the raw XSum supervision mask together with the token stream; in
  `pause*_label`, additionally mask pause positions.
- Pass model context length and the configured pause token through the
  finetune dataset builder.
- Route every Pause XSum grammar through phase-constrained, KV-cached beam
  generation. Real positions cannot emit SEP/NT; pause positions are forced to
  SEP (or to the legacy repeated token when no dedicated pause ID exists).
- Define generation length in real tokens and return real tokens directly.
- Add XSum pipeline contract version 2 with source/config hashes. A completed
  XSum run without a valid v2 contract exits with code 42 instead of being
  reused.
- Add a dual-campaign paired array mode compatible with SIST Shanghai's
  five-submitted-job QOS limit. Even seeds start Pause1; odd seeds start Pause2.

## Validation evidence

- Local syntax/whitespace checks passed.
- Local targeted regression suite: 161 tests passed.
- The migrated SIST snapshot ran the core targeted CPU suite: 15 passed, one
  unrelated old-snapshot test deselected because its script was absent there.
- The final local regression check against the byte-identical SIST source
  snapshot passed all 16 Pause-XSum tests.
- Pause1 smoke job `989152` completed; Pause2 smoke job `989153` completed.
  Each performed a real XSum optimizer step, two-batch XSum generation, a real
  BoolQ optimizer step, and two-batch BoolQ evaluation on four GPUs.
- On four examples from an old affected Pause1 finetune checkpoint, constrained
  decoding improved computed R-AVG from `0.08471` to `0.10431` and mean output
  length from 10.0 to 23.5 words. This isolates a decoding defect, but is not a
  final model-quality claim because the checkpoint itself was trained through
  the defective data path.
- Existing full-test R-AVG values around `0.0278`--`0.0279` are parsed evidence
  of the failure and must not be reported as valid Pause performance.

Smoke metrics are execution-path checks, not accuracy estimates. The final
scientific comparison is the five-seed full-test output from array `989161`.

## SIST paths

- Immutable working snapshot:
  `/public/home/wangpch/TG-Interpolation/artifacts/experiment/pause_xsum_v2_sist_20260829/code`
- Pause1 campaign:
  `/public/home/wangpch/TG-Interpolation/artifacts/experiment/pause1_sep50261_xsum_v2_sist_20260829`
- Pause2 campaign:
  `/public/home/wangpch/TG-Interpolation/artifacts/experiment/pause2_sep50261_xsum_v2_sist_20260829`
- Slurm logs:
  `/public/home/wangpch/TG-Interpolation/artifacts/experiment/pause_xsum_v2_sist_20260829/slurm/dual-xsum-989161_%a.out`

The four core source hashes and all runner hashes were checked byte-for-byte
between the local workspace and this snapshot immediately before submission.
The older SIST code snapshots and affected campaign directories were not
overwritten.
