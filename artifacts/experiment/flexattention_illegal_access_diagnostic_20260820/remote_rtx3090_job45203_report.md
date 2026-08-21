# Remote RTX3090 illegal-memory-access report

## Observation

- Observed: 2026-08-21 11:52:52 +08:00
- Host: `rtx3090` (RTX 3090, 24 GiB per GPU)
- Slurm job: `45203`
- Scheduler result: `FAILED`, elapsed `00:50:59`, exit code `1:0`
- Campaign index: 102
- Run: `tgnomask_mix_tg_100m_xsum_seed31723`
- Model: `tgnomask_mix_tg_100m` (`mixing`: 6 TG heads + 6 TGNoMask heads)
- Task / seed: XSum / 31723
- Pretrained checkpoint:
  `saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded`
- Training contract: fp32 DDP, world size 4, global batch 40, device
  microbatch 1, gradient accumulation 10, three epochs (1341 optimizer steps)
- Environment: PyTorch `2.7.1+cu126`, NVIDIA driver `535.104.05`; `tg_mask`
  imported successfully before submission.

## Failure evidence

The run completed steps 1--1308 and then emitted on rank 0:

```text
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
```

The launcher then terminated ranks 1--3 with `SIGTERM`; rank 0 terminated with
`SIGABRT` (signal 6), and `scripts/train.py` failed. The subsequent
`multiprocessing` semaphore warnings are teardown consequences, not the first
failure.

The final Python frames passed through rotary embedding and `empty_cache()`, but
the CUDA runtime explicitly marks the report asynchronous. They therefore do
not identify the faulty kernel or source operation.

## Scope and consequences

- This is not an OOM: the failure text is an illegal memory access and the
  device microbatch was already reduced to 1.
- No `TRAIN_DONE` marker or final checkpoint was produced.
- Dependent jobs did not run: 45204 (index 102 XSum evaluation), 45205 (index
  103 training), and 45206 (index 103 XSum evaluation).
- This is an additional production-scale reproduction of the campaign's
  FlexAttention illegal-access failure class, but it is not an exact controlled
  reproduction of the fixed-input `tg_100m` diagnostic above. The trigger
  remains unlocalized.

## Source evidence

Remote Slurm output (not copied locally):

```text
/home/wangpch/TG-Interpolation/artifacts/experiment/
scaleup_nonfineweb_multiseed_20260815/slurm/train-remote-rtx3090-mb1-45203.out
```

The local campaign submission record is:

```text
artifacts/experiment/scaleup_nonfineweb_multiseed_20260815/submission.json
```
