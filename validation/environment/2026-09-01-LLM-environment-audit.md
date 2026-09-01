# LLM environment audit and repair — 2026-09-01

Repository revision audited: `99bdb6c` (with pre-existing user changes left
untouched). Environment prefix: `/home/wangpch/.conda/envs/LLM`.

## Outcome

- `conda doctor -n LLM`: healthy; no altered or missing Conda-owned files.
- `python -m pip check`: no broken requirements.
- All 240 applicable pip requirements in `environment.yml` are installed with
  no declared-version drift.
- Core CPU regression: 143 passed.
- Extension/parse regression: 17 passed; GPST rebuild follow-up: 4 passed.
- Repository collection: 751 tests collected without importing historical
  copies under `artifacts/`.
- Benepar `Parser("benepar_en3_large")` loads with protobuf 6.33.6 when the
  parse entry point selects protobuf's Python implementation before imports.
- W&B 0.24.1 completed an offline init/log/finish smoke test outside the
  restricted test sandbox.
- Slurm GPU validation passed in job 3868: `nvidia-smi`, a PyTorch CUDA matrix
  multiplication, and FlashAttention 2.8.2 forward/backward all ran on one
  allocated RTX 3090. The job completed with exit code `0:0`.

## Applied repairs

1. Renamed the declared environment from `LLMH100` to `LLM`.
2. Removed the orphaned CUDA 11.7 pip runtime quartet. No repository code,
   installed distribution, PyTorch library, or FlashAttention library linked
   it. Removed recorded size: 1.345 GiB.
3. Upgraded the yanked W&B 0.24.0 build to 0.24.1.
4. Restored Conda ownership of pip 26.1.2, setuptools 82.0.1, wheel 0.47.0,
   and packaging 26.0. Old duplicate metadata was moved (not deleted) to
   `/tmp/LLM-stale-dist-info-2026-09-01/`.
5. Pinned the tested Google/protobuf package set and made the Benepar protobuf
   workaround execute before the first Benepar import.
6. Replaced the generic FlashAttention source requirement with the official
   2.8.2 wheel for CPython 3.10, torch 2.7, CUDA 12 and CXX11 ABI=true. This
   prevents a fresh environment from falling back to a source build without
   `nvcc`.
7. Installed CMake 3.29.6 with Spack at:
   `/home/wangpch/.spack/opt/spack/linux-ubuntu20.04-icelake/gcc-11.5.0/cmake-3.29.6-vjbss7aibdhilmfpmerbwsjhvolnj5sb`.
8. Made the `tg_mask` CMake build derive Python, Torch paths, and the CXX11 ABI
   from the selected interpreter rather than hard-coded Python 3.10 paths.
9. Embedded the Torch library RPATH in `cppbackend`, so it imports in a clean
   process without importing Torch first.
10. Replaced Release-build-only undefined behavior in `left_most` and
    `right_most` with explicit `std::logic_error` exceptions.

## CUDA layers found

| Layer | Result |
|---|---|
| Hardware | 8 × NVIDIA GeForce RTX 3090 visible on PCIe |
| Kernel module | 535.230.02 loaded |
| User driver libraries | `libcuda.so` and `libnvidia-ml.so` both 535.230.02 |
| PyTorch runtime | torch 2.7.1+cu126 with CUDA 12.6 pip runtime libraries |
| CUDA Toolkit | not installed/discoverable; no `nvcc`, no Spack `cuda` spec |
| GPU usability | passed under Slurm job 3868: RTX 3090, compute capability 8.6, PyTorch matrix multiplication and FlashAttention forward/backward |

The earlier `Failed to initialize NVML: Unknown Error` and zero-device result
occurred in a shell outside a Slurm GPU allocation. Inside job 3868, NVML and
CUDA both worked. The earlier observation was therefore context-specific and
is not evidence that the installed driver is unusable; GPU checks on this
cluster should be run inside an allocation.

Driver 535 meets CUDA 12.x *minor-version compatibility* (minimum 525), and the
actual torch 2.7.1+cu126 and FlashAttention kernels passed. The `CUDA Version:
12.2` banner from `nvidia-smi` describes the driver API capability and is not
the PyTorch wheel's bundled runtime version (`torch.version.cuda == 12.6`). No
driver change is required for the tested workload. The historical Xid 31/43
kernel messages remain worth operational monitoring if a future allocated job
reports a GPU error, but they did not reproduce in this smoke test.

The full Slurm evidence is under `validation/slurm/`; the structured result is
`validation/slurm/results/tg-llm-env_3868.json`.

Official references:

- <https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html>
- <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
- <https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html>
- <https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.2>

## Remaining planned migration

Python 3.10 reaches end of life on 2026-10-04, and this prefix still uses
Python 3.10.0 linked to OpenSSL 1.1.1w. Do not upgrade Python in place during an
experiment: create a parallel Python 3.11 environment, rebuild both CPython
extensions there, reproduce a checkpoint evaluation, and only then switch
training jobs.

NLTK 3.9.3 itself passed imports and tree operations. Its `punkt`, `punkt_tab`,
and tagger models are external data under `nltk_data`, so a new machine needs a
separate data-artifact provisioning/check step even when Conda creation passes.

## Evidence and rollback

The `LLM-before-2026-09-01.*` and `LLM-after-2026-09-01.*` files in this
directory contain explicit Conda records, exported environments, pip freezes,
and doctor reports.

The previous compiled extensions are recoverable from
`/tmp/TG-Interpolation-extension-backup-2026-09-01/`. The three stale package
metadata directories are recoverable from `/tmp/LLM-stale-dist-info-2026-09-01/`.
