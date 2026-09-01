"""Minimal LLM environment smoke test intended to run inside a Slurm GPU job."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


REPO = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO / "validation" / "slurm" / "results"
JOB_ID = os.environ.get("SLURM_JOB_ID", "manual")
RESULT_PATH = RESULT_DIR / f"tg-llm-env_{JOB_ID}.json"

result: dict[str, object] = {
    "job_id": JOB_ID,
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "python": sys.executable,
    "checks": {},
}
failures: list[str] = []


def check(name, function):
    try:
        details = function()
        result["checks"][name] = {"status": "passed", "details": details}
    except Exception as exc:  # keep running to collect all diagnostics
        failures.append(name)
        result["checks"][name] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def command_details(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return {
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_versions():
    packages = [
        "numpy",
        "nltk",
        "protobuf",
        "torch",
        "torchvision",
        "torchaudio",
        "flash-attn",
        "wandb",
        "benepar",
        "spacy",
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    for stale in (
        "nvidia-cublas-cu11",
        "nvidia-cuda-nvrtc-cu11",
        "nvidia-cuda-runtime-cu11",
        "nvidia-cudnn-cu11",
    ):
        try:
            version = importlib.metadata.version(stale)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"unexpected CUDA 11 package remains: {stale}=={version}")
    return versions


def python_stack():
    import numpy as np
    from nltk import Tree

    stacked = np.stack(
        [np.asarray([1, 2]), np.asarray([3, 4])], dtype=np.int64
    )
    tree = Tree.fromstring("(S (NP test) (VP works))")
    assert stacked.dtype == np.int64 and stacked.shape == (2, 2)
    assert tree.leaves() == ["test", "works"]
    return {"numpy_shape": list(stacked.shape), "tree_leaves": tree.leaves()}


def extensions():
    from olmo.data import tg_mask

    extension_dir = REPO / "olmo" / "gpst" / "cpp_extension"
    sys.path.insert(0, str(extension_dir))
    cppbackend = importlib.import_module("cppbackend")
    required_tg = ("SentencepieceVocab", "TG_attention_bias")
    required_gpst = ("TableManager", "SpanTokenizer")
    assert all(hasattr(tg_mask, name) for name in required_tg)
    assert all(hasattr(cppbackend, name) for name in required_gpst)
    return {
        "tg_mask": tg_mask.__file__,
        "cppbackend": cppbackend.__file__,
    }


def torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false inside GPU allocation")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Slurm allocation exposes no CUDA devices")
    device = torch.device("cuda:0")
    a = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
    product = a @ a.T
    torch.cuda.synchronize()
    assert torch.isfinite(product).all().item()
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "matmul_sum": product.sum().item(),
    }


def flash_attention_cuda():
    import torch
    from flash_attn import flash_attn_func

    tensors = [
        torch.randn(
            1,
            16,
            2,
            64,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    output = flash_attn_func(*tensors, dropout_p=0.0, causal=True)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    assert output.shape == tensors[0].shape
    assert torch.isfinite(output).all().item()
    assert all(tensor.grad is not None for tensor in tensors)
    return {"shape": list(output.shape), "dtype": str(output.dtype)}


def benepar_parser():
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") != "python":
        raise RuntimeError("protobuf compatibility mode was not set before imports")
    import benepar

    parser = benepar.Parser("benepar_en3_large")
    parsed = parser.parse("Environment validation works.")
    return {"parser": type(parser).__name__, "root_label": parsed.label()}


def wandb_offline():
    import wandb

    with tempfile.TemporaryDirectory(prefix="tg-wandb-") as directory:
        run = wandb.init(project="tg-interpolation-env-smoke", mode="offline", dir=directory)
        run.log({"loss": 1.0})
        run.finish()
    return {"version": wandb.__version__, "mode": "offline"}


cmake = Path(
    "/home/wangpch/.spack/opt/spack/linux-ubuntu20.04-icelake/gcc-11.5.0/"
    "cmake-3.29.6-vjbss7aibdhilmfpmerbwsjhvolnj5sb/bin/cmake"
)
check("nvidia_smi", lambda: command_details(["nvidia-smi"]))
check("cmake", lambda: command_details([str(cmake), "--version"]))
check(
    "cuda_toolkit",
    lambda: {
        "nvcc": shutil.which("nvcc"),
        "status": "not installed (runtime validation does not require nvcc)",
    },
)
check("package_versions", package_versions)
check("python_stack", python_stack)
check("extensions", extensions)
check("torch_cuda", torch_cuda)
check("flash_attention_cuda", flash_attention_cuda)
check("benepar_parser", benepar_parser)
check("wandb_offline", wandb_offline)

result["status"] = "failed" if failures else "passed"
result["failed_checks"] = failures
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
print(f"result_path={RESULT_PATH}")
raise SystemExit(1 if failures else 0)
