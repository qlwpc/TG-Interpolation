"""Launcher regression tests for distributed GPST training setup."""
from __future__ import annotations

import runpy
from unittest.mock import patch


def test_initialize_distributed_under_torchrun():
    namespace = runpy.run_path("scripts/gpst/run_gpst.py", run_name="gpst_launcher_test")
    initialize_distributed = namespace["initialize_distributed"]

    with patch("torch.distributed.is_initialized", return_value=False), \
            patch("torch.distributed.init_process_group") as init:
        started = initialize_distributed(local_rank=0, device_type="cuda")

    assert started is True
    init.assert_called_once_with(backend="nccl", init_method="env://")


def test_initialize_distributed_is_noop_without_torchrun():
    namespace = runpy.run_path("scripts/gpst/run_gpst.py", run_name="gpst_launcher_test")
    initialize_distributed = namespace["initialize_distributed"]

    with patch("torch.distributed.init_process_group") as init:
        started = initialize_distributed(local_rank=-1, device_type="cpu")

    assert started is False
    init.assert_not_called()
