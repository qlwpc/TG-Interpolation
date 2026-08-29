from __future__ import annotations

import cProfile
import functools
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.hooks
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
    EvaluatorType,
    BeamSearchType
)
from .data import IterableDataset
from .eval import Evaluator, resolve_structure_mode
from .eval.downstream import BLiMP_TASK_LIST
from .exceptions import OLMoConfigurationError
from .model import OLMo
from .transformers_model import HuggingModel
from .optim import Optimizer, Scheduler
from .torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .util import upload
from olmo.data import get_TG_generate_bias_func
from olmo.data.util import is_pause_label

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int,
        num_bck_flops: int,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        # num_fwd_flops and num_bck_flops from the OLMo model computes flops per token
        # converting to GFLOPs here prevents numerical issues while logging
        self.total_training_Gflops = (num_fwd_flops + num_bck_flops) * global_total_tokens / 1e9

        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}

        # plot flops related metrics
        metrics["throughput/total_training_Gflops"] = self.total_training_Gflops
        metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)

        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    mask = (labels != ignore_index)
    if reduction == "mean":
        z_squared = (z_squared * mask).mean()
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


fused_loss_fn: Optional[Callable]

try:
    import flash_attn
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_cross_entropy_loss,  # type: ignore
    )

    def fused_loss_fn(
        logits,
        labels,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

        if ce_loss_use_ignore_index_param:
            ignore_index_kwarg = {"ignore_index": ignore_index}
        else:
            ignore_index_kwarg = {"ignored_index": ignore_index}

        loss, z_loss = flash_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_multiplier,
            inplace_backward=False,
            process_group=None,
            **ignore_index_kwarg,
        )

        mask = labels != ignore_index
        loss_tokens = mask.sum()

        if reduction == "mean":
            loss = loss.sum() / loss_tokens
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not compute_z_loss:
            return loss, None

        if reduction == "mean":
            z_loss = z_loss.sum() / loss_tokens
        elif reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss

except ImportError:
    fused_loss_fn = None


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Union[OLMo, HuggingModel]
    dist_model: Union[DDP, FSDP]
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[Evaluator]
    epoch: Optional[int] = None
    global_step: int = 0
    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    indices_file: Optional[TextIO] = None
    _start_time: float = 0.0
    _gc_init_state: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None

    def __post_init__(self):
        if self.cfg.fused_loss:
            if fused_loss_fn is not None:
                self.loss_fn = fused_loss_fn
            else:
                raise NameError("`fused_loss_fn` is not defined. Please ensure that `flash_attn` is installed.")

    @property
    def dataset(self) -> IterableDataset:
        assert isinstance(self.train_loader.dataset, IterableDataset)
        return self.train_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        if self.batches_per_epoch == 0:
            return 0
        return math.ceil(self.max_steps / self.batches_per_epoch)

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch or 0,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch") or 0
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_step = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        assert self.epoch is not None
        # Reshuffle dataset if needed.
        if self.dataset.epoch != self.epoch:
            log.info(f"Reshuffling data loader for epoch {self.epoch}...")
            self.dataset.reshuffle(self.epoch)

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset, IterableDataset)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        new_learning_rate = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
        )
        for group in self.optim.param_groups:
            group["lr"] = new_learning_rate
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        # Flush data indices file.
        # TODO: upload the indices files?
        if self.indices_file is not None:
            self.indices_file.flush()

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.dist_model,
                self.optim,
                self.trainer_state_dict(),
                upload_to=remote_checkpoint_dir,
            )
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save_overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Remove old checkpoints.
        # For DDP, checkpoint_type being passed to remove_checkpoint is always `unsharded`.
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def _setup_module_output_save_hooks(self, micro_batch_idx: int) -> List[torch.utils.hooks.RemovableHandle]:
        if (
            self.cfg.module_outputs_save_steps is None
            or self.global_step not in self.cfg.module_outputs_save_steps
        ):
            return []

        if micro_batch_idx != 0 or get_global_rank() != 0:
            # Hook is currently only used on the first microbatch of rank 0
            return []

        trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
        if trace_save_folder.exists():
            if self.cfg.save_overwrite:
                shutil.rmtree(trace_save_folder)
            else:
                raise OLMoConfigurationError(
                    f"Attempting to overwrite traces at step {self.global_step} without --save_overwrite"
                )
        trace_save_folder.mkdir(parents=True)

        def trace_outputs_hook(
            module_name: str, _: torch.nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            if len(args) == 0:
                log.info("No input args for module %s, output %s", module_name, output)

            module_input = args[0] if len(args) > 0 else torch.tensor(())
            trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
            trace_save_folder.mkdir(parents=True, exist_ok=True)

            module_occurence_num = 0
            while (
                module_input_filepath := trace_save_folder / f"{module_name}_{module_occurence_num}_input.pt"
            ).exists():
                module_occurence_num += 1
            torch.save(module_input, module_input_filepath)

            module_output_filepath = trace_save_folder / f"{module_name}_{module_occurence_num}_output.pt"
            torch.save(output, module_output_filepath)

        output_hooks = []
        for module_name, module in self.model.named_modules(prefix="model"):
            output_hooks.append(module.register_forward_hook(functools.partial(trace_outputs_hook, module_name)))

        return output_hooks

    def get_labels(self, batch: Dict[str, Any], ignore_id: int = -100) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, ignore_id)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, ignore_id)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=ignore_id)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
        _pushdown_profile = (
            bool(os.environ.get("OLMO_PUSHDOWN_PHASE_PROFILE"))
            and self.cfg.model.transformer_grammar_type == "pushdown"
            and torch.cuda.is_available()
            and get_global_rank() == 0
        )
        if _pushdown_profile:
            torch.cuda.synchronize()
            _pd_forward_start = time.perf_counter()
        # shape: (batch_size, seq_len, vocab_size)
        olmo_out = self.dist_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
            tree_spans=batch.get("tree_spans"),
            pushdown_sentence_ids=batch.get("pushdown_sentence_ids"),
            compute_attachment_logits=(
                self.dist_model.training
                and self.cfg.model.transformer_grammar_type == "pushdown"
                and batch.get("tree_spans") is not None
                and not bool(os.environ.get("OLMO_PUSHDOWN_SKIP_ATTACHMENT"))
            ),
        )
        if _pushdown_profile:
            torch.cuda.synchronize()
            _pd_model_ms = (time.perf_counter() - _pd_forward_start) * 1e3
            _pd_lm_start = time.perf_counter()
        logits = olmo_out.logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        ignore_id = self.cfg.model.pad_token_id
        labels = self.get_labels(batch, ignore_id=ignore_id)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=ignore_id, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)

        # TreeReg auxiliary loss (Nandi et al. 2025): applied on the captured
        # post-block residual at treereg_layer, every k LM steps. Train-only.
        treereg_loss = None
        if (self.dist_model.training
                and self.cfg.model.transformer_grammar_type == "treereg"
                and olmo_out.treereg_hidden is not None
                and batch.get("tree_spans") is not None):
            k = self.cfg.model.treereg_every_k
            step = int(getattr(self, "global_step", 0))
            if k <= 0 or step % k == 0:
                from olmo.treereg import compute_treereg_loss
                tr_hidden = olmo_out.treereg_hidden
                spans = batch["tree_spans"]
                span_mask = batch.get("tree_span_mask")
                if span_mask is None:
                    span_mask = (spans[..., 0] >= 0)
                sentence_ids = batch.get("treereg_sentence_ids")
                word_boundaries = batch.get("treereg_word_boundaries")
                if sentence_ids is None or word_boundaries is None:
                    raise RuntimeError(
                        "TreeReg training requires treereg_sentence_ids and "
                        "treereg_word_boundaries. Rebuild the parse-aligned data "
                        "with scripts/precompute_treereg.py."
                    )
                d_head = self.cfg.model.d_model // self.cfg.model.n_heads
                treereg_loss, sentence_count = compute_treereg_loss(
                    tr_hidden, spans, span_mask,
                    n_heads_subset=self.cfg.model.treereg_n_heads, d_head=d_head,
                    sentence_ids=sentence_ids,
                    word_boundaries=word_boundaries,
                    return_sentence_count=True,
                )
                if loss_reduction == "sum":
                    # The reference objective is a macro average over complete
                    # top-level trees, not over packed rows or LM tokens.
                    treereg_loss = treereg_loss * sentence_count

        # Pushdown attachment-head auxiliary loss (Murty et al. 2023, Eq. 5):
        # cross-entropy of the head's reduce-target logits against the oracle
        # reduce target derived from the gold spans. Train-only (the forward only
        # computed attachment_logits when training + pushdown + tree_spans).
        attachment_loss = None
        if (self.dist_model.training
                and self.cfg.model.transformer_grammar_type == "pushdown"
                and getattr(olmo_out, "attachment_logits", None) is not None
                and batch.get("tree_spans") is not None):
            from olmo.attachment import (
                build_attachment_query_mask,
                compute_attachment_loss,
                derive_oracle_reduce_targets,
            )
            att_logits = olmo_out.attachment_logits          # (B, n, n)
            B, n = att_logits.shape[0], att_logits.shape[1]
            spans = batch["tree_spans"]
            span_mask = batch.get("tree_span_mask")
            if span_mask is None:
                span_mask = (spans[..., 0] >= 0)
            oracle = derive_oracle_reduce_targets(spans, n, span_mask)   # (B, n)
            # Attachment targets exist for words and EOS, not BOS/ROOT. The
            # local tokenizer has distinct IDs (50257/50256), while some GPT-2
            # tokenizers share them; handle both contracts explicitly.
            am = build_attachment_query_mask(
                batch["input_ids"][:, :n],
                batch.get("attention_mask"),
                self.cfg.model.bos_token_id,
                self.cfg.model.eos_token_id,
                batch.get("pushdown_sentence_ids")[:, :n]
                if batch.get("pushdown_sentence_ids") is not None
                else None,
            )
            attachment_loss = compute_attachment_loss(
                att_logits, oracle, am, reduction=loss_reduction
            )
        if _pushdown_profile:
            torch.cuda.synchronize()
            _pd_post_ms = (time.perf_counter() - _pd_lm_start) * 1e3
            print(
                f"[pushdown_forward] model_including_attachment_ms={_pd_model_ms:.1f} "
                f"lm_and_attachment_loss_ms={_pd_post_ms:.1f} "
                f"attachment_loss="
                f"{attachment_loss.detach().item() if attachment_loss is not None else 'none'}",
                flush=True,
            )
        return ce_loss, z_loss, logits, treereg_loss, attachment_loss

    def train_micro_batch(
        self,
        micro_batch: Dict[str, Any],
        batch_size_in_loss_tokens: int,
        treereg_loss_denominator: Optional[float] = None,
        attachment_loss_denominator: Optional[float] = None,
        device_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ce_loss, z_loss, logits, treereg_loss, attachment_loss = self.model_forward(
            micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="sum"
        )
        ce_loss = ce_loss / batch_size_in_loss_tokens

        # Get loss to optimize for.
        if self.cfg.softmax_auxiliary_loss:
            assert z_loss is not None
            z_loss = z_loss / batch_size_in_loss_tokens
            loss = (ce_loss + z_loss) * device_loss_weight
        else:
            loss = ce_loss * device_loss_weight

        # TreeReg is summed over complete top-level trees in this micro-batch
        # and normalized independently of the token-level language-model loss.
        if treereg_loss is not None:
            if treereg_loss_denominator is None or treereg_loss_denominator <= 0:
                raise RuntimeError("missing positive TreeReg sentence denominator")
            tr = treereg_loss / treereg_loss_denominator
            loss = loss + self.cfg.model.treereg_alpha * tr

        # Add Pushdown attachment-head auxiliary loss (already summed over the
        # micro-batch above). Its valid queries are structurally parsed tokens,
        # which are not the same population as label-masked LM targets in SFT.
        # Normalize by the global attachment-query count independently; using the
        # LM denominator makes long XSum sources overweight this term by ~8x.
        if attachment_loss is not None:
            if (
                attachment_loss_denominator is None
                or attachment_loss_denominator <= 0
            ):
                raise RuntimeError("missing positive Pushdown attachment denominator")
            att = attachment_loss / attachment_loss_denominator
            loss = loss + self.cfg.model.pushdown_attachment_weight * att

        # In case this helps with memory utilization.
        del micro_batch
        return loss, ce_loss, z_loss

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        # Whether to normalize the loss by the true per-token count (positions that
        # actually contribute to the loss) rather than by batch numel. Finetune tasks
        # always do this. Pretraining with a pause "_label" grammar type also needs it:
        # the label_mask drops pause slots and padding from the loss, so dividing by
        # numel (which counts those dropped positions) would under-weight the loss.
        # In both cases we rescale by a per-device weight so the DDP-averaged gradient
        # equals sum(loss over all real tokens) / (global #real loss tokens), robust to
        # ranks carrying unequal token counts.
        count_loss_tokens = (
            self.cfg.finetune_task is not None
            or is_pause_label(self.cfg.model.transformer_grammar_type)
            or batch.get("label_mask") is not None
        )
        if count_loss_tokens:
            global_batch_size_in_loss_tokens = (self.get_labels(batch, self.cfg.model.pad_token_id) != self.cfg.model.pad_token_id).sum()
            batch_size_in_loss_tokens = global_batch_size_in_loss_tokens.item()
            dist.all_reduce(global_batch_size_in_loss_tokens)
            device_loss_weight = get_world_size() * batch_size_in_loss_tokens / global_batch_size_in_loss_tokens.item()
        else:
            batch_size_in_loss_tokens = batch["input_ids"].numel()  # fixed: since TG or finetune task exist pad/non-loss tokens
            device_loss_weight = 1.0

        # DDP averages gradients across ranks. Dividing each rank's local
        # sentence-loss sum by global_count/world_size therefore yields exactly
        # the global macro average, even when ranks contain unequal numbers of
        # complete top-level trees.
        treereg_loss_denominator = None
        if (
            self.cfg.model.transformer_grammar_type == "treereg"
            and batch.get("tree_spans") is not None
        ):
            k = self.cfg.model.treereg_every_k
            step = int(getattr(self, "global_step", 0))
            if k <= 0 or step % k == 0:
                sentence_ids = batch.get("treereg_sentence_ids")
                word_boundaries = batch.get("treereg_word_boundaries")
                if sentence_ids is None or word_boundaries is None:
                    raise RuntimeError(
                        "TreeReg training batch is missing sentence/word-boundary "
                        "metadata; rebuild it with scripts/precompute_treereg.py."
                    )
                from olmo.treereg import count_treereg_sentences

                global_sentence_count = count_treereg_sentences(sentence_ids)
                if get_world_size() > 1:
                    dist.all_reduce(global_sentence_count)
                if int(global_sentence_count.item()) <= 0:
                    raise RuntimeError("TreeReg batch contains no complete top-level trees")
                treereg_loss_denominator = (
                    global_sentence_count.item() / get_world_size()
                )

        # Attachment CE is defined over parsed-token queries, not LM loss tokens.
        # Compute its own global denominator before splitting the batch. Dividing
        # each rank's local sum by global_count/world_size makes DDP's averaged
        # gradient equal the global mean over valid attachment queries.
        attachment_loss_denominator = None
        if (
            self.cfg.model.transformer_grammar_type == "pushdown"
            and batch.get("tree_spans") is not None
        ):
            from olmo.attachment import build_attachment_query_mask

            attachment_query_mask = build_attachment_query_mask(
                batch["input_ids"],
                batch.get("attention_mask"),
                self.cfg.model.bos_token_id,
                self.cfg.model.eos_token_id,
                batch.get("pushdown_sentence_ids"),
            )
            global_attachment_count = attachment_query_mask.sum()
            if get_world_size() > 1:
                dist.all_reduce(global_attachment_count)
            if int(global_attachment_count.item()) > 0:
                attachment_loss_denominator = (
                    global_attachment_count.item() / get_world_size()
                )

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            # setup sync context for DDP for all micro-batches except the last
            grad_sync_context = nullcontext
            if (
                self.cfg.distributed_strategy == DistributedStrategy.ddp
                and self.cfg.ddp is not None
                and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
            ):
                if micro_batch_idx != num_micro_batches - 1:
                    grad_sync_context = self.dist_model.no_sync

            # Register output hooks
            output_hooks: List[torch.utils.hooks.RemovableHandle] = []
            output_hooks += self._setup_module_output_save_hooks(micro_batch_idx)

            with grad_sync_context():
                with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                    # Run forward pass.
                    loss, ce_loss, z_loss = self.train_micro_batch(
                        micro_batch,
                        batch_size_in_loss_tokens,
                        treereg_loss_denominator=treereg_loss_denominator,
                        attachment_loss_denominator=attachment_loss_denominator,
                        device_loss_weight=device_loss_weight,
                    )

                    # Update overall CE batch loss.
                    ce_batch_loss += ce_loss.detach()

                    # Update overall Z batch loss.
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()

                # Run backward pass.
                _pushdown_profile = (
                    bool(os.environ.get("OLMO_PUSHDOWN_PHASE_PROFILE"))
                    and torch.cuda.is_available()
                    and get_global_rank() == 0
                )
                if _pushdown_profile:
                    torch.cuda.synchronize()
                    _pd_backward_start = time.perf_counter()
                loss.backward()
                if _pushdown_profile:
                    torch.cuda.synchronize()
                    print(
                        f"[pushdown_backward] total_ms="
                        f"{(time.perf_counter() - _pd_backward_start) * 1e3:.1f}",
                        flush=True,
                    )
                if (
                    bool(os.environ.get("OLMO_GRAD_NAN_PROFILE"))
                    and get_global_rank() == 0
                ):
                    nonfinite = []
                    for name, parameter in self.dist_model.named_parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            grad = parameter.grad.detach()
                            nonfinite.append(
                                f"{name}:nan={torch.isnan(grad).sum().item()},"
                                f"inf={torch.isinf(grad).sum().item()}"
                            )
                    print(
                        f"[pushdown_nonfinite_grads] count={len(nonfinite)} "
                        + " | ".join(nonfinite),
                        flush=True,
                    )

            # Remove output hooks
            for hook in output_hooks:
                hook.remove()

        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Write data-indices to file.
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.dist_model.process_group,
        )

        # Adjust the learning rate.
        for group in self.optim.param_groups:
            # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
            # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
            # the corresponding values from `self.cfg`.
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
            )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits, _, _ = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs
    
    def TG_doc_eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # before move to the right device, make per sent batch attention bias
        # eval must take on exactly one device
        # make sure <bos> occur once one document

        batch_size, T = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
        update_T = batch.get("add_len")
        if batch["doc_id"] != self.cur_doc_id:
            self.kv_to_update = None
            self.doc_kv_cache = None
            self.past_key_values = None
            self.cur_doc_id = batch["doc_id"]
            self.cur_length = 0
            self.last_logProb = None
            self.logits_to_update = None
        # Freeze the preceding sentence's next-token distribution. The cache
        # commit below updates self.last_logProb to the CURRENT sentence, which
        # must not be used to score this sentence's first token.
        prefix_last_log_prob = self.last_logProb
        
        batch = move_to_device(batch, self.device)
        self.num_evaled += batch_size
        samples_per_sentence = int(evaluator.eval_loader.dataset.SENT_SIZE)
        starts_sentence = (
            (self.num_evaled - batch_size) % samples_per_sentence == 0
        )
        ends_sentence = self.num_evaled % samples_per_sentence == 0

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                
                if self.cur_length + T > self.cfg.model.max_sequence_length:
                    remove_len = self.cur_length + T - self.cfg.model.max_sequence_length
                    self.past_key_values = [
                        ( k[:, :, remove_len:, :], v[:, :, remove_len:, :] ) 
                        for k, v in self.doc_kv_cache
                    ]
                else:
                    self.past_key_values = self.doc_kv_cache
                
                out = self.dist_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    attention_bias=batch.get("attention_bias"),
                    doc_lens=batch.get("doc_lens"),
                    max_doc_lens=batch.get("max_doc_lens"),
                    use_cache=starts_sentence,
                    past_key_values = self.past_key_values,
                )
                logits, kv_cache = out.logits, out.attn_key_values
                if starts_sentence:
                    # update input_ids
                    # List[Tuple[torch.Tensor, torch.Tensor]] -> len=model_num_layers, tuple(key, value), 
                    # tensor shape: Batch * n_kv_heads * seq_length * dim_head
                    past_length = self.past_key_values[0][0].shape[-2] if self.past_key_values is not None else 0
                    self.kv_to_update = [
                        ( k[0, :, :past_length + update_T, :].clone().expand(batch_size,-1,-1,-1),
                          v[0, :, :past_length + update_T, :].clone().expand(batch_size,-1,-1,-1) ) 
                        for k, v in kv_cache
                    ]
                    self.logits_to_update = torch.log_softmax(logits[0, update_T - 1, :], dim=-1)
                if ends_sentence:
                    # update past_key_values
                    self.doc_kv_cache = self.kv_to_update
                    self.cur_length = self.doc_kv_cache[0][0].shape[-2]
                    self.last_logProb = self.logits_to_update
                    self.past_key_values = None

                logits_for_loss = logits[..., :-1, :].contiguous()
                # shape: (batch_size * seq_len, vocab_size)
                logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
                # shape: (batch_size, seq_len)
                labels = self.get_labels(batch, self.cfg.model.pad_token_id)
                # shape: (batch_size * seq_len,)
                labels = labels.view(-1)
                ce_loss = F.cross_entropy(
                    logits_for_loss, labels, ignore_index=self.cfg.model.pad_token_id, reduction="none"
                )
                ce_loss = ce_loss.view(batch_size, -1).sum(dim=1)
                if prefix_last_log_prob is not None:
                    ce_loss -= torch.gather(
                        prefix_last_log_prob,
                        dim=0,
                        index=batch["input_ids"][:, 0],
                    )

        evaluator.update_metrics(
            batch, ce_loss, logits
        )

    def _summon_params_ctx(self):
        """Context manager that temporarily unshards FSDP parameters.

        Under FSDP, calling methods on ``self.dist_model.module`` directly
        accesses sharded parameters (the wrapper only intercepts
        ``forward()``).  This context manager gathers all parameters so
        custom methods like ``word_sync_beam_search`` work correctly.

        Under DDP / single-GPU this is a no-op.
        """
        if isinstance(self.dist_model, FSDP):
            return FSDP.summon_full_params(self.dist_model)
        else:
            from contextlib import nullcontext
            return nullcontext()

    def SG_eval_step(
        self,
        batch: List[Dict[str, Any]],
        evaluator: Evaluator,
        eval_cfg=None,
        structure_mode: Optional[str] = None,
    ) -> None:
        # The main eval loop passes the exact config, which is important when a
        # run contains more than one SG evaluator. Keep the fallback for older
        # direct-call smoke utilities.
        if eval_cfg is None:
            eval_cfg = next(
                e
                for e in self.cfg.evaluators
                if e.label == "syntactic_generalization"
            )
        if structure_mode is None:
            structure_mode = resolve_structure_mode(
                eval_cfg, self.cfg.model.transformer_grammar_type
            )
        score_dict = {}
        task_name = batch[0]["task"]
        dataset = evaluator.eval_loader.dataset
        beam_size = getattr(dataset, "samples_per_sent", 300)
        tree_eval_type = getattr(dataset, "tree_eval_type", "default")
        sg_nc_ratio = getattr(dataset, "sg_nc_ratio", 1.0)
        sg_pc = getattr(dataset, "sg_pc", 3)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                for sent in batch:
                    # Plain causal LMs (terminal, pause, pushdown, treereg) use the
                    # teacher-forced CE path. pushdown/treereg are plain causal LMs at
                    # inference (no parse in the SG batch -> tree_spans=None -> pushdown
                    # depth bias vanishes, treereg loss is train-only), so they belong
                    # here, not in word_sync_beam_search (which inserts TG NT tokens the
                    # model never learned).
                    if (
                        self.cfg.model.transformer_grammar_type == "pushdown"
                        and structure_mode != "terminal"
                    ):
                        # pushdown is inference-dependent on tree_spans (its trained
                        # depth-bias path degenerates without it). The SG batch has no
                        # parse, so run a span-tracking beam search that marginalizes
                        # over incremental parses, scoring the tagged continuation CE.
                        sent_d = move_to_device(sent, self.device)
                        # Keep the tag in input coordinates. pushdown_beam_search
                        # skips an already-present BOS internally and maps each
                        # incremental marginalized surprisal back to its original
                        # token position.
                        tag = sent["tag"][0]
                        with self._summon_params_ctx():
                            score = self.dist_model.module.pushdown_beam_search(
                                eval_input_ids=sent_d["input_ids"][0],
                                beam_size=getattr(eval_cfg, "pushdown_beam_size", 20),
                                max_reduce=getattr(eval_cfg, "pushdown_max_reduce", None),
                                bos_id=dataset.vocab.bos,
                                tag=tag,
                                use_attachment_head=self.cfg.model.pushdown_use_attachment_head_inference,
                            )
                        score_dict[sent["condition_name"]] = score
                    elif (
                        structure_mode == "terminal"
                        or self.cfg.model.transformer_grammar_type[:8] == "terminal"
                        or self.cfg.model.ispause
                        or self.cfg.model.transformer_grammar_type in {"treereg", "pushdown"}
                    ):
                        sent = move_to_device(sent, self.device)
                        ce_loss, _ , logits, _, _ = self.model_forward(sent, loss_reduction="none")
                        # Align tag mask with ce_loss positions.
                        # tag and input_ids are built equal-length in prep_examples; ce_loss is one
                        # shorter (predicts tokens 1..L-1). Drop the tag's first position — the BOS
                        # slot for GPT-2, the first real token for Qwen3 — to align with ce_loss.
                        # Unconditional: no runtime length check needed.
                        tag_tensor = torch.LongTensor(sent["tag"][0])[1:].to(self.device)
                        # Clamp per-token CE: bf16 underflow can give a tagged
                        # token probability 0 → CE = inf, which then poisons
                        # score_dict and the formula eval. Cap at a large
                        # finite value (max bf16 finite ~3.4e38; this is well
                        # below and still means "near-zero probability").
                        per_tok_ce = torch.clamp(ce_loss[0], max=1.0e4)
                        score_dict[sent["condition_name"]] = torch.sum(per_tok_ce * tag_tensor).item()
                    else:
                        term_len = sent["input_ids"].shape[1]
                        nc = max(int(sg_nc_ratio * term_len), 5)
                        max_len = max(6 * term_len, 10)
                        with self._summon_params_ctx():
                            surprisal = self.dist_model.module.word_sync_beam_search(
                                vocab=evaluator.eval_loader.dataset.vocab,
                                eval_input_ids=sent["input_ids"][0],
                                max_length=max_len,
                                beam_size=beam_size,
                                nc=nc,
                                pc=sg_pc,
                                generate_TG_bias=get_TG_generate_bias_func(self.cfg, max_length=max_len + 10),
                                tag_start=sent["tag_start"],
                                tag_end=sent["tag_end"],
                                strategy = BeamSearchType.word_sync_dfs,
                                transformer_grammar_type = self.cfg.model.transformer_grammar_type,
                                tree_eval_type=tree_eval_type,
                            )

                        score_dict[sent["condition_name"]] = surprisal

        evaluator.update_metrics(
            task_name, score_dict
        )

    def beam_search_icl_eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        vocab = evaluator.eval_loader.dataset.vocab
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                for idx in range(batch["input_ids"].shape[0]):
                    ctx_len = batch["ctx_len"][idx].item()
                    cont_len = batch["cont_len"][idx].item()
                    doc_id = batch["doc_id"][idx].item()
                    cont_id = batch["cont_id"][idx].item()
                    label_id = batch["label_id"][idx].item()
                    cont_str_len = batch["cont_str_len"][idx].item()
                    cont_byte_len = batch["cont_byte_len"][idx].item()

                    # Force terminal-only format regardless of grammar type.
                    # Beam search will introduce NT structure during decoding.
                    past_input_raw = batch["input_ids"][idx, :ctx_len - 1]
                    past_input = torch.from_numpy(
                        vocab.convert_treenpy_to_terminal(past_input_raw.cpu().numpy())
                    ).long()

                    eval_raw = batch["continuation"][idx, :cont_len]
                    eval_input_ids = torch.from_numpy(
                        vocab.convert_treenpy_to_terminal(eval_raw.cpu().numpy())
                    ).long()
                    cont_len = len(eval_input_ids)

                    # Recompute string lengths from terminal format for
                    # correct len_norm / ce_loss / bpb normalization.
                    dataset = evaluator.eval_loader.dataset
                    term_cont_str = dataset.token_decode(eval_input_ids.tolist())
                    cont_str_len = len(term_cont_str) - 1
                    cont_byte_len = len(term_cont_str[1:].encode("utf-8"))

                    max_length = len(past_input) + 100 * cont_len

                    with self._summon_params_ctx():
                        beams = self.dist_model.module.word_sync_beam_search(
                            vocab=vocab,
                            eval_input_ids=eval_input_ids,
                            past_input=past_input,
                            beam_size=300,
                            generate_TG_bias=self.generate_TG_attention_bias,
                            strategy=BeamSearchType.word_sync_dfs,
                            transformer_grammar_type=self.cfg.model.transformer_grammar_type,
                            max_length=max_length,
                        )

                    if beams:
                        logprobs = torch.tensor(
                            [b["logprob"] for b in beams], device=self.device
                        )
                        log_likelihood = torch.logsumexp(logprobs, dim=0).item()
                    else:
                        log_likelihood = -float("inf")

                    evaluator.update_metrics(
                        (doc_id, cont_id, log_likelihood, label_id,
                         cont_str_len, cont_byte_len), 0.0
                    )

    def BLiMP_beam_eval_step(
        self, batch: Dict[str, Any], evaluator: Evaluator, eval_cfg
    ) -> None:
        """Beam-search BLiMP: score each sentence's terminal sequence via
        ``OLMo.word_sync_beam_search`` (parse-marginalized log-likelihood) and
        scatter the per-sentence ``logsumexp(beam logprob)`` into ``BLiMPMetric``
        via ``update_beam``. ``BLiMPMetric.compute`` (reused unchanged) then
        compares good-vs-bad per minimal pair, exactly as the teacher-forcing
        path does — only the per-sentence score source differs.

        Mirrors ``beam_search_icl_eval_step`` (call signature, beam-marginalized
        LL extraction) and ``SG_eval_step`` (nc/max_length derivation).
        ``device_eval_batch_size`` is forced to 1 for this path
        (``build_downstream_evaluator``), so ``batch`` holds one sentence.
        """
        dataset = evaluator.eval_loader.dataset
        vocab = dataset.vocab
        beam_size = eval_cfg.beam_size
        nc_ratio = eval_cfg.beam_nc_ratio
        pc = eval_cfg.beam_pc
        max_len_factor = eval_cfg.beam_max_len_factor
        tree_eval_type = getattr(dataset, "tree_eval_type", "default")
        # Pushdown is a plain-causal LM at inference; word_sync_beam_search would
        # insert TG NT tokens it never learned. Use pushdown_beam_search instead,
        # which marginalizes over shift-reduce parses and runs the trained
        # _pushdown_attention depth-bias path (tree_spans=None otherwise degenerates
        # -> PPL 78923). Mirrors SG_eval_step (train.py:1133).
        is_pushdown = self.cfg.model.transformer_grammar_type == "pushdown"
        pd_beam_size = getattr(eval_cfg, "pushdown_beam_size", 20)
        pd_max_reduce = getattr(eval_cfg, "pushdown_max_reduce", None)
        # Beam-tree dump (env-gated OLMO_BEAM_DUMP=1 -> save_beam_trees_path set in
        # build_downstream_evaluator). When enabled, decode each sentence's top-N
        # beam trees (bracketed NT sequences) and hand them to the metric, which
        # writes them to JSON in compute() for offline comparison vs blimp_tree_300.
        beam_dump_path = getattr(evaluator.eval_metric, "save_beam_trees_path", None)
        pair_per_task = getattr(dataset, "pair_per_task", 1000)
        ll_list: List[torch.Tensor] = []
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                for idx in range(batch["input_ids"].shape[0]):
                    seq = batch["input_ids"][idx]
                    # Strip trailing pad; keep BOS..EOS. word_sync_beam_search
                    # seeds past_input from seq[0] (BOS) and starts at istart=1,
                    # so the BOS is the seed (not scored) and terminals+EOS at
                    # 1..L-1 are force-scored.
                    nonpad = (seq != self.cfg.model.pad_token_id).nonzero().squeeze(-1)
                    L = int(nonpad[-1]) + 1 if nonpad.numel() else seq.shape[0]
                    eval_input_ids = seq[:L].clone()
                    nc = max(int(nc_ratio * L), 5)
                    max_len = max(max_len_factor * L, 10)
                    if is_pushdown:
                        # Parse-marginalized log p(x) via shift-reduce beam search.
                        # tag=None -> surprisal -log p(x) = -logsumexp_y logprob_y;
                        # log_likelihood = -surprisal = log p(x). update_beam stores
                        # -LL so compute()'s negation yields +LL (matches the TG path).
                        with self._summon_params_ctx():
                            surprisal = self.dist_model.module.pushdown_beam_search(
                                eval_input_ids=eval_input_ids,
                                beam_size=pd_beam_size,
                                max_reduce=pd_max_reduce,
                                bos_id=vocab.bos,
                                tag=None,
                                use_attachment_head=self.cfg.model.pushdown_use_attachment_head_inference,
                            )
                        ll = torch.tensor(-surprisal, device=self.device)
                        ll_list.append(ll)
                        # No beam-tree dump for pushdown (beams carry spans, not NT
                        # bracket tokens — record_beams expects TG tree strings).
                        continue
                    with self._summon_params_ctx():
                        beams = self.dist_model.module.word_sync_beam_search(
                            vocab=vocab,
                            eval_input_ids=eval_input_ids,
                            max_length=max_len,
                            beam_size=beam_size,
                            nc=nc,
                            pc=pc,
                            generate_TG_bias=get_TG_generate_bias_func(self.cfg, max_length=max_len + 10),
                            # word_sync (not word_sync_dfs): the terminal fast-track
                            # (model.py: ``if strategy==word_sync or tag_start``) is
                            # required so the target terminal token always seeds
                            # next_beams. word_sync_dfs lacks it, so for tree-format
                            # models that score NT tokens above the target terminal
                            # (trained on NT-dense tree sequences), every top-k
                            # extension is an NT, next_beams stays empty, and the
                            # beam dies -> empty beams -> LL=-inf. SG_eval_step avoids
                            # this by passing tag_start/tag_end (also enables the
                            # fast-track); BLiMP scores the whole sentence, so use
                            # the standard word_sync strategy instead.
                            strategy=BeamSearchType.word_sync,
                            transformer_grammar_type=self.cfg.model.transformer_grammar_type,
                            tree_eval_type=tree_eval_type,
                        )
                    if beams:
                        lp = torch.tensor([b["logprob"] for b in beams], device=self.device)
                        ll = torch.logsumexp(lp, dim=0)
                    else:
                        ll = torch.tensor(-float("inf"), device=self.device)
                    ll_list.append(ll)

                    if beam_dump_path is not None and beams:
                        self._record_beam_trees(
                            evaluator, batch, idx, eval_input_ids, beams, vocab, pair_per_task
                        )
        evaluator.eval_metric.update_beam(batch, torch.stack(ll_list))

    def pushdown_icl_eval_step(
        self, batch: Dict[str, Any], evaluator: Evaluator, eval_config=None
    ) -> None:
        """Pushdown ICL eval (boolq / downstream classification).

        Downstream strings are converted to terminal tokens (the sequence type
        used in Pushdown training) while their parser-produced TG brackets are
        retained as terminal-coordinate spans by the dataset. Custom datasets
        without retained spans fall back to best-beam parse inference. Logits
        feed ``ICLMetric.update`` unchanged.
        """
        pad_id = self.cfg.model.pad_token_id
        beam_size = getattr(eval_config, "pushdown_beam_size", 20)
        max_reduce = getattr(eval_config, "pushdown_max_reduce", None)
        batch = move_to_device(batch, self.device)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                for idx in range(batch["input_ids"].shape[0]):
                    inp = batch["input_ids"][idx:idx + 1]            # (1, L_pad)
                    real_L = int((inp[0] != pad_id).sum().item())
                    if real_L < 2:
                        continue
                    if batch.get("tree_spans") is not None:
                        spans = batch["tree_spans"][idx]
                        span_mask = batch.get("tree_span_mask")
                        if span_mask is not None:
                            spans = spans[span_mask[idx]]
                        else:
                            spans = spans[spans[:, 0] >= 0]
                    else:
                        # Fallback for downstream dataset subclasses that do not
                        # preserve parser spans in their custom prep_examples.
                        with self._summon_params_ctx():
                            _, spans = self.dist_model.module.pushdown_beam_search(
                                eval_input_ids=inp[0, :real_L],
                                beam_size=beam_size,
                                max_reduce=max_reduce,
                                bos_id=int(inp[0, 0].item()),
                                tag=None,
                                use_attachment_head=(
                                    self.cfg.model.pushdown_use_attachment_head_inference
                                ),
                                return_spans=True,
                            )
                    ts = spans.unsqueeze(0).to(self.device) if spans.shape[0] > 0 else None
                    with self._summon_params_ctx():
                        attn_mask = (inp[0:1] != pad_id)
                        out = self.dist_model.module.forward(
                            input_ids=inp, attention_mask=attn_mask, tree_spans=ts)
                    # 1-row batch for ICLMetric.update (slices continuation logits at
                    # [ctx_len-1 : ctx_len+cont_len-1], positions valid in L_pad).
                    row = {k: (v[idx:idx + 1] if torch.is_tensor(v) else v)
                           for k, v in batch.items()}
                    evaluator.update_metrics(
                        row, torch.zeros(1, device=self.device), out.logits)

    def _get_hf_tokenizer(self):
        """Lazy-init a HF tokenizers.Tokenizer for decoding beam trees to strings.

        ``SentencepieceVocab`` (compiled .so) has no ``decode``; the HF tokenizer
        loaded from the same vocab file can decode token-id lists (including NT
        bracket tokens) to readable strings. Cached on the Trainer for reuse.
        """
        if not hasattr(self, "_hf_tok"):
            from tokenizers import Tokenizer as _HFTokenizer
            vocab_path = str(self.cfg.tokenizer.vocabulary)
            # Configs use a leading "./" which Tokenizer.from_file resolves relative
            # to cwd; the repo root is the cwd for eval runs (sbatch cd's there).
            self._hf_tok = _HFTokenizer.from_file(vocab_path)
        return self._hf_tok

    def _record_beam_trees(self, evaluator, batch, idx, eval_input_ids, beams, vocab, pair_per_task):
        """Decode + record top-N beam trees for one sentence (dump path set)."""
        hf_tok = self._get_hf_tokenizer()
        sent_id = int(batch["sent_id"]) if torch.is_tensor(batch["sent_id"]) else int(batch["sent_id"])
        # Flat layout: sent_id = task_idx*(2*K) + in_task; even=good, odd=bad.
        K = pair_per_task
        task_idx = sent_id // (2 * K)
        pair_id = (sent_id % (2 * K)) // 2
        is_bad = (sent_id % 2) == 1
        task_name = BLiMP_TASK_LIST[task_idx] if task_idx < len(BLiMP_TASK_LIST) else f"task{task_idx}"
        # Terminal sentence: strip NT tokens, decode.
        term_ids = [int(t) for t in eval_input_ids.tolist() if not vocab.is_non_terminal(int(t))]
        terminal_str = hf_tok.decode(term_ids, skip_special_tokens=False)
        # Top-N beams by logprob, decoded to bracketed tree strings.
        top = sorted(beams, key=lambda b: b.get("logprob", float("-inf")), reverse=True)[:5]
        decoded_beams = []
        for b in top:
            tree_str = hf_tok.decode(
                [int(t) for t in b["input_ids"].cpu().tolist()], skip_special_tokens=False
            )
            decoded_beams.append({
                "tree": tree_str,
                "logprob": float(b.get("logprob", 0.0)),
                "terminal_logprob": float(b.get("terminal_logprob", 0.0)),
            })
        evaluator.eval_metric.record_beams(
            sent_id, task_name, pair_id, is_bad, terminal_str, decoded_beams, topk=5
        )


    def summarization_eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                with self._summon_params_ctx():
                    # Decoding path selection:
                    # - terminal / pushdown / treereg: plain autoregressive generate().
                    # - pause variants: constrained pause_generate continues the
                    #   prompt's absolute pause phase, forces the checkpoint's SEP
                    #   token (or the legacy repeated real token), and returns only
                    #   real summary tokens.  This avoids handing a fixed-format
                    #   causal LM to the tree-grammar word-sync decoder.
                    # - TG / tgtree: word_sync_beam_search with NT emission (unchanged).
                    gt = self.cfg.model.transformer_grammar_type
                    if gt in ("terminal", "pushdown", "treereg"):
                        batch = move_to_device(batch, self.device)
                        if gt == "pushdown":
                            # pushdown's depth bias degenerates without tree_spans
                            # (plain generate() -> degenerate "It is, and it is..."
                            # output). Use pushdown_generate, a shift-reduce beam
                            # search that tracks spans during generation so the
                            # _pushdown_attention depth path stays active.
                            gen = self.dist_model.module.pushdown_generate(
                                batch["input_ids"],
                                max_steps=evaluator.eval_loader.dataset.MAX_SUMMARY_LENGTH,
                                beam_size=evaluator.pushdown_beam_size,
                                max_reduce=evaluator.pushdown_max_reduce,
                                eos_token_id=self.cfg.model.eos_token_id,
                                pad_token_id=self.cfg.model.pad_token_id,
                                use_attachment_head=(
                                    self.cfg.model.pushdown_use_attachment_head_inference
                                ),
                                # XSumDataset supplies unary-collapsed gold spans
                                # for article + instruction. They establish the
                                # prompt stack tape; beam search remains only for
                                # generated summary tokens and their attachments.
                                prompt_spans=batch.get("tree_spans"),
                            )
                            predictions = gen.token_ids[:, 0, :].to(self.device)
                        else:
                            predictions = self.dist_model.module.generate(batch["input_ids"],
                                                                           max_steps=evaluator.eval_loader.dataset.MAX_SUMMARY_LENGTH,
                                                                           beam_size=6).token_ids
                            predictions = predictions[:, 0, :].to(self.device)
                    elif gt.startswith("pause"):
                        batch = move_to_device(batch, self.device)
                        p, q = self.cfg.model.pause_spec
                        generated = self.dist_model.module.pause_generate(
                            input_ids=batch["input_ids"],
                            pause_spec=(p, q),
                            max_real_tokens=evaluator.eval_loader.dataset.MAX_SUMMARY_LENGTH,
                            pause_token_id=self.cfg.model.pause_token_id,
                            vocab=evaluator.eval_loader.dataset.vocab,
                            attention_mask=batch.get("attention_mask"),
                            eos_token_id=self.cfg.model.eos_token_id,
                            beam_size=6,
                            score_pause_tokens=not is_pause_label(gt),
                        )
                        predictions = generated.token_ids[:, 0, :].to(self.device)
                    else:
                        # currently only support eval_batch_size==1
                        predictions = self.dist_model.module.word_sync_beam_search(
                                vocab = evaluator.eval_loader.dataset.vocab,
                                past_input = batch["input_ids"][0],
                                max_word_steps = evaluator.eval_loader.dataset.MAX_SUMMARY_LENGTH//2,
                                max_length = evaluator.eval_loader.dataset.MAX_SUMMARY_LENGTH,
                                beam_size=6,
                                generate_TG_bias=self.generate_TG_attention_bias,
                                strategy=BeamSearchType.default,
                                transformer_grammar_type = self.cfg.model.transformer_grammar_type,
                            )
                        predictions = predictions[0]["input_ids"].numpy()
                        predictions = evaluator.eval_loader.dataset.vocab.convert_treenpy_to_terminal(predictions)
                        predictions = torch.tensor(np.expand_dims(predictions, axis=0), device=self.device)

        
        evaluator.update_metrics(
            batch, predictions, batch["gold_summary"]
        )

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.dist_model.eval()
        eval_metrics = {}
        self.generate_TG_attention_bias = get_TG_generate_bias_func(self.cfg)
        for evaluator, eval_config in zip(self.evaluators, self.cfg.evaluators):
            structure_mode = resolve_structure_mode(
                eval_config, self.cfg.model.transformer_grammar_type
            )
            log.info(f"Running evaluation for '{evaluator.label}'...")
            # Reset metrics.
            evaluator.reset_metrics()
            if evaluator.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc):
                evaluator.eval_loader.dataset.reset()
                self.num_evaled = 0
                self.cur_length = 0
                self.cur_doc_id = 0
                self.doc_kv_cache = None
                self.kv_to_update = None
            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                if evaluator.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc):
                    self.TG_doc_eval_step(eval_batch, evaluator)
                elif evaluator.label == "syntactic_generalization":
                    self.SG_eval_step(
                        eval_batch, evaluator, eval_config, structure_mode
                    )
                elif evaluator.label == "BLiMP" and structure_mode == "beam":
                    self.BLiMP_beam_eval_step(eval_batch, evaluator, eval_config)
                elif (evaluator.type == EvaluatorType.downstream
                      and evaluator.label != "BLiMP"
                      and self.cfg.model.transformer_grammar_type == "pushdown"):
                    # BoolQ / other ICL downstream: use parser spans retained in
                    # terminal coordinates (beam fallback for custom datasets)
                    # so the trained depth path remains active. BLiMP is excluded:
                    # it uses marginalized pushdown beam scoring above.
                    self.pushdown_icl_eval_step(eval_batch, evaluator, eval_config)
                elif evaluator.type == EvaluatorType.rouge:
                    self.summarization_eval_step(eval_batch, evaluator)
                elif evaluator.type == EvaluatorType.beam_search_icl:
                    self.beam_search_icl_eval_step(eval_batch, evaluator)
                else:
                    self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")
            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches
            if evaluator.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc):
                del self.kv_to_update
                del self.doc_kv_cache
                del self.cur_doc_id
                del self.cur_length
                del self.last_logProb
                del self.logits_to_update

        # Eval compiles a bunch more versions, and the result is terrible. This way we get back to zero.
        if self.cfg.compile is not None:
            torch.compiler.reset()

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif (
                self.cfg.early_stopping_factor is not None
                and self.global_step > self.cfg.scheduler.t_warmup
                and self.cur_train_loss > self.cfg.early_stopping_factor * self.min_train_loss
            ):
                # Next check if early stopping loss criteria is met.
                should_cancel = True
                cancel_reason = "early stopping from loss increase"
            elif wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from wandb.errors import CommError

                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)
        if self.cfg.stop_at is None:
            self.cfg.stop_at = self.max_steps + 10

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()
        
        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=0, warmup=1, active=2, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                
                # torch.profiler.tensorboard_trace_handler('./log/TG-test')

                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        if self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.dist_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)


        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: int = self.cfg.stop_at if self.cfg.stop_at <= self.max_steps else self.max_steps
        save_checkpoints: bool = True

        with torch_profiler as p:
            # Optional per-step phase timing (set OLMO_STEP_PROFILE=1 to enable).
            # Prints one line/step: data_ms h2d_ms fwd_bwd_ms opt_ms gpu_tr_ms step_ms.
            _step_profile = bool(os.environ.get("OLMO_STEP_PROFILE"))
            _gpu0 = torch.device("cuda", 0) if torch.cuda.is_available() else None
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    if _step_profile:
                        import time as _t
                        _sp_data0 = _t.perf_counter()
                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    if self.cfg.finetune_task is None: # pre-training
                        assert seq_len == self.cfg.model.max_sequence_length
                        assert batch_size == self.cfg.device_train_batch_size
                        global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                        self.global_train_examples_seen_this_epoch += global_batch_size
                        self.global_train_tokens_seen += global_batch_size * seq_len
                        local_tokens = batch_size * seq_len
                    else:  # SFT
                        local_tokens = batch["input_ids"].numel()
                        global_tokens_tensor = torch.tensor(local_tokens, device=self.device)
                        dist.all_reduce(global_tokens_tensor, op=dist.ReduceOp.SUM)
                        self.global_train_examples_seen_this_epoch += batch_size * get_world_size() # Assumption, to get accurate values should use all reduce
                        self.global_train_tokens_seen += global_tokens_tensor.item()
                    self.global_step += 1
                    speed_monitor.batch_start(
                        global_total_tokens=self.global_train_tokens_seen,
                        device_batch_num_tokens=local_tokens,  # num tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        num_fwd_flops=self.model.num_fwd_flops,  # this is per token
                        num_bck_flops=self.model.num_bck_flops,  # this is per token
                        record=not first_batch,
                    )

                    should_log_this_step = self.should_log_this_step()

                    # Run train step on batch.
                    if _step_profile:
                        _sp_data_ms = (_t.perf_counter() - _sp_data0) * 1e3
                        torch.cuda.synchronize() if _gpu0 is not None else None
                        _sp_h2d0 = _t.perf_counter()
                        _sp_gs = (torch.cuda.Event(enable_timing=True) if _gpu0 is not None else None)
                        _sp_ge = (torch.cuda.Event(enable_timing=True) if _gpu0 is not None else None)
                        if _sp_gs is not None:
                            _sp_gs.record()
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)
                    if _step_profile:
                        if _sp_ge is not None:
                            _sp_ge.record(); torch.cuda.synchronize()
                            _sp_gpu_tr = _sp_gs.elapsed_time(_sp_ge)
                        else:
                            _sp_gpu_tr = 0.0
                        _sp_step_ms = (_t.perf_counter() - _sp_data0) * 1e3
                        if get_global_rank() == 0:
                            print(f"[step_profile] step={self.global_step} "
                                  f"data_ms={_sp_data_ms:.1f} "
                                  f"step_total_ms={_sp_step_ms:.1f} "
                                  f"gpu_tr_ms={_sp_gpu_tr:.1f} "
                                  f"(data+overhead={_sp_step_ms - _sp_gpu_tr:.1f})",
                                  flush=True)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())
                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(
                                f"[step={self.global_step}/{self.max_steps},epoch={epoch}]",
                                metrics,
                            )
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps},epoch={epoch}]")

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = min(stop_at, self.global_step + extra_steps)

                    # Maybe save sharded checkpoint.
                    if self.cfg.distributed_strategy != DistributedStrategy.ddp:
                        if save_checkpoints and (
                            cancel_initiated
                            or (
                                self.cfg.save_interval is not None
                                and self.global_step % self.cfg.save_interval == 0
                                and self.cfg.save_num_checkpoints_to_keep != 0
                            )
                        ):
                            log.info("Saving checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Remove any ephemeral checkpoints.
                            while self.ephemeral_checkpoints:
                                self.remove_ephemeral_checkpoint()

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                            # If the run was just canceled this will be the final checkpoint.
                            if cancel_initiated:
                                save_checkpoints = False
                        elif (
                            self.cfg.save_interval_ephemeral is not None
                            and self.global_step % self.cfg.save_interval_ephemeral == 0
                        ):
                            log.info("Saving ephemeral checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    # This code snippet should always execute when running DDP.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        log.info("Saving unsharded checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    if not cancel_initiated and (
                        self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at
                    ):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.dist_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    self.dataset.start_index = 0
                    if self.epoch < self.max_epochs:
                        log.info(f"Reshuffling data loader for epoch {self.epoch}...")
                        self.dataset.reshuffle(self.epoch)
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints and not self.cfg.eval_no_save:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
                and self.cfg.distributed_strategy == DistributedStrategy.fsdp
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
