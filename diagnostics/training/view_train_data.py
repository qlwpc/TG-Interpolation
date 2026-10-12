import gzip
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")

def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    barrier()

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.


    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg)
    tokenizer = Tokenizer.from_train_config(cfg)
    mmp = train_loader.dataset.dataset
    # for batch in train_loader:
    #     input_ids = batch["input_ids"]
    #     print(input_ids)
    #     print(batch["index"])
    #     for b in range(4):
    #         print(tokenizer.decode(input_ids[b].tolist())[:100])        
    #         print(mmp[batch["index"][b]])
    #     break
    idx = 18321345
    print(mmp[idx]["input_ids"].tolist())
    print(tokenizer.decode(mmp[idx]["input_ids"].tolist()))

if __name__ == "__main__":
    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
