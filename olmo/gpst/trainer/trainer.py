"""GPST hard-EM trainer — a compact, self-contained training loop.

Replaces the reference repo's ``ddp_trainer_nosync.py`` ``__main__`` script with
a class-based loop that supports:
- CPU and CUDA (single- or multi-process via DDP);
- supervised (gold merge orders in the batch) and unsupervised (parser-induced)
  modes — the difference is only whether the batch carries ``merge_orders``;
- the two-backward hard-EM pattern with ``WeightedSumFunc.a_ij_require_grad``
  toggling (struct loss: grad flows to ``a_ij``; non-struct loss: grad stopped
  at ``a_ij`` to kill left-branching bias);
- torch 2.7 AMP (``torch.amp``).

Adapted from ant-research/StructuredLM_RTDT trainer/ddp_trainer_nosync.py.
"""
from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW, Optimizer

from olmo.gpst.model.weighted_sum_func import WeightedSumFunc


@dataclass
class TrainConfig:
    lr: float = 5e-5
    parser_lr: float = 1e-3
    warmup: float = 0.01
    max_norm: float = 1.0
    accumulation_steps: int = 1
    log_steps: int = 50
    save_steps: int = 10000
    coeff_start: float = 1.0
    coeff_end: float = 1.0
    coeff_proportion: float = 0.8
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    temperature_proportion: float = 0.8
    amp: bool = True  # only active on CUDA
    max_steps: Optional[int] = None  # cap (for smoke tests)


class _LinearScheduler:
    def __init__(self, start, end, proportion, total_steps):
        self._start, self._end = start, end
        self._total = max(1, total_steps * proportion)

    def update(self, step):
        r = min(1.0, step / self._total)
        return self._start * (1 - r) + self._end * r


def _build_optimizer(model, cfg: TrainConfig) -> Optimizer:
    parser_params, model_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "parser" in name:
            parser_params.append(p)
        else:
            model_params.append(p)
    return AdamW(
        [{"params": model_params},
         {"params": parser_params, "lr": cfg.parser_lr}],
        lr=cfg.lr,
    )


def _warmup_lambda(warmup: float, total: int):
    def f(step):
        if total == 0:
            return 1.0
        warmup_steps = max(1, int(warmup * total))
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total - step) / max(1, total - warmup_steps))
    return f


@contextlib.contextmanager
def _maybe_no_sync(model):
    """Enter ``model.no_sync()`` if the model is DDP-wrapped; else a no-op."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        with model.no_sync():
            yield
    else:
        yield


def train(model, data_loader, cfg: TrainConfig, device, logger=None,
          output_dir: Optional[str] = None, is_master: bool = True):
    """Run the hard-EM training loop. Returns a dict of the last logged metrics."""
    logger = logger or logging.getLogger("gpst")
    model.train()
    optimizer = _build_optimizer(model, cfg)
    total_steps = len(data_loader)
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_lambda(cfg.warmup, total_steps))
    coeff_sched = _LinearScheduler(cfg.coeff_start, cfg.coeff_end,
                                   cfg.coeff_proportion, total_steps)
    temp_sched = _LinearScheduler(cfg.temperature_start, cfg.temperature_end,
                                  cfg.temperature_proportion, total_steps)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics = {}
    step = 0
    while step < total_steps:
        for inputs in data_loader:
            if step >= total_steps:
                break
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            coeff = coeff_sched.update(step)
            temperature = temp_sched.update(step)
            with _maybe_no_sync(model):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    result = model(**inputs, coeff=coeff, temperature=temperature)
                # ---- hard-EM two-backward ----
                WeightedSumFunc.a_ij_require_grad = True
                scaler.scale(result.struct_loss / cfg.accumulation_steps).backward(retain_graph=True)
                WeightedSumFunc.a_ij_require_grad = False
                scaler.scale(result.non_struct_loss / cfg.accumulation_steps).backward()

            if (step + 1) % cfg.accumulation_steps == 0:
                # DDP ``no_sync()`` suppresses the automatic gradient sync across
                # both backward passes above, so the accumulated grads are
                # rank-local. Mirror the reference ``ddp_trainer_nosync.py``
                # and manually all-reduce before unscaling. On CPU / single-GPU
                # (no process group) this is a no-op.
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= world_size
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                optimizer.zero_grad()

            total_loss = float(result.struct_loss.detach()) + float(result.non_struct_loss.detach())
            metrics = {
                "step": step, "total_loss": total_loss,
                "struct_loss": float(result.struct_loss.detach()),
                "non_struct_loss": float(result.non_struct_loss.detach()),
                "gpt_loss": _f(getattr(result, "gpt_loss", None)),
                "parser_loss": _f(getattr(result, "parser_loss", None)),
                "inside_outside_loss": _f(getattr(result, "inside_outside_loss", None)),
                "action_loss": _f(getattr(result, "action_loss", None)),
            }
            if step % cfg.log_steps == 0:
                logger.info(f"step {step}/{total_steps} loss={total_loss:.4f} "
                            f"gpt={metrics['gpt_loss']:.4f} "
                            f"ae={metrics['inside_outside_loss']:.4f} "
                            f"parser={metrics['parser_loss']:.4f} "
                            f"action={metrics['action_loss']:.4f}")
            if is_master and output_dir and step > 0 and step % cfg.save_steps == 0:
                _save(model, os.path.join(output_dir, f"model_{step}.bin"), logger)
            step += 1
    if is_master and output_dir:
        _save(model, os.path.join(output_dir, "model.bin"), logger)
    return metrics


def _f(x):
    if x is None:
        return 0.0
    return float(x.detach()) if hasattr(x, "detach") else float(x)


def _save(model, path, logger):
    try:
        torch.save(model.state_dict(), path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"failed to save {path}: {e}")
