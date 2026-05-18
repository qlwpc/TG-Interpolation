"""Merge (average) multiple model checkpoints into one.

Usage:
    python scripts/merge_checkpoints.py \
        --checkpoints path/step100000-unsharded path/step110000-unsharded path/step120000-unsharded \
        --output path/merged-model

    # With custom weights:
    python scripts/merge_checkpoints.py \
        --checkpoints path/step100000-unsharded path/step120000-unsharded \
        --weights 0.4 0.6 \
        --output path/merged-model
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import torch

from olmo.aliases import PathOrStr

log = logging.getLogger(__name__)


def merge_checkpoints(
    checkpoint_dirs: List[PathOrStr],
    output_dir: PathOrStr,
    weights: Optional[List[float]] = None,
) -> None:
    """Average model weights from multiple checkpoints and save the result.

    Args:
        checkpoint_dirs: List of checkpoint directory paths.
        output_dir: Directory to save the merged checkpoint.
        weights: Optional per-checkpoint weights. Defaults to uniform.
    """
    if len(checkpoint_dirs) < 2:
        raise ValueError("Need at least 2 checkpoints to merge.")

    if weights is None:
        weights = [1.0 / len(checkpoint_dirs)] * len(checkpoint_dirs)
    elif len(weights) != len(checkpoint_dirs):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of checkpoints ({len(checkpoint_dirs)})."
        )

    # Normalize weights to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights]

    # Load all model state dicts
    all_state_dicts = []
    reference_keys = None
    for ckpt_dir, weight in zip(checkpoint_dirs, weights):
        model_path = Path(ckpt_dir) / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        all_state_dicts.append(state_dict)

        if reference_keys is None:
            reference_keys = set(state_dict.keys())
        elif set(state_dict.keys()) != reference_keys:
            raise ValueError(
                f"Key mismatch: {ckpt_dir} has different keys from the first checkpoint."
            )

        log.info("Loaded %s (weight=%.3f), %d parameters", ckpt_dir, weight, len(state_dict))

    # Average state dicts
    merged_state_dict: dict = {}
    for key in reference_keys:
        tensors = [sd[key] for sd in all_state_dicts]
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1:
            # Cast all to the dtype of the first tensor
            target_dtype = tensors[0].dtype
            tensors = [t.to(target_dtype) for t in tensors]

        stacked = torch.stack(tensors, dim=0)
        weighted = stacked * torch.tensor(weights, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        merged_state_dict[key] = weighted.sum(dim=0).to(tensors[0].dtype)

    # Save merged checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged model.pt
    model_output = output_dir / "model.pt"
    torch.save(merged_state_dict, model_output)
    log.info("Saved merged model (%d keys) to %s", len(merged_state_dict), model_output)

    # Copy config.yaml from first checkpoint
    first_ckpt = Path(checkpoint_dirs[0])
    config_src = first_ckpt / "config.yaml"
    if config_src.is_file():
        shutil.copy2(config_src, output_dir / "config.yaml")
        log.info("Copied config.yaml from %s", first_ckpt)
    else:
        log.warning("No config.yaml found in %s", first_ckpt)

    # Copy train.pt from first checkpoint
    train_src = first_ckpt / "train.pt"
    if train_src.is_file():
        shutil.copy2(train_src, output_dir / "train.pt")
        log.info("Copied train.pt from %s", first_ckpt)

    log.info("Merge complete. Merged checkpoint at %s", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Merge (average) multiple model checkpoints into one.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint directories to merge.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the merged checkpoint.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-checkpoint weights (default: uniform).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    merge_checkpoints(args.checkpoints, args.output, args.weights)
