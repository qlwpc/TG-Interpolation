"""Experiment 1: Per-token-type log-probability profiles.

Loads a model checkpoint, runs a forward pass over a validation corpus in tree
format, and computes mean log-probability separately for terminal and
non-terminal tokens. Tests the hypothesis that tree/tgtree models assign
higher probability to non-terminals than to terminals.

Usage:
    python analysis/nonterminal_noise/exp1_token_profile.py \
        --checkpoint saved_models/A800_models/tree_1B/step137217-unsharded \
        --config saved_models/A800_models/tree_1B/step137217-unsharded/config.yaml \
        --data_glob "dataset/bbc-news/tree/*.npy" \
        --max_samples 500 \
        --output analysis-output/exp1/tree_1B_profile.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-token-type log-probability profiles"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to unsharded checkpoint directory")
    p.add_argument("--config", required=True,
                   help="Path to model config.yaml")
    p.add_argument("--data_glob", required=True,
                   help="Glob for .npy memmap files in tree format")
    p.add_argument("--max_samples", type=int, default=500,
                   help="Maximum number of sequences to evaluate")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--output", required=True,
                   help="Output JSON file path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def load_model(checkpoint_dir, config_path, device):
    """Load OLMo model from checkpoint."""
    from olmo.config import TrainConfig
    from olmo.model import OLMo

    cfg = TrainConfig.from_yaml(config_path)
    model = OLMo(cfg.model).to(device)
    model.eval()

    model_path = Path(checkpoint_dir) / "model.pt"
    if model_path.exists():
        state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded model weights from {model_path}")
    else:
        log.warning(f"model.pt not found at {model_path}, using random init")

    return model, cfg


def load_validation_data(data_glob, max_samples, max_seq_len):
    """Load tree-format sequences from memmap files."""
    import glob as glob_mod
    from olmo.data.memmap_dataset import MemMapDataset

    files = sorted(glob_mod.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No files match {data_glob}")
    log.info(f"Found {len(files)} data files")

    dataset = MemMapDataset(
        paths=files,
        chunk_size=max_seq_len + 1,
        memmap_dtype=np.uint32,
        metadata_dtype=np.int32,
    )

    n = min(len(dataset), max_samples)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), n, replace=False)
    samples = [dataset[int(i)] for i in indices]
    log.info(f"Loaded {len(samples)} samples")
    return samples


def classify_tokens(token_ids, vocab):
    """Return boolean array: True for terminal tokens, False for non-terminals."""
    token_ids = np.asarray(token_ids)
    lo = vocab.opening_non_terminals[0]
    hi = vocab.closing_non_terminals[1]
    return (token_ids < lo) | (token_ids > hi)


def compute_profile(model, samples, vocab, device):
    """Run forward pass and collect per-token-type log-probs."""
    all_term_logp = []
    all_nonterm_logp = []
    all_term_count = []
    all_nonterm_count = []

    with torch.no_grad():
        for i, input_ids in enumerate(samples):
            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids_t)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Teacher-forced: log_prob of token[t] given token[:t]
            target_ids = input_ids_t[0, 1:]
            pred_log_probs = log_probs[0, :-1, :]
            token_log_probs = torch.gather(
                pred_log_probs, 1, target_ids.unsqueeze(-1)
            ).squeeze(-1)

            is_terminal = classify_tokens(target_ids.cpu().numpy(), vocab)
            is_terminal_t = torch.tensor(is_terminal, device=device)

            term_lps = token_log_probs[is_terminal_t]
            nonterm_lps = token_log_probs[~is_terminal_t]

            if len(term_lps) > 0:
                all_term_logp.append(term_lps.mean().item())
                all_term_count.append(len(term_lps))
            if len(nonterm_lps) > 0:
                all_nonterm_logp.append(nonterm_lps.mean().item())
                all_nonterm_count.append(len(nonterm_lps))

            if (i + 1) % 50 == 0:
                log.info(f"  Processed {i + 1}/{len(samples)} sequences")

    n_seqs = len(all_term_logp)
    if n_seqs == 0:
        raise RuntimeError("No terminal tokens found in any sequence")

    return {
        "mean_term_logp": float(np.mean(all_term_logp)),
        "mean_nonterm_logp": float(np.mean(all_nonterm_logp)) if all_nonterm_logp else None,
        "std_term_logp": float(np.std(all_term_logp)),
        "std_nonterm_logp": float(np.std(all_nonterm_logp)) if all_nonterm_logp else None,
        "n_term_tokens": int(np.sum(all_term_count)),
        "n_nonterm_tokens": int(np.sum(all_nonterm_count)),
        "n_sequences": n_seqs,
        "per_sequence": [
            {"term_logp": float(t), "nonterm_logp": float(n) if n is not None else None,
             "term_count": int(tc), "nonterm_count": int(nc)}
            for t, n, tc, nc in zip(
                all_term_logp,
                all_nonterm_logp + [None] * (len(all_term_logp) - len(all_nonterm_logp)),
                all_term_count,
                all_nonterm_count + [0] * (len(all_term_count) - len(all_nonterm_count)),
            )
        ],
    }


def main():
    args = parse_args()

    log.info(f"Loading model from {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, args.device)

    from olmo.data.tg_mask import SentencepieceVocab
    vocab = SentencepieceVocab.from_vocab_file(cfg.tokenizer.vocabulary)
    log.info(f"Non-terminal range: {vocab.opening_non_terminals} - {vocab.closing_non_terminals}")

    log.info(f"Loading validation data from {args.data_glob}")
    samples = load_validation_data(args.data_glob, args.max_samples, args.max_seq_len)

    log.info(f"Computing per-token-type profile on {len(samples)} samples")
    profile = compute_profile(model, samples, vocab, args.device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)
    log.info(f"Saved profile to {args.output}")

    gap = (profile["mean_nonterm_logp"] or 0) - profile["mean_term_logp"]
    print(f"\n=== Results ===")
    print(f"Sequences evaluated:     {profile['n_sequences']}")
    print(f"Mean terminal logp:      {profile['mean_term_logp']:.4f}")
    if profile["mean_nonterm_logp"] is not None:
        print(f"Mean non-terminal logp:  {profile['mean_nonterm_logp']:.4f}")
        print(f"Gap (NT - T):            {gap:+.4f}")
        print(f"N non-terminal tokens:   {profile['n_nonterm_tokens']}")
    else:
        print(f"Mean non-terminal logp:  N/A (no non-terminals in data)")
    print(f"N terminal tokens:       {profile['n_term_tokens']}")


if __name__ == "__main__":
    main()
