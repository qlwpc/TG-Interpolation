"""CPU smoke test for beam-search BLiMP + the BLiMPMetric sign convention.

Two parts:

(1) Metric sign-convention test (no checkpoint, no GPU): pins that
    ``BLiMPMetric.update_beam`` stores ``-LL`` so that ``compute()``'s negation
    yields ``+LL`` and the good>bad comparison works. With
    ``LL=[-1.0,-2.0,-1.5,-0.5]`` (good,bad,good,bad across 2 pairs), the
    expected task0 accuracy is 0.5 (pair0 good>bad, pair1 good<bad).

(2) Beam-search smoke test (requires a checkpoint + the compiled tg_mask.so):
    loads the treereg (or pushdown) checkpoint on CPU, takes task 0 pair 0
    (good/bad) from ``blimp_terminal.npy``, calls
    ``OLMo.word_sync_beam_search`` in scoring mode, and asserts the returned
    beams have finite ``logprob`` and that ``logsumexp(logprobs)`` is finite for
    both good and bad. Does NOT assert good>bad (one pair is not statistically
    meaningful) — it checks the plumbing.

Run:
    PYTHONPATH=. python tests/test_blimp_beam_search.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmo.eval.downstream import BLiMPMetric, BLiMP_TASK_LIST  # noqa: E402

VOCAB = "./dataset/bbc-news/TG_GPT2_tokenizer.json"
BLIMP_NPY = "./dataset/BLiMP/tree300/blimp_terminal.npy"
PAD = 50258


def test_metric_sign_convention():
    """update_beam stores -LL; compute() negates to +LL; good>bad -> correct."""
    pair_per_task = 2  # 2 pairs -> 4 slots for task_list[0]
    metric = BLiMPMetric(
        vocab_path=VOCAB,
        dataset_name="terminal",
        device_eval_batch_size=1,
        dataset_length=len(BLiMP_TASK_LIST) * 2 * pair_per_task,
        samples_per_sent=1,
        pair_per_task=pair_per_task,
        tree_eval_type="default",
    )
    metric.to(torch.device("cpu"))

    # 4 sentences: pair0 = (good=-1.0, bad=-2.0) -> good>bad (correct)
    #              pair1 = (good=-1.5, bad=-0.5) -> good<bad (incorrect)
    # These are LOG-PROBS (higher=better), as update_beam expects.
    lls = [-1.0, -2.0, -1.5, -0.5]
    for sid, ll in enumerate(lls):
        metric.update_beam({"sent_id": torch.tensor(sid)}, torch.tensor([ll]))

    acc = metric.compute()
    task0 = BLiMP_TASK_LIST[0]
    task0_key = next(k for k in acc if k.endswith("/" + task0))
    print(f"task0 ({task0}) acc = {acc[task0_key]} ; overall = {acc['overall/overall']}")
    assert abs(acc[task0_key] - 0.5) < 1e-6, f"expected 0.5, got {acc[task0_key]}"
    print("PASS: metric sign convention (update_beam -> compute) correct.")


def test_pair_per_task_subset(tmp_path=None):
    """With pair_per_task=K, compute() divides by K (not 1000): a subset run
    that scores all K pairs of task 0 reports a real per-task accuracy instead
    of being dragged down by 67000-K un-scored (zero) slots."""
    import tempfile
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    K = 2  # 2 pairs/task -> dataset_length = 67*2*2 = 268
    metric = BLiMPMetric(
        vocab_path=VOCAB,
        dataset_name="terminal",
        device_eval_batch_size=1,
        dataset_length=len(BLiMP_TASK_LIST) * 2 * K,
        samples_per_sent=1,
        pair_per_task=K,
        tree_eval_type="default",
    )
    metric.to(torch.device("cpu"))
    # Score the first K pairs (4 sentences) of task 0: good,bad,good,bad.
    # pair0 = (good=-1.0, bad=-2.0) -> correct; pair1 = (good=-1.5, bad=-0.5) -> wrong.
    lls = [-1.0, -2.0, -1.5, -0.5]
    for sid, ll in enumerate(lls):
        metric.update_beam({"sent_id": torch.tensor(sid)}, torch.tensor([ll]))
    acc = metric.compute()
    task0 = BLiMP_TASK_LIST[0]
    task0_key = next(k for k in acc if k.endswith("/" + task0))
    # task0 has 1/2 correct = 0.5 (NOT 1/67000). Pins the subset fix.
    assert abs(acc[task0_key] - 0.5) < 1e-6, f"expected task0 acc 0.5, got {acc[task0_key]}"
    # Overall = 1 correct pair / (K * 67) = 1 / 134 (only task 0 scored).
    expected_overall = 1.0 / (K * len(BLiMP_TASK_LIST))
    assert abs(acc["overall/overall"] - expected_overall) < 1e-6, (
        f"expected overall {expected_overall}, got {acc['overall/overall']}"
    )
    print(f"PASS: pair_per_task subset (K={K}) -> task0={acc[task0_key]}, "
          f"overall={acc['overall/overall']:.6f} (no zero drag-down).")


def test_beam_tree_dump(tmp_path=None):
    """record_beams accumulates JSON-serializable records; compute() writes them
    to disk when save_beam_trees_path is set."""
    import tempfile
    from pathlib import Path
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    K = 1
    dump_path = str(tmp_path / "beam_trees_BLiMP.jsonl")
    metric = BLiMPMetric(
        vocab_path=VOCAB,
        dataset_name="terminal",
        device_eval_batch_size=1,
        dataset_length=len(BLiMP_TASK_LIST) * 2 * K,
        samples_per_sent=1,
        pair_per_task=K,
        tree_eval_type="default",
        save_beam_trees_path=dump_path,
    )
    metric.to(torch.device("cpu"))
    # One good sentence (sent_id=0) with 2 fake beams.
    fake_beams = [
        {"tree": "<(S> a b <S)>", "logprob": -1.0, "terminal_logprob": -0.9},
        {"tree": "<(S> a <(NP> b <NP)> <S)>", "logprob": -1.5, "terminal_logprob": -1.4},
    ]
    metric.record_beams(0, BLiMP_TASK_LIST[0], 0, False, "a b .", fake_beams, topk=5)
    # Feed an LL so compute() has a valid loglikelihoods slot to scatter.
    metric.update_beam({"sent_id": torch.tensor(0)}, torch.tensor([-1.0]))
    metric.compute()
    import json
    import os
    # _save_beam_trees writes per-rank: <base>_rank0.jsonl (inserts _rank{R}
    # before the trailing .jsonl).
    rank0_path = dump_path[:-6] + "_rank0.jsonl" if dump_path.endswith(".jsonl") else dump_path + "_rank0.jsonl"
    assert os.path.exists(rank0_path), f"beam-tree dump file not written at {rank0_path}"
    with open(rank0_path) as f:
        records = json.load(f)
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    r = records[0]
    assert r["task"] == BLiMP_TASK_LIST[0]
    assert r["good_bad"] == "good"
    assert r["terminal"] == "a b ."
    assert len(r["beams"]) == 2
    # Top beam is the higher-logprob one (sorted desc).
    assert r["beams"][0]["tree"] == "<(S> a b <S)>"
    assert abs(r["beams"][0]["logprob"] - (-1.0)) < 1e-9
    print("PASS: beam-tree dump (record_beams -> compute -> JSON) correct.")


def _load_terminal_pair():
    """Return (good_seq, bad_seq) tensors for task 0 pair 0, pad stripped."""
    arr = np.load(BLIMP_NPY, mmap_mode="r")
    good = arr[0, 0]
    bad = arr[0, 1]
    good = torch.LongTensor([int(x) for x in good if int(x) != PAD])
    bad = torch.LongTensor([int(x) for x in bad if int(x) != PAD])
    return good, bad


def _dummy_train_cfg(model_cfg, vocab_path):
    """Minimal shim so get_TG_generate_bias_func(cfg) gets model.transformer_grammar_type."""
    class _C:
        pass
    c = _C()
    c.model = model_cfg
    c.tokenizer = type("t", (), {"vocabulary": vocab_path})()
    return c


def test_beam_search_smoke():
    """Plumbing check: word_sync_beam_search returns finite beam logprobs."""
    ckpt = "saved_models/treereg/step33862-unsharded"
    if not os.path.isdir(ckpt):
        ckpt = "saved_models/pushdown/step33862-unsharded"
    if not os.path.isdir(ckpt):
        print("SKIP: no checkpoint found (saved_models/treereg or /pushdown).")
        return
    if not os.path.exists(BLIMP_NPY):
        print("SKIP: blimp_terminal.npy not found.")
        return

    from olmo.config import ModelConfig, BeamSearchType
    from olmo.model import OLMo
    from olmo.data import get_TG_generate_bias_func
    from olmo.data.tg_mask import SentencepieceVocab

    cfg = ModelConfig.load(os.path.join(ckpt, "config.yaml"), key="model", validate_paths=False)
    cfg.init_device = "cpu"
    model = OLMo(cfg)
    sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu")
    model.load_state_dict(model._make_state_dict_compatible(sd)[0])
    # CPU can't run flex_attention inductor kernels; force SDPA fallback.
    model.config.flex_attention = False
    for block in model.transformer.blocks:
        if hasattr(block, "flex_attention"):
            block.flex_attention = None
    model.eval()

    vocab = SentencepieceVocab.from_vocab_file(VOCAB)
    good, bad = _load_terminal_pair()
    results = {}
    for name, seq in [("good", good), ("bad", bad)]:
        L = seq.shape[0]
        nc = max(int(1.2 * L), 5)
        max_len = max(3 * L, 10)
        with torch.no_grad():
            beams = model.word_sync_beam_search(
                vocab=vocab,
                eval_input_ids=seq,
                max_length=max_len,
                beam_size=20,
                nc=nc,
                pc=4,
                generate_TG_bias=get_TG_generate_bias_func(
                    _dummy_train_cfg(cfg, VOCAB), max_length=max_len + 10
                ),
                strategy=BeamSearchType.word_sync_dfs,
                transformer_grammar_type=cfg.transformer_grammar_type,
                tree_eval_type="default",
            )
        assert beams, f"{name}: no beams returned"
        lps = torch.tensor([b["logprob"] for b in beams])
        ll = torch.logsumexp(lps, dim=0).item()
        assert torch.isfinite(torch.tensor(ll)), f"{name}: non-finite logsumexp {ll}"
        results[name] = ll
        print(f"{name}: L={L} n_beams={len(beams)} logsumexp(logprob)={ll:.4f}")
    assert results["good"] != results["bad"], "good and bad LL identical (suspicious)"
    print("PASS: beam-search smoke (finite, comparable LLs for good/bad).")


if __name__ == "__main__":
    if not os.path.exists(VOCAB):
        print("SKIP: tokenizer not found (run from repo root).")
        sys.exit(0)
    test_metric_sign_convention()
    test_pair_per_task_subset(None)
    test_beam_tree_dump(None)
    test_beam_search_smoke()
