"""Phase 5 end-to-end test: run the launcher (run_gpst.py) for a few steps on
CPU in both unsupervised (synthetic lazy corpus) and supervised (BBC gold trees)
modes. Skips if the needed corpus/tokenizer is unavailable.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_R2D2 = "olmo/gpst/data/en_config/r2d2_256_4_1.json"
_GPT = "olmo/gpst/data/gpt2-small/config.json"
_VOCAB = "olmo/gpst/data/gpt2-small"


def _run(cmd):
    e = dict(os.environ)
    e["PYTHONPATH"] = REPO + (os.pathsep + e.get("PYTHONPATH", ""))
    e["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
    return subprocess.run(cmd, cwd=REPO, env=e, capture_output=True, text=True)


def test_e2e_supervised_cpu():
    tree_npy = "dataset/bbc-news/tree/dev.npy"
    tok = "dataset/bbc-news/TG_GPT2_tokenizer.json"
    if not (os.path.exists(os.path.join(REPO, tree_npy)) and os.path.exists(os.path.join(REPO, tok))):
        pytest.skip("BBC tree corpus not present")
    cmd = [sys.executable, "scripts/gpst/run_gpst.py",
           "--supervised",
           "--tree_npy", tree_npy,
           "--tokenizer_path", tok,
           "--r2d2_config_path", _R2D2,
           "--gpt_config_path", _GPT,
           "--vocab_dir", _VOCAB,
           "--output_dir", "/tmp/gpst_e2e_sup",
           "--batch_size", "4",
           "--num_samples", "8",
           "--max_seq_len", "64",
           "--log_steps", "1",
           "--max_steps", "3",
           "--lr", "5e-5", "--parser_lr", "1e-3"]
    r = _run(cmd)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr[-3000:]}"
    assert "loss=" in r.stderr or "loss=" in r.stdout
    assert os.path.exists("/tmp/gpst_e2e_sup/model.bin")


def test_e2e_unsupervised_cpu(tmp_path):
    import numpy as np
    import pickle
    from transformers import GPT2TokenizerFast
    if not os.path.exists(os.path.join(REPO, _VOCAB, "vocab.json")):
        pytest.skip("GPT-2 vocab not present")
    tok = GPT2TokenizerFast.from_pretrained(os.path.join(REPO, _VOCAB))
    lazy_dir = tmp_path / "tiny.lazy"
    lazy_dir.mkdir()
    data_file = lazy_dir / "data"
    lengths = []
    with open(data_file, "wb") as out:
        for s in ["the cat sat on the mat", "a dog ran fast", "fruit flies like a banana"]:
            ids = tok.encode(s, add_special_tokens=False)
            out.write(np.array(ids, dtype=np.int32).tobytes(order="C"))
            lengths.append(len(ids))
            out.write(np.array([], dtype=np.int32).tobytes(order="C"))
            lengths.append(0)
    with open(lazy_dir / "data.len.pkl", "wb") as f:
        pickle.dump(lengths, f)
    cmd = [sys.executable, "scripts/gpst/run_gpst.py",
           "--unsupervised",
           "--corpus_path", str(lazy_dir),
           "--r2d2_config_path", _R2D2,
           "--gpt_config_path", _GPT,
           "--vocab_dir", _VOCAB,
           "--output_dir", "/tmp/gpst_e2e_unsup",
           "--batch_size", "4",
           "--num_samples", "8",
           "--max_seq_len", "32",
           "--log_steps", "1",
           "--max_steps", "3",
           "--lr", "5e-5", "--parser_lr", "1e-3"]
    r = _run(cmd)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr[-3000:]}"
