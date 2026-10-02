"""CPU contract checks for the public dedup generator, without allocating a model."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from omegaconf import OmegaConf
import pytest

from scripts import prepare_bbc_dedup_500m as prep


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    template = root / "train_configs/templates/tree-500M.yaml"
    template.parent.mkdir(parents=True)
    shutil.copyfile(prep.ROOT / "train_configs/templates/tree-500M.yaml", template)
    tokenizer = root / "tokenizer.json"
    shutil.copyfile(prep.ROOT / "reproducibility/tokenizer-reference.json", tokenizer)
    monkeypatch.setattr(prep, "ROOT", root)
    monkeypatch.setattr(prep, "count_non_embedding_params", lambda _: 507896576)
    for fmt in set(prep.FORMATS.values()):
        folder = root / "data" / fmt
        folder.mkdir(parents=True)
        # Sparse NPY: large enough for the 2000-step warmup without writing GBs.
        train = folder / "train.npy"
        arr = np.lib.format.open_memmap(train, mode="w+", dtype="uint16", shape=(100_000_000,))
        del arr
        receipt = dict(status="complete", tokens=100_000_000, dtype="uint16", format=fmt,
                       sha256="a" * 64, tokenizer_sha256=prep.digest(tokenizer))
        train.with_suffix(".manifest.json").write_text(json.dumps(receipt))
        for split, documents in (("dev", 1000), ("test", 5000)):
            path = folder / f"{split}.npy"
            np.save(path, np.zeros(documents * 3, dtype="uint16"))
            offsets = folder / f"{split}_doc_offsets.npy"
            np.save(offsets, np.arange(documents + 1, dtype="uint64") * 3)
            receipt.update(tokens=documents * 3, documents=documents, sha256=prep.digest(path),
                           doc_offsets_sha256=prep.digest(offsets))
            path.with_suffix(".manifest.json").write_text(json.dumps(receipt))
    return root, ["--data-dir", str(root / "data"), "--eval-dir", str(root / "data"),
                  "--tokenizer", str(tokenizer), "--gpus", "2"]


def test_all_model_contracts_and_nondivisible_fixed_batch(inputs):
    root, args = inputs
    out = root / "campaign"
    prep.main(args + ["--models", *prep.FORMATS, "--fixed-microbatch", "4", "--output-dir", str(out)])
    for name in prep.FORMATS:
        run = out / name
        cfg = OmegaConf.load(run / "config.yaml")
        smoke = OmegaConf.load(run / "smoke.yaml")
        audit = json.loads((run / "protocol.json").read_text())
        typed = name in prep.TYPED_MODELS
        assert cfg.model.tg_typed_attention == typed
        assert cfg.model.flash_attention == (not typed and name != "tgnomask_aug")
        assert cfg.model.flex_attention == (name == "tgnomask_aug")
        assert cfg.model.transformer_grammar_type == ("mixing" if name == "tgnomask_mix_tg" else name)
        assert cfg.global_train_batch_size == 10
        assert audit["batch"]["gradient_accumulation_steps"] == 2
        assert audit["batch"]["last_microbatch"] == 1
        assert cfg.device_train_microbatch_size == 4
        assert cfg.eval_interval == 3000 and cfg.eval_subset_num_batches == -1
        assert cfg.max_duration == "1ep" and cfg.stop_at is None
        assert cfg.wandb.mode == "offline"
        assert cfg.save_overwrite is False
        assert cfg.save_interval is None and cfg.save_interval_unsharded == 3000
        assert smoke.stop_at == 3 and smoke.eval_subset_num_batches == 2
        assert smoke.save_folder != cfg.save_folder
        for data in [cfg.data, *[e.data for e in cfg.evaluators]]:
            assert data.memmap_format == "npy"
            assert data.generate_doc_lengths == (not typed)
            assert data.generate_attention_mask == (name == "tgnomask_aug")
        if name == "tgnomask_mix_tg":
            assert [(g.grammar_type, g.n_heads) for g in cfg.model.mix_head_type] == [("tg", 8), ("tgnomask", 8)]
        assert audit["microbatch_status"] == "pending_gpu_probe"
        for filename, key in (("config.yaml", "config_sha256"), ("base.yaml", "base_config_sha256"),
                              ("smoke.yaml", "smoke_config_sha256")):
            assert prep.digest(run / filename) == audit[key]
        assert "timeout --signal=TERM --kill-after=30s 1800s torchrun" in (run / "smoke.sh").read_text()
        assert audit["smoke"]["world_size"] == 2
        for script in ("launch.sh", "smoke.sh"):
            subprocess.run(["bash", "-n", str(run / script)], check=True)
    with pytest.raises(FileExistsError):
        prep.main(args + ["--output-dir", str(out)])


def test_runtime_paths_and_default_candidate(inputs):
    root, args = inputs
    out = root / "campaign"
    prep.main(args + ["--models", "tree_noont", "--runtime-root", "/remote/repo with spaces",
                      "--output-dir", str(out)])
    cfg = OmegaConf.load(out / "tree_noont/config.yaml")
    assert cfg.device_train_microbatch_size == 1
    assert cfg.data.paths == ["/remote/repo with spaces/data/tree_noont/train.npy"]
    assert cfg.evaluators[0].data.paths == ["/remote/repo with spaces/data/tree_noont/dev.npy"]
    assert cfg.save_folder == "/remote/repo with spaces/campaign/tree_noont/checkpoints"
    launch = (out / "tree_noont/launch.sh").read_text()
    assert "WANDB_MODE=offline" in launch and "'/remote/repo with spaces/campaign/tree_noont/config.yaml'" in launch


@pytest.mark.parametrize("fault", ["tokenizer", "train_manifest", "eval_hash", "offsets", "payload", "train_hash"])
def test_invalid_inputs_leave_no_output(inputs, fault):
    root, args = inputs
    folder = root / "data/tg"
    if fault in ("tokenizer", "train_manifest"):
        path = folder / "train.manifest.json"
        receipt = json.loads(path.read_text())
        receipt["tokenizer_sha256" if fault == "tokenizer" else "tokens"] = "invalid"
        path.write_text(json.dumps(receipt))
    elif fault == "eval_hash":
        arr = np.load(folder / "dev.npy", mmap_mode="r+")
        arr[0] = 1
        arr.flush()
    elif fault == "offsets":
        np.save(folder / "dev_doc_offsets.npy", np.zeros(1001, dtype="uint64"))
    elif fault == "payload":
        with (folder / "train.npy").open("ab") as handle:
            handle.write(b"x")
    else:
        args += ["--verify-train-hash"]
    out = root / "bad"
    with pytest.raises(ValueError):
        prep.main(args + ["--models", "tg", "--output-dir", str(out)])
    assert not out.exists()
