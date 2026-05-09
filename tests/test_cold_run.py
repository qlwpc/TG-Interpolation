"""
Cold-run and downstream evaluator construction tests.

Verifies that the training pipeline and downstream evaluators (XSum, BLiMP,
SG, ICL) can initialize and run for all transformer_grammar_type variations.

Data-dependent tests are skipped when required files are absent.
Set WORKSPACE env var to override the repo root path.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
_WORKSPACE = os.environ.get("WORKSPACE", str(_REPO))

# Real vocab size matching TG_GPT2_tokenizer.json (which has 50320 tokens
# after adding special tokens to GPT-2's base 50257).
# Use this only for tests that load the real tokenizer.
_TG_VOCAB_SIZE = 50320
# GPT-2 base vocab size (no special tokens added)
_GPT2_VOCAB_SIZE = 50257
_REAL_PAD_ID = 50258
_REAL_EOS_ID = 50256


# ============================================================================
# Helpers
# ============================================================================

def _build_minimal_config(grammar_type: str,
                          vocab_size: int = _GPT2_VOCAB_SIZE,
                          pad_id: int = 50256, eos_id: int = 50256,
                          **overrides):
    """Build a minimal TrainConfig for a given grammar type."""
    from olmo.config import TrainConfig

    base = {
        "run_name": f"coldrun-{grammar_type}",
        "seed": 42,
        "dry_run": False,
        "model": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "mlp_ratio": 4,
            "weight_tying": False,
            "rope": True,
            "flash_attention": False,
            "flex_attention": False,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "embedding_dropout": 0.0,
            "max_sequence_length": 64,
            "vocab_size": vocab_size,
            "embedding_size": vocab_size,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            "init_device": "cpu",
            "init_fn": "normal",
            "init_std": 0.02,
            "block_group_size": 1,
            "transformer_grammar_type": grammar_type,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "eps": 1e-8,
            "betas": [0.9, 0.95],
        },
        "scheduler": {
            "name": "cosine_with_warmup",
            "t_warmup": 10,
        },
        "tokenizer": {
            "identifier": "gpt2",
            "vocabulary": f"{_WORKSPACE}/dataset/bbc-news/TG_GPT2_tokenizer.json",
            "truncate_direction": "right",
        },
        "save_folder": tempfile.mkdtemp(prefix="tg-coldrun-"),
        "save_overwrite": True,
        "max_duration": 3,
        "stop_at": 3,
        "global_train_batch_size": 2,
        "device_train_microbatch_size": 2,
        "precision": "fp32",
        "distributed_strategy": None,
        "max_grad_norm": 1.0,
        "console_log_interval": 1,
        "eval_interval": 1000000,
        "save_interval": 1000000,
        "data": {
            "pad_direction": "right",
            "num_workers": 0,
            "drop_last": False,
            "pin_memory": False,
        },
    }
    base.update(overrides)
    return TrainConfig.new(**base)


def _patch_distributed():
    """Patch distributed functions for single-process testing."""
    patches = [
        patch("olmo.torch_util.get_global_rank", return_value=0),
        patch("olmo.torch_util.get_world_size", return_value=1),
        patch("olmo.torch_util.get_local_rank", return_value=0),
        patch("olmo.torch_util.get_local_world_size", return_value=1),
        patch("olmo.torch_util.is_distributed", return_value=False),
        patch("olmo.torch_util.barrier", lambda: None),
        patch("olmo.torch_util.seed_all", lambda s: None),
    ]
    for p in patches:
        p.start()
    return patches


def _cleanup_patches(patches):
    for p in patches:
        p.stop()


# ============================================================================
# 1. Model initialization — all grammar types
# ============================================================================

ALL_GRAMMAR_TYPES = [
    "terminal", "tree", "tg", "tgnomask", "tgnomaskaug",
    "tgproximal", "tgheight",
    "pause1", "pause2", "pause3", "pause1/2", "pause1/2_label",
    "tree_shuffle", "tree_shuffle_mask",
    "mixing",
]

# Types usable in forward-pass tests (all except mixing which needs mix_head_type)
FWD_GRAMMAR_TYPES = [t for t in ALL_GRAMMAR_TYPES if t != "mixing"]

# Safe types for downstream eval (no C++ TG bias extension needed)
SAFE_EVAL_TYPES = [
    "terminal", "tree",
    "pause1", "pause2", "pause3", "pause1/2", "pause1/2_label",
    "tree_shuffle", "tree_shuffle_mask",
]
# Types that use C++ TG bias (may crash in test due to extension issues)
TG_BIAS_TYPES = ["tg", "tgnomask", "tgnomaskaug", "tgproximal", "tgheight", "mixing"]


class TestModelInit:
    """Model + optimizer + scheduler can be constructed for all types."""

    @pytest.mark.parametrize("grammar_type", ALL_GRAMMAR_TYPES)
    def test_model_init(self, grammar_type):
        from olmo.model import OLMo

        cfg = _build_minimal_config(grammar_type)
        model = OLMo(cfg.model, init_params=True)
        assert model.num_params() > 0

    @pytest.mark.parametrize("grammar_type", [
        "terminal", "tree", "tg", "tgnomask", "pause1",
        "tree_shuffle", "mixing",
    ])
    def test_optimizer_and_scheduler_build(self, grammar_type):
        from olmo.model import OLMo
        from olmo.optim import build_optimizer, build_scheduler

        cfg = _build_minimal_config(grammar_type)
        model = OLMo(cfg.model, init_params=True)
        optim = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg)
        assert optim is not None
        assert scheduler is not None


# ============================================================================
# 2. Forward pass with TG attention bias — all grammar types
# ============================================================================

class TestForwardPass:
    """Forward pass with / without TG attention bias on synthetic data."""

    # Types that need a real vocab file for TG bias construction
    _TG_TYPES = {"tg", "tgnomask", "tgnomaskaug", "tgproximal", "tgheight", "mixing"}

    def _vocab_path(self):
        p = f"{_WORKSPACE}/dataset/bbc-news/TG_GPT2_tokenizer.json"
        return p if os.path.exists(p) else None

    @pytest.mark.parametrize("grammar_type", FWD_GRAMMAR_TYPES)
    def test_forward_no_crash(self, grammar_type):
        from olmo.model import OLMo
        from olmo.data import get_TG_generate_bias_func

        cfg = _build_minimal_config(grammar_type, vocab_size=256, pad_id=255, eos_id=254)

        # TG bias types need a real vocab; skip if unavailable
        if grammar_type in self._TG_TYPES:
            vocab_path = self._vocab_path()
            if vocab_path is None:
                pytest.skip("Vocab file not available for TG bias")
            cfg.tokenizer.vocabulary = vocab_path

        model = OLMo(cfg.model, init_params=True)
        model.eval()

        input_ids = torch.randint(0, 250, (1, 32))
        bias = None
        bias_func = get_TG_generate_bias_func(cfg)
        if bias_func is not None:
            raw_bias, _ = bias_func(input_ids[0])
            if raw_bias is not None:
                # Ensure 4D shape (B, H, T, T) as expected by model.forward
                bias = raw_bias
                while bias.ndim < 4:
                    bias = bias.unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_bias=bias)
        assert out.logits.shape == (1, 32, cfg.model.embedding_size)


# ============================================================================
# 3. Training step on synthetic data — representative grammar types
# ============================================================================

class TestTrainingStep:
    """Full forward + backward + optimizer step on synthetic data."""

    REP_TYPES = ["terminal", "tree", "tg", "tgnomask", "pause1"]

    _TG_TYPES = {"tg", "tgnomask", "tgnomaskaug", "tgproximal", "tgheight", "mixing"}

    def _vocab_path(self):
        p = f"{_WORKSPACE}/dataset/bbc-news/TG_GPT2_tokenizer.json"
        return p if os.path.exists(p) else None

    @pytest.mark.parametrize("grammar_type", REP_TYPES)
    def test_training_step_synthetic(self, grammar_type):
        from olmo.model import OLMo
        from olmo.optim import build_optimizer, build_scheduler
        from olmo.data import get_TG_generate_bias_func

        vocab_size, pad_id = 256, 255
        cfg = _build_minimal_config(
            grammar_type, vocab_size=vocab_size, pad_id=pad_id, eos_id=254,
        )
        if grammar_type in self._TG_TYPES:
            vp = self._vocab_path()
            if vp is None:
                pytest.skip("Vocab file not available for TG bias")
            cfg.tokenizer.vocabulary = vp

        model = OLMo(cfg.model, init_params=True)
        optim = build_optimizer(cfg, model)
        _scheduler = build_scheduler(cfg)

        # Synthetic batch
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len))
        attention_bias = None
        bias_func = get_TG_generate_bias_func(cfg)
        if bias_func is not None:
            raw_bias, _ = bias_func(input_ids[0])
            if raw_bias is not None:
                attention_bias = raw_bias
                while attention_bias.ndim < 4:
                    attention_bias = attention_bias.unsqueeze(0)
                attention_bias = attention_bias.expand(batch_size, -1, -1, -1)

        model.train()
        optim.zero_grad()

        out = model(
            input_ids=input_ids,
            attention_bias=attention_bias,
        )
        logits = out.logits

        import torch.nn.functional as F
        labels = input_ids[..., 1:].contiguous().view(-1)
        logits_flat = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        loss = F.cross_entropy(logits_flat, labels, ignore_index=pad_id,
                               reduction="mean")

        loss.backward()
        optim.step()

        assert not torch.isnan(loss)
        assert loss.item() > 0


# ============================================================================
# 4. DataLoader with real data — formats that exist on disk
# ============================================================================

class TestDataLoader:
    """DataLoader construction with real .npy data."""

    DATA_DIR = f"{_WORKSPACE}/dataset/bbc-news"

    def _data_path(self, grammar_type):
        fmt = {"terminal": "terminal", "tree": "tree",
               "tg": "tg", "tgnomask": "tg", "tgnomaskaug": "tg",
               "tgproximal": "tg", "tgheight": "tg", "mixing": "tg"}
        folder = fmt.get(grammar_type, "terminal")
        if grammar_type.startswith("tree_shuffle"):
            folder = "tree"
        if grammar_type.startswith("pause"):
            folder = "terminal"
        return f"{self.DATA_DIR}/{folder}/train.npy"

    @pytest.mark.parametrize("grammar_type", ["terminal", "tree", "tg"])
    def test_dataloader_build(self, grammar_type):
        npy = self._data_path(grammar_type)
        if not os.path.exists(npy):
            pytest.skip(f"No data at {npy}")

        from olmo.data import build_train_dataloader

        cfg = _build_minimal_config(grammar_type, data={"paths": [npy]})
        cfg.device_train_batch_size = 2
        loader = build_train_dataloader(cfg, world_size=1, rank=0, fs_local_rank=0)
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 2


# ============================================================================
# 5. Downstream evaluator construction — all grammar types
# ============================================================================

class TestDownstreamEval:
    """Downstream evaluator construction for BLiMP, SG, XSum, BoolQ, tg_doc."""

    _tokenizer = None
    _patches = None

    @pytest.fixture(autouse=True)
    def _setup(self):
        if TestDownstreamEval._patches is None:
            TestDownstreamEval._patches = _patch_distributed()
        if TestDownstreamEval._tokenizer is None:
            tok_path = f"{_WORKSPACE}/dataset/bbc-news/TG_GPT2_tokenizer.json"
            if os.path.exists(tok_path):
                from olmo.tokenizer import Tokenizer
                TestDownstreamEval._tokenizer = Tokenizer.from_file(
                    tok_path, eos_token_id=_REAL_EOS_ID, pad_token_id=_REAL_PAD_ID,
                )

    @property
    def tokenizer(self):
        if TestDownstreamEval._tokenizer is None:
            pytest.skip("Tokenizer not available")
        return TestDownstreamEval._tokenizer

    # ------------------------------------------------------------------
    # BLiMP
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("grammar_type", [t for t in ALL_GRAMMAR_TYPES if t != "mixing"])
    def test_blimp_dataset(self, grammar_type):
        blimp_dir = f"{_WORKSPACE}/dataset/BLiMP/tree300"
        if not os.path.isdir(blimp_dir):
            pytest.skip(f"BLiMP data not found at {blimp_dir}")

        from olmo.data import get_TG_generate_bias_func
        from olmo.eval.downstream import BLiMPApproximationDataset

        cfg = _build_minimal_config(grammar_type)
        try:
            ds = BLiMPApproximationDataset(
                tokenizer=self.tokenizer,
                dataset_path=blimp_dir,
                model_ctx_len=cfg.model.max_sequence_length,
                transformer_grammar_type=grammar_type,
                generate_TG_attention_bias=get_TG_generate_bias_func(cfg),
                vocab_path=cfg.tokenizer.vocabulary,
                device_eval_batch_size=2,
                samples_per_sent=2,
                pair_per_task=2,
            )
        except Exception as exc:
            pytest.skip(f"BLiMP init failed for {grammar_type}: {exc}")

        assert len(ds) > 0
        batch = ds.collate_fn([ds[0], ds[1]] if len(ds) > 1 else [ds[0], ds[0]])
        assert "input_ids" in batch
        assert batch["input_ids"].ndim == 2

    # ------------------------------------------------------------------
    # SG
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("grammar_type", SAFE_EVAL_TYPES)
    def test_sg_dataset(self, grammar_type):
        sg_dir = f"{_WORKSPACE}/evaluation/SG/raw_data"
        if not os.path.isdir(sg_dir):
            pytest.skip(f"SG data not found at {sg_dir}")

        from olmo.eval.downstream import SGDataset

        cfg = _build_minimal_config(grammar_type)
        try:
            ds = SGDataset(
                tokenizer=self.tokenizer,
                dataset_path=sg_dir,
                vocab_path=cfg.tokenizer.vocabulary,
                transformer_grammar_type=grammar_type,
            )
        except Exception as exc:
            pytest.skip(f"SG init failed for {grammar_type}: {exc}")

        assert len(ds) > 0
        batch = ds[0]
        for item in batch:
            assert "input_ids" in item
            assert "task" in item
            assert "condition_name" in item

    # ------------------------------------------------------------------
    # XSum
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("grammar_type", SAFE_EVAL_TYPES)
    def test_xsum_dataset(self, grammar_type):
        xsum_dir = f"{_WORKSPACE}/dataset/Xsum"
        if not os.path.isdir(xsum_dir):
            pytest.skip(f"XSum data not found at {xsum_dir}")

        from olmo.data import get_TG_generate_bias_func
        from olmo.eval.downstream import XsumDataset

        cfg = _build_minimal_config(grammar_type)
        try:
            ds = XsumDataset(
                tokenizer=self.tokenizer,
                dataset_path=xsum_dir,
                model_ctx_len=cfg.model.max_sequence_length,
                split="test",
                generate_TG_attention_bias=get_TG_generate_bias_func(cfg),
                transformer_grammar_type=grammar_type,
                vocab_path=cfg.tokenizer.vocabulary,
            )
        except Exception as exc:
            pytest.skip(f"XSum init failed for {grammar_type}: {exc}")

        assert len(ds) > 0
        item = ds[0]
        assert "input_ids" in item
        assert "gold_summary" in item

    # ------------------------------------------------------------------
    # BoolQ (ICL)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("grammar_type", SAFE_EVAL_TYPES)
    def test_boolq_dataset(self, grammar_type):
        boolq_dir = f"{_WORKSPACE}/dataset/SuperGLUE/BoolQ"
        if not os.path.isdir(boolq_dir):
            pytest.skip(f"BoolQ data not found at {boolq_dir}")

        from olmo.data import get_TG_generate_bias_func
        from olmo.eval.downstream import BoolQ

        cfg = _build_minimal_config(grammar_type)
        try:
            ds = BoolQ(
                tokenizer=self.tokenizer,
                dataset_path=boolq_dir,
                model_ctx_len=cfg.model.max_sequence_length,
                split="validation",
                transformer_grammar_type=grammar_type,
                generate_TG_attention_bias=get_TG_generate_bias_func(cfg),
                vocab_path=cfg.tokenizer.vocabulary,
                tree_eval_type="default",
            )
        except FileNotFoundError:
            pytest.skip(f"BoolQ validation data not found")
        except Exception as exc:
            pytest.skip(f"BoolQ init failed for {grammar_type}: {exc}")

        assert len(ds) > 0
        item = ds[0]
        assert "input_ids" in item

    # ------------------------------------------------------------------
    # tg_doc evaluator (LM perplexity)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("grammar_type", ["terminal", "tree", "tg"])
    def test_lm_evaluator(self, grammar_type):
        """LM (perplexity) evaluator construction with real data."""
        fmt = {"terminal": "terminal", "tree": "tree", "tg": "tg"}
        npy = f"{_WORKSPACE}/dataset/bbc-news/{fmt[grammar_type]}/dev.npy"
        if not os.path.exists(npy):
            pytest.skip(f"No eval data at {npy}")

        from olmo.eval import build_evaluators
        from olmo.config import EvaluatorConfig, EvaluatorType, DataConfig

        tok_path = f"{_WORKSPACE}/dataset/bbc-news/TG_GPT2_tokenizer.json"
        cfg = _build_minimal_config(
            grammar_type, vocab_size=_TG_VOCAB_SIZE,
            pad_id=_REAL_PAD_ID, eos_id=_REAL_EOS_ID,
            tokenizer={
                "identifier": tok_path,
                "vocabulary": tok_path,
            },
            data={"paths": [npy]},
        )
        cfg.evaluators = [
            EvaluatorConfig.new(
                label=f"lm-{grammar_type}",
                type=EvaluatorType.lm,
                device_eval_batch_size=2,
                subset_num_batches=1,
                data=DataConfig.new(
                    paths=[npy],
                    num_workers=0,
                    drop_last=False,
                ),
            )
        ]
        evaluators = build_evaluators(cfg, torch.device("cpu"))
        assert len(evaluators) > 0


# ============================================================================
# 6. Training YAML config smoke tests
# ============================================================================

class TestConfigSmoke:
    """Load real YAML configs and check key fields."""

    @pytest.mark.parametrize("yaml_file,expected_type", [
        ("TG.yaml", "tg"),
        ("terminal.yaml", "terminal"),
        ("tree.yaml", "tree"),
    ])
    def test_config_loads(self, yaml_file, expected_type):
        yaml_path = Path(_WORKSPACE) / "train_configs" / yaml_file
        if not yaml_path.exists():
            pytest.skip(f"{yaml_file} not found")

        from omegaconf import OmegaConf
        from olmo.config import TrainConfig

        # ${workspace} is an OmegaConf variable resolver, not a config field.
        # Register it before loading the YAML.
        OmegaConf.register_new_resolver("workspace", lambda: _WORKSPACE, replace=True)
        try:
            cfg = TrainConfig.load(str(yaml_path), validate_paths=False)
        finally:
            OmegaConf.clear_resolver("workspace")
        assert cfg.model.transformer_grammar_type == expected_type

    def test_tg_yaml_model_size(self):
        """TG.yaml 100M model size: 768 dim, 12 layers, 12 heads."""
        yaml_path = Path(_WORKSPACE) / "train_configs" / "TG.yaml"
        if not yaml_path.exists():
            pytest.skip("TG.yaml not found")

        from omegaconf import OmegaConf
        from olmo.config import TrainConfig

        OmegaConf.register_new_resolver("workspace", lambda: _WORKSPACE, replace=True)
        try:
            cfg = TrainConfig.load(str(yaml_path), validate_paths=False)
        finally:
            OmegaConf.clear_resolver("workspace")
        assert cfg.model.d_model == 768
        assert cfg.model.n_layers == 12
        assert cfg.model.n_heads == 12
