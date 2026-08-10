"""TDD for the OLMo backbone option of GPST.

The GPST generative model's type/token transformer stacks can optionally be
backed by OLMo blocks (``olmo/model.py:OLMoBlock``) instead of the HF GPT2
wrapper. These tests pin the ``OLMoStack`` contract (same interface as
``GPT2Model``) and the end-to-end hard-EM loop under that backbone.

CPU-only; mirrors the style of ``test_gpst_trainer.py``.
"""
from __future__ import annotations

import torch

import olmo.gpst  # noqa: F401  (ensures cpp backend bootstrap path runs)
from olmo.gpst.model.weighted_sum_func import WeightedSumFunc

_R2D2_CFG = "olmo/gpst/data/en_config/r2d2_256_4_1.json"
_GPT_CFG = "olmo/gpst/data/gpt2-small/config.json"


def _small_stack(n_layers=2, d_model=64, n_heads=4, max_seq=32):
    from olmo.config import ModelConfig, BlockType, ActivationType, InitFnType
    from olmo.gpst.model.olmo_stack import OLMoStack
    cfg = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        rope=False, alibi=False,
        block_type=BlockType.sequential,
        activation_type=ActivationType.gelu,
        init_fn=InitFnType.normal, init_device="cpu",
        attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
        max_sequence_length=max_seq,
    )
    stack = OLMoStack(cfg, n_layers=n_layers, no_embedding=True,
                      no_layer_norm=False, add_position=True)
    # reset block params (OLMoBlock.build does not auto-init when init_device!='meta')
    for blk in stack.blocks:
        blk.reset_parameters()
    stack.eval()
    return stack


def test_olmo_stack_forward_shape():
    # Arrange
    torch.manual_seed(0)
    stack = _small_stack()
    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D)
    pos = torch.arange(T).unsqueeze(0).expand(B, T)
    # Act
    out = stack(inputs_embeds=x, position_ids=pos)
    # Assert
    assert out.last_hidden_state.shape == (B, T, D)
    assert torch.isfinite(out.last_hidden_state).all()


def test_olmo_stack_position_ids_honored():
    # Arrange
    torch.manual_seed(0)
    stack = _small_stack()
    B, T, D = 1, 6, 64
    x = torch.randn(B, T, D)
    pos_a = torch.arange(T).unsqueeze(0)
    pos_b = torch.arange(T, 0, -1) - 1  # reversed
    # Act
    with torch.no_grad():
        out_a = stack(inputs_embeds=x, position_ids=pos_a).last_hidden_state
        out_b = stack(inputs_embeds=x, position_ids=pos_b).last_hidden_state
    # Assert: different positions => different outputs (wpe is actually used)
    assert not torch.allclose(out_a, out_b, atol=1e-6), \
        "position_ids had no effect on output (wpe not applied?)"


def test_olmo_stack_past_key_values():
    # Arrange
    torch.manual_seed(0)
    stack = _small_stack(n_layers=3)
    B, T, D = 2, 5, 64
    x = torch.randn(B, T, D)
    pos = torch.arange(T).unsqueeze(0).expand(B, T)
    with torch.no_grad():
        # Act (full)
        out_full = stack(inputs_embeds=x, position_ids=pos)
        # Act (chunked with KV cache: first half then second half)
        out_first = stack(inputs_embeds=x[:, :2], position_ids=pos[:, :2], use_cache=True)
        past = out_first.past_key_values
        assert len(past) == 3, f"expected 3 layer caches, got {len(past)}"
        out_second = stack(inputs_embeds=x[:, 2:], position_ids=pos[:, 2:],
                           past_key_values=past, use_cache=True)
        recon = torch.cat([out_first.last_hidden_state, out_second.last_hidden_state], dim=1)
    # Assert: chunked-with-cache reconstruction matches a single full forward
    assert torch.allclose(out_full.last_hidden_state, recon, atol=1e-4), \
        "KV-cache chunked forward disagrees with full forward"


def test_olmo_stack_causal():
    # Arrange
    torch.manual_seed(0)
    stack = _small_stack()
    B, T, D = 1, 6, 64
    x = torch.randn(B, T, D)
    pos = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        out1 = stack(inputs_embeds=x, position_ids=pos).last_hidden_state
        x2 = x.clone()
        x2[:, -1] = torch.randn(B, D)  # perturb only the last (future) token
        out2 = stack(inputs_embeds=x2, position_ids=pos).last_hidden_state
    # Assert: past tokens unaffected by a future-token change (causality)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6), \
        "non-causal: changing a future token altered past outputs"


def test_olmo_stack_padding():
    # Arrange
    torch.manual_seed(0)
    stack = _small_stack()
    B, T, D = 2, 6, 64
    x = torch.randn(B, T, D)
    pos = torch.arange(T).unsqueeze(0).expand(B, T)
    mask = torch.ones(B, T)
    mask[1, 3:] = 0  # second sequence padded after position 2
    with torch.no_grad():
        out_full = stack(inputs_embeds=x, position_ids=pos, attention_mask=mask).last_hidden_state
        # truncate to the valid prefix of the padded sequence
        x_trunc = x[1:2, :3]
        pos_trunc = pos[1:2, :3]
        out_trunc = stack(inputs_embeds=x_trunc, position_ids=pos_trunc).last_hidden_state
    # Assert: valid positions of the padded forward equal the unpadded prefix forward
    assert torch.allclose(out_full[1, :3], out_trunc[0], atol=1e-5), \
        "padding positions leaked into valid outputs"


def test_create_model_olmo_backbone():
    # Arrange
    from olmo.gpst.model.model_factory import create_model
    from olmo.gpst.model.olmo_stack import OLMoStack
    # Act
    model = create_model("r2d2-gen-fast", _R2D2_CFG, _GPT_CFG, backbone="olmo")
    # Assert
    assert isinstance(model.action_layers, OLMoStack)
    assert isinstance(model.generation_layers, OLMoStack)
    # type/token layer split matches gpt2 path (action_layer_num + remainder)
    assert len(model.action_layers.blocks) < len(model.generation_layers.blocks)
    assert model.action_layers.add_position is True
    assert model.generation_layers.add_position is False
    assert model.action_layers.config is not model.generation_layers.config


def test_olmo_stack_gradient_checkpointing_enabled():
    stack = _small_stack()
    stack.gradient_checkpointing = True
    stack.train()
    x = torch.randn(1, 4, 64, requires_grad=True)
    out = stack(inputs_embeds=x, position_ids=torch.arange(4).unsqueeze(0))
    out.last_hidden_state.sum().backward()

    assert stack.gradient_checkpointing is True
    assert x.grad is not None


def test_word_sync_beam_searcher_olmo_bos_step():
    from transformers import AutoConfig
    from olmo.gpst.model.model_factory import create_model
    from olmo.gpst.utils.generator_factory import create_generator

    model = create_model("r2d2-gen-fast", _R2D2_CFG, _GPT_CFG, backbone="olmo")
    config = AutoConfig.from_pretrained(_GPT_CFG)
    generator = create_generator(
        "r2d2-gen-fast", model, torch.device("cpu"), config,
        beam_size=2, sampling=False, word_sync=True,
    )
    action_logits, token_logits, action_kv, token_kv = generator.bos_step(1)

    assert action_logits.shape == (1, 1, 2)
    assert token_logits.shape[:2] == (1, 1)
    assert len(action_kv) == config.action_layer_num
    assert len(token_kv) == config.n_layer - config.action_layer_num

    states = generator.beam_search(
        target_ids=torch.tensor([[10, 20, 30]]),
        target_masks=torch.ones(1, 3, dtype=torch.long),
    )
    assert len(states) == 1
    assert all(state.is_finished and state.token_offset == 3 for state in states[0])


def test_word_sync_beam_searcher_gpt2_cache_compatibility():
    from transformers import AutoConfig
    from olmo.gpst.model.model_factory import create_model
    from olmo.gpst.utils.generator_factory import create_generator

    model = create_model("r2d2-gen-fast", _R2D2_CFG, _GPT_CFG, backbone="gpt2")
    generator = create_generator(
        "r2d2-gen-fast", model, torch.device("cpu"),
        AutoConfig.from_pretrained(_GPT_CFG), beam_size=2,
        sampling=False, word_sync=True,
    )
    states = generator.beam_search(
        target_ids=torch.tensor([[10, 20, 30]]),
        target_masks=torch.ones(1, 3, dtype=torch.long),
    )

    assert len(states) == 1
    assert all(state.is_finished and state.token_offset == 3 for state in states[0])


def _unsup_batch():
    from olmo.gpst.reader.data_collator import DefaultCollator
    items = [
        {"text": [10, 20, 30, 40, 50], "sentence_splits": [3, 5]},
        {"text": [60, 70, 80, 90], "sentence_splits": [2, 4]},
    ]
    c = DefaultCollator(enable_group=True, external_vocab_path=None)
    return c.generative_r2d2_collate_fn_ext(items)


def _loader(batch):
    class _L:
        def __iter__(self):
            yield batch
            yield batch
        def __len__(self):
            return 2
    return _L()


def _to_dev(batch, dev):
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(dev)
    return batch


def test_trainer_olmo_backbone_cpu():
    # Arrange
    from olmo.gpst.model.model_factory import create_model
    from olmo.gpst.trainer.trainer import train, TrainConfig
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_model("r2d2-gen-fast", _R2D2_CFG, _GPT_CFG, backbone="olmo").to(dev)
    batch = _to_dev(_unsup_batch(), dev)
    cfg = TrainConfig(lr=5e-5, parser_lr=1e-3, accumulation_steps=1,
                      log_steps=1, save_steps=100000, amp=False, max_steps=2)
    # Act
    m = train(model, _loader(batch), cfg, dev)
    # Assert
    assert torch.isfinite(torch.tensor(m["total_loss"]))
    assert WeightedSumFunc.a_ij_require_grad is False


def test_grad_stop_toggle_olmo():
    # Arrange
    from olmo.gpst.model.model_factory import create_model
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_model("r2d2-gen-fast", _R2D2_CFG, _GPT_CFG, backbone="olmo").to(dev)
    batch = _to_dev(_unsup_batch(), dev)
    seen = []
    orig = WeightedSumFunc.a_ij_require_grad
    # Act
    WeightedSumFunc.a_ij_require_grad = True
    r = model(coeff=1.0, temperature=1.0, **batch)
    r.struct_loss.backward(retain_graph=True)
    seen.append(WeightedSumFunc.a_ij_require_grad)
    WeightedSumFunc.a_ij_require_grad = False
    r2 = model(coeff=1.0, temperature=1.0, **batch)
    r2.non_struct_loss.backward()
    seen.append(WeightedSumFunc.a_ij_require_grad)
    WeightedSumFunc.a_ij_require_grad = orig
    # Assert
    assert seen == [True, False], seen
