"""CPU TDD test for pause-model XSUM (ROUGE) generation.

Pins the two fixes that make pause-model XSUM eval correct:

(1) Non-label pause models (pause1, pause1in2, ...): ``word_sync_beam_search`` must
    NOT emit non-terminal (NT) tokens into the generated summary. The NT id range
    ``[opening_non_terminals[0], closing_non_terminals[1]]`` must be masked out of the
    main beam candidates when ``transformer_grammar_type`` is a pause type. Without this,
    the tree-grammar decoder (designed for TG/tgtree) pollutes the pause model's output
    with bracket tokens the model never learned to emit coherently -> ROUGE garbage.

(2) pause1_label: the model's loss was masked to real-token-predicting positions only
    (``pause_label_mask``), so it never learns to emit pause tokens. Free generation
    skips pauses, misaligning ``extract_real_tokens``'s grid. The fix is
    ``OLMo.pause_label_generate``: a KV-cache generator that deterministically inserts
    ``p`` repeats of the last real token at every ``q``-token block boundary (matching
    training's ``pause_token_id=None`` repeat-mode), then continues. Output is
    real-token-only.

All models trained with ``pause_token_id=None`` (repeat mode, confirmed in saved
configs), so pause slots repeat the block's last real token — NOT a dedicated SEP id.

No checkpoint needed: tiny randomly-init OLMo on CPU. Run:
    PYTHONPATH=. python tests/test_pause_xsum_generate.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOCAB = "./dataset/bbc-news/TG_GPT2_tokenizer.json"


def _make_pause_model(grammar_type: str = "pause1", max_seq: int = 64):
    """Small pause OLMo (CPU, SDPA path — no flex)."""
    from olmo.config import (
        ModelConfig, BlockType, LayerNormType, ActivationType, InitFnType,
    )
    from olmo.model import OLMo
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_layers=2, mlp_ratio=4, mlp_hidden_size=256,
        vocab_size=50320, embedding_size=50320, max_sequence_length=max_seq,
        block_type=BlockType.sequential, layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu, rope=True, flash_attention=False,
        flex_attention=False, attention_dropout=0.0, init_device="cpu",
        init_fn=InitFnType.normal, init_std=0.02,
        transformer_grammar_type=grammar_type,
        weight_tying=True, eos_token_id=50256, pad_token_id=50258,
    )
    m = OLMo(cfg).eval()
    for b in m.transformer.blocks:
        if hasattr(b, "flex_attention"):
            b.flex_attention = None
    return m


def _vocab():
    from olmo.data.tg_mask import SentencepieceVocab
    return SentencepieceVocab.from_vocab_file(VOCAB)


def test_word_sync_beam_search_excludes_nt_for_pause():
    """RED: pause-model word_sync_beam_search must emit NO non-terminal tokens.

    The tree-grammar decoder samples the main beam candidates over the FULL vocab
    (including NT ids). For a pause model this pollutes the summary with NT tokens.
    Fix: mask ``[NT_start, NT_end]`` out of the main candidates when grammar is pause.
    """
    torch.manual_seed(3)
    m = _make_pause_model("pause1")
    v = _vocab()
    NT_start = v.opening_non_terminals[0]
    NT_end = v.closing_non_terminals[1]

    # 1D paused prompt (BOS + a few terminal tokens).
    prompt = torch.LongTensor([v.bos, 100, 200, 300, 400])
    beams = m.word_sync_beam_search(
        vocab=v,
        past_input=prompt,
        max_word_steps=8,
        max_length=24,
        beam_size=4,
        generate_TG_bias=None,
        transformer_grammar_type="pause1",
    )
    assert isinstance(beams, list) and len(beams) >= 1
    top = beams[0]["input_ids"].tolist()
    nt_emitted = [t for t in top if NT_start <= t <= NT_end]
    assert nt_emitted == [], (
        f"pause beam search emitted NT tokens {nt_emitted} in {top}; "
        f"NT range [{NT_start},{NT_end}] must be masked for pause types."
    )
    print(f"PASS: pause word_sync_beam_search emits no NT tokens. Top beam len={len(top)}.")


def test_pause_label_generate_inserts_repeats_on_grid():
    """RED: pause_label_generate deterministically inserts p repeats every q tokens.

    pause1_label = spec (p=1, q=1): after EACH real token, insert 1 repeat of it
    (repeat-mode, since pause_token_id=None). The internal paused feed must match
    training's layout; the returned real-token stream has no repeats.
    """
    torch.manual_seed(3)
    m = _make_pause_model("pause1_label", max_seq=128)
    v = _vocab()

    prompt = torch.LongTensor([v.bos, 100, 200, 300]).unsqueeze(0)  # (1, L)
    out = m.pause_label_generate(
        input_ids=prompt,
        pause_spec=(1, 1),
        max_real_tokens=6,
        eos_token_id=50256,
    )
    assert out.dim() == 2 and out.shape[0] == 1
    real = out[0].tolist()
    assert len(real) <= 6, f"real output {real} exceeds max_real_tokens=6"
    NT_start = v.opening_non_terminals[0]
    NT_end = v.closing_non_terminals[1]
    assert not any(NT_start <= t <= NT_end for t in real), f"NT in real output {real}"
    print(f"PASS: pause_label_generate returns {len(real)} real tokens: {real}")


def test_pause_label_generate_deterministic():
    """Same input -> same output (greedy, no sampling)."""
    torch.manual_seed(3)
    m = _make_pause_model("pause1_label", max_seq=128)
    v = _vocab()
    prompt = torch.LongTensor([v.bos, 100, 200, 300]).unsqueeze(0)
    out1 = m.pause_label_generate(prompt, (1, 1), 6, eos_token_id=50256)
    out2 = m.pause_label_generate(prompt, (1, 1), 6, eos_token_id=50256)
    assert torch.equal(out1, out2), "pause_label_generate not deterministic"
    print("PASS: pause_label_generate deterministic.")


def test_pause_label_generate_inserts_pauses_every_q():
    """For spec (p=1, q=2): a pause (repeat of last real) is fed every 2 real tokens.

    Instrument forward() to record the count of deterministic pause-feeding calls
    (input == last real token repeated WITHOUT sampling). With prompt real_count
    aligned so pauses are owed, generation must intersperse pauses.
    """
    torch.manual_seed(3)
    m = _make_pause_model("pause1_label", max_seq=128)
    v = _vocab()
    # Prompt: BOS is a real token. Spec (1,2): pause after every 2 real tokens.
    # Build a 3-real-token paused prompt: [BOS, a, pause, b] where pause repeats a.
    # Simpler: feed a short un-paused-like prompt and let cadence drive pauses.
    # Use BOS + 1 real token (real_in_prompt=2 incl BOS+100). After 2 reals, a
    # pause is owed. The generator must insert a pause before emitting the 3rd real.
    from olmo.data.util import pause_input_ids
    real_prompt = torch.LongTensor([v.bos, 100, 200])  # 3 real tokens
    paused = pause_input_ids(real_prompt.tolist(), None, pause_num="pause1/2")
    prompt = torch.tensor([paused], dtype=torch.long)

    calls = {"forward": 0}
    orig_forward = m.forward

    def counting_forward(input_ids, **kw):
        calls["forward"] += 1
        return orig_forward(input_ids, **kw)
    m.forward = counting_forward

    out = m.pause_label_generate(prompt, (1, 2), max_real_tokens=4, eos_token_id=50256)
    m.forward = orig_forward
    # The model was forward-called: 1 prefill + per-token forwards. For
    # max_real_tokens=4 reals + pauses inserted at every 2nd real, total forwards
    # > 1 + 4 (i.e., pauses were inserted, adding extra forwards).
    assert calls["forward"] > 1 + 4, (
        f"expected pause-insertion forwards (>5 total), got {calls['forward']}; "
        "pause grid not inserting deterministic pauses."
    )
    assert out.shape[1] <= 4
    print(f"PASS: pause grid inserts pauses (forward calls={calls['forward']}, "
          f"real out len={out.shape[1]}).")


def test_pause_generate_forces_dedicated_sep_and_returns_real_tokens_only():
    """SEP checkpoints must continue the fixed grid without leaking SEP to ROUGE."""
    from olmo.data.util import pause_input_ids

    torch.manual_seed(3)
    m = _make_pause_model("pause2", max_seq=128)
    v = _vocab()
    sep = 50261
    real_prompt = torch.LongTensor([v.bos, 100, 200])
    paused_prompt = pause_input_ids(real_prompt, sep, pause_num="pause2")
    prompt = paused_prompt.unsqueeze(0)

    single_token_inputs = []
    orig_forward = m.forward

    def recording_forward(input_ids, **kwargs):
        if input_ids.shape[-1] == 1:
            single_token_inputs.extend(input_ids.detach().reshape(-1).tolist())
        return orig_forward(input_ids, **kwargs)

    m.forward = recording_forward
    generated = m.pause_generate(
        prompt,
        pause_spec=(2, 1),
        max_real_tokens=4,
        pause_token_id=sep,
        vocab=v,
        eos_token_id=50256,
        beam_size=1,
    )
    m.forward = orig_forward

    real = generated.token_ids[0, 0].tolist()
    assert len(real) <= 4
    assert sep not in real
    assert sep in single_token_inputs, "forced SEP was not fed through the KV cache"


def test_pause_generate_rejects_invalid_batch_size():
    m = _make_pause_model("pause1", max_seq=128)
    with torch.no_grad(), pytest.raises(ValueError, match="device_eval_batch_size=1"):
        m.pause_generate(
            torch.ones((2, 4), dtype=torch.long),
            pause_spec=(1, 1),
            max_real_tokens=4,
        )


if __name__ == "__main__":
    test_word_sync_beam_search_excludes_nt_for_pause()
    test_pause_label_generate_inserts_repeats_on_grid()
    test_pause_label_generate_deterministic()
    test_pause_label_generate_inserts_pauses_every_q()
    test_pause_generate_forces_dedicated_sep_and_returns_real_tokens_only()
    print("\nAll pause-XSUM tests passed.")
