"""End-to-end CPU smoke for the sentence-local Pushdown PPL matrix."""

from types import SimpleNamespace

import torch

from olmo.config import ActivationType, BlockType, InitFnType, LayerNormType, ModelConfig
from olmo.eval.pushdown_document_ppl import PushdownGoldCandidate
from olmo.model import OLMo
from scripts.evaluate_pushdown_ppl_2x2 import evaluate_matrix


def _candidate(spans):
    return PushdownGoldCandidate(
        tokens=(7, 1, 2, 3),
        spans=spans,
        sentence_ids=(-1, 0, 0, 0),
        # The matrix localizer deliberately rebuilds both fields from spans.
        attachment_targets=(-1, 1, 2, 3),
        legal_attachment_targets=((), (1,), (2, 1), (3, 2, 1)),
    )


class TinyCorpus:
    vocab = SimpleNamespace(bos=7, eos=8)

    def __init__(self):
        self.rows = ((
            _candidate(((1, 1, 2), (1, 2, 3))),
            _candidate(((2, 2, 3), (1, 1, 3))),
        ),)

    def __len__(self):
        return len(self.rows)

    def sentence_candidates(self, index):
        return self.rows[index]


def _model():
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        mlp_hidden_size=32,
        vocab_size=16,
        embedding_size=16,
        max_sequence_length=16,
        block_type=BlockType.sequential,
        layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu,
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_device="cpu",
        rope=True,
        flash_attention=False,
        flex_attention=False,
        pushdown_use_flex=False,
        transformer_grammar_type="pushdown",
        pushdown_max_depth=8,
        weight_tying=True,
        bos_token_id=7,
        eos_token_id=8,
        pad_token_id=9,
    )
    return OLMo(config).eval()


def test_tiny_sentence_matrix_has_four_finite_cells_and_one_denominator():
    torch.manual_seed(11)
    result = evaluate_matrix(
        _model(),
        TinyCorpus(),
        device="cpu",
        beam_size=16,
        max_reduce=None,
        eval_batch_size=2,
        max_batch_tokens=64,
        progress_every=0,
    )
    assert result["status"] == "complete"
    assert result["counts"]["sentence_count"] == 1
    assert result["counts"]["terminal_count"] == 3
    assert result["counts"]["supplied_candidate_slots"] == 2
    assert result["contract"]["candidate_aggregation"].endswith("no_divide_by_k")
    assert set(result["cells"]) == {"beam_search", "teacher_forced"}
    for source in result["cells"].values():
        assert set(source) == {"stack_legal", "sentence_causal"}
        for cell in source.values():
            assert torch.isfinite(torch.tensor(cell["perplexity"]))
    assert (
        result["cells"]["teacher_forced"]["stack_legal"]["perplexity"]
        <= result["cells"]["teacher_forced"]["sentence_causal"]["perplexity"]
    )
