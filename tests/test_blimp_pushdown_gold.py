"""BLiMP gold300 adapter tests for terminal-only Pushdown checkpoints."""

from pathlib import Path

import pytest
import torch

from olmo.eval.downstream import BLiMPApproximationDataset
from olmo.tokenizer import Tokenizer


VOCAB = Path("dataset/bbc-news/TG_GPT2_tokenizer.json")
BLIMP_DIR = Path("dataset/BLiMP/tree300")


@pytest.mark.skipif(
    not VOCAB.exists() or not (BLIMP_DIR / "blimp_tree_300.npy").exists(),
    reason="local BLiMP gold300 data is unavailable",
)
def test_pushdown_gold300_emits_terminals_and_valid_spans():
    tokenizer = Tokenizer.from_file(
        str(VOCAB), pad_token_id=50258, eos_token_id=50256
    )
    dataset = BLiMPApproximationDataset(
        tokenizer=tokenizer,
        dataset_path=str(BLIMP_DIR),
        transformer_grammar_type="pushdown",
        vocab_path=str(VOCAB),
        samples_per_sent=300,
        pair_per_task=1,
        pushdown_gold=True,
        parse_binarize_direction="right",
    )

    items = [dataset[i] for i in range(4)]
    assert dataset.dataset_name == "tree_300"
    assert dataset.SENT_SIZE == 300
    for item in items:
        ids = item["input_ids"]
        spans = item["tree_spans"]
        assert ids[0].item() == 50257
        assert ids[-1].item() == 50256
        assert spans.ndim == 2 and spans.shape[1] == 3
        assert torch.all(spans[:, 0] <= spans[:, 1])
        assert torch.all(spans[:, 1] <= spans[:, 2])
        assert torch.all(spans[:, 2] < ids.numel())
        # Unary/preterminal constituents are SHIFTs, not Pushdown REDUCEs.
        assert not torch.any(spans[:, 0] == spans[:, 2])

    batch = dataset.collate_fn(items)
    assert batch["input_ids"].shape[0] == len(items)
    assert batch["tree_spans"].shape[:2] == batch["tree_span_mask"].shape
    assert torch.all(batch["tree_spans"][batch["tree_span_mask"]] >= 0)
