from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from olmo.checkpoint import FullCheckpointer


class FakeDDP:
    def __init__(self):
        self.module = MagicMock()
        self.module.config = SimpleNamespace(transformer_grammar_type="terminal")


def test_full_checkpointer_skips_trainer_state_for_model_only_restore():
    dist_model = FakeDDP()
    optim = MagicMock()

    with patch("olmo.checkpoint.DDP", FakeDDP), patch(
        "olmo.checkpoint.load_state_dict", return_value={}
    ) as load_state, patch("olmo.checkpoint.barrier"), patch(
        "olmo.checkpoint.torch.cuda.empty_cache"
    ):
        state = FullCheckpointer(MagicMock()).restore_checkpoint(
            "/model-only/step1-unsharded",
            dist_model,
            optim,
            load_optimizer_state=False,
            load_trainer_state=False,
        )

    assert state == {}
    assert [call.args[1] for call in load_state.call_args_list] == ["model.pt"]
    dist_model.module.load_state_dict.assert_called_once_with({}, strict=True)
