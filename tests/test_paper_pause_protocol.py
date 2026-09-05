"""Paper Pause aliases must identify SEP weights before applying overrides."""

import pytest

from olmo.exceptions import OLMoCliError
from scripts.init_cfg_and_sbatch import _load_checkpoint_config, generate_config, model_paths
from scripts.prepare_paper_pretraining import REPO_ROOT, load_manifest


@pytest.mark.parametrize("variant", ["pause1", "pause2"])
def test_paper_eval_alias_matches_sep_pretraining_checkpoint(variant):
    runs = {run["id"]: run for run in load_manifest()["runs"]}
    run = runs[f"bbc_100m_{variant}_sep"]
    assert model_paths[variant].lstrip("/") == run["source_checkpoint"]
    cfg = _load_checkpoint_config(REPO_ROOT / run["source_config"], [], variant)
    assert cfg.model.pause_token_id == 50261
    assert cfg.model.max_sequence_length == run["sequence_length"]


@pytest.mark.parametrize("variant", ["pause1", "pause2"])
def test_repeat_weights_cannot_be_relabelled_by_a_sep_override(variant):
    runs = {run["id"]: run for run in load_manifest()["runs"]}
    source = REPO_ROOT / runs[f"bbc_100m_{variant}_repeat"]["source_config"]
    with pytest.raises(OLMoCliError, match="never relabel repeat-token weights"):
        _load_checkpoint_config(source, ["model.pause_token_id=50261"], variant)
    cfg = _load_checkpoint_config(source, [], f"{variant}-repeat")
    assert cfg.model.pause_token_id is None
    assert model_paths[f"{variant}-repeat"].lstrip("/") == str(source.parent.relative_to(REPO_ROOT))


@pytest.mark.parametrize("override", ["model.pause_token_id=null", "model.transformer_grammar_type=pause2"])
def test_sep_eval_overrides_cannot_change_checkpoint_identity(override):
    source = REPO_ROOT / "train_configs/paper_sources/bbc_100m_pause1_sep.yaml"
    with pytest.raises(OLMoCliError, match="requires checkpoint grammar"):
        _load_checkpoint_config(source, [override], "pause1")


@pytest.mark.parametrize("variant", ["pause1", "pause2"])
@pytest.mark.parametrize("task", ["xsum_finetune", "xsum_test", "boolq"])
def test_paper_downstream_aliases_require_the_versioned_pause_campaign(tmp_path, variant, task):
    with pytest.raises(OLMoCliError, match="scripts/pause_eval_campaign.py prepare"):
        generate_config(tmp_path / "run.yaml", [], Device="RTX3090", modelname=variant, task=task)

