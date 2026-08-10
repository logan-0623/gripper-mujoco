import pytest
import torch

pytest.importorskip("lerobot")

from interaction_vla.lerobot_bridge.act_smoke import run_one_batch_check


def test_one_batch_act_update_and_reload_are_finite(
    tiny_lerobot_dataset, tmp_path
) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    result = run_one_batch_check(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_dir=tmp_path / "checkpoint",
        device=torch.device("cpu"),
        batch_size=1,
        seed=0,
        architecture="test",
    )

    assert result.loss >= 0.0
    assert result.gradient_norm > 0.0
    assert result.reload_max_abs_error <= 1e-5
    assert (tmp_path / "checkpoint" / "model.safetensors").is_file()
    assert (tmp_path / "checkpoint" / "policy_preprocessor.json").is_file()
    assert (tmp_path / "checkpoint" / "policy_postprocessor.json").is_file()
    assert (tmp_path / "checkpoint" / "bridge_checkpoint.json").is_file()
