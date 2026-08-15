from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interaction_vla.lerobot_bridge import act_diagnostics as diagnostics_module
from interaction_vla.lerobot_bridge.act_diagnostics import (
    action_chunk_metrics,
    evaluate_checkpoint_actions,
    partition_action_metrics,
)
from interaction_vla.lerobot_bridge.interaction_phase import PHASE_IDS


def test_action_chunk_metrics_mask_padding_and_report_direction() -> None:
    target = np.zeros((1, 3, 7), dtype=np.float32)
    predicted = np.zeros_like(target)
    target[0, 0, :3] = (1.0, 2.0, 0.0)
    predicted[0, 0, :3] = (1.0, -2.0, 0.0)
    target[0, 1, :3] = (0.5, 0.0, 0.0)
    predicted[0, 1, :3] = (0.5, 0.0, 0.0)
    predicted[0, 2] = 1000.0
    is_pad = np.asarray([[False, False, True]])

    metrics = action_chunk_metrics(predicted, target, is_pad)

    assert metrics["valid_actions"] == 2
    assert metrics["translation_mae_y"] == pytest.approx(2.0)
    assert metrics["first_translation_mae_y"] == pytest.approx(4.0)
    assert metrics["first_translation_sign_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["translation_direction_cosine"] < 1.0
    assert metrics["gripper_mae"] == pytest.approx(0.0)


def test_action_chunk_metrics_reject_nonfinite_or_empty_input() -> None:
    values = np.zeros((1, 2, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="valid action"):
        action_chunk_metrics(values, values, np.ones((1, 2), dtype=np.bool_))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        action_chunk_metrics(
            values,
            np.zeros_like(values),
            np.zeros((1, 2), dtype=np.bool_),
        )


def test_partition_action_metrics_reports_causal_phases() -> None:
    target = np.zeros((2, 2, 7), dtype=np.float32)
    predicted = target.copy()
    predicted[1, 0, 0] = 2.0
    padding = np.zeros((2, 2), dtype=np.bool_)

    report = partition_action_metrics(
        predicted,
        target,
        padding,
        np.asarray([PHASE_IDS["approach"], PHASE_IDS["grasp"]]),
    )

    assert report["overall"]["valid_actions"] == 4
    assert report["by_phase"]["approach"]["translation_mae_x"] == 0.0
    assert report["by_phase"]["grasp"]["first_translation_mae_x"] == 2.0


def test_checkpoint_action_evaluation_loads_once_and_writes_partitions(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "diagnostics"
    config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        dataset=SimpleNamespace(
            root=tmp_path / "dataset",
            repo_id="local/test",
            episodes=50,
        ),
        act=SimpleNamespace(device="cpu", batch_size=2, seed=0),
        recovery=SimpleNamespace(output_dir=output_dir),
    )
    monkeypatch.setattr(
        diagnostics_module, "load_bridge_config", lambda path: config
    )
    monkeypatch.setattr(
        diagnostics_module, "validate_dataset_root", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        diagnostics_module,
        "pilot_episode_split",
        lambda **kwargs: {"train": [0], "validation": [1], "test": [2]},
    )
    monkeypatch.setattr(
        diagnostics_module,
        "phase_lookup_from_manifest",
        lambda root: {
            (0, 0): PHASE_IDS["approach"],
            (1, 0): PHASE_IDS["grasp"],
        },
    )

    def sample(episode: int) -> dict[str, object]:
        return {
            "action": torch.zeros(8, 7),
            "action_is_pad": torch.zeros(8, dtype=torch.bool),
            "episode_index": torch.tensor(episode),
            "frame_index": torch.tensor(0),
        }

    monkeypatch.setattr(
        diagnostics_module,
        "load_act_dataset",
        lambda **kwargs: [sample(int(kwargs["episodes"][0]))],
    )

    class Policy:
        reset_calls = 0

        def eval(self):
            return self

        def reset(self) -> None:
            self.reset_calls += 1

        def predict_action_chunk(self, batch):
            return batch["action"]

    policy = Policy()
    load_calls = []

    def load_bundle(**kwargs):
        load_calls.append(kwargs)
        return policy, lambda batch: batch, lambda actions: actions, {}

    monkeypatch.setattr(diagnostics_module, "_load_checkpoint_bundle", load_bundle)

    report = evaluate_checkpoint_actions("config.yaml", tmp_path / "checkpoint")

    assert len(load_calls) == 1
    assert policy.reset_calls == 2
    assert set(report) >= {"train", "validation", "checkpoint"}
    assert report["train"]["by_phase"]["approach"]["valid_actions"] == 8
    assert report["validation"]["by_phase"]["grasp"]["valid_actions"] == 8
    assert (output_dir / "action_diagnostics.json").is_file()
