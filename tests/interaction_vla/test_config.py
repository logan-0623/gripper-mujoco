from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from interaction_vla.config import (
    EnvironmentConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    PhysicsConfig,
    RandomizationConfig,
    RecordingConfig,
    RecoveryConfig,
    SequenceConfig,
    TrainConfig,
    load_config,
)
from interaction_vla.device import resolve_device


def test_invalid_capacity_fails_early() -> None:
    with pytest.raises(ValueError, match="max_objects"):
        ExperimentConfig(max_objects=1)


def test_auto_device_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto").type == "cuda"


def test_auto_device_uses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto").type == "mps"


def test_device_falls_back_to_cpu_when_accelerators_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"


def test_explicit_unavailable_mps_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_explicit_unavailable_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_experiment_config_accepts_cuda_training() -> None:
    assert ExperimentConfig(device="cuda").device == "cuda"


def test_action_dimension_must_match_the_selected_backend() -> None:
    assert ModelConfig(action_dim=7).action_dim == 7

    with pytest.raises(ValueError, match="action_dim=4"):
        ExperimentConfig(model=ModelConfig(action_dim=7))
    with pytest.raises(ValueError, match="action_dim=7"):
        ExperimentConfig(backend="franka_contact", model=ModelConfig(action_dim=4))


def test_physics_config_declares_the_7d_500hz_contract() -> None:
    cfg = load_config("configs/physics_smoke_macos.yaml")

    assert cfg.backend == "franka_contact"
    assert cfg.model.action_dim == 7
    assert cfg.physics.timestep == pytest.approx(0.002)
    assert cfg.physics.policy_hz == 20
    assert cfg.physics.substeps == 25
    assert cfg.physics.translation_delta == pytest.approx(0.02)
    assert cfg.physics.rotation_delta == pytest.approx(0.05235987755982988)
    assert cfg.recording.cameras == (
        "agentview",
        "wristview",
        "sideview",
        "topview",
    )


def test_physics_frequency_product_is_validated() -> None:
    with pytest.raises(ValueError, match="frequency"):
        PhysicsConfig(timestep=0.002, policy_hz=20, substeps=24)


def test_randomization_ranges_and_recording_shape_are_validated() -> None:
    with pytest.raises(ValueError, match="mass"):
        RandomizationConfig(object_mass_scale=(1.2, 0.8))
    with pytest.raises(ValueError, match="cameras"):
        RecordingConfig(cameras=("agentview", "wristview"))


def test_ood_counts_cannot_overlap_training_counts() -> None:
    with pytest.raises(ValueError, match="OOD"):
        ExperimentConfig(
            train=TrainConfig(object_counts=(2, 3)),
            eval=EvalConfig(
                object_counts=(2, 3, 4),
                ood_object_counts=(3, 4),
                crowded_object_counts=(4,),
            ),
        )


def test_invalid_workspace_fails_during_config_loading() -> None:
    with pytest.raises(ValueError, match="workspace"):
        EnvironmentConfig(workspace_low=(0.0, 0.0, 0.0), workspace_high=(0.0, 1.0, 1.0))


def test_crowded_distances_are_validated() -> None:
    cfg = EnvironmentConfig(
        crowded_anchor_min_distance=0.085,
        crowded_anchor_max_distance=0.105,
    )

    assert cfg.crowded_anchor_min_distance < cfg.crowded_anchor_max_distance
    with pytest.raises(ValueError, match="crowded"):
        EnvironmentConfig(
            crowded_anchor_min_distance=0.11,
            crowded_anchor_max_distance=0.09,
        )


def test_crowded_eval_counts_must_be_configured_and_held_out() -> None:
    cfg = EvalConfig(
        object_counts=(2, 3, 4, 5),
        ood_object_counts=(4, 5),
        crowded_object_counts=(4, 5),
    )
    assert cfg.crowded_object_counts == (4, 5)

    with pytest.raises(ValueError, match="crowded"):
        EvalConfig(
            object_counts=(2, 3, 4),
            ood_object_counts=(4,),
            crowded_object_counts=(5,),
        )
    with pytest.raises(ValueError, match="crowded"):
        ExperimentConfig(
            train=TrainConfig(object_counts=(2, 3)),
            eval=EvalConfig(
                object_counts=(2, 3, 4),
                ood_object_counts=(4,),
                crowded_object_counts=(3, 4),
            ),
        )


def test_recovery_variant_count_is_validated() -> None:
    with pytest.raises(ValueError, match="variants"):
        RecoveryConfig(variants_per_episode=-1)
    with pytest.raises(ValueError, match="variants"):
        RecoveryConfig(enabled=True, variants_per_episode=0)


def test_recovery_acceptance_rate_is_validated() -> None:
    assert RecoveryConfig(min_acceptance_rate=0.8).min_acceptance_rate == 0.8
    with pytest.raises(ValueError, match="acceptance"):
        RecoveryConfig(min_acceptance_rate=-0.1)
    with pytest.raises(ValueError, match="acceptance"):
        RecoveryConfig(min_acceptance_rate=1.1)


def test_sequence_config_encodes_the_shared_h8_controller() -> None:
    config = SequenceConfig(
        enabled=True,
        horizon=8,
        future_loss_decay=0.9,
        temporal_decay=0.25,
        gripper_close_threshold=0.35,
        gripper_open_threshold=0.65,
        recovery_loss_fraction=0.25,
        rollout_device="cpu",
    )

    assert config.horizon == 8
    assert config.recovery_loss_fraction == 0.25
    assert config.rollout_device == "cpu"


@pytest.mark.parametrize(
    "changes",
    [
        {"horizon": 0},
        {"future_loss_decay": 0.0},
        {"future_loss_decay": 1.1},
        {"temporal_decay": -0.1},
        {"gripper_close_threshold": 0.7, "gripper_open_threshold": 0.6},
        {"recovery_loss_fraction": 1.0},
        {"rollout_device": "mps"},
    ],
)
def test_enabled_sequence_config_rejects_unfair_or_invalid_values(changes) -> None:
    base = SequenceConfig(enabled=True, horizon=8, recovery_loss_fraction=0.25)

    with pytest.raises(ValueError):
        replace(base, **changes)