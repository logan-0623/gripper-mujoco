from __future__ import annotations

import numpy as np
import pytest
import torch

from interaction_vla.chunked_controller import (
    ChunkedPolicyController,
    TemporalActionEnsembler,
)
from interaction_vla.config import PhysicsConfig
from interaction_vla.graph.builder import SceneGraphBuilder
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.train import TrainingStatistics


def constant_chunk(value: float, *, horizon: int = 3, gripper: float = 1.0) -> np.ndarray:
    chunk = np.full((horizon, 7), value, dtype=np.float32)
    chunk[:, 6] = gripper
    return chunk


def make_ensemble(*, horizon: int = 3, initially_open: bool = True):
    ensemble = TemporalActionEnsembler(
        horizon=horizon,
        temporal_decay=0.25,
        gripper_close_threshold=0.35,
        gripper_open_threshold=0.65,
    )
    ensemble.reset(gripper_open=initially_open)
    return ensemble


def test_temporal_ensemble_uses_all_predictions_for_the_current_step() -> None:
    ensemble = make_ensemble()
    ensemble.add(0, constant_chunk(0.1))
    ensemble.add(1, constant_chunk(0.2))
    ensemble.add(2, constant_chunk(0.3))

    action, diagnostics = ensemble.action_for_step(2)

    weights = np.exp(-0.25 * np.arange(3))
    expected = np.average((0.3, 0.2, 0.1), weights=weights)
    np.testing.assert_allclose(action[:6], expected, atol=1e-7)
    assert diagnostics.ensemble_size == 3
    np.testing.assert_allclose(diagnostics.raw_first_action[:6], 0.3)
    np.testing.assert_allclose(diagnostics.aggregated_action, action)


def test_h1_ensemble_matches_the_current_prediction_and_expires_old_chunks() -> None:
    ensemble = make_ensemble(horizon=1)
    ensemble.add(0, constant_chunk(0.2, horizon=1))
    first, _ = ensemble.action_for_step(0)
    ensemble.add(1, constant_chunk(-0.4, horizon=1, gripper=0.0))
    second, diagnostics = ensemble.action_for_step(1)

    np.testing.assert_allclose(first[:6], 0.2)
    np.testing.assert_allclose(second[:6], -0.4)
    assert diagnostics.ensemble_size == 1


def test_gripper_hysteresis_holds_state_inside_deadband() -> None:
    ensemble = make_ensemble(initially_open=True)

    assert ensemble.resolve_gripper(0.34) == 0.0
    assert ensemble.resolve_gripper(0.50) == 0.0
    assert ensemble.resolve_gripper(0.66) == 1.0
    assert ensemble.resolve_gripper(0.50) == 1.0
    assert ensemble.gripper_switch_count == 2


class FakeChunkPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.scene_encoder = object()
        self.last_device: torch.device | None = None

    def predict_action_chunk(self, scene, proprioception):
        self.last_device = proprioception.device
        chunk = torch.zeros((len(proprioception), 8, 7), device=proprioception.device)
        chunk[:, :, 2] = 0.1 + self.anchor
        chunk[:, :, 6] = 1.0
        return chunk


def physics_statistics() -> TrainingStatistics:
    return TrainingStatistics(
        node_mean=np.zeros(23, dtype=np.float32),
        node_std=np.ones(23, dtype=np.float32),
        edge_mean=np.zeros(18, dtype=np.float32),
        edge_std=np.ones(18, dtype=np.float32),
        proprio_mean=np.zeros(23, dtype=np.float32),
        proprio_std=np.ones(23, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
    )


def test_chunked_policy_controller_runs_cpu_normalization_hysteresis_and_ik() -> None:
    env = FrankaContactEnv(
        max_steps=5,
        physics=PhysicsConfig(settle_steps=5),
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
    )
    env.reset(seed=11, object_count=2)
    policy = FakeChunkPolicy()
    controller = ChunkedPolicyController(
        policy=policy,
        statistics=physics_statistics(),
        builder=SceneGraphBuilder(max_objects=5, feature_schema="physics_v2"),
        horizon=8,
        temporal_decay=0.25,
        gripper_close_threshold=0.35,
        gripper_open_threshold=0.65,
        device="cpu",
    )
    controller.reset(env)

    action, diagnostics = controller.act(env)

    assert policy.last_device == torch.device("cpu")
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert np.all(np.abs(action[:6]) <= 1.0)
    assert action[6] in {0.0, 1.0}
    assert diagnostics.ik_projection_scale <= 1.0
    assert diagnostics.ensemble_size == 1


def test_chunked_controller_rejects_non_cpu_rollout() -> None:
    with pytest.raises(ValueError, match="CPU"):
        ChunkedPolicyController(
            policy=FakeChunkPolicy(),
            statistics=physics_statistics(),
            builder=SceneGraphBuilder(max_objects=5, feature_schema="physics_v2"),
            horizon=8,
            temporal_decay=0.25,
            gripper_close_threshold=0.35,
            gripper_open_threshold=0.65,
            device="mps",
        )
