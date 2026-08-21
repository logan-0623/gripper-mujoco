from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from interaction_vla.representation_study.rl.environment import (
    ResidualMujocoRuntime,
    residual_clipping,
    interaction_potential,
    sparse_task_reward,
)


def test_sparse_reward_only_rewards_strict_success() -> None:
    assert sparse_task_reward("success") == 1.0
    assert sparse_task_reward("timeout") == 0.0
    assert sparse_task_reward("dropped") == 0.0


def test_interaction_potential_switches_from_gripper_to_receptacle() -> None:
    snapshot = SimpleNamespace(
        target_object=SimpleNamespace(position=np.asarray([1.0, 0.0, 0.0])),
        gripper=SimpleNamespace(position=np.asarray([0.0, 0.0, 0.0])),
        receptacle=SimpleNamespace(position=np.asarray([0.5, 0.0, 0.0])),
    )
    assert interaction_potential(snapshot, in_hand=False) == -1.0
    assert interaction_potential(snapshot, in_hand=True) == -0.5


def test_residual_clipping_reports_policy_bound_intervention() -> None:
    local, clipped = residual_clipping(
        np.asarray([0.95] * 6 + [0.9]),
        np.ones(7),
        np.asarray([0.2] * 7),
    )
    assert clipped is True
    assert np.all(local[:6] == 1.0)
    assert local[6] == 1.0


def test_residual_clipping_preserves_upstream_policy_clipping() -> None:
    _, clipped = residual_clipping(
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        base_was_clipped=True,
    )
    assert clipped is True


def test_environment_generator_state_is_resume_exact_without_serializing_mujoco() -> None:
    runtime = ResidualMujocoRuntime.__new__(ResidualMujocoRuntime)
    runtime.rng = np.random.default_rng(9)
    state = runtime.rng_state()
    expected = runtime.rng.integers(0, 1000, size=8)
    runtime.restore_rng_state(state)
    assert np.array_equal(runtime.rng.integers(0, 1000, size=8), expected)
