from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import interaction_vla.representation_study.rl.environment as environment_module
from interaction_vla.representation_study.rl.distributions import RecoveryCase
from interaction_vla.representation_study.rl.environment import ResidualMujocoRuntime
from interaction_vla.representation_study.rl.rewards import (
    recovery_reward,
    terminal_reward,
)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("success", 1.0), ("dropped", -1.0), ("wrong_object", -1.0), ("timeout", 0.0)],
)
def test_terminal_reward_signs(reason: str, expected: float) -> None:
    assert terminal_reward(reason) == expected


def test_reward_matches_registered_decomposition() -> None:
    terms = recovery_reward(
        reason="running",
        previous_potential=-0.5,
        next_potential=-0.4,
        residual=np.ones(7),
        residual_scale=np.full(7, 0.1),
        gamma=0.99,
    )
    assert terms.total == pytest.approx(
        0.10 * (0.99 * -0.4 + 0.5) - 0.01 * 0.07
    )
    assert terms.terminal == 0.0
    assert terms.progress > 0.0
    assert terms.residual < 0.0


def test_physics_failure_cannot_be_silently_optimized() -> None:
    with pytest.raises(ValueError, match="physics_failure"):
        terminal_reward("physics_failure")


def test_reward_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        recovery_reward(
            reason="running",
            previous_potential=0.0,
            next_potential=np.nan,
            residual=np.zeros(7),
            residual_scale=np.ones(7),
            gamma=0.99,
        )


def test_v2_runtime_step_records_reward_terms_and_next_oracle(monkeypatch) -> None:
    identity = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

    def snapshot(target_x: float):
        return SimpleNamespace(
            target_object=SimpleNamespace(
                name="object_0", position=np.asarray((target_x, 0.0, 0.0))
            ),
            receptacle=SimpleNamespace(position=np.asarray((0.0, 0.0, 0.0))),
            gripper=SimpleNamespace(
                position=np.asarray((0.0, 0.0, 0.0)), orientation=identity
            ),
        )

    class FakeOracle:
        def encode_snapshot(self, env, value, case, *, progress):
            del env, value, case
            return np.full(36, progress, dtype=np.float32)

    initial = snapshot(1.0)
    following = snapshot(0.8)
    fake_env = SimpleNamespace(
        controller=object(),
        grasp_state=SimpleNamespace(stable_object=None),
        target_name="object_0",
        step=lambda action: SimpleNamespace(
            snapshot=following,
            reason=SimpleNamespace(value="success"),
            done=True,
        ),
    )
    monkeypatch.setattr(
        environment_module,
        "project_cartesian_action",
        lambda controller, action: SimpleNamespace(action=action, scale=1.0),
    )
    runtime = ResidualMujocoRuntime.__new__(ResidualMujocoRuntime)
    runtime.env = fake_env
    runtime.snapshot = initial
    runtime.current_observation = {"task": ["test"]}
    runtime.residual_scale = np.full(7, 0.1, dtype=np.float32)
    runtime.base_action_was_clipped = False
    runtime.gripper = SimpleNamespace(resolve=lambda value: value)
    runtime.clipped_steps = 0
    runtime.projection_scales = []
    runtime.episode_return = 0.0
    runtime.episode_length = 0
    runtime.previous_potential = -1.0
    runtime.intervention_start_potential = -1.0
    runtime.reward_mode = "sparse"
    runtime.progress_reward_scale = 0.0
    runtime.recovery_gamma = 0.99
    runtime.recovery_progress_coefficient = 0.10
    runtime.recovery_residual_coefficient = 0.01
    runtime.active_case = RecoveryCase(
        case_id="training:11:nominal:0:nominal",
        partition="training",
        family="nominal",
        source_seed=11,
        variant_id=0,
        object_count=2,
        layout="normal",
        phase="approach",
        intervention_kind="nominal",
        severity=0.0,
    )
    runtime.oracle_codec = FakeOracle()
    runtime.current_oracle_state = np.zeros(36, dtype=np.float32)

    transition = runtime.step(
        base_action=np.zeros(7, dtype=np.float32),
        latent=torch.zeros(1, 4),
        residual=np.ones(7, dtype=np.float32),
    )

    assert transition.reward_terminal == 1.0
    assert transition.reward_progress > 0.0
    assert transition.reward_residual < 0.0
    assert transition.reward == pytest.approx(
        transition.reward_terminal
        + transition.reward_progress
        + transition.reward_residual
    )
    np.testing.assert_array_equal(transition.oracle_state, 0.0)
    assert np.all(transition.next_oracle_state > 0.0)
