from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import interaction_vla.physics_action_safety as safety
from interaction_vla.franka_controller import ControllerDiagnostics


class NormLimitedController:
    def __init__(self, maximum_pose_norm: float) -> None:
        self.maximum_pose_norm = float(maximum_pose_norm)
        self.calls: list[np.ndarray] = []

    def apply_action(self, action: np.ndarray) -> ControllerDiagnostics:
        values = np.asarray(action, dtype=np.float64).copy()
        self.calls.append(values)
        norm = float(np.linalg.norm(values[:6]))
        return ControllerDiagnostics(
            ik_limited=norm > self.maximum_pose_norm,
            position_error=norm,
            orientation_error=0.0,
            iterations=1,
            joint_target=np.zeros(7, dtype=np.float64),
        )


def test_physics_action_safety_module_exists() -> None:
    assert importlib.util.find_spec("interaction_vla.physics_action_safety") is not None


def test_full_scale_action_is_returned_when_ik_is_feasible() -> None:
    controller = NormLimitedController(maximum_pose_norm=1.0)
    action = np.asarray((0.2, 0, 0, 0, 0, 0, 1), dtype=np.float32)

    result = safety.project_cartesian_action(controller, action)

    assert result.scale == 1.0
    assert np.array_equal(result.raw_action, action)
    assert np.array_equal(result.action, action)
    assert result.raw_diagnostics.ik_limited is False
    assert result.projected_diagnostics.ik_limited is False
    assert len(controller.calls) == 1


def test_projection_selects_first_feasible_scale_and_preserves_gripper() -> None:
    controller = NormLimitedController(maximum_pose_norm=0.21)
    action = np.asarray((0.8, 0, 0, 0, 0, 0, 1), dtype=np.float32)

    result = safety.project_cartesian_action(
        controller,
        action,
        scales=(1.0, 0.5, 0.25, 0.0),
    )

    assert result.scale == 0.25
    assert result.action[6] == 1.0
    assert np.array_equal(result.action[:6], result.raw_action[:6] * 0.25)
    assert result.raw_diagnostics.ik_limited is True
    assert result.projected_diagnostics.ik_limited is False
    assert [call[6] for call in controller.calls] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "scales",
    [
        (),
        (1.0,),
        (0.5, 0.0),
        (1.0, 0.5),
        (1.0, 0.5, 0.5, 0.0),
        (1.0, 0.25, 0.5, 0.0),
        (1.0, 1.1, 0.0),
        (1.0, float("nan"), 0.0),
    ],
)
def test_projection_rejects_invalid_scale_schedules(
    scales: tuple[float, ...],
) -> None:
    controller = NormLimitedController(maximum_pose_norm=1.0)

    with pytest.raises(ValueError, match="scale schedule"):
        safety.project_cartesian_action(
            controller,
            np.zeros(7, dtype=np.float32),
            scales=scales,
        )

    assert controller.calls == []


@pytest.mark.parametrize(
    "action",
    [np.zeros(6, dtype=np.float32), np.asarray((0, 0, 0, 0, 0, 0, np.nan))],
)
def test_projection_rejects_invalid_actions(action: np.ndarray) -> None:
    controller = NormLimitedController(maximum_pose_norm=1.0)

    with pytest.raises(ValueError, match=r"finite vector with shape \(7,\)"):
        safety.project_cartesian_action(controller, action)

    assert controller.calls == []


def test_projection_raises_when_zero_pose_is_ik_limited() -> None:
    controller = NormLimitedController(maximum_pose_norm=-1.0)

    with pytest.raises(RuntimeError, match="zero-pose Cartesian action"):
        safety.project_cartesian_action(
            controller,
            np.asarray((0.8, 0, 0, 0, 0, 0, 0), dtype=np.float32),
            scales=(1.0, 0.0),
        )

    assert len(controller.calls) == 2
