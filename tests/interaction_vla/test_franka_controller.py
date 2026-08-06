from __future__ import annotations

import mujoco
import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig
from interaction_vla.franka import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    FRANKA_SCENE_PATH,
    HOME_QPOS,
)
from interaction_vla.franka_controller import (
    CartesianCommand,
    FrankaCartesianController,
    so3_exp,
)


def initialized_model_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    data = mujoco.MjData(model)
    for value, name in zip(HOME_QPOS, ARM_JOINT_NAMES, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    for name in FINGER_JOINT_NAMES:
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.04
    for index in range(5):
        address = model.jnt_qposadr[model.joint(f"object_{index}_joint").id]
        data.qpos[address : address + 3] = (0.0, 0.0, -2.0 - index)
        data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return model, data


def test_cartesian_action_is_scaled_and_clipped_by_vector_norm() -> None:
    command = CartesianCommand.from_action(
        np.asarray((1.0, -1.0, 0.5, 1.0, 1.0, 0.0, 0.49)),
        translation_delta=0.02,
        rotation_delta=np.deg2rad(3.0),
    )

    np.testing.assert_allclose(command.translation, (0.02, -0.02, 0.01))
    assert np.linalg.norm(command.rotation_vector) == pytest.approx(np.deg2rad(3.0))
    assert command.gripper_open is False


@pytest.mark.parametrize(
    "action",
    (np.zeros(6), np.zeros(8), np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 1.0))),
)
def test_cartesian_action_rejects_wrong_shape_or_non_finite_values(action: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        CartesianCommand.from_action(
            action,
            translation_delta=0.02,
            rotation_delta=np.deg2rad(3.0),
        )


def test_so3_exp_returns_the_requested_axis_angle_rotation() -> None:
    rotation = so3_exp(np.asarray((0.0, 0.0, np.pi / 2.0)))

    np.testing.assert_allclose(
        rotation @ np.asarray((1.0, 0.0, 0.0)),
        (0.0, 1.0, 0.0),
        atol=1e-7,
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)


def test_controller_composes_body_rotation_and_only_writes_actuator_targets() -> None:
    model, data = initialized_model_data()
    controller = FrankaCartesianController(
        model,
        data,
        PhysicsConfig(),
        workspace_low=np.asarray((0.25, -0.35, 0.23)),
        workspace_high=np.asarray((0.78, 0.35, 0.75)),
    )
    qpos_before = data.qpos.copy()
    position_before = controller.target_position.copy()
    rotation_before = controller.target_rotation.copy()

    diagnostics = controller.apply_action(
        np.asarray((0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    )

    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_allclose(
        controller.target_position,
        np.clip(position_before + (0.01, 0.0, 0.0), controller.workspace_low, controller.workspace_high),
    )
    np.testing.assert_allclose(
        controller.target_rotation,
        rotation_before @ so3_exp((np.deg2rad(3.0), 0.0, 0.0)),
        atol=1e-8,
    )
    np.testing.assert_allclose(data.ctrl[:7], diagnostics.joint_target)
    assert data.ctrl[7] == 0.0
    assert diagnostics.joint_target.shape == (7,)
    assert np.isfinite(diagnostics.joint_target).all()
    for target, name in zip(diagnostics.joint_target, ARM_JOINT_NAMES, strict=True):
        joint = model.joint(name)
        low, high = model.jnt_range[joint.id]
        assert low <= target <= high


def test_gripper_threshold_maps_to_the_upstream_actuator_range() -> None:
    model, data = initialized_model_data()
    controller = FrankaCartesianController(
        model,
        data,
        PhysicsConfig(),
        workspace_low=np.asarray((0.25, -0.35, 0.23)),
        workspace_high=np.asarray((0.78, 0.35, 0.75)),
    )

    controller.apply_action(np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)))
    assert data.ctrl[7] == 255.0
    controller.apply_action(np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.499)))
    assert data.ctrl[7] == 0.0


def test_delta_pose_is_anchored_to_measured_tcp_instead_of_accumulating_target_error() -> None:
    model, data = initialized_model_data()
    controller = FrankaCartesianController(
        model,
        data,
        PhysicsConfig(),
        workspace_low=np.asarray((0.25, -0.35, 0.23)),
        workspace_high=np.asarray((0.78, 0.35, 0.75)),
    )
    measured_position, _ = controller.tcp_pose()
    action = np.asarray((0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

    controller.apply_action(action)
    controller.apply_action(action)

    np.testing.assert_allclose(
        controller.target_position,
        np.clip(
            measured_position + (0.01, 0.0, 0.0),
            controller.workspace_low,
            controller.workspace_high,
        ),
    )
