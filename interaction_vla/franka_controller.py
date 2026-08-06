from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import PhysicsConfig
from .franka import (
    ARM_ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    FINGER_ACTUATOR_NAME,
    FINGER_JOINT_NAMES,
    TCP_OFFSET_IN_HAND,
)


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape {shape}")
    return result


def so3_exp(rotation_vector: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    vector = _finite_vector(np.asarray(rotation_vector), (3,), "rotation vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        skew = np.asarray(
            (
                (0.0, -vector[2], vector[1]),
                (vector[2], 0.0, -vector[0]),
                (-vector[1], vector[0], 0.0),
            ),
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + skew
    axis = vector / angle
    skew = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite matrix with shape (3, 3)")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * vee
    if np.pi - angle < 1e-6:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = np.copysign(axis[0], vee[0] if abs(vee[0]) > 1e-12 else 1.0)
        axis[1] = np.copysign(axis[1], vee[1] if abs(vee[1]) > 1e-12 else 1.0)
        axis[2] = np.copysign(axis[2], vee[2] if abs(vee[2]) > 1e-12 else 1.0)
        norm = float(np.linalg.norm(axis))
        return angle * axis / max(norm, 1e-12)
    return angle * vee / (2.0 * np.sin(angle))


def relative_body_rotvec(current_rotation: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
    current = np.asarray(current_rotation, dtype=np.float64)
    target = np.asarray(target_rotation, dtype=np.float64)
    if current.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("current and target rotations must have shape (3, 3)")
    return so3_log(current.T @ target)


@dataclass(frozen=True)
class CartesianCommand:
    translation: np.ndarray
    rotation_vector: np.ndarray
    gripper_open: bool

    @classmethod
    def from_action(
        cls,
        action: np.ndarray,
        *,
        translation_delta: float,
        rotation_delta: float,
    ) -> CartesianCommand:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Cartesian action must be a finite vector with shape (7,)")
        if translation_delta <= 0.0 or rotation_delta <= 0.0:
            raise ValueError("action scales must be positive")

        translation = np.clip(values[:3], -1.0, 1.0) * translation_delta
        rotation_command = np.clip(values[3:6], -1.0, 1.0)
        norm = float(np.linalg.norm(rotation_command))
        if norm > 1.0:
            rotation_command = rotation_command / norm
        return cls(
            translation=translation,
            rotation_vector=rotation_command * rotation_delta,
            gripper_open=bool(values[6] >= 0.5),
        )


@dataclass(frozen=True)
class ControllerDiagnostics:
    ik_limited: bool
    position_error: float
    orientation_error: float
    iterations: int
    joint_target: np.ndarray


class FrankaCartesianController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: PhysicsConfig,
        *,
        workspace_low: np.ndarray,
        workspace_high: np.ndarray,
        max_joint_delta: float = 0.12,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self.workspace_low = _finite_vector(workspace_low, (3,), "workspace_low").copy()
        self.workspace_high = _finite_vector(workspace_high, (3,), "workspace_high").copy()
        if np.any(self.workspace_low >= self.workspace_high):
            raise ValueError("workspace_low must be below workspace_high")
        if not np.isfinite(max_joint_delta) or max_joint_delta <= 0.0:
            raise ValueError("max_joint_delta must be finite and positive")
        self.max_joint_delta = float(max_joint_delta)

        self.arm_joint_ids = np.asarray(
            [model.joint(name).id for name in ARM_JOINT_NAMES], dtype=np.int32
        )
        self.arm_qpos_addresses = model.jnt_qposadr[self.arm_joint_ids].astype(np.int32)
        self.arm_dof_addresses = model.jnt_dofadr[self.arm_joint_ids].astype(np.int32)
        self.arm_actuator_ids = np.asarray(
            [model.actuator(name).id for name in ARM_ACTUATOR_NAMES], dtype=np.int32
        )
        self.finger_qpos_addresses = np.asarray(
            [model.jnt_qposadr[model.joint(name).id] for name in FINGER_JOINT_NAMES],
            dtype=np.int32,
        )
        self.finger_actuator_id = model.actuator(FINGER_ACTUATOR_NAME).id
        self.hand_body_id = model.body("hand").id
        self.joint_ranges = model.jnt_range[self.arm_joint_ids].copy()
        self.ik_data = mujoco.MjData(model)
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_rotation = np.eye(3, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        position, rotation = self.tcp_pose(self.data)
        self.target_position = position
        self.target_rotation = rotation
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_addresses]
        finger_open = float(np.mean(self.data.qpos[self.finger_qpos_addresses])) >= 0.02
        self.data.ctrl[self.finger_actuator_id] = 255.0 if finger_open else 0.0

    def tcp_pose(self, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        state = self.data if data is None else data
        rotation = np.asarray(state.xmat[self.hand_body_id], dtype=np.float64).reshape(3, 3).copy()
        position = (
            np.asarray(state.xpos[self.hand_body_id], dtype=np.float64)
            + rotation @ TCP_OFFSET_IN_HAND
        )
        return position.copy(), rotation

    def apply_action(self, action: np.ndarray) -> ControllerDiagnostics:
        command = CartesianCommand.from_action(
            action,
            translation_delta=self.config.translation_delta,
            rotation_delta=self.config.rotation_delta,
        )
        measured_position, measured_rotation = self.tcp_pose(self.data)
        self.target_position = np.clip(
            measured_position + command.translation,
            self.workspace_low,
            self.workspace_high,
        )
        self.target_rotation = measured_rotation @ so3_exp(command.rotation_vector)
        joint_target, position_error, orientation_error, iterations = self._solve_ik()
        self.data.ctrl[self.arm_actuator_ids] = joint_target
        self.data.ctrl[self.finger_actuator_id] = 255.0 if command.gripper_open else 0.0
        return ControllerDiagnostics(
            ik_limited=bool(
                position_error > self.config.ik_position_tolerance
                or orientation_error > self.config.ik_orientation_tolerance
            ),
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=iterations,
            joint_target=joint_target.copy(),
        )

    def _solve_ik(self) -> tuple[np.ndarray, float, float, int]:
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        mujoco.mj_fwdPosition(self.model, self.ik_data)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        position_error_norm = np.inf
        orientation_error_norm = np.inf
        iterations = 0

        for iterations in range(1, self.config.ik_iterations + 1):
            current_position, current_rotation = self.tcp_pose(self.ik_data)
            position_error = self.target_position - current_position
            orientation_error = so3_log(self.target_rotation @ current_rotation.T)
            position_error_norm = float(np.linalg.norm(position_error))
            orientation_error_norm = float(np.linalg.norm(orientation_error))
            if (
                position_error_norm <= self.config.ik_position_tolerance
                and orientation_error_norm <= self.config.ik_orientation_tolerance
            ):
                break

            mujoco.mj_jac(
                self.model,
                self.ik_data,
                jacobian_position,
                jacobian_rotation,
                current_position,
                self.hand_body_id,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, self.arm_dof_addresses],
                    jacobian_rotation[:, self.arm_dof_addresses],
                )
            )
            error = np.concatenate((position_error, orientation_error))
            regularized = (
                jacobian @ jacobian.T
                + self.config.ik_damping**2 * np.eye(6, dtype=np.float64)
            )
            delta = jacobian.T @ np.linalg.solve(regularized, error)
            candidate = self.ik_data.qpos[self.arm_qpos_addresses] + delta
            self.ik_data.qpos[self.arm_qpos_addresses] = np.clip(
                candidate,
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
            )
            mujoco.mj_fwdPosition(self.model, self.ik_data)

        current = self.data.qpos[self.arm_qpos_addresses]
        target = np.clip(
            self.ik_data.qpos[self.arm_qpos_addresses],
            current - self.max_joint_delta,
            current + self.max_joint_delta,
        )
        target = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        actuator_ranges = self.model.actuator_ctrlrange[self.arm_actuator_ids]
        target = np.clip(target, actuator_ranges[:, 0], actuator_ranges[:, 1])

        self.ik_data.qpos[self.arm_qpos_addresses] = target
        mujoco.mj_fwdPosition(self.model, self.ik_data)
        final_position, final_rotation = self.tcp_pose(self.ik_data)
        position_error_norm = float(np.linalg.norm(self.target_position - final_position))
        orientation_error_norm = float(
            np.linalg.norm(so3_log(self.target_rotation @ final_rotation.T))
        )
        return target.astype(np.float64, copy=True), position_error_norm, orientation_error_norm, iterations
