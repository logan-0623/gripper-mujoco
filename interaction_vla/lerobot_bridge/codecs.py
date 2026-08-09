from __future__ import annotations

from typing import Any

import numpy as np

from interaction_vla.franka import FINGER_JOINT_NAMES
from interaction_vla.graph.schema import SceneSnapshot


FINGER_POSITION_SLICE = slice(13, 15)
FINGER_POSITION_LOW = 0.0
FINGER_POSITION_HIGH = 0.04


def _finite(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _rotation(value: np.ndarray) -> np.ndarray:
    rotation = _finite(value, (3, 3), "gripper_rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("gripper_rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("gripper_rotation must be right-handed")
    return rotation


def validate_finger_joint_ranges(model: Any) -> None:
    expected = np.asarray((FINGER_POSITION_LOW, FINGER_POSITION_HIGH))
    for joint_name in FINGER_JOINT_NAMES:
        joint_range = np.asarray(model.joint(joint_name).range, dtype=np.float64)
        if joint_range.shape != (2,) or not np.allclose(
            joint_range, expected, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"{joint_name} range must remain [0.0, 0.04], got {joint_range.tolist()}"
            )


class EndEffectorStateCodec:
    @staticmethod
    def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        quat = _finite(quaternion, (4,), "quaternion")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise ValueError("quaternion norm must be non-zero")
        w, x, y, z = quat / norm
        return np.asarray(
            (
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def encode_rotation(rotation: np.ndarray) -> np.ndarray:
        return _rotation(rotation)[:, :2].T.reshape(6).astype(np.float32)

    @staticmethod
    def decode_rotation(rotation_6d: np.ndarray) -> np.ndarray:
        columns = _finite(rotation_6d, (6,), "rotation_6d").reshape(2, 3)
        first = columns[0]
        first_norm = float(np.linalg.norm(first))
        if first_norm < 1e-8:
            raise ValueError("rotation_6d first column must be non-zero")
        first = first / first_norm
        second = columns[1] - first * float(np.dot(first, columns[1]))
        second_norm = float(np.linalg.norm(second))
        if second_norm < 1e-8:
            raise ValueError("rotation_6d columns must not be near-collinear")
        second = second / second_norm
        third = np.cross(first, second)
        return np.column_stack((first, second, third))

    @classmethod
    def encode(
        cls,
        position: np.ndarray,
        rotation: np.ndarray,
        aperture: float,
    ) -> np.ndarray:
        position_array = _finite(position, (3,), "position")
        aperture_value = float(aperture)
        if not np.isfinite(aperture_value) or not 0.0 <= aperture_value <= 1.0:
            raise ValueError("aperture must be finite and within [0, 1]")
        return np.concatenate(
            (
                position_array,
                cls.encode_rotation(rotation),
                np.asarray((aperture_value,)),
            )
        ).astype(np.float32)

    @classmethod
    def encode_snapshot(
        cls,
        snapshot: SceneSnapshot,
        proprioception: np.ndarray,
    ) -> np.ndarray:
        proprio = _finite(proprioception, (23,), "proprioception")
        fingers = proprio[FINGER_POSITION_SLICE]
        aperture = float(
            np.clip(
                np.mean(
                    (fingers - FINGER_POSITION_LOW)
                    / (FINGER_POSITION_HIGH - FINGER_POSITION_LOW)
                ),
                0.0,
                1.0,
            )
        )
        rotation = cls.quaternion_to_matrix(snapshot.gripper.orientation)
        return cls.encode(snapshot.gripper.position, rotation, aperture)


class LocalCartesianActionCodec:
    @staticmethod
    def encode(action_world: np.ndarray, gripper_rotation: np.ndarray) -> np.ndarray:
        action = _finite(action_world, (7,), "action_world").copy()
        rotation = _rotation(gripper_rotation)
        action[:3] = rotation.T @ action[:3]
        return action.astype(np.float32)

    @staticmethod
    def decode(action_local: np.ndarray, gripper_rotation: np.ndarray) -> np.ndarray:
        action = _finite(action_local, (7,), "action_local").copy()
        rotation = _rotation(gripper_rotation)
        action[:3] = rotation @ action[:3]
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = float(action[6] >= 0.5)
        return action.astype(np.float32)
