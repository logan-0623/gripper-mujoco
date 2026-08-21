from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    FINGER_POSITION_HIGH,
    FINGER_POSITION_LOW,
    FINGER_POSITION_SLICE,
)

from .distributions import RecoveryCase


ORACLE_STATE_WIDTH = 36
ORACLE_SLICES = {
    "gripper_target": slice(0, 10),
    "target_receptacle": slice(10, 16),
    "interaction": slice(16, 20),
    "distractor": slice(20, 21),
    "phase": slice(21, 27),
    "intervention": slice(27, 34),
    "recovery": slice(34, 36),
}
PHASES = ("approach", "grasp", "lift", "transport", "place", "release")
INTERVENTIONS = (
    "nominal",
    "approach_offset",
    "grasp_offset",
    "lift_offset",
    "wrong_way_transport",
    "premature_open",
    "receptacle_misalignment",
)


def _positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class OracleNormalization:
    gripper_target_translation_scale: float = 0.50
    target_receptacle_translation_scale: float = 0.50
    distance_scale: float = 0.75
    placement_xy_scale: float = 0.20
    distractor_clearance_scale: float = 0.50
    progress_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_finite(float(value), name)

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": "compact_oracle_normalization_v1",
            "oracle_state_width": ORACLE_STATE_WIDTH,
            **asdict(self),
        }


def _position(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3D position")
    return result


def _rotation(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.allclose(result.T @ result, np.eye(3), atol=1.0e-6)
        or not np.isclose(np.linalg.det(result), 1.0, atol=1.0e-6)
    ):
        raise ValueError(f"{name} must be a finite right-handed rotation")
    return result


def _one_hot(value: str, vocabulary: tuple[str, ...], name: str) -> np.ndarray:
    try:
        index = vocabulary.index(value)
    except ValueError as error:
        raise ValueError(f"unknown {name}: {value}") from error
    result = np.zeros(len(vocabulary), dtype=np.float64)
    result[index] = 1.0
    return result


@dataclass(frozen=True)
class CompactOracleStateCodec:
    normalization: OracleNormalization = field(default_factory=OracleNormalization)

    def encode(
        self,
        *,
        gripper_position: object,
        gripper_rotation: object,
        target_position: object,
        target_rotation: object,
        receptacle_position: object,
        receptacle_rotation: object,
        distractor_positions: object,
        gripper_open_fraction: float,
        bilateral_target_contact: bool,
        stable_target_grasp: bool,
        target_support: bool,
        phase: str,
        intervention_kind: str,
        severity: float,
        progress: float,
    ) -> np.ndarray:
        gripper_xyz = _position(gripper_position, "gripper_position")
        target_xyz = _position(target_position, "target_position")
        receptacle_xyz = _position(receptacle_position, "receptacle_position")
        gripper_rot = _rotation(gripper_rotation, "gripper_rotation")
        target_rot = _rotation(target_rotation, "target_rotation")
        receptacle_rot = _rotation(receptacle_rotation, "receptacle_rotation")
        distractors = np.asarray(distractor_positions, dtype=np.float64)
        if distractors.ndim != 2 or distractors.shape[1:] != (3,) or not np.isfinite(distractors).all():
            raise ValueError("distractor_positions must be a finite [objects, 3] array")
        aperture = float(gripper_open_fraction)
        severity_value = float(severity)
        progress_value = float(progress)
        if not np.isfinite(aperture) or not 0.0 <= aperture <= 1.0:
            raise ValueError("gripper_open_fraction must lie within [0, 1]")
        if not np.isfinite(severity_value) or not 0.0 <= severity_value <= 1.0:
            raise ValueError("severity must lie within [0, 1]")
        if not np.isfinite(progress_value):
            raise ValueError("progress must be finite")

        gripper_target_world = target_xyz - gripper_xyz
        gripper_target_local = gripper_rot.T @ gripper_target_world
        relative_rotation = gripper_rot.T @ target_rot
        gripper_target = np.concatenate(
            (
                np.clip(
                    gripper_target_local
                    / self.normalization.gripper_target_translation_scale,
                    -1.0,
                    1.0,
                ),
                EndEffectorStateCodec.encode_rotation(relative_rotation),
                np.asarray(
                    (
                        np.clip(
                            np.linalg.norm(gripper_target_world)
                            / self.normalization.distance_scale,
                            0.0,
                            1.0,
                        ),
                    )
                ),
            )
        )

        target_receptacle_world = target_xyz - receptacle_xyz
        target_receptacle_local = receptacle_rot.T @ target_receptacle_world
        upright_alignment = float(
            np.clip(np.dot(target_rot[:, 2], receptacle_rot[:, 2]), -1.0, 1.0)
        )
        placement_alignment = 1.0 - float(
            np.clip(
                np.linalg.norm(target_receptacle_local[:2])
                / self.normalization.placement_xy_scale,
                0.0,
                1.0,
            )
        )
        target_receptacle = np.concatenate(
            (
                np.clip(
                    target_receptacle_local
                    / self.normalization.target_receptacle_translation_scale,
                    -1.0,
                    1.0,
                ),
                np.asarray(
                    (
                        np.clip(
                            np.linalg.norm(target_receptacle_world)
                            / self.normalization.distance_scale,
                            0.0,
                            1.0,
                        ),
                        upright_alignment,
                        placement_alignment,
                    )
                ),
            )
        )

        nearest_clearance = (
            self.normalization.distractor_clearance_scale
            if distractors.shape[0] == 0
            else float(np.min(np.linalg.norm(distractors[:, :2] - target_xyz[:2], axis=1)))
        )
        encoded = np.concatenate(
            (
                gripper_target,
                target_receptacle,
                np.asarray(
                    (
                        aperture,
                        float(bool(bilateral_target_contact)),
                        float(bool(stable_target_grasp)),
                        float(bool(target_support)),
                    )
                ),
                np.asarray(
                    (
                        np.clip(
                            nearest_clearance
                            / self.normalization.distractor_clearance_scale,
                            0.0,
                            1.0,
                        ),
                    )
                ),
                _one_hot(str(phase), PHASES, "phase"),
                _one_hot(str(intervention_kind), INTERVENTIONS, "intervention kind"),
                np.asarray(
                    (
                        severity_value,
                        np.clip(
                            progress_value / self.normalization.progress_scale,
                            -1.0,
                            1.0,
                        ),
                    )
                ),
            )
        ).astype(np.float32)
        if encoded.shape != (ORACLE_STATE_WIDTH,) or not np.isfinite(encoded).all():
            raise ValueError("Compact Oracle-State must be finite with width 36")
        return encoded

    def encode_snapshot(
        self,
        env: Any,
        snapshot: Any,
        case: RecoveryCase,
        *,
        progress: float,
    ) -> np.ndarray:
        target_name = snapshot.target_object.name
        proprioception = np.asarray(env.proprioception(), dtype=np.float64)
        if proprioception.shape != (23,) or not np.isfinite(proprioception).all():
            raise ValueError("runtime proprioception must be finite with width 23")
        fingers = proprioception[FINGER_POSITION_SLICE]
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
        distractors = np.asarray(
            [item.position for item in snapshot.objects if item.name != target_name],
            dtype=np.float64,
        ).reshape(-1, 3)
        contacts = env.contact_diagnostics
        grasp = env.grasp_state
        return self.encode(
            gripper_position=snapshot.gripper.position,
            gripper_rotation=EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.gripper.orientation
            ),
            target_position=snapshot.target_object.position,
            target_rotation=EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.target_object.orientation
            ),
            receptacle_position=snapshot.receptacle.position,
            receptacle_rotation=EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.receptacle.orientation
            ),
            distractor_positions=distractors,
            gripper_open_fraction=aperture,
            bilateral_target_contact=grasp.bilateral_object == target_name,
            stable_target_grasp=grasp.stable_object == target_name,
            target_support=(
                target_name in contacts.object_table
                or target_name in contacts.object_receptacle
            ),
            phase=case.phase,
            intervention_kind=case.intervention_kind,
            severity=case.severity,
            progress=progress,
        )

    def encode_runtime(
        self,
        env: Any,
        prepared: Any,
        case: RecoveryCase,
    ) -> np.ndarray:
        return self.encode_snapshot(
            env,
            prepared.snapshot,
            case,
            progress=0.0,
        )
