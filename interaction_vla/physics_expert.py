from __future__ import annotations

from enum import Enum

import numpy as np

from .config import PhysicsConfig
from .contact_physics import ContactDiagnostics, GraspState
from .franka_controller import relative_body_rotvec
from .graph.schema import SceneSnapshot


class PhysicsExpertPhase(str, Enum):
    APPROACH = "approach"
    ALIGN = "align"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    TRANSPORT = "transport"
    RELEASE = "release"
    RETREAT = "retreat"
    OPEN_RECOVER = "open_recover"


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (4,) or not np.isfinite(value).all() or norm < 1e-12:
        raise ValueError("orientation quaternion must be finite and non-zero")
    w, x, y, z = value / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _clip_by_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum else vector * (maximum / norm)


def delta_pose_action(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    gripper_open: bool,
    translation_delta: float,
    rotation_delta: float,
) -> np.ndarray:
    translation = _clip_by_norm(
        (np.asarray(target_position) - np.asarray(current_position)) / translation_delta,
        1.0,
    )
    rotation = relative_body_rotvec(current_rotation, target_rotation)
    rotation = _clip_by_norm(rotation / rotation_delta, 1.0)
    return np.concatenate(
        (translation, rotation, np.asarray((float(gripper_open),)))
    ).astype(np.float32)


class PhysicsScriptedExpert:
    def __init__(self, physics: PhysicsConfig, *, max_retries: int = 3) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.physics = physics
        self.max_retries = int(max_retries)
        self.reset(seed=0)

    def reset(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.phase = PhysicsExpertPhase.APPROACH
        self.phase_steps = 0
        self.retry_count = 0
        self._target_rotation: np.ndarray | None = None

    def act(
        self,
        snapshot: SceneSnapshot,
        contacts: ContactDiagnostics,
        grasp: GraspState,
    ) -> np.ndarray:
        current_position = snapshot.gripper.position.astype(np.float64)
        current_rotation = _quaternion_matrix(snapshot.gripper.orientation)
        if self._target_rotation is None:
            self._target_rotation = current_rotation.copy()
        target = snapshot.target_object
        target_name = target.name
        self._resynchronize_post_grasp(snapshot, grasp)
        correction = self._retry_correction()
        gripper_open = True

        if self.phase is PhysicsExpertPhase.APPROACH:
            waypoint = target.position.astype(np.float64) + correction + (0.0, 0.0, 0.13)
            if self._near(current_position, waypoint, 0.012):
                self._set_phase(PhysicsExpertPhase.ALIGN)
        elif self.phase is PhysicsExpertPhase.ALIGN:
            waypoint = target.position.astype(np.float64) + correction + (0.0, 0.0, 0.08)
            if self._near(current_position, waypoint, 0.010):
                self._set_phase(PhysicsExpertPhase.DESCEND)
        elif self.phase is PhysicsExpertPhase.DESCEND:
            waypoint = target.position.astype(np.float64) + correction + (0.0, 0.0, 0.002)
            if self._near(current_position, waypoint, 0.008):
                self._set_phase(PhysicsExpertPhase.CLOSE)
        elif self.phase is PhysicsExpertPhase.CLOSE:
            gripper_open = False
            waypoint = target.position.astype(np.float64) + correction + (0.0, 0.0, 0.002)
            if grasp.bilateral_object == target_name:
                self._set_phase(PhysicsExpertPhase.LIFT)
            elif self.phase_steps >= 20:
                self._start_recovery()
                gripper_open = True
        elif self.phase is PhysicsExpertPhase.LIFT:
            gripper_open = False
            if grasp.bilateral_object != target_name:
                self._start_recovery()
                gripper_open = True
                waypoint = target.position.astype(np.float64) + (0.0, 0.0, 0.13)
            else:
                waypoint = target.position.astype(np.float64).copy()
                waypoint[2] = 0.365
                if grasp.stable_object == target_name and current_position[2] >= 0.355:
                    self._set_phase(PhysicsExpertPhase.TRANSPORT)
        elif self.phase is PhysicsExpertPhase.TRANSPORT:
            gripper_open = False
            if grasp.bilateral_object != target_name:
                self._start_recovery()
                gripper_open = True
                waypoint = target.position.astype(np.float64) + (0.0, 0.0, 0.13)
            else:
                waypoint = snapshot.receptacle.position.astype(np.float64).copy()
                waypoint[2] = 0.39
                target_goal_distance = float(
                    np.linalg.norm(
                        target.position[:2] - snapshot.receptacle.position[:2]
                    )
                )
                if (
                    self._near(current_position, waypoint, 0.012)
                    and grasp.bilateral_object == target_name
                    and target_goal_distance <= 0.065
                ):
                    self._set_phase(PhysicsExpertPhase.RELEASE)
        elif self.phase is PhysicsExpertPhase.RELEASE:
            gripper_open = True
            waypoint = current_position
            if target_name in contacts.object_receptacle:
                self._set_phase(PhysicsExpertPhase.RETREAT)
        elif self.phase is PhysicsExpertPhase.RETREAT:
            gripper_open = True
            waypoint = current_position.copy()
            waypoint[2] = 0.50
        else:
            gripper_open = True
            waypoint = target.position.astype(np.float64) + (0.0, 0.0, 0.15)
            if self._near(current_position, waypoint, 0.015) or self.phase_steps >= 10:
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    self._set_phase(PhysicsExpertPhase.APPROACH)

        self.phase_steps += 1
        return delta_pose_action(
            current_position,
            current_rotation,
            waypoint,
            self._target_rotation,
            gripper_open=gripper_open,
            translation_delta=self.physics.translation_delta,
            rotation_delta=self.physics.rotation_delta,
        )

    def _start_recovery(self) -> None:
        self._set_phase(PhysicsExpertPhase.OPEN_RECOVER)

    def _resynchronize_post_grasp(
        self, snapshot: SceneSnapshot, grasp: GraspState
    ) -> None:
        target = snapshot.target_object
        target_name = target.name
        target_goal_distance = float(
            np.linalg.norm(target.position[:2] - snapshot.receptacle.position[:2])
        )
        holding_target = grasp.bilateral_object == target_name
        grasp_established = holding_target and (
            grasp.stable_object == target_name or grasp.ever_stable_target
        )
        if grasp_established:
            if (
                self.phase is PhysicsExpertPhase.RELEASE
                and target_goal_distance <= 0.065
            ):
                return
            if self.phase is PhysicsExpertPhase.LIFT:
                desired_phase = PhysicsExpertPhase.LIFT
            elif self.phase is PhysicsExpertPhase.TRANSPORT:
                desired_phase = (
                    PhysicsExpertPhase.LIFT
                    if snapshot.gripper.position[2] < 0.335
                    else PhysicsExpertPhase.TRANSPORT
                )
            else:
                desired_phase = (
                    PhysicsExpertPhase.LIFT
                    if snapshot.gripper.position[2] < 0.355
                    else PhysicsExpertPhase.TRANSPORT
                )
            if self.phase is not desired_phase:
                self._set_phase(desired_phase)
        elif (
            self.phase is PhysicsExpertPhase.RELEASE
            and target_goal_distance > 0.065
        ):
            self._start_recovery()

    def _set_phase(self, phase: PhysicsExpertPhase) -> None:
        self.phase = phase
        self.phase_steps = 0

    def _retry_correction(self) -> np.ndarray:
        if self.retry_count == 0:
            return np.zeros(3, dtype=np.float64)
        rng = np.random.default_rng(
            np.random.SeedSequence((self.seed, self.retry_count, 0x45585054))
        )
        angle = float(rng.uniform(-np.pi, np.pi))
        radius = 0.005
        return np.asarray(
            (radius * np.cos(angle), radius * np.sin(angle), 0.0),
            dtype=np.float64,
        )

    @staticmethod
    def _near(current: np.ndarray, target: np.ndarray, tolerance: float) -> bool:
        return bool(np.linalg.norm(current - target) <= tolerance)
