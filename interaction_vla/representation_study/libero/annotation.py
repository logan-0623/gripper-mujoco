from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .schema import (
    ContactLabels,
    EntityLabels,
    FactorApplicability,
    GeometryLabels,
    InteractionLabels,
    NextRelation,
)
from .task_semantics import TaskSemantics


@dataclass(frozen=True)
class AnnotationThresholds:
    stable_window_frames: int = 5
    relative_translation_drift_m: float = 0.015
    relative_rotation_drift_deg: float = 15.0
    minimum_comotion_m: float = 0.005
    lift_clearance_m: float = 0.01
    approach_surface_distance_m: float = 0.05
    hysteresis_m: float = 0.01
    grasp_aperture_threshold: float = 0.55
    minimum_finger_groups: int = 2


@dataclass(frozen=True)
class PrivilegedFrame:
    frame_index: int
    gripper_pose: np.ndarray
    target_pose: np.ndarray
    goal_pose: np.ndarray
    gripper_target_surface_distance: float
    target_goal_surface_distance: float
    finger_contact_groups: tuple[str, ...]
    target_goal_contact: bool
    target_source_contact: bool
    gripper_aperture: float
    source_supported: bool
    goal_satisfied: bool

    def __post_init__(self) -> None:
        for name in ("gripper_pose", "target_pose", "goal_pose"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 4x4 transform")
        scalars = np.asarray(
            (
                self.gripper_target_surface_distance,
                self.target_goal_surface_distance,
                self.gripper_aperture,
            )
        )
        if not np.isfinite(scalars).all() or np.any(scalars[:2] < 0):
            raise ValueError("annotation distances/aperture must be finite and non-negative")


def relative_pose_9d(reference: np.ndarray, target: np.ndarray) -> tuple[float, ...]:
    relative = np.linalg.inv(np.asarray(reference, dtype=np.float64)) @ np.asarray(
        target, dtype=np.float64
    )
    rotation6d = relative[:3, :2].T.reshape(-1)
    result = np.concatenate((relative[:3, 3], rotation6d))
    if not np.isfinite(result).all():
        raise ValueError("relative pose must be finite")
    return tuple(float(item) for item in result)


def _rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _stable_grasp(
    frames: Sequence[PrivilegedFrame], index: int, thresholds: AnnotationThresholds
) -> bool | None:
    start = index - thresholds.stable_window_frames + 1
    if start < 0:
        return None
    window = frames[start : index + 1]
    if any(
        len(set(frame.finger_contact_groups)) < thresholds.minimum_finger_groups
        or frame.gripper_aperture > thresholds.grasp_aperture_threshold
        for frame in window
    ):
        return False
    relatives = [
        np.linalg.inv(frame.gripper_pose) @ frame.target_pose for frame in window
    ]
    translations = np.stack([pose[:3, 3] for pose in relatives])
    translation_drift = float(
        np.max(np.linalg.norm(translations - translations[0], axis=1), initial=0.0)
    )
    rotation_drift = max(
        (_rotation_angle_deg(relatives[0], pose) for pose in relatives), default=0.0
    )
    target_motion = float(
        np.linalg.norm(window[-1].target_pose[:3, 3] - window[0].target_pose[:3, 3])
    )
    physical_motion = (
        target_motion >= thresholds.minimum_comotion_m
        or not window[-1].source_supported
    )
    return bool(
        translation_drift <= thresholds.relative_translation_drift_m
        and rotation_drift <= thresholds.relative_rotation_drift_deg
        and physical_motion
    )


def _phase(
    frame: PrivilegedFrame,
    stable: bool | None,
    target_motion: float,
    near_target: bool,
    thresholds: AnnotationThresholds,
) -> str:
    contact = bool(frame.finger_contact_groups)
    if frame.goal_satisfied and not stable and not contact:
        return "release_retreat"
    if frame.goal_satisfied and (stable or contact):
        return "place"
    if stable and not frame.source_supported:
        return "transport" if target_motion >= thresholds.lift_clearance_m else "lift"
    if stable and frame.source_supported:
        return "secure"
    if contact:
        return "contact"
    if near_target:
        return "align_precontact"
    return "approach"


def _next_relation(
    frame: PrivilegedFrame,
    stable: bool | None,
    semantics: TaskSemantics,
    near_target: bool,
    near_goal: bool,
    thresholds: AnnotationThresholds,
) -> NextRelation:
    contact = bool(frame.finger_contact_groups)
    if frame.goal_satisfied and not stable and not contact:
        return NextRelation(0, "gripper", "clearance", "target", "increase")
    if frame.goal_satisfied and (stable or contact):
        return NextRelation(0, "gripper", "contact", "target", "clear")
    if stable and frame.source_supported:
        return NextRelation(0, "target", "off_support", "source", "establish")
    if stable and not frame.goal_satisfied:
        if not near_goal:
            return NextRelation(0, "target", "near", "goal", "establish")
        return NextRelation(
            0, "target", semantics.goal_predicate, "goal", "establish"
        )
    if contact:
        return NextRelation(0, "gripper", "stable_grasp", "target", "establish")
    if near_target:
        return NextRelation(0, "gripper", "contact", "target", "establish")
    return NextRelation(0, "gripper", "near", "target", "establish")


def annotate_relocation_episode(
    frames: Sequence[PrivilegedFrame],
    semantics: TaskSemantics,
    thresholds: AnnotationThresholds,
) -> tuple[InteractionLabels, ...]:
    if semantics.task_family != "relocation":
        raise ValueError("relocation annotator requires relocation task semantics")
    if not frames:
        raise ValueError("cannot annotate an empty episode")
    initial_target = np.asarray(frames[0].target_pose[:3, 3], dtype=np.float64)
    near_target = False
    near_goal = False
    labels: list[InteractionLabels] = []
    for index, frame in enumerate(frames):
        near_target = (
            frame.gripper_target_surface_distance
            <= thresholds.approach_surface_distance_m
            if not near_target
            else frame.gripper_target_surface_distance
            <= thresholds.approach_surface_distance_m + thresholds.hysteresis_m
        )
        near_goal = (
            frame.target_goal_surface_distance
            <= thresholds.approach_surface_distance_m
            if not near_goal
            else frame.target_goal_surface_distance
            <= thresholds.approach_surface_distance_m + thresholds.hysteresis_m
        )
        stable = _stable_grasp(frames, index, thresholds)
        applicability = FactorApplicability.all_applicable()
        if stable is None:
            applicability = replace(applicability, stable_grasp=False)
        target_motion = float(np.linalg.norm(frame.target_pose[:3, 3] - initial_target))
        contact = ContactLabels(
            gripper_target=bool(frame.finger_contact_groups),
            target_goal=frame.target_goal_contact,
            target_source=frame.target_source_contact,
            finger_groups=tuple(sorted(set(frame.finger_contact_groups))),
        )
        labels.append(
            InteractionLabels(
                applicability=applicability,
                entity=EntityLabels(
                    target=semantics.target,
                    goal=semantics.goal,
                    source=semantics.source,
                    distractors=semantics.distractors,
                ),
                geometry=GeometryLabels(
                    gripper_to_target=relative_pose_9d(
                        frame.gripper_pose, frame.target_pose
                    ),
                    target_to_goal=relative_pose_9d(frame.target_pose, frame.goal_pose),
                    gripper_target_distance=frame.gripper_target_surface_distance,
                    target_goal_distance=frame.target_goal_surface_distance,
                ),
                contact=contact,
                stable_grasp=stable,
                phase=_phase(frame, stable, target_motion, near_target, thresholds),
                next_relation=_next_relation(
                    frame, stable, semantics, near_target, near_goal, thresholds
                ),
            )
        )
    return tuple(labels)
