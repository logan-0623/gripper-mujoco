import numpy as np

from interaction_vla.representation_study.libero.annotation import (
    AnnotationThresholds,
    PrivilegedFrame,
    annotate_relocation_episode,
    relative_pose_9d,
)
from interaction_vla.representation_study.libero.task_semantics import (
    GoalAtom,
    TaskSemanticsRegistry,
)


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (x, y, z)
    return pose


def _semantics():
    return TaskSemanticsRegistry.default().resolve(
        suite="libero_spatial",
        task_id=0,
        task_name="fixture",
        language="put bowl on plate",
        goal_atoms=(GoalAtom("on", ("bowl", "plate")),),
    )


def _frame(
    index: int,
    *,
    gripper_x: float,
    target_x: float = 0.0,
    goal_x: float = 0.5,
    fingers: tuple[str, ...] = (),
    aperture: float = 0.8,
    source_supported: bool = True,
    goal_satisfied: bool = False,
) -> PrivilegedFrame:
    return PrivilegedFrame(
        frame_index=index,
        gripper_pose=_pose(gripper_x),
        target_pose=_pose(target_x),
        goal_pose=_pose(goal_x),
        gripper_target_surface_distance=abs(gripper_x - target_x),
        target_goal_surface_distance=abs(target_x - goal_x),
        finger_contact_groups=fingers,
        target_goal_contact=goal_satisfied,
        target_source_contact=source_supported,
        gripper_aperture=aperture,
        source_supported=source_supported,
        goal_satisfied=goal_satisfied,
    )


def test_relative_pose_is_translation_plus_rotation6d() -> None:
    result = relative_pose_9d(_pose(1.0), _pose(1.5, 2.0, 3.0))
    assert result[:3] == (0.5, 2.0, 3.0)
    assert result[3:] == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def test_contact_is_not_stable_grasp_without_windowed_comotion() -> None:
    frames = [
        _frame(i, gripper_x=0.0, fingers=("left", "right"), aperture=0.2)
        for i in range(5)
    ]
    labels = annotate_relocation_episode(frames, _semantics(), AnnotationThresholds())
    assert all(label.contact.gripper_target for label in labels if label.contact)
    assert all(label.stable_grasp is None for label in labels[:-1])
    assert labels[-1].stable_grasp is False
    assert labels[-1].phase == "contact"


def test_stable_grasp_requires_bilateral_contact_pose_stability_and_comotion() -> None:
    frames = [
        _frame(
            i,
            gripper_x=0.01 * i,
            target_x=0.01 * i,
            fingers=("left", "right"),
            aperture=0.2,
            source_supported=i < 2,
        )
        for i in range(6)
    ]
    labels = annotate_relocation_episode(frames, _semantics(), AnnotationThresholds())
    assert labels[0].stable_grasp is None
    assert labels[-1].stable_grasp is True
    assert labels[-1].phase in {"lift", "transport"}
    assert labels[-1].next_relation.predicate == "near"
    assert labels[-1].next_relation.subject_role == "target"


def test_stable_grasp_is_not_biased_by_target_width() -> None:
    frames = [
        _frame(
            i,
            gripper_x=0.01 * i,
            target_x=0.01 * i,
            fingers=("left", "right"),
            aperture=0.80,
            source_supported=i < 2,
        )
        for i in range(6)
    ]

    labels = annotate_relocation_episode(frames, _semantics(), AnnotationThresholds())

    assert labels[-1].stable_grasp is True
    assert labels[-1].phase in {"lift", "transport"}


def test_release_relation_is_clearance_not_approach() -> None:
    frames = [
        _frame(i, gripper_x=0.48, target_x=0.5, goal_x=0.5, goal_satisfied=True)
        for i in range(5)
    ]
    labels = annotate_relocation_episode(frames, _semantics(), AnnotationThresholds())
    assert labels[-1].phase == "release_retreat"
    assert labels[-1].next_relation.predicate == "clearance"
    assert labels[-1].next_relation.operator == "increase"


def test_near_relation_uses_registered_hysteresis() -> None:
    thresholds = AnnotationThresholds(
        approach_surface_distance_m=0.05,
        hysteresis_m=0.01,
    )
    frames = [
        _frame(0, gripper_x=0.049),
        _frame(1, gripper_x=0.055),
        _frame(2, gripper_x=0.061),
    ]
    labels = annotate_relocation_episode(frames, _semantics(), thresholds)
    assert labels[0].phase == "align_precontact"
    assert labels[1].phase == "align_precontact"
    assert labels[2].phase == "approach"
