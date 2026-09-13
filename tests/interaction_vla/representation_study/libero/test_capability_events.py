import numpy as np

from interaction_vla.representation_study.libero.annotation import (
    AnnotationThresholds,
    PrivilegedFrame,
    annotate_relocation_episode,
)
from interaction_vla.representation_study.libero.capability_events import (
    EpisodeEventTracker,
    _summarize,
)
from interaction_vla.representation_study.libero.task_semantics import GoalAtom, TaskSemanticsRegistry


def frame(i, z, *, contact=False, goal=False):
    pose = np.eye(4)
    target = pose.copy(); target[2, 3] = z
    return PrivilegedFrame(
        frame_index=i, gripper_pose=target.copy(), target_pose=target,
        goal_pose=pose.copy(), gripper_target_surface_distance=0.0,
        target_goal_surface_distance=0.0, finger_contact_groups=("left", "right") if contact else (),
        target_goal_contact=goal, target_source_contact=z == 0.0,
        gripper_aperture=0.2, source_supported=z == 0.0, goal_satisfied=goal,
    )


def test_summarizes_grasp_lift_normal_release_and_drop() -> None:
    thresholds = AnnotationThresholds(stable_window_frames=2, minimum_comotion_m=0.001)
    semantics = TaskSemanticsRegistry.default().resolve(
        suite="libero_spatial", task_id=0, task_name="fixture", language="fixture",
        goal_atoms=(GoalAtom("on", ("target", "goal")),),
    )
    frames = [frame(0, 0.0), frame(1, 0.01, contact=True), frame(2, 0.02, contact=True), frame(3, 0.0, goal=True)]
    tracker = EpisodeEventTracker(0, 10, 20, thresholds, 0.02, frames)
    result = _summarize(tracker, annotate_relocation_episode(frames, semantics, thresholds), success=True)
    assert result["stable_grasp"] and result["lift"] and result["normal_release"]
    assert not result["unintended_drop"]

    dropped = frames[:3] + [frame(3, 0.0)]
    tracker.frames = dropped
    result = _summarize(tracker, annotate_relocation_episode(dropped, semantics, thresholds), success=False)
    assert result["unintended_drop"] and not result["normal_release"]


def test_drop_then_regrasp_success_preserves_drop_and_records_recovery() -> None:
    thresholds = AnnotationThresholds(stable_window_frames=2, minimum_comotion_m=0.001)
    semantics = TaskSemanticsRegistry.default().resolve(
        suite="libero_spatial", task_id=0, task_name="fixture", language="fixture",
        goal_atoms=(GoalAtom("on", ("target", "goal")),),
    )
    frames = [
        frame(0, 0.0), frame(1, 0.02, contact=True), frame(2, 0.04, contact=True),
        frame(3, 0.0), frame(4, 0.01, contact=True), frame(5, 0.02, contact=True),
        frame(6, 0.02, goal=True),
    ]
    tracker = EpisodeEventTracker(0, 10, 20, thresholds, 0.02, frames)
    result = _summarize(
        tracker, annotate_relocation_episode(frames, semantics, thresholds), success=True
    )
    assert result["unintended_drop"] is True
    assert result["recovered_after_drop"] is True
    assert result["normal_release"] is True
    assert result["geometric_lift"] is True
    assert result["supported_lift"] is True
