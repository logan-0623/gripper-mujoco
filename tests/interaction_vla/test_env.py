from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.env import KinematicTabletopEnv, TerminationReason


def object_positions(snapshot) -> np.ndarray:
    return np.stack([entity.position for entity in snapshot.objects])


def minimum_pairwise_distance(positions: np.ndarray) -> float:
    distances = [
        np.linalg.norm(positions[left, :2] - positions[right, :2])
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    ]
    return min(distances)


def test_reset_is_reproducible_and_non_overlapping() -> None:
    first = KinematicTabletopEnv(max_objects=5).reset(seed=17, object_count=4)
    second = KinematicTabletopEnv(max_objects=5).reset(seed=17, object_count=4)

    np.testing.assert_allclose(object_positions(first), object_positions(second))
    assert minimum_pairwise_distance(object_positions(first)) >= 0.12
    assert first.target_object.name == second.target_object.name


def test_crowded_reset_is_deterministic_and_places_anchor_near_target() -> None:
    first = KinematicTabletopEnv(max_objects=5).reset(
        seed=71, object_count=5, layout_mode="crowded"
    )
    second = KinematicTabletopEnv(max_objects=5).reset(
        seed=71, object_count=5, layout_mode="crowded"
    )

    np.testing.assert_allclose(object_positions(first), object_positions(second))
    target = first.target_object
    nearest = min(
        np.linalg.norm(entity.position[:2] - target.position[:2])
        for entity in first.objects
        if entity.name != target.name
    )
    assert 0.085 <= nearest <= 0.105


def test_normal_layout_retains_original_spacing() -> None:
    snapshot = KinematicTabletopEnv(max_objects=5).reset(
        seed=71, object_count=5, layout_mode="normal"
    )

    assert minimum_pairwise_distance(object_positions(snapshot)) >= 0.12


def test_closing_near_target_establishes_holding_relation() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=4, object_count=2)
    target = snapshot.target_object
    env.set_gripper_position(target.position + np.array([0.0, 0.0, 0.055], dtype=np.float32))

    result = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    assert result.snapshot.held_object == target.name
    assert frozenset(("gripper", target.name)) in result.snapshot.contacts
    assert not result.done


def test_held_object_follows_state_consistent_gripper_perturbation() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=4, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    )
    grasped = env.step(np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32))
    assert grasped.snapshot.held_object == snapshot.target_object.name
    step_count_before = env.step_count

    perturbed = env.perturb_gripper_state(
        np.asarray((0.04, -0.02, -0.02), dtype=np.float32)
    )

    np.testing.assert_allclose(
        perturbed.target_object.position,
        perturbed.gripper.position + env.hold_offset,
    )
    assert perturbed.held_object == snapshot.target_object.name
    assert env.step_count == step_count_before


def test_closing_near_distractor_reports_wrong_object() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=6, object_count=3)
    distractor = next(entity for entity in snapshot.objects if not entity.target)
    env.set_gripper_position(distractor.position + np.array([0.0, 0.0, 0.055], dtype=np.float32))

    result = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    assert result.done
    assert result.reason is TerminationReason.WRONG_OBJECT


def test_invalid_action_is_rejected() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    env.reset(seed=1, object_count=2)

    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([0.0, np.nan, 0.0, 1.0], dtype=np.float32))


def test_custom_workspace_bounds_are_enforced() -> None:
    env = KinematicTabletopEnv(
        max_objects=5,
        workspace_low=(-0.20, -0.10, 0.10),
        workspace_high=(0.20, 0.10, 0.30),
    )
    env.reset(seed=1, object_count=2)

    env.set_gripper_position(np.asarray((1.0, -1.0, 1.0), dtype=np.float32))

    np.testing.assert_allclose(env.gripper_position, (0.20, -0.10, 0.30))
