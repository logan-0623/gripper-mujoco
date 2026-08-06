from __future__ import annotations

import numpy as np
import pytest

import interaction_vla.physics_recovery as physics_recovery_module
from interaction_vla.physics_recovery import (
    PhysicsRecoveryKind,
    make_physics_recovery_spec,
)


def test_post_grasp_specs_are_deterministic_and_balanced_per_source() -> None:
    first = tuple(make_physics_recovery_spec(42, index) for index in range(3))
    second = tuple(make_physics_recovery_spec(42, index) for index in range(3))

    assert [spec.kind for spec in first] == list(PhysicsRecoveryKind)[:3]
    for left, right in zip(first, second, strict=True):
        assert left.metadata() == right.metadata()

    wrong_way, premature_open, misalignment = first
    assert wrong_way.translation_steps == 3
    assert wrong_way.open_substeps == 0
    assert premature_open.translation_steps == 0
    assert premature_open.open_substeps == 1
    assert misalignment.translation_steps == 2
    assert misalignment.open_substeps == 0


def test_terminal_reclose_is_the_fourth_deterministic_local_variant() -> None:
    specs = tuple(
        make_physics_recovery_spec(42, index, kind_index=index)
        for index in range(4)
    )

    assert [spec.kind for spec in specs] == list(PhysicsRecoveryKind)
    terminal = specs[3]
    assert terminal.kind is PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE
    assert terminal.trigger_phase == "retreat"
    assert terminal.translation_steps == 0
    assert terminal.open_substeps == 0
    assert terminal.close_descent_steps == 5
    assert physics_recovery_module.recovery_trigger_ready(
        terminal,
        phase="retreat",
        stable_target=False,
        distance=0.04,
        supported_target=True,
    )
    assert not physics_recovery_module.recovery_trigger_ready(
        terminal,
        phase="retreat",
        stable_target=False,
        distance=0.04,
        supported_target=False,
    )


def test_recovery_trigger_requires_transport_stable_target_and_distance_band() -> None:
    wrong_way = make_physics_recovery_spec(42, 0)
    misalignment = make_physics_recovery_spec(42, 2)

    assert not physics_recovery_module.recovery_trigger_ready(
        wrong_way, phase="lift", stable_target=True, distance=0.30
    )
    assert not physics_recovery_module.recovery_trigger_ready(
        wrong_way, phase="transport", stable_target=False, distance=0.30
    )
    assert physics_recovery_module.recovery_trigger_ready(
        wrong_way, phase="transport", stable_target=True, distance=0.30
    )
    assert not physics_recovery_module.recovery_trigger_ready(
        misalignment, phase="transport", stable_target=True, distance=0.11
    )
    assert physics_recovery_module.recovery_trigger_ready(
        misalignment, phase="transport", stable_target=True, distance=0.10
    )


def test_recovery_translation_is_goal_relative() -> None:
    target = np.asarray((0.30, 0.10), dtype=np.float32)
    receptacle = np.asarray((0.60, 0.10), dtype=np.float32)
    toward_goal = receptacle - target

    wrong_way = physics_recovery_module.recovery_translation_direction(
        make_physics_recovery_spec(42, 0), target, receptacle
    )
    tangent = physics_recovery_module.recovery_translation_direction(
        make_physics_recovery_spec(42, 2), target, receptacle
    )

    assert float(np.dot(wrong_way, toward_goal)) < 0.0
    assert float(np.dot(tangent, toward_goal)) == pytest.approx(0.0, abs=1e-7)


def test_recovery_specs_reject_invalid_identifiers() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_physics_recovery_spec(-1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        make_physics_recovery_spec(0, -1)
