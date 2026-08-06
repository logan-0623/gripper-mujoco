from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.expert import ExpertPhase
from interaction_vla.recovery import (
    PerturbationKind,
    apply_recovery_spec,
    make_recovery_spec,
)


def test_recovery_specs_are_deterministic_and_round_robin() -> None:
    first = make_recovery_spec(source_seed=42, variant_id=0)
    second = make_recovery_spec(source_seed=42, variant_id=0)

    assert first.kind == second.kind
    np.testing.assert_array_equal(first.delta, second.delta)
    assert first.gripper_open == second.gripper_open
    assert [make_recovery_spec(0, variant).kind for variant in range(4)] == [
        PerturbationKind.ALIGN_OFFSET,
        PerturbationKind.FAILED_CLOSE,
        PerturbationKind.LIFT_OFFSET,
        PerturbationKind.TRANSPORT_OFFSET,
    ]


@pytest.mark.parametrize("variant_id", range(4))
def test_recovery_spec_magnitudes_stay_within_family_bounds(variant_id: int) -> None:
    spec = make_recovery_spec(source_seed=0, variant_id=variant_id)
    lateral = float(np.linalg.norm(spec.delta[:2]))

    if spec.kind is PerturbationKind.ALIGN_OFFSET:
        assert 0.04 <= lateral <= 0.06
        assert 0.02 <= spec.delta[2] <= 0.04
        assert spec.gripper_open == pytest.approx(1.0)
        assert spec.injection_phase is ExpertPhase.ALIGN
    elif spec.kind is PerturbationKind.FAILED_CLOSE:
        assert 0.075 <= lateral <= 0.09
        assert spec.delta[2] == pytest.approx(0.0)
        assert spec.gripper_open == pytest.approx(0.0)
        assert spec.injection_phase is ExpertPhase.CLOSE
    elif spec.kind is PerturbationKind.LIFT_OFFSET:
        assert 0.03 <= lateral <= 0.05
        assert spec.delta[2] == pytest.approx(-0.02)
        assert spec.gripper_open is None
        assert spec.injection_phase is ExpertPhase.LIFT
    else:
        assert 0.05 <= lateral <= 0.07
        assert spec.delta[2] == pytest.approx(0.0)
        assert spec.gripper_open is None
        assert spec.injection_phase is ExpertPhase.TRANSPORT


def test_failed_close_does_not_create_a_grasp() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=15, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    )

    perturbed = apply_recovery_spec(env, make_recovery_spec(0, 1))

    assert perturbed.held_object is None
    assert perturbed.gripper.gripper_open == pytest.approx(0.0)


@pytest.mark.parametrize("variant_id", (2, 3))
def test_held_recovery_offsets_preserve_attachment(variant_id: int) -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=16, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    )
    grasped = env.step(np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32))
    assert grasped.snapshot.held_object == snapshot.target_object.name

    perturbed = apply_recovery_spec(env, make_recovery_spec(0, variant_id))

    assert perturbed.held_object == snapshot.target_object.name
    np.testing.assert_allclose(
        perturbed.target_object.position,
        perturbed.gripper.position + env.hold_offset,
    )
