from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig, RandomizationConfig
from interaction_vla.contact_physics import NonFiniteContactForceError
from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.physics_env import FrankaContactEnv


WORKSPACE_LOW = (0.25, -0.35, 0.23)
WORKSPACE_HIGH = (0.78, 0.35, 0.75)


def make_env(*, physics: PhysicsConfig | None = None, max_steps: int = 20) -> FrankaContactEnv:
    return FrankaContactEnv(
        max_objects=5,
        max_steps=max_steps,
        min_object_distance=0.12,
        workspace_low=WORKSPACE_LOW,
        workspace_high=WORKSPACE_HIGH,
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
        physics=physics or PhysicsConfig(settle_steps=50),
    )


def object_positions(env: FrankaContactEnv) -> np.ndarray:
    return np.stack([entity.position for entity in env.snapshot().objects])


def set_free_object_pose(
    env: FrankaContactEnv,
    name: str,
    *,
    position: tuple[float, float, float],
) -> None:
    joint = env.model.joint(f"{name}_joint")
    qpos_address = int(env.model.jnt_qposadr[joint.id])
    dof_address = int(env.model.jnt_dofadr[joint.id])
    env.data.qpos[qpos_address : qpos_address + 3] = position
    env.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
    env.data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.contact_diagnostics = env.contact_parser.parse(env.data)


def test_reset_is_seed_deterministic_for_layout_target_and_physics() -> None:
    randomization = RandomizationConfig(
        enabled=True,
        object_mass_scale=(0.8, 1.2),
        friction_scale=(0.8, 1.2),
        joint_damping_scale=(0.9, 1.1),
    )
    physics = replace(PhysicsConfig(settle_steps=25), randomization=randomization)
    first = make_env(physics=physics)
    second = make_env(physics=physics)

    first_snapshot = first.reset(seed=71, object_count=4, layout_mode=LayoutMode.CROWDED)
    second_snapshot = second.reset(seed=71, object_count=4, layout_mode=LayoutMode.CROWDED)

    assert first_snapshot.target_object.name == second_snapshot.target_object.name
    np.testing.assert_array_equal(object_positions(first), object_positions(second))
    np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
    np.testing.assert_array_equal(first.data.qvel, second.data.qvel)
    assert first.physics_metadata() == second.physics_metadata()
    metadata = first.physics_metadata()
    assert all(0.8 <= value <= 1.2 for value in metadata["object_mass_scales"])
    assert 0.8 <= metadata["friction_scale"] <= 1.2
    assert 0.9 <= metadata["joint_damping_scale"] <= 1.1


def test_different_seeds_change_the_physical_layout() -> None:
    first = make_env()
    second = make_env()
    first.reset(seed=11, object_count=3)
    second.reset(seed=12, object_count=3)

    assert not np.array_equal(object_positions(first), object_positions(second))


def test_one_policy_action_executes_exactly_25_physics_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)
    real_step = mujoco.mj_step
    calls = 0

    def counted_step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        nonlocal calls
        calls += 1
        real_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    result = env.step(np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))

    assert calls == 25
    assert result.info["physics_substeps"] == 25
    assert env.step_count == 1


def test_unlabelled_intervention_advances_physics_without_policy_step() -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)
    start_time = float(env.data.time)

    result = env.advance_intervention(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        substeps=4,
    )

    assert env.step_count == 0
    assert float(env.data.time) == pytest.approx(
        start_time + 4 * env.physics.timestep
    )
    assert result.snapshot.gripper.name == "gripper"
    assert result.controller_diagnostics is not None
    assert result.physics_failure is None


def test_unlabelled_intervention_reports_existing_physics_failure() -> None:
    env = make_env()
    before = env.reset(seed=11, object_count=2)
    env.data.qpos[0] = np.nan

    result = env.advance_intervention(np.zeros(7), substeps=4)

    assert result.snapshot is before
    assert result.controller_diagnostics is None
    assert result.physics_failure == "non_finite_state"


@pytest.mark.parametrize("substeps", [0, 26])
def test_unlabelled_intervention_rejects_invalid_substep_count(
    substeps: int,
) -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)

    with pytest.raises(ValueError, match="substeps"):
        env.advance_intervention(np.zeros(7), substeps=substeps)


def test_physics_env_exposes_23d_proprioception_and_rejects_4d_action() -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)

    proprioception = env.proprioception()
    assert proprioception.shape == (23,)
    assert np.isfinite(proprioception).all()
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        env.step(np.zeros(4, dtype=np.float32))


def test_wrist_camera_is_synchronized_and_looks_along_gripper_approach_axis() -> None:
    env = make_env(physics=PhysicsConfig(settle_steps=5))
    env.reset(seed=2_140_049, object_count=4, layout_mode=LayoutMode.CROWDED)

    camera_id = env.model.camera("wristview").id
    camera_rotation = np.asarray(env.data.cam_xmat[camera_id]).reshape(3, 3)
    tcp_position, hand_rotation = env.controller.tcp_pose()

    np.testing.assert_allclose(env.data.cam_xpos[camera_id], tcp_position, atol=1e-10)
    np.testing.assert_allclose(-camera_rotation[:, 2], hand_rotation[:, 2], atol=1e-10)


def test_non_finite_state_terminates_as_physics_failure() -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)
    env.data.qpos[0] = np.nan

    result = env.step(np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))

    assert result.done
    assert result.reason is TerminationReason.PHYSICS_FAILURE
    assert result.info["physics_failure"] == "non_finite_state"


def test_non_finite_contact_force_terminates_as_physics_failure(monkeypatch) -> None:
    env = make_env()
    before = env.reset(seed=11, object_count=2)

    def invalid_force(*args, **kwargs):
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")

    monkeypatch.setattr(env.contact_parser, "parse", invalid_force)
    result = env.step(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    )

    assert result.done
    assert result.reason is TerminationReason.PHYSICS_FAILURE
    assert result.info["physics_failure"] == "non_finite_contact_force"
    assert result.snapshot is before


def test_non_finite_contact_force_stops_intervention(monkeypatch) -> None:
    env = make_env()
    before = env.reset(seed=11, object_count=2)

    def invalid_force(*args, **kwargs):
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")

    monkeypatch.setattr(env.contact_parser, "parse", invalid_force)
    result = env.advance_intervention(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        substeps=1,
    )

    assert result.snapshot is before
    assert result.physics_failure == "non_finite_contact_force"


def test_severe_penetration_is_a_physics_failure_with_current_failure_state() -> None:
    env = make_env()
    before = env.reset(seed=11, object_count=2)
    address = env.model.jnt_qposadr[env.model.joint("object_0_joint").id]
    env.data.qpos[address + 2] = env.table_top - 0.02
    mujoco.mj_forward(env.model, env.data)

    result = env.step(np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))

    assert result.done
    assert result.reason is TerminationReason.PHYSICS_FAILURE
    assert result.info["physics_failure"] == "severe_penetration"
    assert result.snapshot.objects[0].position[2] != before.objects[0].position[2]
    assert "failure_qpos" in result.info and "failure_qvel" in result.info


def test_reset_objects_settle_on_the_table_without_scripted_attachment() -> None:
    env = make_env(physics=PhysicsConfig(settle_steps=250))
    snapshot = env.reset(seed=31, object_count=3)

    for entity in snapshot.objects:
        assert entity.position[2] == pytest.approx(env.table_top + 0.022, abs=2e-3)
        assert entity.name in env.contact_diagnostics.object_table
    assert snapshot.held_object is None


def test_exterior_wall_contact_never_accumulates_strict_placement() -> None:
    env = make_env(physics=PhysicsConfig(settle_steps=5))
    env.reset(seed=11, object_count=2, target_index=0)
    set_free_object_pose(env, "object_0", position=(0.584, -0.12, 0.273))

    for _ in range(12):
        env._update_placement_frames()

    assert env.contact_diagnostics.object_receptacle_wall == frozenset(("object_0",))
    assert env.last_placement.fully_contained is False
    assert env.last_placement.strict_stable is False
    assert env._placement_frames == 0


def test_base_contact_full_containment_can_reach_strict_success() -> None:
    env = make_env(physics=PhysicsConfig(settle_steps=5))
    env.reset(seed=11, object_count=2, target_index=0)
    set_free_object_pose(env, "object_0", position=(0.67, -0.12, 0.2725))

    for _ in range(10):
        env._update_placement_frames()

    assert env.last_placement.fully_contained
    assert env.last_placement.base_contact
    assert env.last_placement.strict_stable
    assert env._is_success()
