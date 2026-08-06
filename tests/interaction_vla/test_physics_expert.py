from __future__ import annotations

from dataclasses import replace

import numpy as np

from interaction_vla.config import PhysicsConfig
from interaction_vla.contact_physics import ContactDiagnostics, GraspState
from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_expert import PhysicsExpertPhase, PhysicsScriptedExpert


def make_env() -> FrankaContactEnv:
    return FrankaContactEnv(
        max_steps=180,
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
        physics=PhysicsConfig(settle_steps=100),
    )


def empty_contacts(
    *, left: frozenset[str] = frozenset(), right: frozenset[str] = frozenset()
) -> ContactDiagnostics:
    return ContactDiagnostics(
        left_objects=left,
        right_objects=right,
        object_table=frozenset(),
        object_receptacle=frozenset(),
        interactions=(),
    )


def grasp_state(
    *, bilateral: str | None = None, stable: str | None = None
) -> GraspState:
    return GraspState(
        bilateral_object=bilateral,
        stable_object=stable,
        stable_frames=10 if stable else 0,
        ever_stable_target=stable is not None,
        dropped_target=False,
    )


def test_expert_always_returns_a_bounded_7d_command() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)

    action = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)

    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action[:6]) <= 1.0)
    assert action[6] in {0.0, 1.0}


def test_close_transitions_to_lift_on_bilateral_contact_before_stable_grasp() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.CLOSE

    action = expert.act(
        snapshot,
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        grasp_state(bilateral=target),
    )

    assert expert.phase is PhysicsExpertPhase.LIFT
    assert action[6] == 0.0


def test_close_keeps_correcting_tcp_alignment_while_waiting_for_bilateral_contact() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.CLOSE
    shifted_gripper = replace(
        snapshot.gripper,
        position=snapshot.gripper.position + np.asarray((-0.02, 0.0, 0.0)),
    )

    action = expert.act(
        replace(snapshot, gripper=shifted_gripper),
        empty_contacts(),
        grasp_state(),
    )

    assert action[0] > 0.0
    assert action[6] == 0.0


def test_lift_requires_stable_grasp_and_contact_loss_starts_recovery() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.LIFT

    expert.act(snapshot, empty_contacts(), grasp_state())
    assert expert.phase is PhysicsExpertPhase.OPEN_RECOVER

    expert.phase = PhysicsExpertPhase.LIFT
    expert.act(
        snapshot,
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        grasp_state(bilateral=target, stable=target),
    )
    assert expert.phase is PhysicsExpertPhase.TRANSPORT


def test_stable_grasp_remains_in_lift_until_the_tcp_reaches_safe_height() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.LIFT
    low_gripper = replace(
        snapshot.gripper,
        position=np.asarray(
            (snapshot.gripper.position[0], snapshot.gripper.position[1], 0.30),
            dtype=np.float32,
        ),
    )

    action = expert.act(
        replace(snapshot, gripper=low_gripper),
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        grasp_state(bilateral=target, stable=target),
    )

    assert expert.phase is PhysicsExpertPhase.LIFT
    assert action[2] > 0.0


def test_lift_does_not_enter_transport_before_original_safe_height() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.LIFT
    below_entry_height = replace(
        snapshot.gripper,
        position=np.asarray(
            (
                snapshot.gripper.position[0],
                snapshot.gripper.position[1],
                0.340,
            ),
            dtype=np.float32,
        ),
    )

    action = expert.act(
        replace(snapshot, gripper=below_entry_height),
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        grasp_state(bilateral=target, stable=target),
    )

    assert expert.phase is PhysicsExpertPhase.LIFT
    assert action[2] > 0.0
    assert action[6] == 0.0


def test_transport_tolerates_a_transient_stable_counter_reset_with_bilateral_contact() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.TRANSPORT

    action = expert.act(
        snapshot,
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        GraspState(
            bilateral_object=target,
            stable_object=None,
            stable_frames=3,
            ever_stable_target=True,
            dropped_target=False,
        ),
    )

    assert expert.phase is PhysicsExpertPhase.TRANSPORT
    assert action[6] == 0.0


def test_transport_does_not_oscillate_to_lift_near_safe_height() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.TRANSPORT
    near_safe_height = replace(
        snapshot.gripper,
        position=np.asarray(
            (
                snapshot.gripper.position[0],
                snapshot.gripper.position[1],
                0.354,
            ),
            dtype=np.float32,
        ),
    )

    action = expert.act(
        replace(snapshot, gripper=near_safe_height),
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        grasp_state(bilateral=target, stable=target),
    )

    assert expert.phase is PhysicsExpertPhase.TRANSPORT
    assert action[2] > 0.0
    assert action[6] == 0.0


def test_release_away_from_receptacle_resynchronizes_to_closed_transport() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.RELEASE

    action = expert.act(
        snapshot,
        empty_contacts(left=frozenset((target,)), right=frozenset((target,))),
        grasp_state(bilateral=target, stable=target),
    )

    assert expert.phase is PhysicsExpertPhase.TRANSPORT
    assert action[6] == 0.0


def test_scripted_expert_completes_a_real_contact_pick_and_place() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2, layout_mode=LayoutMode.NORMAL)
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    phases: set[PhysicsExpertPhase] = set()

    for _ in range(180):
        phases.add(expert.phase)
        action = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
        transition = env.step(action)
        snapshot = transition.snapshot
        if transition.done:
            break

    assert transition.reason is TerminationReason.SUCCESS
    assert PhysicsExpertPhase.CLOSE in phases
    assert PhysicsExpertPhase.LIFT in phases
    assert PhysicsExpertPhase.TRANSPORT in phases
    assert env.grasp_state.ever_stable_target
