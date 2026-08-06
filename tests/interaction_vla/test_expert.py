from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.env import KinematicTabletopEnv, TerminationReason
from interaction_vla.expert import ExpertPhase, ScriptedExpert


def test_scripted_expert_completes_pick_and_place() -> None:
    env = KinematicTabletopEnv(max_objects=5, max_steps=120)
    snapshot = env.reset(seed=9, object_count=3)
    expert = ScriptedExpert()
    visited = {expert.phase}

    result = None
    for _ in range(120):
        action = expert.act(snapshot)
        result = env.step(action)
        snapshot = result.snapshot
        visited.add(expert.phase)
        if result.done:
            break

    assert result is not None
    assert result.reason is TerminationReason.SUCCESS
    assert ExpertPhase.CLOSE in visited
    assert ExpertPhase.TRANSPORT in visited
    assert ExpertPhase.RELEASE in visited
    assert ExpertPhase.RETREAT in visited


@pytest.mark.parametrize(
    ("object_count", "seed"),
    ((2, 11), (3, 23), (4, 37), (5, 71)),
)
def test_scripted_expert_solves_crowded_layouts(object_count: int, seed: int) -> None:
    env = KinematicTabletopEnv(max_objects=5, max_steps=120)
    snapshot = env.reset(seed=seed, object_count=object_count, layout_mode="crowded")
    expert = ScriptedExpert()

    result = None
    for _ in range(120):
        result = env.step(expert.act(snapshot))
        snapshot = result.snapshot
        if result.done:
            break

    assert result is not None
    assert result.reason is TerminationReason.SUCCESS


def test_closed_empty_gripper_reopens_and_realigns() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=12, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.04, 0.0, 0.055), dtype=np.float32)
    )
    closed_unheld_offset_snapshot = env.perturb_gripper_state(
        np.zeros(3, dtype=np.float32), gripper_open=0.0
    )
    expert = ScriptedExpert()
    expert.phase = ExpertPhase.CLOSE

    action = expert.act(closed_unheld_offset_snapshot)

    assert action[3] == pytest.approx(1.0)
    assert expert.phase is ExpertPhase.ALIGN


def test_held_target_below_lift_height_returns_to_lift() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=13, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    )
    grasped = env.step(np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32))
    assert grasped.snapshot.held_object == snapshot.target_object.name
    perturbed_held_snapshot = env.perturb_gripper_state(
        np.asarray((0.02, -0.02, -0.02), dtype=np.float32)
    )
    expert = ScriptedExpert()
    expert.phase = ExpertPhase.TRANSPORT

    action = expert.act(perturbed_held_snapshot)

    assert action[2] > 0
    assert action[3] == pytest.approx(0.0)
    assert expert.phase is ExpertPhase.LIFT


def test_held_target_away_from_receptacle_cannot_release() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=14, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    )
    grasped = env.step(np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32))
    assert grasped.snapshot.held_object == snapshot.target_object.name
    raised = env.perturb_gripper_state(np.asarray((0.0, 0.0, 0.17), dtype=np.float32))
    expert = ScriptedExpert()
    expert.phase = ExpertPhase.RELEASE

    action = expert.act(raised)

    assert expert.phase is ExpertPhase.TRANSPORT
    assert action[3] == pytest.approx(0.0)
