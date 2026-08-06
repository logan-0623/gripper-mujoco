from __future__ import annotations

import mujoco
import numpy as np

from interaction_vla.contact_physics import (
    ContactDiagnostics,
    ContactParser,
    StableGraspTracker,
)
from interaction_vla.franka import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    FRANKA_SCENE_PATH,
    HOME_QPOS,
)
from interaction_vla.graph.schema import InteractionSignal


def empty_contacts(
    *,
    left: frozenset[str] = frozenset(),
    right: frozenset[str] = frozenset(),
    table: frozenset[str] = frozenset(),
    receptacle: frozenset[str] = frozenset(),
) -> ContactDiagnostics:
    return ContactDiagnostics(
        left_objects=left,
        right_objects=right,
        object_table=table,
        object_receptacle=receptacle,
        interactions=(),
    )


def test_interaction_signal_rejects_negative_or_non_finite_force() -> None:
    for value in (-1.0, np.nan):
        try:
            InteractionSignal("gripper", "object_0", normal_force=value)
        except ValueError as error:
            assert "force" in str(error)
        else:
            raise AssertionError("invalid force was accepted")


def test_stable_grasp_requires_bilateral_lift_for_ten_physics_frames() -> None:
    tracker = StableGraspTracker(required_frames=10, lift_height=0.01)
    tracker.reset(target_name="object_0", table_top=0.225)
    bilateral = empty_contacts(
        left=frozenset(("object_0",)),
        right=frozenset(("object_0",)),
    )

    on_table = tracker.update(
        bilateral,
        object_bottom_heights={"object_0": 0.225},
    )
    assert on_table.bilateral_object == "object_0"
    assert on_table.stable_object is None
    assert on_table.stable_frames == 0

    for _ in range(9):
        state = tracker.update(
            bilateral,
            object_bottom_heights={"object_0": 0.236},
        )
        assert state.stable_object is None
    state = tracker.update(
        bilateral,
        object_bottom_heights={"object_0": 0.236},
    )
    assert state.stable_object == "object_0"
    assert state.stable_frames == 10
    assert state.ever_stable_target


def test_lost_bilateral_target_contact_on_table_reports_drop() -> None:
    tracker = StableGraspTracker(required_frames=1, lift_height=0.01)
    tracker.reset(target_name="object_0", table_top=0.225)
    bilateral = empty_contacts(
        left=frozenset(("object_0",)),
        right=frozenset(("object_0",)),
    )
    stable = tracker.update(
        bilateral,
        object_bottom_heights={"object_0": 0.236},
    )
    assert stable.stable_object == "object_0"

    dropped = tracker.update(
        empty_contacts(table=frozenset(("object_0",))),
        object_bottom_heights={"object_0": 0.225},
    )
    assert dropped.bilateral_object is None
    assert dropped.stable_object is None
    assert dropped.dropped_target


def test_wrong_object_stable_event_is_latched_across_later_contact_loss() -> None:
    tracker = StableGraspTracker(required_frames=2, lift_height=0.01)
    tracker.reset(target_name="object_0", table_top=0.225)
    simultaneous = empty_contacts(
        left=frozenset(("object_0", "object_1")),
        right=frozenset(("object_0", "object_1")),
    )
    heights = {"object_0": 0.236, "object_1": 0.236}

    tracker.update(simultaneous, object_bottom_heights=heights)
    stable = tracker.update(simultaneous, object_bottom_heights=heights)
    lost = tracker.update(empty_contacts(), object_bottom_heights=heights)

    assert stable.stable_object == "object_0"
    assert stable.wrong_stable_object == "object_1"
    assert stable.ever_stable_wrong_object
    assert stable.ever_bilateral_contact
    assert lost.stable_object is None
    assert lost.wrong_stable_object is None
    assert lost.ever_stable_wrong_object
    assert lost.ever_bilateral_contact


def test_tracker_separates_target_and_distractor_bilateral_contact() -> None:
    tracker = StableGraspTracker(required_frames=2, lift_height=0.01)
    tracker.reset(target_name="object_0", table_top=0.225)
    wrong = empty_contacts(
        left=frozenset(("object_1",)),
        right=frozenset(("object_1",)),
    )
    target = empty_contacts(
        left=frozenset(("object_0",)),
        right=frozenset(("object_0",)),
    )

    first = tracker.update(
        wrong,
        object_bottom_heights={"object_0": 0.236, "object_1": 0.236},
    )
    second = tracker.update(
        target,
        object_bottom_heights={"object_0": 0.236, "object_1": 0.236},
    )
    stable = tracker.update(
        target,
        object_bottom_heights={"object_0": 0.236, "object_1": 0.236},
    )

    assert first.ever_bilateral_contact
    assert first.first_bilateral_object == "object_1"
    assert first.ever_bilateral_wrong_object
    assert not first.ever_bilateral_target_contact
    assert first.total_bilateral_wrong_substeps == 1
    assert first.total_bilateral_target_substeps == 0
    assert second.ever_bilateral_target_contact
    assert second.first_bilateral_target_substep == 2
    assert stable.first_stable_target_substep == 3
    assert stable.total_stable_target_substeps == 1
    assert stable.longest_stable_target_run == 1
    assert stable.total_bilateral_target_substeps == 2
    events = tracker.interaction_events_since(0)
    assert events[0].substep == 1
    assert events[0].bilateral_objects == ("object_1",)
    assert events[1].substep == 2
    assert events[1].bilateral_objects == ("object_0",)
    assert events[2].stable_objects == ("object_0",)


def test_contact_parser_reads_non_negative_mujoco_contact_force() -> None:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    data = mujoco.MjData(model)
    for value, name in zip(HOME_QPOS, ARM_JOINT_NAMES, strict=True):
        address = model.jnt_qposadr[model.joint(name).id]
        data.qpos[address] = value
        data.ctrl[model.actuator(f"actuator{int(name.removeprefix('joint'))}").id] = value
    for name in FINGER_JOINT_NAMES:
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.04
    data.ctrl[model.actuator("actuator8").id] = 255.0
    for index in range(5):
        address = model.jnt_qposadr[model.joint(f"object_{index}_joint").id]
        data.qpos[address : address + 3] = (
            (0.45, 0.10, 0.247) if index == 0 else (0.0, 0.0, -2.0 - index)
        )
        data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    for _ in range(25):
        mujoco.mj_step(model, data)

    diagnostics = ContactParser(model).parse(data)

    assert "object_0" in diagnostics.object_table
    signal = next(
        signal
        for signal in diagnostics.interactions
        if signal.key == frozenset(("object_0", "table"))
    )
    assert signal.contact
    assert np.isfinite(signal.normal_force)
    assert np.isfinite(signal.tangential_force)
    assert signal.normal_force >= 0.0
    assert signal.tangential_force >= 0.0


def test_contact_parser_distinguishes_receptacle_base_from_wall() -> None:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    data = mujoco.MjData(model)
    for index in range(5):
        address = model.jnt_qposadr[model.joint(f"object_{index}_joint").id]
        if index == 0:
            position = (0.67, -0.12, 0.2725)
        elif index == 1:
            position = (0.584, -0.12, 0.273)
        else:
            position = (0.0, 0.0, -2.0 - index)
        data.qpos[address : address + 3] = position
        data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    diagnostics = ContactParser(model).parse(data)

    assert "object_0" in diagnostics.object_receptacle_base
    assert "object_0" not in diagnostics.object_receptacle_wall
    assert "object_1" in diagnostics.object_receptacle_wall
    assert diagnostics.object_receptacle == frozenset(("object_0", "object_1"))
