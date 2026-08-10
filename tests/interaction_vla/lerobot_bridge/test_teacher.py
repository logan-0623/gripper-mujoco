import inspect
from dataclasses import replace

import mujoco
import numpy as np

from interaction_vla.config import PhysicsConfig
from interaction_vla.lerobot_bridge.capture import (
    CameraFrame,
    DualViewFrame,
    camera_calibration,
)
from interaction_vla.lerobot_bridge.teacher import (
    TCTIGTeacherExtractor,
    transform_snapshot_passive,
)
from interaction_vla.physics_env import FrankaContactEnv


def test_extractor_uses_task_roles_and_sparse_relation_slots() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    extractor = TCTIGTeacherExtractor.from_defaults()

    frame = extractor.extract_geometry(
        snapshot, frame_index=0, timestamp=0.0, state_hash="state"
    )

    assert frame.entity_mask.tolist() == [True, True, True, True, True, True]
    assert frame.relation_mask.tolist() == [True] * 8
    assert np.isfinite(frame.relation_values).all()


def test_extractor_source_never_reads_privileged_interaction_state() -> None:
    source = inspect.getsource(TCTIGTeacherExtractor).lower()
    forbidden = (
        "contact_diagnostics",
        "grasp_state",
        "held_object",
        "stable_grasp",
        "normal_force",
        "expert_phase",
        "termination_reason",
    )
    assert all(fragment not in source for fragment in forbidden)


def test_local_relations_are_invariant_to_passive_translation_and_yaw() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    transformed = transform_snapshot_passive(
        snapshot,
        translation=np.asarray((1.2, -0.7, 0.3)),
        yaw_radians=0.8,
    )

    first = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        snapshot, frame_index=0, timestamp=0.0, state_hash="first"
    )
    second = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        transformed, frame_index=0, timestamp=0.0, state_hash="second"
    )

    np.testing.assert_allclose(
        first.relation_values[first.relation_mask],
        second.relation_values[second.relation_mask],
        atol=1e-5,
    )


def test_distractor_slots_do_not_depend_on_input_object_order() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    reversed_snapshot = replace(snapshot, objects=tuple(reversed(snapshot.objects)))

    first = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        snapshot, frame_index=0, timestamp=0.0, state_hash="first"
    )
    second = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        reversed_snapshot, frame_index=0, timestamp=0.0, state_hash="second"
    )

    np.testing.assert_allclose(first.entity_pose, second.entity_pose, atol=1e-6)
    np.testing.assert_allclose(first.relation_values, second.relation_values, atol=1e-6)


def test_raw_geom_pairs_map_to_canonical_task_instances() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    target_body_id = env.model.body(snapshot.target_object.name).id
    target_geom_id = int(np.flatnonzero(env.model.geom_bodyid == target_body_id)[0])
    raw = np.full((8, 8, 2), -1, dtype=np.int32)
    raw[0, 0] = (target_geom_id, int(mujoco.mjtObj.mjOBJ_GEOM))
    camera = CameraFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        segmentation=raw,
    )
    capture = DualViewFrame(
        policy_step=0,
        timestamp=0.0,
        state_hash="state",
        views={"agent": camera, "wrist": camera},
    )
    extractor = TCTIGTeacherExtractor.from_defaults(
        model=env.model,
        calibration=camera_calibration(env, width=8, height=8),
    )

    frame = extractor.extract(snapshot, capture, state=np.zeros(10, dtype=np.float32))

    assert frame.instance_agent[0, 0] == 2
    assert frame.instance_wrist[0, 0] == 2
    assert frame.entity_visibility[1].tolist() == [1 / 64, 1 / 64]
