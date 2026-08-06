from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from interaction_vla.mujoco_env import MujocoTabletopEnv


def test_mujoco_adapter_reset_is_deterministic_and_headless() -> None:
    first_env = MujocoTabletopEnv(max_objects=5)
    second_env = MujocoTabletopEnv(max_objects=5)

    first = first_env.reset(seed=23, object_count=4)
    second = second_env.reset(seed=23, object_count=4)

    np.testing.assert_allclose(
        np.stack([entity.position for entity in first.objects]),
        np.stack([entity.position for entity in second.objects]),
    )
    assert not hasattr(first_env, "viewer")
    assert first_env.model.nmocap == 1


def test_mujoco_adapter_mirrors_step_into_mujoco_data() -> None:
    env = MujocoTabletopEnv(max_objects=5)
    env.reset(seed=8, object_count=2)

    result = env.step(np.asarray((0.02, -0.01, 0.0, 1.0), dtype=np.float32))

    np.testing.assert_allclose(env.data.mocap_pos[0], result.snapshot.gripper.position, atol=1e-6)
    object_body = env.data.body("object_0")
    np.testing.assert_allclose(object_body.xpos, result.snapshot.objects[0].position, atol=1e-6)


def test_mujoco_adapter_forwards_crowded_layout() -> None:
    env = MujocoTabletopEnv(max_objects=5)

    snapshot = env.reset(seed=71, object_count=5, layout_mode="crowded")

    target = snapshot.target_object
    nearest = min(
        np.linalg.norm(entity.position[:2] - target.position[:2])
        for entity in snapshot.objects
        if entity.name != target.name
    )
    assert 0.085 <= nearest <= 0.105


def test_mujoco_adapter_syncs_held_object_after_perturbation() -> None:
    env = MujocoTabletopEnv(max_objects=5)
    snapshot = env.reset(seed=4, object_count=2)
    env.set_gripper_position(
        snapshot.target_object.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    )
    grasped = env.step(np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32))
    assert grasped.snapshot.held_object == snapshot.target_object.name

    perturbed = env.perturb_gripper_state(
        np.asarray((0.04, -0.02, -0.02), dtype=np.float32)
    )

    np.testing.assert_allclose(env.data.mocap_pos[0], perturbed.gripper.position, atol=1e-6)
    np.testing.assert_allclose(
        env.data.body(snapshot.target_object.name).xpos,
        perturbed.target_object.position,
        atol=1e-6,
    )


def test_mujoco_framebuffer_supports_documented_gif_resolution() -> None:
    env = MujocoTabletopEnv(max_objects=5)

    assert env.model.vis.global_.offwidth >= 320
    assert env.model.vis.global_.offheight >= 240
