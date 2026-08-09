import os

import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig
from interaction_vla.lerobot_bridge.capture import (
    CameraFrame,
    DualViewCapture,
    camera_calibration,
)
from interaction_vla.physics_env import FrankaContactEnv


def test_camera_frame_requires_rgb_metric_depth_and_raw_segmentation() -> None:
    frame = CameraFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        segmentation=np.full((8, 8, 2), -1, dtype=np.int32),
    )

    assert frame.segmentation.shape == (8, 8, 2)
    with pytest.raises(ValueError, match="segmentation"):
        CameraFrame(frame.rgb, frame.depth, np.zeros((8, 8), dtype=np.int32))


def test_wrist_calibration_recomposes_at_two_end_effector_poses() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    env.reset(seed=11, object_count=2)

    first = camera_calibration(env, width=64, height=64)
    env.step(np.asarray((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32))
    second = camera_calibration(env, width=64, height=64)

    assert tuple(first) == ("agent", "wrist")
    assert first["agent"]["intrinsics"].shape == (3, 3)
    assert first["wrist"]["camera_to_rig"].shape == (4, 4)
    assert not np.array_equal(
        first["wrist"]["camera_to_base"], second["wrist"]["camera_to_base"]
    )


@pytest.mark.skipif(
    bool(os.environ.get("CODEX_SANDBOX")),
    reason="macOS CoreGraphics is unavailable inside the Codex sandbox",
)
def test_capture_has_two_views_and_does_not_advance_mujoco() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    env.reset(seed=11, object_count=2)
    qpos_before = env.data.qpos.copy()
    qvel_before = env.data.qvel.copy()
    ctrl_before = env.data.ctrl.copy()
    time_before = float(env.data.time)
    capture = DualViewCapture(env.model, width=64, height=64)
    try:
        frame = capture.capture(env, include_teacher=True)
    finally:
        capture.close()

    assert tuple(frame.views) == ("agent", "wrist")
    assert frame.policy_step == 0
    assert frame.timestamp == pytest.approx(0.0)
    for view in frame.views.values():
        assert view.rgb.shape == (64, 64, 3)
        assert view.depth.shape == (64, 64)
        assert view.segmentation.shape == (64, 64, 2)
        assert np.all(view.depth > 0.0)
    np.testing.assert_array_equal(env.data.qpos, qpos_before)
    np.testing.assert_array_equal(env.data.qvel, qvel_before)
    np.testing.assert_array_equal(env.data.ctrl, ctrl_before)
    assert env.data.time == time_before
