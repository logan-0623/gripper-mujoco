from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_recording import (
    MultiViewFrame,
    MultiViewRecorder,
    RGBDFrame,
    compose_dashboard_frame,
    metric_depth_from_buffer,
)


def synthetic_frame(step: int, *, height: int = 6, width: int = 8) -> MultiViewFrame:
    colors = {
        "agent": (255, 0, 0),
        "wrist": (0, 255, 0),
        "side": (0, 0, 255),
        "top": (255, 255, 0),
    }
    views = {
        name: RGBDFrame(
            rgb=np.broadcast_to(np.asarray(color, dtype=np.uint8), (height, width, 3)).copy(),
            depth=np.full((height, width), 0.5 + step, dtype=np.float32),
        )
        for name, color in colors.items()
    }
    return MultiViewFrame(
        policy_step=step,
        simulation_time=step / 20.0,
        state_hash=f"state-{step}",
        views=views,
    )


def test_raw_depth_conversion_returns_metric_distance() -> None:
    raw = np.asarray((0.0, 0.5, 1.0), dtype=np.float32)

    metric = metric_depth_from_buffer(raw, near=0.1, far=10.0)

    np.testing.assert_allclose(metric, (0.1, 0.1980198, 10.0), rtol=1e-5)
    assert metric.dtype == np.float32


def test_rgbd_episode_round_trip_uses_four_synchronized_view_keys(tmp_path) -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    recorder = MultiViewRecorder(env.model, width=8, height=6)
    destination = tmp_path / "rgbd_episode.npz"

    recorder.save_episode((synthetic_frame(0), synthetic_frame(1)), destination)

    with np.load(destination, allow_pickle=False) as archive:
        for name in ("agent", "wrist", "side", "top"):
            assert archive[f"observation_{name}_rgb"].shape == (2, 6, 8, 3)
            assert archive[f"observation_{name}_rgb"].dtype == np.uint8
            assert archive[f"observation_{name}_depth"].shape == (2, 6, 8)
            assert archive[f"observation_{name}_depth"].dtype == np.float32
        np.testing.assert_array_equal(archive["policy_step"], (0, 1))
        np.testing.assert_allclose(archive["simulation_time"], (0.0, 0.05))
        np.testing.assert_array_equal(archive["state_hash"], ("state-0", "state-1"))


def test_dashboard_places_agent_wrist_side_top_in_the_documented_order() -> None:
    frame = synthetic_frame(0)

    dashboard = np.asarray(compose_dashboard_frame(frame, overlay="expert · running"))

    height, width = frame.views["agent"].rgb.shape[:2]
    assert dashboard.shape == (2 * (height + 24), 2 * width, 3)
    np.testing.assert_array_equal(dashboard[24, 0], (255, 0, 0))
    np.testing.assert_array_equal(dashboard[24, width], (0, 255, 0))
    np.testing.assert_array_equal(dashboard[height + 48, 0], (0, 0, 255))
    np.testing.assert_array_equal(dashboard[height + 48, width], (255, 255, 0))


@pytest.mark.skipif(
    bool(os.environ.get("CODEX_SANDBOX")),
    reason="macOS CoreGraphics is unavailable inside the Codex sandbox",
)
def test_capture_returns_metric_rgbd_without_advancing_physics() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=25))
    env.reset(seed=11, object_count=2)
    qpos_before = env.data.qpos.copy()
    qvel_before = env.data.qvel.copy()
    time_before = env.data.time
    recorder = MultiViewRecorder(env.model, width=64, height=64)
    try:
        frame = recorder.capture(env)
    finally:
        recorder.close()

    assert set(frame.views) == {"agent", "wrist", "side", "top"}
    for view in frame.views.values():
        assert view.rgb.shape == (64, 64, 3)
        assert view.rgb.dtype == np.uint8
        assert view.depth.shape == (64, 64)
        assert view.depth.dtype == np.float32
        assert np.isfinite(view.depth).all()
        assert np.all(view.depth > 0.0)
    np.testing.assert_array_equal(env.data.qpos, qpos_before)
    np.testing.assert_array_equal(env.data.qvel, qvel_before)
    assert env.data.time == time_before
