from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .physics_env import FrankaContactEnv


VIEW_TO_CAMERA = {
    "agent": "agentview",
    "wrist": "wristview",
    "side": "sideview",
    "top": "topview",
}
VIEW_LABELS = {
    "agent": "Agent View",
    "wrist": "Wrist / Egocentric View",
    "side": "Side View",
    "top": "Top View",
}


def metric_depth_from_buffer(
    depth_buffer: np.ndarray, *, near: float, far: float
) -> np.ndarray:
    values = np.asarray(depth_buffer, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("depth buffer values must be finite and inside [0, 1]")
    if not np.isfinite(near) or not np.isfinite(far) or not 0.0 < near < far:
        raise ValueError("depth clipping planes must satisfy 0 < near < far")
    metric = near / (1.0 - values * (1.0 - near / far))
    return metric.astype(np.float32)


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB frame must be HxWx3 uint8")
        if depth.shape != rgb.shape[:2] or depth.dtype != np.float32:
            raise ValueError("depth frame must match RGB height/width and use float32")
        if not np.isfinite(depth).all():
            raise ValueError("depth frame must be finite")
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "depth", depth.copy())


@dataclass(frozen=True)
class MultiViewFrame:
    policy_step: int
    simulation_time: float
    state_hash: str
    views: dict[str, RGBDFrame]

    def __post_init__(self) -> None:
        if self.policy_step < 0 or not np.isfinite(self.simulation_time):
            raise ValueError("frame timestamp must be finite and non-negative")
        if not self.state_hash:
            raise ValueError("state_hash must not be empty")
        if tuple(self.views) != ("agent", "wrist", "side", "top"):
            raise ValueError("views must be ordered as agent, wrist, side, top")
        shapes = {view.rgb.shape for view in self.views.values()}
        if len(shapes) != 1:
            raise ValueError("all synchronized camera frames must have the same shape")


class MultiViewRecorder:
    def __init__(self, model: mujoco.MjModel, *, width: int = 256, height: int = 256) -> None:
        if width < 1 or height < 1:
            raise ValueError("recording width and height must be positive")
        missing = [name for name in VIEW_TO_CAMERA.values() if model.camera(name).id < 0]
        if missing:
            raise ValueError(f"MuJoCo model is missing cameras: {missing}")
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._renderer: mujoco.Renderer | None = None

    def capture(self, env: FrankaContactEnv) -> MultiViewFrame:
        if env.model is not self.model:
            raise ValueError("recorder and environment must share the same MjModel")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, width=self.width, height=self.height
            )
        views: dict[str, RGBDFrame] = {}
        for view_name, camera_name in VIEW_TO_CAMERA.items():
            self._renderer.disable_depth_rendering()
            self._renderer.update_scene(env.data, camera=camera_name)
            rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(env.data, camera=camera_name)
            depth = np.asarray(self._renderer.render(), dtype=np.float32).copy()
            views[view_name] = RGBDFrame(rgb=rgb, depth=depth)
        self._renderer.disable_depth_rendering()
        return MultiViewFrame(
            policy_step=env.step_count,
            simulation_time=float(env.data.time),
            state_hash=self.state_hash(env),
            views=views,
        )

    @staticmethod
    def save_episode(frames: Iterable[MultiViewFrame], path: str | Path) -> Path:
        values = tuple(frames)
        if not values:
            raise ValueError("cannot save an RGB-D episode with no frames")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "policy_step": np.asarray(
                [frame.policy_step for frame in values], dtype=np.int32
            ),
            "simulation_time": np.asarray(
                [frame.simulation_time for frame in values], dtype=np.float64
            ),
            "state_hash": np.asarray([frame.state_hash for frame in values]),
        }
        for name in VIEW_TO_CAMERA:
            arrays[f"observation_{name}_rgb"] = np.stack(
                [frame.views[name].rgb for frame in values]
            ).astype(np.uint8, copy=False)
            arrays[f"observation_{name}_depth"] = np.stack(
                [frame.views[name].depth for frame in values]
            ).astype(np.float32, copy=False)
        np.savez_compressed(destination, **arrays)
        return destination

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @staticmethod
    def state_hash(env: FrankaContactEnv) -> str:
        digest = hashlib.sha256()
        for array in (env.data.qpos, env.data.qvel, env.data.ctrl):
            digest.update(np.asarray(array, dtype=np.float64).tobytes())
        digest.update(np.asarray((env.data.time, env.step_count), dtype=np.float64).tobytes())
        return digest.hexdigest()


def compose_dashboard_frame(
    frame: MultiViewFrame, *, overlay: str = ""
) -> Image.Image:
    height, width, _ = frame.views["agent"].rgb.shape
    panel_height = height + 24
    dashboard = Image.new("RGB", (2 * width, 2 * panel_height), color="white")
    positions = {
        "agent": (0, 0),
        "wrist": (width, 0),
        "side": (0, panel_height),
        "top": (width, panel_height),
    }
    for name, position in positions.items():
        x, y = position
        dashboard.paste(Image.fromarray(frame.views[name].rgb, mode="RGB"), (x, y + 24))
        label_bar = Image.new("RGB", (width, 24), color="white")
        label_draw = ImageDraw.Draw(label_bar)
        label_draw.text((5, 1), VIEW_LABELS[name], fill="black")
        if name == "agent" and overlay:
            label_draw.text((5, 12), overlay, fill="black")
        dashboard.paste(label_bar, (x, y))
    return dashboard
