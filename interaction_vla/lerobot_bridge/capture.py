from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_recording import MultiViewRecorder


POLICY_VIEW_TO_CAMERA = {
    "agent": "agentview",
    "wrist": "wristview",
}


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    return values.copy()


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray

    def __post_init__(self) -> None:
        rgb = _validate_rgb(self.rgb)
        depth = np.asarray(self.depth)
        segmentation = np.asarray(self.segmentation)
        if depth.shape != rgb.shape[:2] or depth.dtype != np.float32:
            raise ValueError("depth must match RGB height/width and use float32")
        if not np.isfinite(depth).all() or np.any(depth <= 0.0):
            raise ValueError("depth must contain positive finite metric distances")
        if segmentation.shape != (*rgb.shape[:2], 2) or segmentation.dtype != np.int32:
            raise ValueError("segmentation must be an HxWx2 int32 object-id/type array")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth", depth.copy())
        object.__setattr__(self, "segmentation", segmentation.copy())


@dataclass(frozen=True)
class PolicyCameraFrame:
    rgb: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rgb", _validate_rgb(self.rgb))


@dataclass(frozen=True)
class DualViewFrame:
    policy_step: int
    timestamp: float
    state_hash: str
    views: dict[str, CameraFrame | PolicyCameraFrame]
    calibration: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.policy_step < 0:
            raise ValueError("policy_step must be non-negative")
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        if not self.state_hash:
            raise ValueError("state_hash must not be empty")
        if tuple(self.views) != ("agent", "wrist"):
            raise ValueError("views must be ordered as agent, wrist")
        frame_types = {type(frame) for frame in self.views.values()}
        if len(frame_types) != 1:
            raise ValueError("agent and wrist must expose the same frame type")
        shapes = {frame.rgb.shape for frame in self.views.values()}
        if len(shapes) != 1:
            raise ValueError("agent and wrist RGB frames must have the same shape")
        object.__setattr__(self, "views", dict(self.views))


def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def camera_calibration(
    env: FrankaContactEnv,
    *,
    width: int = 256,
    height: int = 256,
) -> dict[str, object]:
    if width < 1 or height < 1:
        raise ValueError("camera width and height must be positive")
    near = float(env.model.vis.map.znear * env.model.stat.extent)
    far = float(env.model.vis.map.zfar * env.model.stat.extent)
    result: dict[str, object] = {}
    for view_name, camera_name in POLICY_VIEW_TO_CAMERA.items():
        camera_id = int(env.model.camera(camera_name).id)
        if camera_id < 0:
            raise ValueError(f"MuJoCo model is missing camera: {camera_name}")
        fovy = float(env.model.cam_fovy[camera_id])
        focal = 0.5 * height / math.tan(0.5 * math.radians(fovy))
        intrinsics = np.asarray(
            (
                (focal, 0.0, (width - 1) / 2.0),
                (0.0, focal, (height - 1) / 2.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        camera_to_base = _pose_matrix(
            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
        )
        view: dict[str, object] = {
            "camera_name": camera_name,
            "intrinsics": intrinsics,
            "camera_to_base": camera_to_base,
            "znear_m": near,
            "zfar_m": far,
        }
        if view_name == "wrist":
            rig_id = int(env.model.body("wrist_camera_rig").id)
            if int(env.model.cam_bodyid[camera_id]) != rig_id:
                raise ValueError("wristview must be attached to wrist_camera_rig")
            camera_rotation_rig = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(camera_rotation_rig, env.model.cam_quat[camera_id])
            camera_to_rig = _pose_matrix(
                env.model.cam_pos[camera_id], camera_rotation_rig
            )
            rig_to_base = _pose_matrix(
                env.data.xpos[rig_id], env.data.xmat[rig_id]
            )
            recomposed = rig_to_base @ camera_to_rig
            if not np.allclose(recomposed, camera_to_base, rtol=0.0, atol=1e-6):
                raise ValueError("wrist camera runtime pose violates its XML calibration")
            view["camera_to_rig"] = camera_to_rig
        result[view_name] = view
    return result


class DualViewCapture:
    def __init__(self, model: mujoco.MjModel, *, width: int = 256, height: int = 256) -> None:
        if width < 1 or height < 1:
            raise ValueError("capture width and height must be positive")
        missing = [
            camera_name
            for camera_name in POLICY_VIEW_TO_CAMERA.values()
            if model.camera(camera_name).id < 0
        ]
        if missing:
            raise ValueError(f"MuJoCo model is missing cameras: {missing}")
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._renderer: mujoco.Renderer | None = None

    def capture(
        self,
        env: FrankaContactEnv,
        *,
        include_teacher: bool,
    ) -> DualViewFrame:
        if env.model is not self.model:
            raise ValueError("capture and environment must share the same MjModel")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, width=self.width, height=self.height
            )
        views: dict[str, CameraFrame | PolicyCameraFrame] = {}
        for view_name, camera_name in POLICY_VIEW_TO_CAMERA.items():
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()
            self._renderer.update_scene(env.data, camera=camera_name)
            rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
            if include_teacher:
                self._renderer.enable_depth_rendering()
                self._renderer.update_scene(env.data, camera=camera_name)
                depth = np.asarray(self._renderer.render(), dtype=np.float32).copy()
                self._renderer.disable_depth_rendering()

                self._renderer.enable_segmentation_rendering()
                self._renderer.update_scene(env.data, camera=camera_name)
                segmentation = np.asarray(
                    self._renderer.render(), dtype=np.int32
                ).copy()
                self._renderer.disable_segmentation_rendering()
                views[view_name] = CameraFrame(rgb, depth, segmentation)
            else:
                views[view_name] = PolicyCameraFrame(rgb)
        return DualViewFrame(
            policy_step=env.step_count,
            timestamp=env.step_count / env.policy_hz,
            state_hash=MultiViewRecorder.state_hash(env),
            views=views,
            calibration=self.camera_calibration(env),
        )

    def camera_calibration(self, env: FrankaContactEnv) -> dict[str, object]:
        if env.model is not self.model:
            raise ValueError("capture and environment must share the same MjModel")
        return camera_calibration(env, width=self.width, height=self.height)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
