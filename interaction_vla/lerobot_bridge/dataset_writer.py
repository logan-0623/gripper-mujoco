from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np


STATE_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "rot_c0_x",
    "rot_c0_y",
    "rot_c0_z",
    "rot_c1_x",
    "rot_c1_y",
    "rot_c1_z",
    "gripper_aperture",
]
ACTION_NAMES = [
    "dx_local",
    "dy_local",
    "dz_local",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
]


def standard_features(*, width: int, height: int) -> dict[str, dict[str, object]]:
    if (width, height) != (256, 256):
        raise ValueError("the first LeRobot bridge requires 256x256 images")
    return {
        "observation.images.agent": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
    }


class LeRobotEpisodeWriter:
    def __init__(
        self,
        dataset: Any,
        *,
        root: Path,
        width: int,
        height: int,
    ) -> None:
        self._dataset = dataset
        self.root = root
        self.width = int(width)
        self.height = int(height)
        self._finalized = False

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        root: str | Path,
        fps: int,
        width: int,
        height: int,
    ) -> LeRobotEpisodeWriter:
        destination = Path(root)
        if destination.exists():
            raise FileExistsError(f"LeRobot writer requires a new dataset root: {destination}")
        if not repo_id:
            raise ValueError("repo_id must not be empty")
        if fps != 20:
            raise ValueError("the first LeRobot bridge requires 20 Hz")
        if importlib.metadata.version("lerobot") != "0.6.1":
            raise RuntimeError("LeRobotEpisodeWriter requires lerobot==0.6.1")
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=destination,
            fps=fps,
            robot_type="franka_mujoco",
            features=standard_features(width=width, height=height),
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=2,
        )
        return cls(dataset, root=destination, width=width, height=height)

    def add_frame(
        self,
        *,
        agent_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        task: str,
    ) -> None:
        self._require_open("add frames")
        agent = self._image(agent_rgb, "agent_rgb")
        wrist = self._image(wrist_rgb, "wrist_rgb")
        state_values = self._vector(state, (10,), "state")
        action_values = self._vector(action, (7,), "action")
        if not task.strip():
            raise ValueError("task must not be empty")
        self._dataset.add_frame(
            {
                "observation.images.agent": agent,
                "observation.images.wrist": wrist,
                "observation.state": state_values,
                "action": action_values,
                "task": task,
            }
        )

    def clear_episode(self) -> None:
        self._require_open("clear the episode buffer")
        self._dataset.clear_episode_buffer()

    def save_episode(self) -> None:
        self._require_open("save episodes")
        self._dataset.save_episode(parallel_encoding=False)

    def finalize(self) -> None:
        if not self._finalized:
            self._dataset.finalize()
            self._finalized = True

    def _require_open(self, operation: str) -> None:
        if self._finalized:
            raise RuntimeError(f"cannot {operation} after finalization")

    def _image(self, value: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(value)
        expected = (self.height, self.width, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 with shape {expected}")
        return image.copy()

    @staticmethod
    def _vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        vector = np.asarray(value)
        if vector.shape != shape or vector.dtype != np.float32:
            raise ValueError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must be finite")
        return vector.copy()
