from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from interaction_vla.lerobot_bridge.provenance import sha256_file
from interaction_vla.lerobot_bridge.teacher_schema import SCHEMA_VERSION, TeacherFrame


@dataclass(frozen=True)
class TeacherSidecarRecord:
    path: str
    episode_index: int
    frames: int
    sha256: str
    first_state_hash: str
    last_state_hash: str
    schema_version: str
    seed: int
    object_count: int
    task_id: int


class TeacherSidecarWriter:
    def __init__(self, root: str | Path, *, fps: int = 20) -> None:
        if fps != 20:
            raise ValueError("TC-TIG sidecars require 20 Hz")
        self.root = Path(root)
        self.fps = int(fps)

    def write_episode(
        self,
        episode_index: int,
        frames: Iterable[TeacherFrame],
        relation_goals: np.ndarray,
        *,
        seed: int = 0,
        object_count: int = 0,
        task_id: int = 0,
    ) -> TeacherSidecarRecord:
        values = tuple(frames)
        if not values:
            raise ValueError("cannot write an empty teacher sidecar")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        frame_indices = np.asarray(
            [frame.frame_index for frame in values], dtype=np.int32
        )
        expected_indices = np.arange(len(values), dtype=np.int32)
        if not np.array_equal(frame_indices, expected_indices):
            raise ValueError("teacher frame indices must be contiguous and start at zero")
        timestamps = np.asarray(
            [frame.timestamp for frame in values], dtype=np.float64
        )
        if not np.allclose(
            timestamps,
            expected_indices / self.fps,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("teacher timestamps must match frame_index / fps")
        goals = np.asarray(relation_goals)
        if goals.shape != (len(values), 5) or goals.dtype != np.float32:
            raise ValueError("relation goals must be float32 with shape [T, 5]")
        if not np.isfinite(goals).all():
            raise ValueError("relation goals must be finite")

        arrays: dict[str, np.ndarray] = {
            "frame_index": frame_indices,
            "timestamp": timestamps,
            "state_hash": np.asarray([frame.state_hash for frame in values]),
            "annotation.tc_tig.entity_pose": self._stack(values, "entity_pose"),
            "annotation.tc_tig.entity_size": self._stack(values, "entity_size"),
            "annotation.tc_tig.entity_role": self._stack(values, "entity_role"),
            "annotation.tc_tig.entity_visibility": self._stack(
                values, "entity_visibility"
            ),
            "annotation.tc_tig.entity_mask": self._stack(values, "entity_mask"),
            "annotation.tc_tig.relation_values": self._stack(
                values, "relation_values"
            ),
            "annotation.tc_tig.relation_type": self._stack(values, "relation_type"),
            "annotation.tc_tig.relation_mask": self._stack(values, "relation_mask"),
            "annotation.tc_tig.instance_agent": self._stack(values, "instance_agent"),
            "annotation.tc_tig.instance_wrist": self._stack(values, "instance_wrist"),
            "annotation.tc_tig.depth_agent": self._stack(values, "depth_agent"),
            "annotation.tc_tig.depth_wrist": self._stack(values, "depth_wrist"),
            "annotation.tc_tig.camera_intrinsics": self._stack(
                values, "camera_intrinsics"
            ),
            "annotation.tc_tig.camera_extrinsics_base": self._stack(
                values, "camera_extrinsics_base"
            ),
            "annotation.tc_tig.relation_goal": goals.copy(),
        }
        if any(array.dtype == np.dtype(object) for array in arrays.values()):
            raise ValueError("teacher sidecars must never contain object arrays")

        relative_path = Path("teacher") / f"episode_{episode_index:06d}.npz"
        destination = self.root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        return TeacherSidecarRecord(
            path=relative_path.as_posix(),
            episode_index=int(episode_index),
            frames=len(values),
            sha256=sha256_file(destination),
            first_state_hash=values[0].state_hash,
            last_state_hash=values[-1].state_hash,
            schema_version=SCHEMA_VERSION,
            seed=int(seed),
            object_count=int(object_count),
            task_id=int(task_id),
        )

    @staticmethod
    def _stack(frames: tuple[TeacherFrame, ...], field_name: str) -> np.ndarray:
        return np.stack([getattr(frame, field_name) for frame in frames])


def load_teacher_sidecar(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError(f"teacher sidecar SHA-256 mismatch: {source}")
    try:
        with np.load(source, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid teacher sidecar: {source}") from error
    if any(array.dtype == np.dtype(object) for array in arrays.values()):
        raise ValueError("teacher sidecar contains a forbidden object array")
    frame_index = arrays.get("frame_index")
    if frame_index is None or frame_index.ndim != 1:
        raise ValueError("teacher sidecar is missing a one-dimensional frame_index")
    for name, array in arrays.items():
        if array.ndim < 1 or len(array) != len(frame_index):
            raise ValueError(f"teacher sidecar array has inconsistent rows: {name}")
    return arrays
