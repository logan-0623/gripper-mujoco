from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interaction_vla.env import LayoutMode
from interaction_vla.graph_control.tracing import load_trace_episode
from interaction_vla.lerobot_bridge.capture import DualViewCapture
from interaction_vla.lerobot_bridge.config import BridgeConfig
from interaction_vla.physics_env import FrankaContactEnv

from .schema import StateBankRecord


@dataclass(frozen=True)
class MaterializedObservation:
    record: StateBankRecord
    agent_rgb: torch.Tensor
    wrist_rgb: torch.Tensor
    robot_state: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("agent_rgb", "wrist_rgb"):
            value = getattr(self, name)
            if value.shape != (3, 256, 256) or value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32 CHW 256x256")
            if not torch.isfinite(value).all() or torch.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must be finite in [0, 1]")
        if self.robot_state.shape != (10,) or self.robot_state.dtype != torch.float32:
            raise ValueError("materialized robot_state must be float32 with shape [10]")


def _chw(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.shape == (256, 256, 3):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape != (3, 256, 256):
        raise ValueError(f"{name} must have shape CHW or HWC 256x256")
    tensor = tensor.to(dtype=torch.float32)
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.contiguous()


def collate_observations(values: Sequence[MaterializedObservation]) -> dict[str, object]:
    if not values:
        raise ValueError("cannot collate an empty State Bank batch")
    return {
        "observation.images.agent": torch.stack([value.agent_rgb for value in values]),
        "observation.images.wrist": torch.stack([value.wrist_rgb for value in values]),
        "observation.state": torch.stack([value.robot_state for value in values]),
        "task": [value.record.instruction for value in values],
    }


def validate_replayed_position(
    expected: object, actual: object, *, tolerance: float
) -> float:
    first = np.asarray(expected, dtype=np.float64)
    second = np.asarray(actual, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("replay position values must be finite three-vectors")
    error = float(np.max(np.abs(first - second)))
    if error > tolerance:
        raise ValueError(
            f"deterministic trace replay drifted by {error:.6g} m (tolerance {tolerance:.6g})"
        )
    return error


def _make_env(config: BridgeConfig, *, max_steps: int) -> FrankaContactEnv:
    source = config.source
    return FrankaContactEnv(
        max_objects=source.max_objects,
        max_steps=max_steps,
        min_object_distance=source.environment.min_object_distance,
        workspace_low=source.environment.workspace_low,
        workspace_high=source.environment.workspace_high,
        crowded_anchor_min_distance=source.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=source.environment.crowded_anchor_max_distance,
        physics=source.physics,
    )


class StateBankMaterializer:
    def __init__(
        self,
        *,
        dataset_root: str | Path,
        repo_id: str,
        bridge_config: BridgeConfig,
        replay_position_tolerance: float,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.repo_id = str(repo_id)
        self.bridge_config = bridge_config
        self.replay_position_tolerance = float(replay_position_tolerance)
        self._dataset: Any | None = None

    def _expert(self, record: StateBankRecord) -> MaterializedObservation:
        if self._dataset is None:
            from lerobot.datasets import LeRobotDataset

            self._dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)
        sample = self._dataset[record.observation.source_index]
        return MaterializedObservation(
            record=record,
            agent_rgb=_chw(sample[record.observation.agent_rgb_key], "agent RGB"),
            wrist_rgb=_chw(sample[record.observation.wrist_rgb_key], "wrist RGB"),
            robot_state=torch.tensor(record.robot_state, dtype=torch.float32),
        )

    def _replay(
        self, path: Path, selected: Sequence[StateBankRecord]
    ) -> Iterable[MaterializedObservation]:
        rows = load_trace_episode(path)
        by_step = {record.observation.source_index: record for record in selected}
        if len(by_step) != len(selected):
            raise ValueError("trace replay records contain duplicate steps")
        if any(step < 0 or step >= len(rows) for step in by_step):
            raise ValueError("trace replay step is outside the source episode")
        first = rows[0]
        env = _make_env(self.bridge_config, max_steps=len(rows))
        snapshot = env.reset(
            seed=int(first["environment_seed"]),
            object_count=int(first["object_count"]),
            layout_mode=LayoutMode(str(first["layout"])),
        )
        capture = DualViewCapture(
            env.model,
            width=self.bridge_config.dataset.image_size[1],
            height=self.bridge_config.dataset.image_size[0],
        )
        try:
            for step, row in enumerate(rows):
                if step in by_step:
                    record = by_step[step]
                    validate_replayed_position(
                        row["end_effector_position"],
                        snapshot.gripper.position,
                        tolerance=self.replay_position_tolerance,
                    )
                    frame = capture.capture(env, include_teacher=False)
                    yield MaterializedObservation(
                        record=record,
                        agent_rgb=_chw(frame.views["agent"].rgb, "replay agent RGB"),
                        wrist_rgb=_chw(frame.views["wrist"].rgb, "replay wrist RGB"),
                        robot_state=torch.tensor(record.robot_state, dtype=torch.float32),
                    )
                transition = env.step(
                    np.asarray(row["executed_world_action"], dtype=np.float64)
                )
                snapshot = transition.snapshot
                if transition.done and step != len(rows) - 1:
                    raise ValueError("trace replay terminated before its source episode")
        finally:
            capture.close()

    def iter_records(
        self, records: Sequence[StateBankRecord]
    ) -> Iterable[MaterializedObservation]:
        trace_groups: dict[str, list[StateBankRecord]] = defaultdict(list)
        for record in records:
            if record.domain == "expert_support":
                yield self._expert(record)
            else:
                trace_groups[record.observation.source_uri].append(record)
        for uri in sorted(trace_groups):
            yield from self._replay(
                Path(uri),
                sorted(trace_groups[uri], key=lambda record: record.frame_index),
            )

