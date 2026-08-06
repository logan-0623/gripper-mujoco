from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .evaluate import shuffle_valid_edge_assignments
from .graph.builder import SceneGraphBuilder
from .models.encoders import SceneBatch
from .models.policy import ActionPolicy
from .physics_action_safety import project_cartesian_action
from .physics_env import FrankaContactEnv

if TYPE_CHECKING:
    from .train import TrainingStatistics


@dataclass(frozen=True)
class ChunkControllerDiagnostics:
    ensemble_size: int
    raw_first_action: np.ndarray
    aggregated_action: np.ndarray
    raw_gripper_score: float
    gripper_command: float
    gripper_switch_count: int
    smoothing_delta_norm: float
    ik_projection_scale: float


class TemporalActionEnsembler:
    def __init__(
        self,
        *,
        horizon: int,
        temporal_decay: float,
        gripper_close_threshold: float,
        gripper_open_threshold: float,
    ) -> None:
        if horizon < 1:
            raise ValueError("temporal ensemble horizon must be positive")
        if not math.isfinite(temporal_decay) or temporal_decay < 0.0:
            raise ValueError("temporal decay must be finite and non-negative")
        if not (
            0.0 <= gripper_close_threshold < gripper_open_threshold <= 1.0
        ):
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        self.horizon = int(horizon)
        self.temporal_decay = float(temporal_decay)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.reset()

    def reset(self, *, gripper_open: bool = True) -> None:
        self._chunks: dict[int, np.ndarray] = {}
        self._gripper_command = float(bool(gripper_open))
        self.gripper_switch_count = 0

    def add(self, issued_step: int, chunk: np.ndarray) -> None:
        values = np.asarray(chunk, dtype=np.float32)
        if issued_step < 0:
            raise ValueError("issued policy step must be non-negative")
        if values.shape != (self.horizon, 7) or not np.isfinite(values).all():
            raise ValueError(
                f"action chunk must be finite with shape ({self.horizon}, 7)"
            )
        self._chunks[int(issued_step)] = values.copy()

    def resolve_gripper(self, score: float) -> float:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("gripper score must be finite")
        previous = self._gripper_command
        if value <= self.gripper_close_threshold:
            self._gripper_command = 0.0
        elif value >= self.gripper_open_threshold:
            self._gripper_command = 1.0
        if self._gripper_command != previous:
            self.gripper_switch_count += 1
        return self._gripper_command

    def action_for_step(
        self,
        step: int,
    ) -> tuple[np.ndarray, ChunkControllerDiagnostics]:
        if step < 0:
            raise ValueError("policy step must be non-negative")
        self._chunks = {
            issued: chunk
            for issued, chunk in self._chunks.items()
            if 0 <= step - issued < self.horizon
        }
        if step not in self._chunks:
            raise RuntimeError("current policy step has no newly issued action chunk")
        candidates: list[np.ndarray] = []
        weights: list[float] = []
        for issued in sorted(self._chunks, reverse=True):
            age = step - issued
            candidates.append(self._chunks[issued][age])
            weights.append(math.exp(-self.temporal_decay * age))
        candidate_array = np.stack(candidates).astype(np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        pose = np.average(candidate_array[:, :6], axis=0, weights=weight_array)
        gripper_score = float(
            np.average(candidate_array[:, 6], axis=0, weights=weight_array)
        )
        gripper_command = self.resolve_gripper(gripper_score)
        aggregated = np.concatenate((np.clip(pose, -1.0, 1.0), (gripper_command,)))
        aggregated = aggregated.astype(np.float32)
        raw_first = self._chunks[step][0].copy()
        diagnostics = ChunkControllerDiagnostics(
            ensemble_size=len(candidates),
            raw_first_action=raw_first,
            aggregated_action=aggregated.copy(),
            raw_gripper_score=gripper_score,
            gripper_command=gripper_command,
            gripper_switch_count=self.gripper_switch_count,
            smoothing_delta_norm=float(
                np.linalg.norm(aggregated[:6] - raw_first[:6])
            ),
            ik_projection_scale=1.0,
        )
        return aggregated, diagnostics


class ChunkedPolicyController:
    def __init__(
        self,
        *,
        policy: ActionPolicy,
        statistics: TrainingStatistics,
        builder: SceneGraphBuilder,
        horizon: int,
        temporal_decay: float,
        gripper_close_threshold: float,
        gripper_open_threshold: float,
        device: torch.device | str = "cpu",
        edge_shuffle: bool = False,
        edge_shuffle_seed: int = 0,
    ) -> None:
        resolved_device = torch.device(device)
        if resolved_device.type != "cpu":
            raise ValueError("chunked learned rollouts require a CPU device")
        self.policy = policy.to(resolved_device).eval()
        self.statistics = statistics
        self.builder = builder
        self.device = resolved_device
        self.edge_shuffle = bool(edge_shuffle)
        self.edge_shuffle_seed = int(edge_shuffle_seed)
        self.horizon = int(horizon)
        self.ensemble = TemporalActionEnsembler(
            horizon=horizon,
            temporal_decay=temporal_decay,
            gripper_close_threshold=gripper_close_threshold,
            gripper_open_threshold=gripper_open_threshold,
        )
        self.step_index = 0

    def reset(self, env: FrankaContactEnv) -> None:
        proprioception = env.proprioception()
        gripper_open = bool(float(np.mean(proprioception[13:15])) >= 0.02)
        self.ensemble.reset(gripper_open=gripper_open)
        self.step_index = 0

    def _normalized_inputs(
        self,
        env: FrankaContactEnv,
    ) -> tuple[SceneBatch, torch.Tensor]:
        graph = self.builder.build(env.snapshot())
        scene = SceneBatch(
            node_features=torch.from_numpy(graph.node_features[None]).float(),
            edge_index=torch.from_numpy(graph.edge_index).long(),
            edge_features=torch.from_numpy(graph.edge_features[None]).float(),
            node_mask=torch.from_numpy(graph.node_mask[None]).bool(),
            edge_mask=torch.from_numpy(graph.edge_mask[None]).bool(),
        )
        scene = self.statistics.normalize_scene(scene)
        if self.edge_shuffle:
            scene = shuffle_valid_edge_assignments(
                scene,
                seed=self.edge_shuffle_seed ^ self.step_index ^ 0x45444745,
            )
        proprioception = self.statistics.normalize_proprioception(
            torch.from_numpy(env.proprioception()[None]).float()
        )
        return scene.to(self.device), proprioception.to(self.device)

    @torch.no_grad()
    def act(
        self,
        env: FrankaContactEnv,
    ) -> tuple[np.ndarray, ChunkControllerDiagnostics]:
        scene, proprioception = self._normalized_inputs(env)
        chunk = self.policy.predict_action_chunk(
            scene if self.policy.scene_encoder is not None else None,
            proprioception,
        )[0].detach().cpu().numpy().astype(np.float32)
        self.ensemble.add(self.step_index, chunk)
        aggregated, diagnostics = self.ensemble.action_for_step(self.step_index)
        projection = project_cartesian_action(env.controller, aggregated)
        diagnostics = replace(
            diagnostics,
            ik_projection_scale=float(projection.scale),
        )
        self.step_index += 1
        return projection.action.astype(np.float32), diagnostics
