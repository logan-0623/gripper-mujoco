from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np

from interaction_vla.lerobot_bridge.interaction_phase import PHASE_IDS, PHASE_NAMES
from interaction_vla.lerobot_bridge.teacher_schema import (
    ENTITY_SLOTS,
    OPERATOR_IDS,
    PREDICATE_IDS,
    RELATION_SLOTS,
)


GRAPH_SCHEMA_VERSION = "mujoco_interaction_graph_v2"
TOKEN_SCHEMA_VERSION = "interaction_graph_control_v2"
# Transitional public name used by the existing fine-tune pipeline. All new artifacts
# are bound to the v2 value even before the remaining pipeline modules are migrated.
SCHEMA_VERSION = GRAPH_SCHEMA_VERSION

SEMANTIC_CHANNELS = tuple(range(12, 22))
ENTITY_COUNT = len(ENTITY_SLOTS)
RELATION_COUNT = len(RELATION_SLOTS)
SEMANTIC_DIM = len(SEMANTIC_CHANNELS)

_TOKEN_WIDTHS = (
    ("entity_presence", 6),
    ("entity_visibility", 12),
    ("relation_presence", 8),
    ("gripper_target_geometry", 8),
    ("target_receptacle_geometry", 10),
    ("distractor_geometry", 14),
    ("phase", 6),
    ("relation_trends", 4),
    ("next_relation", 8),
    ("relation_operator", 5),
    ("predicate", 7),
    ("goal_residual", 1),
)


def _token_slices() -> Mapping[str, slice]:
    cursor = 0
    result: dict[str, slice] = {}
    for name, width in _TOKEN_WIDTHS:
        result[name] = slice(cursor, cursor + width)
        cursor += width
    return MappingProxyType(result)


TOKEN_SLICES: Final[Mapping[str, slice]] = _token_slices()
TOKEN_DIM: Final[int] = TOKEN_SLICES["goal_residual"].stop
TOKEN_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{name}_{index}"
    for name, bounds in TOKEN_SLICES.items()
    for index in range(bounds.stop - bounds.start)
)


def _array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if result.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _bounded(values: np.ndarray, name: str) -> None:
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} probabilities/confidences must lie within [0, 1]")


def _float_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class GraphV2Targets:
    entity_mask: np.ndarray
    entity_visibility: np.ndarray
    relation_mask: np.ndarray
    gripper_target_geometry: np.ndarray
    target_receptacle_geometry: np.ndarray
    distractor_geometry: np.ndarray
    phase: np.ndarray
    relation_trends: np.ndarray
    goal_relation: np.ndarray
    goal_operator: np.ndarray
    goal_predicate: np.ndarray
    goal_residual: np.ndarray

    def __post_init__(self) -> None:
        entity_mask = np.asarray(self.entity_mask)
        if entity_mask.ndim != 2 or entity_mask.shape[1] != ENTITY_COUNT:
            raise ValueError("entity_mask must have shape [frames, 6]")
        frames = len(entity_mask)
        if frames < 1:
            raise ValueError("Graph v2 targets must contain at least one frame")
        specs = {
            "entity_mask": ((frames, ENTITY_COUNT), np.dtype(np.bool_)),
            "entity_visibility": (
                (frames, ENTITY_COUNT, 2),
                np.dtype(np.float32),
            ),
            "relation_mask": ((frames, RELATION_COUNT), np.dtype(np.bool_)),
            "gripper_target_geometry": ((frames, 8), np.dtype(np.float32)),
            "target_receptacle_geometry": ((frames, 10), np.dtype(np.float32)),
            "distractor_geometry": ((frames, 2, 7), np.dtype(np.float32)),
            "phase": ((frames,), np.dtype(np.int64)),
            "relation_trends": ((frames, 4), np.dtype(np.float32)),
            "goal_relation": ((frames,), np.dtype(np.int64)),
            "goal_operator": ((frames,), np.dtype(np.int64)),
            "goal_predicate": ((frames,), np.dtype(np.int64)),
            "goal_residual": ((frames,), np.dtype(np.float32)),
        }
        values = {
            name: _array(
                getattr(self, name),
                shape=shape,
                dtype=dtype,
                name=name,
            )
            for name, (shape, dtype) in specs.items()
        }
        _bounded(values["entity_visibility"], "entity_visibility")
        _bounded(
            values["gripper_target_geometry"][:, 5:8],
            "gripper_target_geometry",
        )
        _bounded(
            values["target_receptacle_geometry"][:, 8:10],
            "target_receptacle_geometry",
        )
        _bounded(
            values["distractor_geometry"][:, :, 4:7],
            "distractor_geometry",
        )
        categories = {
            "phase": len(PHASE_NAMES),
            "goal_relation": RELATION_COUNT,
            "goal_operator": len(OPERATOR_IDS),
            "goal_predicate": len(PREDICATE_IDS),
        }
        for name, count in categories.items():
            if np.any((values[name] < 0) | (values[name] >= count)):
                raise ValueError(f"{name} contains an out-of-range ID")
        for distractor, relation_index in enumerate((3, 5)):
            inactive = ~values["relation_mask"][:, relation_index]
            if np.any(values["distractor_geometry"][inactive, distractor] != 0.0):
                raise ValueError("inactive distractor geometry must be zero")
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GraphV2Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    workspace_scale: float
    velocity_scale: float

    def __post_init__(self) -> None:
        state_mean = _float_array(self.state_mean, (10,), "state_mean")
        state_std = _float_array(self.state_std, (10,), "state_std")
        if np.any(state_std <= 0.0):
            raise ValueError("state_std must be positive")
        scales: dict[str, float] = {}
        for name in ("workspace_scale", "velocity_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            scales[name] = value
        object.__setattr__(self, "state_mean", state_mean)
        object.__setattr__(self, "state_std", state_std)
        for name, value in scales.items():
            object.__setattr__(self, name, value)


def normalized_graph_v2_frame(
    targets: GraphV2Targets,
    frame: int,
    normalization: GraphV2Normalization,
) -> dict[str, np.ndarray]:
    index = int(frame)
    frames = len(targets.phase)
    if index != frame or not 0 <= index < frames:
        raise IndexError(f"Graph v2 frame must lie within [0, {frames})")
    workspace = np.float32(normalization.workspace_scale)
    velocity = np.float32(normalization.velocity_scale)

    gripper_target = targets.gripper_target_geometry[index].copy()
    gripper_target[:4] /= workspace
    gripper_target[4] /= velocity

    target_receptacle = targets.target_receptacle_geometry[index].copy()
    target_receptacle[:8] /= workspace

    distractors = targets.distractor_geometry[index].copy()
    distractors[:, :4] /= workspace

    trends = targets.relation_trends[index].copy()
    trends[[0, 2, 3]] /= workspace

    residual = np.float32(targets.goal_residual[index])
    if int(targets.phase[index]) in {
        PHASE_IDS["approach"],
        PHASE_IDS["transport"],
    }:
        residual /= workspace
    residual = np.clip(residual, -1.0, 1.0).astype(np.float32)

    result = {
        "gripper_target_geometry": gripper_target,
        "target_receptacle_geometry": target_receptacle,
        "distractor_geometry": distractors,
        "relation_trends": trends,
        "goal_residual": np.asarray(residual, dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("normalized Graph v2 target must be finite")
    return result


def pack_oracle_target(
    targets: GraphV2Targets,
    frame: int,
    normalization: GraphV2Normalization,
) -> np.ndarray:
    index = int(frame)
    normalized = normalized_graph_v2_frame(targets, index, normalization)
    token = np.zeros(TOKEN_DIM, dtype=np.float32)
    token[TOKEN_SLICES["entity_presence"]] = targets.entity_mask[index]
    token[TOKEN_SLICES["entity_visibility"]] = (
        targets.entity_visibility[index].reshape(-1)
    )
    token[TOKEN_SLICES["relation_presence"]] = targets.relation_mask[index]
    token[TOKEN_SLICES["gripper_target_geometry"]] = normalized[
        "gripper_target_geometry"
    ]
    token[TOKEN_SLICES["target_receptacle_geometry"]] = normalized[
        "target_receptacle_geometry"
    ]
    token[TOKEN_SLICES["distractor_geometry"]] = normalized[
        "distractor_geometry"
    ].reshape(-1)
    token[TOKEN_SLICES["phase"].start + int(targets.phase[index])] = 1.0
    token[TOKEN_SLICES["relation_trends"]] = normalized["relation_trends"]
    token[
        TOKEN_SLICES["next_relation"].start + int(targets.goal_relation[index])
    ] = 1.0
    token[
        TOKEN_SLICES["relation_operator"].start
        + int(targets.goal_operator[index])
    ] = 1.0
    token[
        TOKEN_SLICES["predicate"].start + int(targets.goal_predicate[index])
    ] = 1.0
    token[TOKEN_SLICES["goal_residual"]] = normalized["goal_residual"]
    if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
        raise ValueError(f"Graph v2 token must be finite with shape [{TOKEN_DIM}]")
    return token


@dataclass(frozen=True)
class MuJoCoGraphTargets:
    """Temporary v1 target container removed when data extraction migrates in Task 2."""

    entity_mask: np.ndarray
    entity_visibility: np.ndarray
    relation_mask: np.ndarray
    relation_semantics: np.ndarray
    goal_relation: np.ndarray
    goal_operator: np.ndarray
    goal_predicate: np.ndarray
    goal_residual: np.ndarray

    def __post_init__(self) -> None:
        entity_mask = np.asarray(self.entity_mask)
        if entity_mask.ndim != 2 or entity_mask.shape[1] != ENTITY_COUNT:
            raise ValueError("entity_mask must have shape [frames, 6]")
        frames = len(entity_mask)
        if frames < 1:
            raise ValueError("graph targets must contain at least one frame")
        specs = {
            "entity_mask": ((frames, ENTITY_COUNT), np.dtype(np.bool_)),
            "entity_visibility": (
                (frames, ENTITY_COUNT, 2),
                np.dtype(np.float32),
            ),
            "relation_mask": ((frames, RELATION_COUNT), np.dtype(np.bool_)),
            "relation_semantics": (
                (frames, RELATION_COUNT, SEMANTIC_DIM),
                np.dtype(np.float32),
            ),
            "goal_relation": ((frames,), np.dtype(np.int64)),
            "goal_operator": ((frames,), np.dtype(np.int64)),
            "goal_predicate": ((frames,), np.dtype(np.int64)),
            "goal_residual": ((frames,), np.dtype(np.float32)),
        }
        values = {
            name: _array(
                getattr(self, name),
                shape=shape,
                dtype=dtype,
                name=name,
            )
            for name, (shape, dtype) in specs.items()
        }
        _bounded(values["entity_visibility"], "entity_visibility")
        categories = {
            "goal_relation": RELATION_COUNT,
            "goal_operator": len(OPERATOR_IDS),
            "goal_predicate": len(PREDICATE_IDS),
        }
        for name, count in categories.items():
            if np.any((values[name] < 0) | (values[name] >= count)):
                raise ValueError(f"{name} contains an out-of-range ID")
        for name, value in values.items():
            object.__setattr__(self, name, value)
