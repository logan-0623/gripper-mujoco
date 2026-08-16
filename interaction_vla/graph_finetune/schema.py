from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np

from interaction_vla.lerobot_bridge.interaction_phase import PHASE_NAMES
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
