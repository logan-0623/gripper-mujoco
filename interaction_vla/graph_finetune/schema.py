from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interaction_vla.lerobot_bridge.teacher_schema import (
    ENTITY_SLOTS,
    OPERATOR_IDS,
    PREDICATE_IDS,
    RELATION_SLOTS,
)


SCHEMA_VERSION = "mujoco_semantic_graph_v1"
SEMANTIC_CHANNELS = tuple(range(12, 22))
ENTITY_COUNT = len(ENTITY_SLOTS)
RELATION_COUNT = len(RELATION_SLOTS)
SEMANTIC_DIM = len(SEMANTIC_CHANNELS)


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


@dataclass(frozen=True)
class MuJoCoGraphTargets:
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
        values = {
            "entity_mask": _array(
                entity_mask,
                shape=(frames, ENTITY_COUNT),
                dtype=np.dtype(np.bool_),
                name="entity_mask",
            ),
            "entity_visibility": _array(
                self.entity_visibility,
                shape=(frames, ENTITY_COUNT, 2),
                dtype=np.dtype(np.float32),
                name="entity_visibility",
            ),
            "relation_mask": _array(
                self.relation_mask,
                shape=(frames, RELATION_COUNT),
                dtype=np.dtype(np.bool_),
                name="relation_mask",
            ),
            "relation_semantics": _array(
                self.relation_semantics,
                shape=(frames, RELATION_COUNT, SEMANTIC_DIM),
                dtype=np.dtype(np.float32),
                name="relation_semantics",
            ),
            "goal_relation": _array(
                self.goal_relation,
                shape=(frames,),
                dtype=np.dtype(np.int64),
                name="goal_relation",
            ),
            "goal_operator": _array(
                self.goal_operator,
                shape=(frames,),
                dtype=np.dtype(np.int64),
                name="goal_operator",
            ),
            "goal_predicate": _array(
                self.goal_predicate,
                shape=(frames,),
                dtype=np.dtype(np.int64),
                name="goal_predicate",
            ),
            "goal_residual": _array(
                self.goal_residual,
                shape=(frames,),
                dtype=np.dtype(np.float32),
                name="goal_residual",
            ),
        }
        visibility = values["entity_visibility"]
        if np.any((visibility < 0.0) | (visibility > 1.0)):
            raise ValueError("entity_visibility must lie within [0, 1]")
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
