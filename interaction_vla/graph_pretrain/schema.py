from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interaction_vla.lerobot_bridge.teacher_schema import OPERATOR_IDS, PREDICATE_IDS


SCHEMA_VERSION = "reflect_semantic_graph_v1"
OBJECT_COUNT = 6
NONE_OBJECT_INDEX = OBJECT_COUNT
STATE_NAMES = ("unknown", "ready", "blocked", "bad", "done")
UPRIGHT_NAMES = ("unknown", "false", "true")
PHASE_NAMES = ("pick", "place", "reorient", "insert")
STATE_IDS = {name: index for index, name in enumerate(STATE_NAMES)}
UPRIGHT_IDS = {name: index for index, name in enumerate(UPRIGHT_NAMES)}
PHASE_IDS = {name: index for index, name in enumerate(PHASE_NAMES)}


def _integer_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class ReflectGraphTarget:
    target_index: int
    in_hand_index: int
    object_mask: np.ndarray
    state_ids: np.ndarray
    upright_ids: np.ndarray
    dependency: np.ndarray
    phase_id: int
    goal_operator_id: int
    goal_predicate_id: int

    def __post_init__(self) -> None:
        state_ids = _integer_array(self.state_ids, (OBJECT_COUNT,), "state_ids")
        object_mask = np.asarray(self.object_mask, dtype=np.bool_)
        if object_mask.shape != (OBJECT_COUNT,) or not object_mask.any():
            raise ValueError("object_mask must select at least one of six object slots")
        upright_ids = _integer_array(
            self.upright_ids, (OBJECT_COUNT,), "upright_ids"
        )
        dependency = np.asarray(self.dependency, dtype=np.float32)
        if dependency.shape != (OBJECT_COUNT, OBJECT_COUNT):
            raise ValueError("dependency must have shape (6, 6)")
        if not np.isfinite(dependency).all() or np.any(
            (dependency != 0.0) & (dependency != 1.0)
        ):
            raise ValueError("dependency must be a finite binary matrix")
        if np.any(np.diag(dependency) != 0.0):
            raise ValueError("dependency must not contain self edges")
        if not 0 <= int(self.target_index) < OBJECT_COUNT:
            raise ValueError("target_index is outside the object slots")
        if not object_mask[int(self.target_index)]:
            raise ValueError("target_index must select a present object")
        if not 0 <= int(self.in_hand_index) <= NONE_OBJECT_INDEX:
            raise ValueError("in_hand_index is outside the object slots")
        if (
            int(self.in_hand_index) != NONE_OBJECT_INDEX
            and not object_mask[int(self.in_hand_index)]
        ):
            raise ValueError("in_hand_index must select a present object or none")
        if np.any((state_ids < 0) | (state_ids >= len(STATE_NAMES))):
            raise ValueError("state_ids contains an unknown state")
        if np.any((upright_ids < 0) | (upright_ids >= len(UPRIGHT_NAMES))):
            raise ValueError("upright_ids contains an unknown state")
        if not 0 <= int(self.phase_id) < len(PHASE_NAMES):
            raise ValueError("phase_id is unknown")
        if int(self.goal_operator_id) not in OPERATOR_IDS.values():
            raise ValueError("goal_operator_id is unknown")
        if int(self.goal_predicate_id) not in PREDICATE_IDS.values():
            raise ValueError("goal_predicate_id is unknown")
        object.__setattr__(self, "target_index", int(self.target_index))
        object.__setattr__(self, "in_hand_index", int(self.in_hand_index))
        object.__setattr__(self, "object_mask", object_mask.copy())
        object.__setattr__(self, "state_ids", state_ids)
        object.__setattr__(self, "upright_ids", upright_ids)
        object.__setattr__(self, "dependency", dependency.copy())
        object.__setattr__(self, "phase_id", int(self.phase_id))
        object.__setattr__(self, "goal_operator_id", int(self.goal_operator_id))
        object.__setattr__(self, "goal_predicate_id", int(self.goal_predicate_id))
