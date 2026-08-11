from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from interaction_vla.lerobot_bridge.teacher_schema import SCHEMA_VERSION as TEACHER_SCHEMA

from .schema import MuJoCoGraphTargets, SEMANTIC_CHANNELS


_TARGET_KEYS = {
    "annotation.tc_tig.entity_mask",
    "annotation.tc_tig.entity_visibility",
    "annotation.tc_tig.relation_mask",
    "annotation.tc_tig.relation_values",
    "annotation.tc_tig.relation_goal",
}


def semantic_targets(arrays: Mapping[str, np.ndarray]) -> MuJoCoGraphTargets:
    missing = _TARGET_KEYS - set(arrays)
    if missing:
        raise ValueError("teacher arrays are missing: " + ", ".join(sorted(missing)))
    relation_values = np.asarray(
        arrays["annotation.tc_tig.relation_values"], dtype=np.float32
    )
    if relation_values.ndim != 3 or relation_values.shape[1:] != (8, 24):
        raise ValueError("teacher relation_values must have shape [frames, 8, 24]")
    goals = np.asarray(arrays["annotation.tc_tig.relation_goal"], dtype=np.float32)
    if goals.shape != (len(relation_values), 5) or not np.isfinite(goals).all():
        raise ValueError("teacher relation_goal must be finite with shape [frames, 5]")
    if not np.equal(goals[:, :3], np.floor(goals[:, :3])).all():
        raise ValueError("teacher relation_goal categorical IDs must be integers")
    return MuJoCoGraphTargets(
        entity_mask=np.asarray(
            arrays["annotation.tc_tig.entity_mask"], dtype=np.bool_
        ),
        entity_visibility=np.asarray(
            arrays["annotation.tc_tig.entity_visibility"], dtype=np.float32
        ),
        relation_mask=np.asarray(
            arrays["annotation.tc_tig.relation_mask"], dtype=np.bool_
        ),
        relation_semantics=relation_values[:, :, SEMANTIC_CHANNELS],
        goal_relation=goals[:, 0].astype(np.int64),
        goal_operator=goals[:, 1].astype(np.int64),
        goal_predicate=goals[:, 2].astype(np.int64),
        goal_residual=goals[:, 3].astype(np.float32),
    )


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    values = np.asarray(ratios, dtype=np.float64)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("split ratios must be three positive values summing to one")


def split_episode_indices(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[int]]:
    _validate_ratios(ratios)
    if len(records) < 3:
        raise ValueError("episode split requires at least three episodes")
    episode_indices = [int(record.get("episode_index", -1)) for record in records]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("teacher episode indices must be unique")
    if any(record.get("schema_version") != TEACHER_SCHEMA for record in records):
        raise ValueError("teacher manifest schema version is incompatible")

    def digest(record: Mapping[str, object]) -> bytes:
        episode_index = int(record["episode_index"])
        teacher_seed = int(record["seed"])
        return hashlib.sha256(
            f"{int(seed)}:{episode_index}:{teacher_seed}".encode()
        ).digest()

    ordered = sorted(records, key=lambda record: (digest(record), int(record["episode_index"])))
    validation_count = max(1, int(math.floor(len(ordered) * ratios[1])))
    test_count = max(1, int(math.floor(len(ordered) * ratios[2])))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise ValueError("split ratios leave no training episode")
    values = [int(record["episode_index"]) for record in ordered]
    return {
        "train": sorted(values[:train_count]),
        "validation": sorted(values[train_count : train_count + validation_count]),
        "test": sorted(values[train_count + validation_count :]),
    }


def select_training_fraction(
    episodes: Sequence[int], *, fraction: float, seed: int
) -> list[int]:
    values = [int(value) for value in episodes]
    if not values or len(set(values)) != len(values):
        raise ValueError("training episodes must be non-empty and unique")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("training fraction must lie within (0, 1]")
    ordered = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{int(seed)}:{value}".encode()).digest(),
            value,
        ),
    )
    count = max(1, math.ceil(len(ordered) * float(fraction)))
    return sorted(ordered[:count])
