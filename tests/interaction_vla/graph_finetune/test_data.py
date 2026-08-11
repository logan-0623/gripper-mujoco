from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph_finetune.data import (
    select_training_fraction,
    semantic_targets,
    split_episode_indices,
)
from interaction_vla.graph_finetune.schema import SEMANTIC_CHANNELS


def teacher_arrays(frames: int = 3) -> dict[str, np.ndarray]:
    relation_values = np.arange(frames * 8 * 24, dtype=np.float32).reshape(
        frames, 8, 24
    )
    relation_values[:, :, 16:20] = 0.5
    relation_values[:, :, 20:22] = 0.25
    relation_values[:, :, 22:24] = 1.0
    relation_goal = np.zeros((frames, 5), dtype=np.float32)
    relation_goal[:, 0] = np.arange(frames) % 8
    relation_goal[:, 1] = np.arange(frames) % 5
    relation_goal[:, 2] = np.arange(frames) % 7
    relation_goal[:, 3] = np.linspace(-0.3, 0.0, frames, dtype=np.float32)
    relation_goal[:, 4] = 1.0
    return {
        "annotation.tc_tig.entity_mask": np.ones((frames, 6), dtype=np.bool_),
        "annotation.tc_tig.entity_visibility": np.full(
            (frames, 6, 2), 0.25, dtype=np.float32
        ),
        "annotation.tc_tig.relation_mask": np.ones((frames, 8), dtype=np.bool_),
        "annotation.tc_tig.relation_values": relation_values,
        "annotation.tc_tig.relation_goal": relation_goal,
    }


def manifest_records(count: int = 5) -> list[dict[str, object]]:
    return [
        {
            "episode_index": episode,
            "seed": 1000 + episode,
            "frames": 3,
            "schema_version": "tc_tig_teacher_v1",
            "task": "Pick up the target and place it in the receptacle.",
        }
        for episode in range(count)
    ]


def test_semantic_target_selects_only_coordinate_invariant_channels() -> None:
    arrays = teacher_arrays()

    target = semantic_targets(arrays)

    assert SEMANTIC_CHANNELS == tuple(range(12, 22))
    np.testing.assert_array_equal(
        target.relation_semantics,
        arrays["annotation.tc_tig.relation_values"][:, :, 12:22],
    )
    assert target.relation_semantics.shape == (3, 8, 10)
    assert not hasattr(target, "entity_pose")
    assert not hasattr(target, "depth_agent")
    assert not hasattr(target, "action")


def test_semantic_target_extracts_goal_categories_and_residual() -> None:
    target = semantic_targets(teacher_arrays())

    assert target.goal_relation.tolist() == [0, 1, 2]
    assert target.goal_operator.tolist() == [0, 1, 2]
    assert target.goal_predicate.tolist() == [0, 1, 2]
    np.testing.assert_allclose(target.goal_residual, [-0.3, -0.15, 0.0])


def test_semantic_target_rejects_out_of_range_goal_id() -> None:
    arrays = teacher_arrays()
    arrays["annotation.tc_tig.relation_goal"][0, 0] = 8

    with pytest.raises(ValueError, match="goal_relation"):
        semantic_targets(arrays)


def test_episode_split_is_deterministic_and_never_splits_frames() -> None:
    records = manifest_records()

    first = split_episode_indices(records, seed=7, ratios=(0.6, 0.2, 0.2))
    second = split_episode_indices(records, seed=7, ratios=(0.6, 0.2, 0.2))

    assert first == second
    assert {name: len(value) for name, value in first.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert sorted(first["train"] + first["validation"] + first["test"]) == list(
        range(5)
    )
    assert set(first["train"]).isdisjoint(first["validation"] + first["test"])


def test_episode_split_rejects_duplicate_episode_indices() -> None:
    records = manifest_records()
    records[-1]["episode_index"] = 0

    with pytest.raises(ValueError, match="unique"):
        split_episode_indices(records, seed=0, ratios=(0.6, 0.2, 0.2))


def test_training_fraction_selects_whole_episodes_deterministically() -> None:
    episodes = list(range(10))

    first = select_training_fraction(episodes, fraction=0.25, seed=3)
    second = select_training_fraction(episodes, fraction=0.25, seed=3)

    assert first == second
    assert len(first) == 3
    assert set(first).issubset(episodes)
