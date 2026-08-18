from __future__ import annotations

import numpy as np

from interaction_vla.graph_control.ablation import (
    representation_transform,
    resample_sequence_nearest,
    stratified_episode_permutation,
)
from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES


def _active_groups(tokens: np.ndarray) -> set[str]:
    return {
        name
        for name, bounds in TOKEN_SLICES.items()
        if np.any(tokens[:, bounds] != 0.0)
    }


def test_progressive_representation_transforms_keep_fixed_width_and_exact_groups() -> None:
    source = np.ones((3, TOKEN_DIM), dtype=np.float32)

    flat = representation_transform(source, "flat")
    entity = representation_transform(source, "entity_geometry")
    interaction = representation_transform(source, "interaction_state")
    full = representation_transform(source, "full_graph")

    assert flat.shape == entity.shape == interaction.shape == full.shape == source.shape
    assert _active_groups(flat) == set()
    assert _active_groups(entity) == {
        "entity_presence",
        "entity_visibility",
        "gripper_target_geometry",
        "target_receptacle_geometry",
        "distractor_geometry",
    }
    assert _active_groups(interaction) == _active_groups(entity) | {
        "relation_presence",
        "phase",
    }
    assert _active_groups(full) == set(TOKEN_SLICES)
    assert not np.shares_memory(entity, source)


def test_episode_permutation_is_deterministic_deranged_and_length_stratified() -> None:
    lengths = {episode: 10 + episode for episode in range(12)}

    first, first_strata = stratified_episode_permutation(lengths, seed=17)
    second, second_strata = stratified_episode_permutation(lengths, seed=17)

    assert first == second
    assert first_strata == second_strata
    assert set(first) == set(lengths)
    assert set(first.values()) == set(lengths)
    assert all(destination != source for destination, source in first.items())
    assert all(
        first_strata[destination] == first_strata[source]
        for destination, source in first.items()
    )


def test_short_partition_coalesces_strata_without_self_pairing() -> None:
    lengths = {episode: 20 + episode for episode in range(5)}

    permutation, strata = stratified_episode_permutation(lengths, seed=23)

    assert all(destination != source for destination, source in permutation.items())
    assert len(set(strata.values())) == 2
    assert all(strata[destination] == strata[source] for destination, source in permutation.items())


def test_nearest_progress_resampling_preserves_values_without_interpolation() -> None:
    source = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32)

    expanded = resample_sequence_nearest(source, destination_length=5)
    contracted = resample_sequence_nearest(source, destination_length=2)

    np.testing.assert_array_equal(expanded.ravel(), [0.0, 0.0, 10.0, 20.0, 20.0])
    np.testing.assert_array_equal(contracted.ravel(), [0.0, 20.0])
    assert set(expanded.ravel()) <= set(source.ravel())

