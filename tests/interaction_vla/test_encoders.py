from __future__ import annotations

import torch

from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.graph.builder import SceneGraphBuilder
from interaction_vla.models.encoders import (
    GraphEncoder,
    build_matched_encoders,
    count_parameters,
    permute_scene_batch,
    scene_graphs_to_batch,
)


def sample_batch():
    env = KinematicTabletopEnv(max_objects=5)
    builder = SceneGraphBuilder(max_objects=5)
    first = builder.build(env.reset(seed=1, object_count=2))
    second = builder.build(env.reset(seed=2, object_count=3))
    return scene_graphs_to_batch((first, second))


def test_matched_encoders_emit_the_same_shape() -> None:
    batch = sample_batch()
    flat, graph = build_matched_encoders(
        max_nodes=8,
        max_edges=56,
        node_feature_dim=23,
        edge_feature_dim=10,
        graph_hidden_dim=32,
        embedding_dim=24,
        message_rounds=2,
    )

    assert flat(batch).shape == (2, 24)
    assert graph(batch).shape == (2, 24)


def test_graph_pooling_is_invariant_to_valid_node_permutation() -> None:
    torch.manual_seed(0)
    encoder = GraphEncoder(
        node_feature_dim=23,
        edge_feature_dim=10,
        hidden_dim=32,
        embedding_dim=24,
        message_rounds=2,
    ).eval()
    batch = sample_batch()
    permutation = torch.tensor([2, 0, 1, 7, 3, 4, 5, 6])

    original = encoder(batch)
    permuted = encoder(permute_scene_batch(batch, permutation))

    torch.testing.assert_close(original, permuted, atol=1e-5, rtol=1e-5)


def test_padding_values_cannot_change_encoder_outputs() -> None:
    torch.manual_seed(3)
    flat, graph = build_matched_encoders(
        max_nodes=8,
        max_edges=56,
        node_feature_dim=23,
        edge_feature_dim=10,
        graph_hidden_dim=24,
        embedding_dim=16,
        message_rounds=1,
    )
    batch = sample_batch()
    changed = batch.clone()
    changed.node_features[~changed.node_mask] = 999.0
    changed.edge_features[~changed.edge_mask] = -999.0

    torch.testing.assert_close(flat(batch), flat(changed))
    torch.testing.assert_close(graph(batch), graph(changed))


def test_flat_and_graph_parameter_counts_are_within_ten_percent() -> None:
    flat, graph = build_matched_encoders(
        max_nodes=8,
        max_edges=56,
        node_feature_dim=23,
        edge_feature_dim=10,
        graph_hidden_dim=32,
        embedding_dim=24,
        message_rounds=2,
    )
    flat_count = count_parameters(flat)
    graph_count = count_parameters(graph)

    assert abs(flat_count - graph_count) / max(flat_count, graph_count) <= 0.10


def test_small_physics_encoders_are_matched_without_relaxing_tolerance() -> None:
    flat, graph = build_matched_encoders(
        max_nodes=8,
        max_edges=56,
        node_feature_dim=23,
        edge_feature_dim=18,
        graph_hidden_dim=16,
        embedding_dim=16,
        message_rounds=1,
    )

    flat_count = count_parameters(flat)
    graph_count = count_parameters(graph)
    assert abs(flat_count - graph_count) / max(flat_count, graph_count) <= 0.10
