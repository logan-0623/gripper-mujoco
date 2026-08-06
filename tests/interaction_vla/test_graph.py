from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from interaction_vla.graph.builder import SceneGraphBuilder
from interaction_vla.graph.schema import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    EntityState,
    SceneSnapshot,
)


def entity(
    name: str,
    entity_type: str,
    position: tuple[float, float, float],
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    movable: bool = False,
    target: bool = False,
    gripper_open: float = 0.0,
) -> EntityState:
    return EntityState(
        name=name,
        entity_type=entity_type,
        position=np.asarray(position, dtype=np.float32),
        orientation=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        linear_velocity=np.asarray(velocity, dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        size=np.asarray((0.04, 0.04, 0.08), dtype=np.float32),
        movable=movable,
        target=target,
        gripper_open=gripper_open,
    )


def example_snapshot(object_count: int = 2) -> SceneSnapshot:
    objects = tuple(
        entity(
            f"object_{index}",
            "object",
            (0.12 * index, 0.05, 0.04),
            velocity=(0.01 * index, 0.0, 0.0),
            movable=True,
            target=index == 0,
        )
        for index in range(object_count)
    )
    return SceneSnapshot(
        gripper=entity("gripper", "gripper", (0.0, 0.0, 0.2), gripper_open=1.0),
        objects=objects,
        receptacle=entity("receptacle", "receptacle", (0.3, -0.2, 0.02)),
        support=entity("table", "support", (0.0, 0.0, 0.0)),
        contacts=frozenset({frozenset(("gripper", "object_0"))}),
        held_object="object_0",
        support_relations=frozenset({("object_1", "table")}) if object_count > 1 else frozenset(),
    )


def edge_row(graph, source: int, destination: int) -> np.ndarray:
    matches = np.flatnonzero(
        (graph.edge_index[0] == source) & (graph.edge_index[1] == destination)
    )
    assert len(matches) == 1
    return graph.edge_features[matches[0]]


def test_builder_creates_complete_directed_masked_graph() -> None:
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(2))

    assert graph.node_features.shape == (8, NODE_FEATURE_DIM)
    assert graph.edge_features.shape == (56, EDGE_FEATURE_DIM)
    assert graph.node_mask.sum() == 5
    assert graph.edge_mask.sum() == 20
    graph.validate()


def test_builder_encodes_current_relations_and_relative_geometry() -> None:
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(2))

    gripper_to_target = edge_row(graph, 0, 1)
    np.testing.assert_allclose(gripper_to_target[:3], (0.0, 0.05, -0.16), atol=1e-6)
    assert gripper_to_target[7] == pytest.approx(1.0)  # contact
    assert gripper_to_target[8] == pytest.approx(1.0)  # holding

    object_to_table = edge_row(graph, 2, 7)
    assert object_to_table[9] == pytest.approx(1.0)


def test_flat_payload_contains_exact_graph_values_and_masks() -> None:
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(3))
    payload = graph.flat_payload()
    node_end = graph.node_features.size
    edge_end = node_end + graph.edge_features.size

    np.testing.assert_array_equal(payload[:node_end], graph.node_features.ravel())
    np.testing.assert_array_equal(payload[node_end:edge_end], graph.edge_features.ravel())
    np.testing.assert_array_equal(
        payload[edge_end : edge_end + graph.node_mask.size], graph.node_mask.astype(np.float32)
    )


def test_validate_rejects_non_finite_features() -> None:
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(2))
    graph.node_features[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        graph.validate()


def test_builder_rejects_too_many_objects() -> None:
    snapshot = example_snapshot(5)
    extra = entity("object_5", "object", (0.4, 0.1, 0.04), movable=True)

    with pytest.raises(ValueError, match="max_objects"):
        SceneGraphBuilder(max_objects=5).build(replace(snapshot, objects=snapshot.objects + (extra,)))


def test_validate_rejects_non_complete_or_self_loop_edge_index() -> None:
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(2))
    graph.edge_index[:, 0] = graph.edge_index[:, 1]

    with pytest.raises(ValueError, match="complete directed"):
        graph.validate()
