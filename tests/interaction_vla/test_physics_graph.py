from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph.builder import SceneGraphBuilder
from interaction_vla.graph.schema import (
    PHYSICS_EDGE_FEATURE_DIM,
    EntityState,
    InteractionSignal,
    SceneSnapshot,
)


def entity(
    name: str,
    entity_type: str,
    position: tuple[float, float, float],
    *,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target: bool = False,
    movable: bool = False,
) -> EntityState:
    return EntityState(
        name=name,
        entity_type=entity_type,
        position=np.asarray(position),
        orientation=np.asarray(orientation),
        linear_velocity=np.asarray(linear_velocity),
        angular_velocity=np.asarray(angular_velocity),
        size=np.asarray((0.04, 0.04, 0.04)),
        target=target,
        movable=movable,
    )


def physical_snapshot() -> SceneSnapshot:
    half_sqrt = np.sqrt(0.5)
    return SceneSnapshot(
        gripper=entity(
            "gripper",
            "gripper",
            (0.0, 0.0, 0.0),
            linear_velocity=(0.1, 0.2, 0.3),
            angular_velocity=(0.2, 0.1, 0.0),
        ),
        objects=(
            entity(
                "object_0",
                "object",
                (1.0, 2.0, 3.0),
                orientation=(half_sqrt, 0.0, 0.0, half_sqrt),
                linear_velocity=(0.4, 0.6, 0.8),
                angular_velocity=(0.5, 0.4, 0.3),
                target=True,
                movable=True,
            ),
        ),
        receptacle=entity("receptacle", "receptacle", (0.5, -0.2, 0.0)),
        support=entity("table", "support", (0.0, 0.0, -0.1)),
        interactions=(
            InteractionSignal(
                "gripper",
                "object_0",
                contact=True,
                normal_force=2.0,
                tangential_force=0.5,
                stable_grasp=True,
            ),
            InteractionSignal(
                "object_0",
                "table",
                contact=True,
                normal_force=1.0,
                support=True,
            ),
        ),
    )


def edge_row(graph, source: int, destination: int) -> np.ndarray:
    matches = np.flatnonzero(
        (graph.edge_index[0] == source) & (graph.edge_index[1] == destination)
    )
    assert len(matches) == 1
    return graph.edge_features[matches[0]]


def test_physics_graph_has_the_exact_18d_interaction_payload() -> None:
    graph = SceneGraphBuilder(max_objects=5, feature_schema="physics_v2").build(
        physical_snapshot()
    )

    assert graph.feature_schema == "physics_v2"
    assert graph.edge_features.shape == (56, PHYSICS_EDGE_FEATURE_DIM)
    row = edge_row(graph, 0, 1)
    np.testing.assert_allclose(row[0:3], (1.0, 2.0, 3.0), atol=1e-6)
    np.testing.assert_allclose(row[3:6], (0.0, 0.0, np.pi / 2), atol=1e-6)
    np.testing.assert_allclose(row[6:9], (0.3, 0.4, 0.5), atol=1e-6)
    np.testing.assert_allclose(row[9:12], (0.3, 0.3, 0.3), atol=1e-6)
    assert row[12] == pytest.approx(np.sqrt(14.0))
    np.testing.assert_allclose(row[13:18], (1.0, 2.0, 0.5, 1.0, 0.0))
    assert edge_row(graph, 1, 7)[17] == 1.0


def test_physics_flat_payload_contains_the_exact_same_graph_values() -> None:
    graph = SceneGraphBuilder(max_objects=5, feature_schema="physics_v2").build(
        physical_snapshot()
    )
    payload = graph.flat_payload()
    node_end = graph.node_features.size
    edge_end = node_end + graph.edge_features.size

    np.testing.assert_array_equal(payload[:node_end], graph.node_features.ravel())
    np.testing.assert_array_equal(payload[node_end:edge_end], graph.edge_features.ravel())
    np.testing.assert_array_equal(
        payload[edge_end : edge_end + graph.node_mask.size],
        graph.node_mask.astype(np.float32),
    )


def test_unknown_edge_feature_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="feature_schema"):
        SceneGraphBuilder(max_objects=5, feature_schema="unknown")
