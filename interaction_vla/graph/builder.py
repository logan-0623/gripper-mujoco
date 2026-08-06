from __future__ import annotations

import numpy as np

from .schema import (
    EDGE_SCHEMAS,
    LEGACY_EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NODE_TYPES,
    EntityState,
    SceneGraph,
    SceneSnapshot,
)


class SceneGraphBuilder:
    def __init__(self, max_objects: int = 5, feature_schema: str = "kinematic_v1"):
        if max_objects < 1:
            raise ValueError("max_objects must be positive")
        if feature_schema not in EDGE_SCHEMAS:
            raise ValueError(f"unknown feature_schema: {feature_schema}")
        self.max_objects = max_objects
        self.feature_schema = feature_schema
        self.edge_feature_dim = EDGE_SCHEMAS[feature_schema]
        self.max_nodes = max_objects + 3
        self.max_edges = self.max_nodes * (self.max_nodes - 1)
        edges = [(source, destination) for source in range(self.max_nodes) for destination in range(self.max_nodes) if source != destination]
        self.edge_index = np.asarray(edges, dtype=np.int64).T

    def build(self, snapshot: SceneSnapshot) -> SceneGraph:
        if len(snapshot.objects) > self.max_objects:
            raise ValueError(
                f"snapshot has {len(snapshot.objects)} objects, greater than max_objects={self.max_objects}"
            )
        snapshot.target_object

        entities: list[EntityState | None] = [snapshot.gripper]
        entities.extend(sorted(snapshot.objects, key=lambda entity: entity.name))
        entities.extend([None] * (self.max_objects - len(snapshot.objects)))
        entities.extend((snapshot.receptacle, snapshot.support))

        node_features = np.zeros((self.max_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        node_mask = np.zeros(self.max_nodes, dtype=np.bool_)
        names: list[str | None] = []
        for index, entity in enumerate(entities):
            names.append(None if entity is None else entity.name)
            if entity is None:
                continue
            node_features[index] = self._node_features(entity)
            node_mask[index] = True

        edge_features = np.zeros(
            (self.max_edges, self.edge_feature_dim), dtype=np.float32
        )
        edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
        for edge_id in np.flatnonzero(edge_mask):
            source = entities[int(self.edge_index[0, edge_id])]
            destination = entities[int(self.edge_index[1, edge_id])]
            assert source is not None and destination is not None
            edge_features[edge_id] = self._edge_features(source, destination, snapshot)

        graph = SceneGraph(
            node_features=node_features,
            edge_index=self.edge_index.copy(),
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask.astype(np.bool_, copy=False),
            entity_names=tuple(names),
            feature_schema=self.feature_schema,
        )
        graph.validate()
        return graph

    @staticmethod
    def _node_features(entity: EntityState) -> np.ndarray:
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        features[NODE_TYPES.index(entity.entity_type)] = 1.0
        features[4:7] = entity.position
        features[7:11] = entity.orientation
        features[11:14] = entity.linear_velocity
        features[14:17] = entity.angular_velocity
        features[17:20] = entity.size
        features[20] = float(entity.gripper_open)
        features[21] = float(entity.movable)
        features[22] = float(entity.target)
        return features

    def _edge_features(
        self, source: EntityState, destination: EntityState, snapshot: SceneSnapshot
    ) -> np.ndarray:
        if self.feature_schema == "physics_v2":
            return self._physics_edge_features(source, destination, snapshot)
        return self._legacy_edge_features(source, destination, snapshot)

    @staticmethod
    def _legacy_edge_features(
        source: EntityState, destination: EntityState, snapshot: SceneSnapshot
    ) -> np.ndarray:
        relative_position = destination.position - source.position
        relative_velocity = destination.linear_velocity - source.linear_velocity
        features = np.zeros(LEGACY_EDGE_FEATURE_DIM, dtype=np.float32)
        features[:3] = relative_position
        features[3:6] = relative_velocity
        features[6] = np.linalg.norm(relative_position)
        features[7] = float(frozenset((source.name, destination.name)) in snapshot.contacts)
        features[8] = float(
            source.entity_type == "gripper" and destination.name == snapshot.held_object
        )
        features[9] = float((source.name, destination.name) in snapshot.support_relations)
        return features

    @staticmethod
    def _physics_edge_features(
        source: EntityState, destination: EntityState, snapshot: SceneSnapshot
    ) -> np.ndarray:
        relative_position = destination.position - source.position
        relative_linear_velocity = (
            destination.linear_velocity - source.linear_velocity
        )
        relative_angular_velocity = (
            destination.angular_velocity - source.angular_velocity
        )
        relative_rotation = SceneGraphBuilder._relative_rotation_vector(
            source.orientation, destination.orientation
        )
        signal_by_pair = {signal.key: signal for signal in snapshot.interactions}
        signal = signal_by_pair.get(frozenset((source.name, destination.name)))

        features = np.zeros(EDGE_SCHEMAS["physics_v2"], dtype=np.float32)
        features[0:3] = relative_position
        features[3:6] = relative_rotation
        features[6:9] = relative_linear_velocity
        features[9:12] = relative_angular_velocity
        features[12] = np.linalg.norm(relative_position)
        if signal is not None:
            features[13] = float(signal.contact)
            features[14] = signal.normal_force
            features[15] = signal.tangential_force
            features[16] = float(
                signal.stable_grasp
                and source.entity_type == "gripper"
                and destination.entity_type == "object"
            )
            features[17] = float(
                signal.support
                and source.entity_type == "object"
                and destination.entity_type in {"support", "receptacle"}
            )
        return features

    @staticmethod
    def _relative_rotation_vector(
        source_quaternion: np.ndarray, destination_quaternion: np.ndarray
    ) -> np.ndarray:
        source = SceneGraphBuilder._quaternion_matrix(source_quaternion)
        destination = SceneGraphBuilder._quaternion_matrix(destination_quaternion)
        rotation = source.T @ destination
        cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        vee = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        )
        if angle < 1e-8:
            return (0.5 * vee).astype(np.float32)
        if np.pi - angle < 1e-6:
            diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
            axis = np.sqrt(diagonal)
            for index in range(3):
                if abs(vee[index]) > 1e-12:
                    axis[index] = np.copysign(axis[index], vee[index])
            norm = float(np.linalg.norm(axis))
            return (angle * axis / max(norm, 1e-12)).astype(np.float32)
        return (angle * vee / (2.0 * np.sin(angle))).astype(np.float32)

    @staticmethod
    def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
        value = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if value.shape != (4,) or not np.isfinite(value).all() or norm < 1e-12:
            raise ValueError("orientation quaternion must be finite and non-zero")
        w, x, y, z = value / norm
        return np.asarray(
            (
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
