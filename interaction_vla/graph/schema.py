from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NODE_TYPES = ("gripper", "object", "receptacle", "support")
NODE_FEATURE_DIM = 23
LEGACY_EDGE_FEATURE_DIM = 10
PHYSICS_EDGE_FEATURE_DIM = 18
EDGE_SCHEMAS = {
    "kinematic_v1": LEGACY_EDGE_FEATURE_DIM,
    "physics_v2": PHYSICS_EDGE_FEATURE_DIM,
}
EDGE_FEATURE_DIM = LEGACY_EDGE_FEATURE_DIM


@dataclass(frozen=True)
class EntityState:
    name: str
    entity_type: str
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    size: np.ndarray
    movable: bool = False
    target: bool = False
    gripper_open: float = 0.0

    def __post_init__(self) -> None:
        if self.entity_type not in NODE_TYPES:
            raise ValueError(f"unknown entity_type: {self.entity_type}")
        for field_name, expected_shape in (
            ("position", (3,)),
            ("orientation", (4,)),
            ("linear_velocity", (3,)),
            ("angular_velocity", (3,)),
            ("size", (3,)),
        ):
            value = np.asarray(getattr(self, field_name), dtype=np.float32)
            if value.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape {expected_shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class InteractionSignal:
    first: str
    second: str
    contact: bool = False
    normal_force: float = 0.0
    tangential_force: float = 0.0
    stable_grasp: bool = False
    support: bool = False

    def __post_init__(self) -> None:
        if not self.first or not self.second or self.first == self.second:
            raise ValueError("interaction names must be non-empty and distinct")
        for name in ("normal_force", "tangential_force"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("interaction force must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.first, self.second))


@dataclass(frozen=True)
class SceneSnapshot:
    gripper: EntityState
    objects: tuple[EntityState, ...]
    receptacle: EntityState
    support: EntityState
    contacts: frozenset[frozenset[str]] = frozenset()
    held_object: str | None = None
    support_relations: frozenset[tuple[str, str]] = frozenset()
    interactions: tuple[InteractionSignal, ...] = ()

    @property
    def target_object(self) -> EntityState:
        targets = [entity for entity in self.objects if entity.target]
        if len(targets) != 1:
            raise ValueError("snapshot must contain exactly one target object")
        return targets[0]


@dataclass
class SceneGraph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    entity_names: tuple[str | None, ...]
    feature_schema: str = "kinematic_v1"

    def validate(self) -> None:
        max_nodes = len(self.entity_names)
        max_edges = max_nodes * (max_nodes - 1)
        if self.feature_schema not in EDGE_SCHEMAS:
            raise ValueError(f"unknown feature_schema: {self.feature_schema}")
        expected = {
            "node_features": (max_nodes, NODE_FEATURE_DIM),
            "edge_index": (2, max_edges),
            "edge_features": (max_edges, EDGE_SCHEMAS[self.feature_schema]),
            "node_mask": (max_nodes,),
            "edge_mask": (max_edges,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(self.node_features).all() or not np.isfinite(self.edge_features).all():
            raise ValueError("scene graph features must be finite")
        if self.node_mask.dtype != np.bool_ or self.edge_mask.dtype != np.bool_:
            raise ValueError("scene graph masks must be boolean")
        if np.any(self.edge_index < 0) or np.any(self.edge_index >= max_nodes):
            raise ValueError("edge_index contains an invalid node index")
        expected_edge_index = np.asarray(
            [
                (source, destination)
                for source in range(max_nodes)
                for destination in range(max_nodes)
                if source != destination
            ],
            dtype=np.int64,
        ).T
        if not np.array_equal(self.edge_index, expected_edge_index):
            raise ValueError(
                "edge_index must be the canonical complete directed graph without self-loops"
            )
        expected_edge_mask = self.node_mask[self.edge_index[0]] & self.node_mask[self.edge_index[1]]
        if not np.array_equal(expected_edge_mask, self.edge_mask):
            raise ValueError("edge_mask is inconsistent with node_mask")

    def flat_payload(self) -> np.ndarray:
        self.validate()
        return np.concatenate(
            (
                self.node_features.reshape(-1),
                self.edge_features.reshape(-1),
                self.node_mask.astype(np.float32),
                self.edge_mask.astype(np.float32),
            )
        ).astype(np.float32, copy=False)
