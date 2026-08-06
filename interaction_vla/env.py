from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .graph.schema import EntityState, SceneSnapshot


class TerminationReason(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    WRONG_OBJECT = "wrong_object"
    DROPPED = "dropped"
    TIMEOUT = "timeout"
    PHYSICS_FAILURE = "physics_failure"


class LayoutMode(str, Enum):
    NORMAL = "normal"
    CROWDED = "crowded"


@dataclass(frozen=True)
class EnvStep:
    snapshot: SceneSnapshot
    done: bool
    reason: TerminationReason
    info: dict[str, object]


class KinematicTabletopEnv:
    """Fast deterministic manipulation environment for representation experiments.

    Cartesian actions are `(dx, dy, dz, gripper_open)`. A held object is attached
    kinematically below the gripper. This deliberately removes unstable contact
    dynamics from the first representation comparison.
    """

    def __init__(
        self,
        max_objects: int = 5,
        max_steps: int = 120,
        min_object_distance: float = 0.12,
        action_scale: float = 0.04,
        workspace_low: tuple[float, float, float] = (-0.45, -0.35, 0.04),
        workspace_high: tuple[float, float, float] = (0.45, 0.35, 0.55),
        crowded_anchor_min_distance: float = 0.085,
        crowded_anchor_max_distance: float = 0.105,
    ) -> None:
        if max_objects < 2:
            raise ValueError("max_objects must be at least 2")
        self.max_objects = max_objects
        self.max_steps = max_steps
        self.min_object_distance = min_object_distance
        self.action_scale = action_scale
        self.crowded_anchor_min_distance = crowded_anchor_min_distance
        self.crowded_anchor_max_distance = crowded_anchor_max_distance
        if not (
            0.08 <= crowded_anchor_min_distance
            < crowded_anchor_max_distance
            < min_object_distance
        ):
            raise ValueError(
                "crowded anchor distances must satisfy "
                "0.08 <= min < max < min_object_distance"
            )
        self.workspace_low = np.asarray(workspace_low, dtype=np.float32)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float32)
        if (
            self.workspace_low.shape != (3,)
            or self.workspace_high.shape != (3,)
            or not np.isfinite(self.workspace_low).all()
            or not np.isfinite(self.workspace_high).all()
            or np.any(self.workspace_low >= self.workspace_high)
        ):
            raise ValueError("workspace bounds must be finite 3D vectors with low < high")
        self.receptacle_position = np.asarray((0.30, -0.18, 0.02), dtype=np.float32)
        self.receptacle_radius = 0.10
        self.grasp_radius = 0.07
        self.hold_offset = np.asarray((0.0, 0.0, -0.055), dtype=np.float32)
        self._initialized = False

    def reset(
        self,
        seed: int,
        object_count: int,
        target_index: int | None = None,
        layout_mode: LayoutMode | str = LayoutMode.NORMAL,
    ) -> SceneSnapshot:
        if object_count < 2 or object_count > self.max_objects:
            raise ValueError(f"object_count must be between 2 and {self.max_objects}")
        layout_mode = LayoutMode(layout_mode)
        rng = np.random.default_rng(seed)
        if layout_mode is LayoutMode.NORMAL:
            positions = self._sample_normal_positions(rng, object_count)
            if target_index is None:
                target_index = int(rng.integers(0, object_count))
        else:
            if target_index is None:
                target_index = int(rng.integers(0, object_count))
            positions = self._sample_crowded_positions(rng, object_count, target_index)

        if not 0 <= target_index < object_count:
            raise ValueError("target_index is outside the object range")

        self.seed = int(seed)
        self.object_count = object_count
        self.target_index = target_index
        self.layout_mode = layout_mode
        self.object_positions = np.stack(positions)
        self.object_velocities = np.zeros((object_count, 3), dtype=np.float32)
        self.gripper_position = np.clip(
            np.asarray((-0.35, 0.0, 0.28), dtype=np.float32),
            self.workspace_low,
            self.workspace_high,
        )
        self.gripper_velocity = np.zeros(3, dtype=np.float32)
        self.gripper_open = 1.0
        self.held_object: int | None = None
        self.step_count = 0
        self.ever_grasped = False
        self.drop_count = 0
        self._initialized = True
        return self.snapshot()

    def _sample_normal_positions(
        self, rng: np.random.Generator, object_count: int
    ) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        for _ in range(object_count):
            for _attempt in range(500):
                candidate = np.asarray(
                    (rng.uniform(-0.20, 0.12), rng.uniform(-0.25, 0.25), 0.04),
                    dtype=np.float32,
                )
                if all(
                    np.linalg.norm(candidate[:2] - existing[:2]) >= self.min_object_distance
                    for existing in positions
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError(f"could not sample {object_count} non-overlapping objects")
        return positions

    def _sample_crowded_positions(
        self,
        rng: np.random.Generator,
        object_count: int,
        target_index: int,
    ) -> list[np.ndarray]:
        if not 0 <= target_index < object_count:
            raise ValueError("target_index is outside the object range")

        distractor_indices = [index for index in range(object_count) if index != target_index]
        anchor_index = int(rng.choice(distractor_indices))
        spawn_low = np.asarray((-0.20, -0.25), dtype=np.float32)
        spawn_high = np.asarray((0.12, 0.25), dtype=np.float32)

        for _attempt in range(500):
            target_xy = rng.uniform(spawn_low, spawn_high).astype(np.float32)
            radius = float(
                rng.uniform(
                    self.crowded_anchor_min_distance,
                    self.crowded_anchor_max_distance,
                )
            )
            angle = float(rng.uniform(-np.pi, np.pi))
            anchor_xy = target_xy + radius * np.asarray(
                (np.cos(angle), np.sin(angle)), dtype=np.float32
            )
            if np.all(anchor_xy >= spawn_low) and np.all(anchor_xy <= spawn_high):
                break
        else:
            raise RuntimeError("could not sample a crowded target-anchor pair")

        positions_by_index = {
            target_index: np.asarray((*target_xy, 0.04), dtype=np.float32),
            anchor_index: np.asarray((*anchor_xy, 0.04), dtype=np.float32),
        }
        for index in range(object_count):
            if index in positions_by_index:
                continue
            for _attempt in range(500):
                candidate = np.asarray(
                    (rng.uniform(-0.20, 0.12), rng.uniform(-0.25, 0.25), 0.04),
                    dtype=np.float32,
                )
                if all(
                    np.linalg.norm(candidate[:2] - existing[:2]) >= self.min_object_distance
                    for existing in positions_by_index.values()
                ):
                    positions_by_index[index] = candidate
                    break
            else:
                raise RuntimeError(f"could not sample {object_count} crowded objects")

        return [positions_by_index[index] for index in range(object_count)]

    def set_gripper_position(self, position: np.ndarray) -> None:
        self._require_initialized()
        value = np.asarray(position, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("gripper position must be a finite vector with shape (3,)")
        self.gripper_position = np.clip(value, self.workspace_low, self.workspace_high)
        self.gripper_velocity = np.zeros(3, dtype=np.float32)

    def perturb_gripper_state(
        self,
        delta: np.ndarray,
        gripper_open: float | None = None,
    ) -> SceneSnapshot:
        """Apply an instantaneous, deterministic state perturbation without stepping time."""

        self._require_initialized()
        value = np.asarray(delta, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("gripper perturbation must be a finite vector with shape (3,)")
        if gripper_open is not None and not np.isfinite(gripper_open):
            raise ValueError("gripper_open must be finite")

        previous_gripper = self.gripper_position.copy()
        self.gripper_position = np.clip(
            self.gripper_position + value,
            self.workspace_low,
            self.workspace_high,
        )
        self.gripper_velocity = (self.gripper_position - previous_gripper).astype(
            np.float32, copy=True
        )
        if gripper_open is not None:
            self.gripper_open = float(np.clip(gripper_open, 0.0, 1.0))

        self.object_velocities *= 0.0
        if self.held_object is not None:
            held = self.held_object
            previous_object = self.object_positions[held].copy()
            self.object_positions[held] = self.gripper_position + self.hold_offset
            self.object_velocities[held] = self.object_positions[held] - previous_object
        return self.snapshot()

    def step(self, action: np.ndarray) -> EnvStep:
        self._require_initialized()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError("action must have shape (4,)")
        if not np.isfinite(action).all():
            raise ValueError("action must be finite")

        self.step_count += 1
        delta = np.clip(action[:3], -self.action_scale, self.action_scale)
        self.gripper_position = np.clip(
            self.gripper_position + delta, self.workspace_low, self.workspace_high
        )
        self.gripper_velocity = delta.astype(np.float32, copy=True)
        requested_open = float(np.clip(action[3], 0.0, 1.0))

        released_away_from_goal = False
        if requested_open >= 0.5 and self.held_object is not None:
            held = self.held_object
            self.object_positions[held] = self.gripper_position + self.hold_offset
            self.object_positions[held, 2] = 0.04
            released_away_from_goal = not self._is_over_receptacle(self.object_positions[held])
            self.held_object = None
            if released_away_from_goal:
                self.drop_count += 1

        if requested_open < 0.5 and self.held_object is None:
            distances = np.linalg.norm(self.object_positions - self.gripper_position, axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= self.grasp_radius:
                self.held_object = nearest
                self.ever_grasped = True

        self.gripper_open = requested_open
        if self.held_object is not None:
            held = self.held_object
            previous = self.object_positions[held].copy()
            self.object_positions[held] = self.gripper_position + self.hold_offset
            self.object_velocities[held] = self.object_positions[held] - previous
        for index in range(self.object_count):
            if index != self.held_object:
                self.object_velocities[index] *= 0.0

        reason = TerminationReason.RUNNING
        done = False
        if self.held_object is not None and self.held_object != self.target_index:
            reason, done = TerminationReason.WRONG_OBJECT, True
        elif released_away_from_goal:
            reason, done = TerminationReason.DROPPED, True
        elif self._is_success():
            reason, done = TerminationReason.SUCCESS, True
        elif self.step_count >= self.max_steps:
            reason, done = TerminationReason.TIMEOUT, True

        return EnvStep(
            snapshot=self.snapshot(),
            done=done,
            reason=reason,
            info={
                "step_count": self.step_count,
                "ever_grasped": self.ever_grasped,
                "drop_count": self.drop_count,
                "target_index": self.target_index,
            },
        )

    def proprioception(self) -> np.ndarray:
        self._require_initialized()
        return np.concatenate(
            (self.gripper_position, self.gripper_velocity, np.asarray((self.gripper_open,), dtype=np.float32))
        ).astype(np.float32)

    def snapshot(self) -> SceneSnapshot:
        self._require_initialized()
        objects = tuple(
            self._entity(
                name=f"object_{index}",
                entity_type="object",
                position=self.object_positions[index],
                velocity=self.object_velocities[index],
                size=(0.04, 0.04, 0.08),
                movable=True,
                target=index == self.target_index,
            )
            for index in range(self.object_count)
        )
        contacts: set[frozenset[str]] = set()
        support_relations: set[tuple[str, str]] = set()
        for index, entity in enumerate(objects):
            if index == self.held_object:
                contacts.add(frozenset(("gripper", entity.name)))
                continue
            support = "receptacle" if self._is_over_receptacle(entity.position) else "table"
            support_relations.add((entity.name, support))
            contacts.add(frozenset((entity.name, support)))
            if np.linalg.norm(entity.position - self.gripper_position) <= self.grasp_radius:
                contacts.add(frozenset(("gripper", entity.name)))

        return SceneSnapshot(
            gripper=self._entity(
                name="gripper",
                entity_type="gripper",
                position=self.gripper_position,
                velocity=self.gripper_velocity,
                size=(0.035, 0.035, 0.06),
                gripper_open=self.gripper_open,
            ),
            objects=objects,
            receptacle=self._entity(
                name="receptacle",
                entity_type="receptacle",
                position=self.receptacle_position,
                velocity=(0.0, 0.0, 0.0),
                size=(self.receptacle_radius, self.receptacle_radius, 0.02),
            ),
            support=self._entity(
                name="table",
                entity_type="support",
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                size=(0.5, 0.4, 0.02),
            ),
            contacts=frozenset(contacts),
            held_object=None if self.held_object is None else f"object_{self.held_object}",
            support_relations=frozenset(support_relations),
        )

    def _is_over_receptacle(self, position: np.ndarray) -> bool:
        return bool(np.linalg.norm(position[:2] - self.receptacle_position[:2]) <= self.receptacle_radius)

    def _is_success(self) -> bool:
        target = self.object_positions[self.target_index]
        return bool(
            self.held_object is None
            and self.gripper_open >= 0.5
            and self.gripper_position[2] >= 0.18
            and self._is_over_receptacle(target)
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("call reset before using the environment")

    @staticmethod
    def _entity(
        *,
        name: str,
        entity_type: str,
        position: np.ndarray | tuple[float, float, float],
        velocity: np.ndarray | tuple[float, float, float],
        size: tuple[float, float, float],
        movable: bool = False,
        target: bool = False,
        gripper_open: float = 0.0,
    ) -> EntityState:
        return EntityState(
            name=name,
            entity_type=entity_type,
            position=np.asarray(position, dtype=np.float32).copy(),
            orientation=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            linear_velocity=np.asarray(velocity, dtype=np.float32).copy(),
            angular_velocity=np.zeros(3, dtype=np.float32),
            size=np.asarray(size, dtype=np.float32),
            movable=movable,
            target=target,
            gripper_open=gripper_open,
        )
