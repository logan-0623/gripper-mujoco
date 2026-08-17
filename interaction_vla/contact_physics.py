from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mujoco
import numpy as np

from .franka import OBJECT_NAMES
from .graph.schema import InteractionSignal


class NonFiniteContactForceError(FloatingPointError):
    """MuJoCo returned a contact force that cannot label a valid graph."""


def contact_force_components(force: np.ndarray) -> tuple[float, float]:
    values = np.asarray(force, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")
    normal = abs(float(values[0]))
    tangential = float(np.linalg.norm(values[1:3]))
    if not np.isfinite(normal) or not np.isfinite(tangential):
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")
    return normal, tangential


@dataclass(frozen=True)
class ContactDiagnostics:
    left_objects: frozenset[str]
    right_objects: frozenset[str]
    object_table: frozenset[str]
    object_receptacle: frozenset[str]
    interactions: tuple[InteractionSignal, ...]
    object_receptacle_base: frozenset[str] = frozenset()
    object_receptacle_wall: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GraspState:
    bilateral_object: str | None
    stable_object: str | None
    stable_frames: int
    ever_stable_target: bool
    dropped_target: bool
    ever_stable_wrong_object: bool = False
    wrong_stable_object: str | None = None
    ever_bilateral_contact: bool = False
    first_bilateral_object: str | None = None
    ever_bilateral_target_contact: bool = False
    ever_bilateral_wrong_object: bool = False
    first_bilateral_target_substep: int | None = None
    first_stable_target_substep: int | None = None
    total_stable_target_substeps: int = 0
    longest_stable_target_run: int = 0
    tracker_substep: int = 0
    total_bilateral_target_substeps: int = 0
    total_bilateral_wrong_substeps: int = 0


@dataclass(frozen=True)
class InteractionSubstepEvent:
    substep: int
    bilateral_objects: tuple[str, ...] = ()
    stable_objects: tuple[str, ...] = ()
    dropped_target: bool = False


class StableGraspTracker:
    def __init__(self, *, required_frames: int, lift_height: float) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        if not np.isfinite(lift_height) or lift_height <= 0.0:
            raise ValueError("lift_height must be finite and positive")
        self.required_frames = int(required_frames)
        self.lift_height = float(lift_height)
        self._initialized = False

    def reset(self, *, target_name: str, table_top: float) -> None:
        if not target_name:
            raise ValueError("target_name must not be empty")
        if not np.isfinite(table_top):
            raise ValueError("table_top must be finite")
        self.target_name = target_name
        self.table_top = float(table_top)
        self._stable_frames_by_object: dict[str, int] = {}
        self._ever_stable_target = False
        self._ever_stable_wrong_object = False
        self._ever_bilateral_contact = False
        self._first_bilateral_object: str | None = None
        self._ever_bilateral_target_contact = False
        self._ever_bilateral_wrong_object = False
        self._first_bilateral_target_substep: int | None = None
        self._first_stable_target_substep: int | None = None
        self._total_stable_target_substeps = 0
        self._current_stable_target_run = 0
        self._longest_stable_target_run = 0
        self._tracker_substep = 0
        self._interaction_events: list[InteractionSubstepEvent] = []
        self._total_bilateral_target_substeps = 0
        self._total_bilateral_wrong_substeps = 0
        self._dropped_target = False
        self._initialized = True

    def interaction_events_since(
        self,
        substep: int,
    ) -> tuple[InteractionSubstepEvent, ...]:
        if not self._initialized:
            raise RuntimeError("call reset before reading interaction events")
        if substep < 0:
            raise ValueError("interaction event substep must be non-negative")
        return tuple(
            event
            for event in self._interaction_events
            if event.substep > int(substep)
        )

    def update(
        self,
        contacts: ContactDiagnostics,
        *,
        object_bottom_heights: Mapping[str, float],
    ) -> GraspState:
        if not self._initialized:
            raise RuntimeError("call reset before updating stable grasp state")
        self._tracker_substep += 1
        common = sorted(contacts.left_objects & contacts.right_objects)
        self._ever_bilateral_contact |= bool(common)
        if common and self._first_bilateral_object is None:
            self._first_bilateral_object = (
                self.target_name if self.target_name in common else common[0]
            )
        if self.target_name in common:
            self._total_bilateral_target_substeps += 1
            self._ever_bilateral_target_contact = True
            if self._first_bilateral_target_substep is None:
                self._first_bilateral_target_substep = self._tracker_substep
        if any(name != self.target_name for name in common):
            self._ever_bilateral_wrong_object = True
            self._total_bilateral_wrong_substeps += 1
        lifted = {
            name
            for name in common
            if name in object_bottom_heights
            and float(object_bottom_heights[name])
            >= self.table_top + self.lift_height
        }
        tracked = set(self._stable_frames_by_object) | set(common)
        for name in tracked:
            self._stable_frames_by_object[name] = (
                self._stable_frames_by_object.get(name, 0) + 1
                if name in lifted
                else 0
            )
        stable_objects = sorted(
            name
            for name, frames in self._stable_frames_by_object.items()
            if frames >= self.required_frames
        )
        wrong_stable = next(
            (name for name in stable_objects if name != self.target_name), None
        )
        if wrong_stable is not None:
            self._ever_stable_wrong_object = True
        stable = (
            self.target_name
            if self.target_name in stable_objects
            else (stable_objects[0] if stable_objects else None)
        )
        bilateral = (
            self.target_name
            if self.target_name in common
            else max(
                common,
                key=lambda name: (self._stable_frames_by_object.get(name, 0), name),
                default=None,
            )
        )
        stable_frames = (
            self._stable_frames_by_object.get(stable or bilateral, 0)
            if stable is not None or bilateral is not None
            else 0
        )
        if stable == self.target_name:
            self._ever_stable_target = True
            if self._first_stable_target_substep is None:
                self._first_stable_target_substep = self._tracker_substep
            self._total_stable_target_substeps += 1
            self._current_stable_target_run += 1
            self._longest_stable_target_run = max(
                self._longest_stable_target_run,
                self._current_stable_target_run,
            )
        else:
            self._current_stable_target_run = 0
        dropped_before = self._dropped_target
        if (
            self._ever_stable_target
            and bilateral != self.target_name
            and self.target_name in contacts.object_table
            and self.target_name not in contacts.object_receptacle
        ):
            self._dropped_target = True
        if common or stable_objects or self._dropped_target != dropped_before:
            self._interaction_events.append(
                InteractionSubstepEvent(
                    substep=self._tracker_substep,
                    bilateral_objects=tuple(common),
                    stable_objects=tuple(stable_objects),
                    dropped_target=(
                        self._dropped_target and not dropped_before
                    ),
                )
            )
        return GraspState(
            bilateral_object=bilateral,
            stable_object=stable,
            stable_frames=stable_frames,
            ever_stable_target=self._ever_stable_target,
            dropped_target=self._dropped_target,
            ever_stable_wrong_object=self._ever_stable_wrong_object,
            wrong_stable_object=wrong_stable,
            ever_bilateral_contact=self._ever_bilateral_contact,
            first_bilateral_object=self._first_bilateral_object,
            ever_bilateral_target_contact=self._ever_bilateral_target_contact,
            ever_bilateral_wrong_object=self._ever_bilateral_wrong_object,
            first_bilateral_target_substep=self._first_bilateral_target_substep,
            first_stable_target_substep=self._first_stable_target_substep,
            total_stable_target_substeps=self._total_stable_target_substeps,
            longest_stable_target_run=self._longest_stable_target_run,
            tracker_substep=self._tracker_substep,
            total_bilateral_target_substeps=(
                self._total_bilateral_target_substeps
            ),
            total_bilateral_wrong_substeps=(
                self._total_bilateral_wrong_substeps
            ),
        )


class ContactParser:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.left_body_id = model.body("left_finger").id
        self.right_body_id = model.body("right_finger").id
        self.receptacle_body_id = model.body("receptacle").id
        self.object_body_by_name = {
            name: model.body(name).id for name in OBJECT_NAMES
        }
        self.object_name_by_body = {
            body_id: name for name, body_id in self.object_body_by_name.items()
        }

    def parse(
        self,
        data: mujoco.MjData,
        *,
        stable_object: str | None = None,
    ) -> ContactDiagnostics:
        left_objects: set[str] = set()
        right_objects: set[str] = set()
        object_table: set[str] = set()
        object_receptacle: set[str] = set()
        object_receptacle_base: set[str] = set()
        object_receptacle_wall: set[str] = set()
        aggregates: dict[frozenset[str], list[float]] = {}

        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            first = self._semantic_label(int(contact.geom1))
            second = self._semantic_label(int(contact.geom2))
            if first is None or second is None or first == second:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, data, contact_id, force)
            normal, tangential = contact_force_components(force)

            finger: str | None = None
            object_name: str | None = None
            if first in {"left_finger", "right_finger"} and second.startswith("object_"):
                finger, object_name = first, second
            elif second in {"left_finger", "right_finger"} and first.startswith("object_"):
                finger, object_name = second, first
            if finger is not None and object_name is not None:
                if finger == "left_finger":
                    left_objects.add(object_name)
                else:
                    right_objects.add(object_name)
                self._accumulate(aggregates, "gripper", object_name, normal, tangential)
                continue

            labels = {first, second}
            object_names = [label for label in labels if label.startswith("object_")]
            if len(object_names) != 1:
                continue
            object_name = object_names[0]
            if "table" in labels:
                object_table.add(object_name)
                self._accumulate(aggregates, object_name, "table", normal, tangential)
            elif labels & {"receptacle", "receptacle_base", "receptacle_wall"}:
                object_receptacle.add(object_name)
                if "receptacle_base" in labels:
                    object_receptacle_base.add(object_name)
                if "receptacle_wall" in labels:
                    object_receptacle_wall.add(object_name)
                self._accumulate(
                    aggregates, object_name, "receptacle", normal, tangential
                )

        interactions = tuple(
            InteractionSignal(
                first=names[0],
                second=names[1],
                contact=True,
                normal_force=values[0],
                tangential_force=values[1],
                stable_grasp=(
                    stable_object is not None
                    and frozenset(names) == frozenset(("gripper", stable_object))
                ),
                support=("table" in names or "receptacle" in names),
            )
            for pair, values in sorted(
                aggregates.items(), key=lambda item: tuple(sorted(item[0]))
            )
            for names in (tuple(sorted(pair)),)
        )
        return ContactDiagnostics(
            left_objects=frozenset(left_objects),
            right_objects=frozenset(right_objects),
            object_table=frozenset(object_table),
            object_receptacle=frozenset(object_receptacle),
            interactions=interactions,
            object_receptacle_base=frozenset(object_receptacle_base),
            object_receptacle_wall=frozenset(object_receptacle_wall),
        )

    def _semantic_label(self, geom_id: int) -> str | None:
        geom_name = self.model.geom(geom_id).name
        body_id = int(self.model.geom_bodyid[geom_id])
        if geom_name == "table":
            return "table"
        if geom_name == "receptacle_base":
            return "receptacle_base"
        if geom_name.startswith("receptacle_wall_"):
            return "receptacle_wall"
        if self._is_descendant(body_id, self.left_body_id):
            return "left_finger"
        if self._is_descendant(body_id, self.right_body_id):
            return "right_finger"
        if self._is_descendant(body_id, self.receptacle_body_id):
            return "receptacle"
        return self.object_name_by_body.get(body_id)

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        current = body_id
        while current != 0:
            if current == ancestor_id:
                return True
            current = int(self.model.body_parentid[current])
        return ancestor_id == 0 and body_id == 0

    @staticmethod
    def _accumulate(
        aggregates: dict[frozenset[str], list[float]],
        first: str,
        second: str,
        normal: float,
        tangential: float,
    ) -> None:
        values = aggregates.setdefault(frozenset((first, second)), [0.0, 0.0])
        updated = np.asarray(values, dtype=np.float64) + np.asarray(
            (normal, tangential), dtype=np.float64
        )
        if not np.isfinite(updated).all():
            raise NonFiniteContactForceError(
                "aggregated contact force is non-finite"
            )
        values[0], values[1] = float(updated[0]), float(updated[1])
