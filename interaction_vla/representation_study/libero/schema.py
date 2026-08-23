from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


STATE_BANK_SCHEMA = "libero_interaction_state_bank_v1"
FACTORS = ("entity", "geometry", "contact", "stable_grasp", "phase", "next_relation")
PHASES = (
    "approach",
    "align_precontact",
    "contact",
    "secure",
    "actuate",
    "lift",
    "transport",
    "place",
    "release_retreat",
)
RELATION_PREDICATES = (
    "near",
    "contact",
    "stable_grasp",
    "off_support",
    "on",
    "inside",
    "open",
    "closed",
    "powered_on",
    "powered_off",
    "clearance",
)
RELATION_OPERATORS = ("establish", "maintain", "increase", "decrease", "clear")


def _finite(values: tuple[float, ...] | list[float], name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in values)
    if not np.isfinite(np.asarray(result, dtype=np.float64)).all():
        raise ValueError(f"{name} must be finite")
    return result


def _sha(value: str, name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class FactorApplicability:
    entity: bool
    geometry: bool
    contact: bool
    stable_grasp: bool
    phase: bool
    next_relation: bool

    @classmethod
    def all_applicable(cls) -> "FactorApplicability":
        return cls(**{factor: True for factor in FACTORS})

    def to_dict(self) -> dict[str, bool]:
        return {factor: bool(getattr(self, factor)) for factor in FACTORS}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FactorApplicability":
        if set(value) != set(FACTORS):
            raise ValueError(f"applicability must contain exactly {FACTORS}")
        return cls(**{factor: bool(value[factor]) for factor in FACTORS})


@dataclass(frozen=True)
class EntityLabels:
    target: str
    goal: str | None
    source: str | None
    distractors: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityLabels":
        return cls(
            target=str(value["target"]),
            goal=None if value.get("goal") is None else str(value["goal"]),
            source=None if value.get("source") is None else str(value["source"]),
            distractors=tuple(str(item) for item in value.get("distractors", ())),
        )


@dataclass(frozen=True)
class GeometryLabels:
    gripper_to_target: tuple[float, ...]
    target_to_goal: tuple[float, ...]
    gripper_target_distance: float
    target_goal_distance: float

    def __post_init__(self) -> None:
        if len(_finite(self.gripper_to_target, "gripper_to_target")) != 9:
            raise ValueError("gripper_to_target must contain translation3 + rotation6D")
        if len(_finite(self.target_to_goal, "target_to_goal")) != 9:
            raise ValueError("target_to_goal must contain translation3 + rotation6D")
        _finite(
            (self.gripper_target_distance, self.target_goal_distance),
            "geometry distances",
        )
        if self.gripper_target_distance < 0 or self.target_goal_distance < 0:
            raise ValueError("geometry distances must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeometryLabels":
        return cls(
            gripper_to_target=_finite(value["gripper_to_target"], "gripper_to_target"),
            target_to_goal=_finite(value["target_to_goal"], "target_to_goal"),
            gripper_target_distance=float(value["gripper_target_distance"]),
            target_goal_distance=float(value["target_goal_distance"]),
        )


@dataclass(frozen=True)
class ContactLabels:
    gripper_target: bool
    target_goal: bool
    target_source: bool
    finger_groups: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContactLabels":
        return cls(
            gripper_target=bool(value["gripper_target"]),
            target_goal=bool(value["target_goal"]),
            target_source=bool(value["target_source"]),
            finger_groups=tuple(str(item) for item in value.get("finger_groups", ())),
        )


@dataclass(frozen=True)
class NextRelation:
    active_goal_index: int
    subject_role: str
    predicate: str
    object_role: str
    operator: str

    def __post_init__(self) -> None:
        if self.active_goal_index < 0:
            raise ValueError("active_goal_index must be non-negative")
        if self.predicate not in RELATION_PREDICATES:
            raise ValueError(f"unknown next-relation predicate: {self.predicate}")
        if self.operator not in RELATION_OPERATORS:
            raise ValueError(f"unknown next-relation operator: {self.operator}")
        if not self.subject_role or not self.object_role:
            raise ValueError("next-relation roles must be non-empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextRelation":
        return cls(
            active_goal_index=int(value["active_goal_index"]),
            subject_role=str(value["subject_role"]),
            predicate=str(value["predicate"]),
            object_role=str(value["object_role"]),
            operator=str(value["operator"]),
        )


@dataclass(frozen=True)
class InteractionLabels:
    applicability: FactorApplicability
    entity: EntityLabels | None
    geometry: GeometryLabels | None
    contact: ContactLabels | None
    stable_grasp: bool | None
    phase: str | None
    next_relation: NextRelation | None

    def __post_init__(self) -> None:
        for factor in FACTORS:
            applicable = bool(getattr(self.applicability, factor))
            value = getattr(self, factor)
            if not applicable and value is not None:
                raise ValueError(f"masked factor {factor} must be null")
            if applicable and value is None:
                raise ValueError(f"applicable factor {factor} must have a label")
        if self.phase is not None and self.phase not in PHASES:
            raise ValueError(f"unknown phase: {self.phase}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicability": self.applicability.to_dict(),
            "entity": None if self.entity is None else asdict(self.entity),
            "geometry": None if self.geometry is None else asdict(self.geometry),
            "contact": None if self.contact is None else asdict(self.contact),
            "stable_grasp": self.stable_grasp,
            "phase": self.phase,
            "next_relation": (
                None if self.next_relation is None else asdict(self.next_relation)
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InteractionLabels":
        if "recovery" in value:
            raise ValueError("nominal LIBERO labels must not contain recovery")
        applicability = FactorApplicability.from_dict(value["applicability"])
        entity = value.get("entity")
        geometry = value.get("geometry")
        contact = value.get("contact")
        relation = value.get("next_relation")
        return cls(
            applicability=applicability,
            entity=None if entity is None else EntityLabels.from_dict(entity),
            geometry=None if geometry is None else GeometryLabels.from_dict(geometry),
            contact=None if contact is None else ContactLabels.from_dict(contact),
            stable_grasp=(
                None if value.get("stable_grasp") is None else bool(value["stable_grasp"])
            ),
            phase=None if value.get("phase") is None else str(value["phase"]),
            next_relation=(
                None if relation is None else NextRelation.from_dict(relation)
            ),
        )


@dataclass(frozen=True)
class ObservationReference:
    dataset_index: int
    global_rgb_key: str
    wrist_rgb_key: str | None
    robot_state: tuple[float, ...]
    action: tuple[float, ...]
    timestamp: float

    def __post_init__(self) -> None:
        if self.dataset_index < 0:
            raise ValueError("dataset_index must be non-negative")
        if len(_finite(self.robot_state, "robot_state")) != 8:
            raise ValueError("robot_state must have LIBERO dimension 8")
        if len(_finite(self.action, "action")) != 7:
            raise ValueError("action must have LIBERO dimension 7")
        _finite((self.timestamp,), "timestamp")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationReference":
        return cls(
            dataset_index=int(value["dataset_index"]),
            global_rgb_key=str(value["global_rgb_key"]),
            wrist_rgb_key=(
                None if value.get("wrist_rgb_key") is None else str(value["wrist_rgb_key"])
            ),
            robot_state=_finite(value["robot_state"], "robot_state"),
            action=_finite(value["action"], "action"),
            timestamp=float(value["timestamp"]),
        )


@dataclass(frozen=True)
class ReplayReference:
    hdf5_relative_path: str
    demo_key: str
    simulator_state_index: int
    action_index: int
    model_xml_sha256: str
    initial_state_sha256: str

    def __post_init__(self) -> None:
        if self.simulator_state_index < 0 or self.action_index < 0:
            raise ValueError("replay row indices must be non-negative")
        _sha(self.model_xml_sha256, "model_xml_sha256")
        _sha(self.initial_state_sha256, "initial_state_sha256")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayReference":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class StateRecord:
    state_id: str
    suite: str
    task_id: int
    task_name: str
    language: str
    source_episode_id: str
    lerobot_episode_index: int
    frame_index: int
    simulator_seed: int | None
    observation: ObservationReference
    replay: ReplayReference
    labels: InteractionLabels
    source_revision: str
    annotator_sha256: str
    schema_version: str = STATE_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATE_BANK_SCHEMA:
            raise ValueError(f"schema_version must be {STATE_BANK_SCHEMA}")
        if not self.state_id.startswith(f"libero:{self.suite}:{self.task_id}:"):
            raise ValueError("state_id is inconsistent with suite/task")
        if min(self.task_id, self.lerobot_episode_index, self.frame_index) < 0:
            raise ValueError("task, episode, and frame indices must be non-negative")
        if not self.language or not self.source_episode_id or not self.source_revision:
            raise ValueError("language and source identities must be non-empty")
        _sha(self.annotator_sha256, "annotator_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_id": self.state_id,
            "suite": self.suite,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "language": self.language,
            "source_episode_id": self.source_episode_id,
            "lerobot_episode_index": self.lerobot_episode_index,
            "frame_index": self.frame_index,
            "simulator_seed": self.simulator_seed,
            "observation": asdict(self.observation),
            "replay": asdict(self.replay),
            "labels": self.labels.to_dict(),
            "source_revision": self.source_revision,
            "annotator_sha256": self.annotator_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateRecord":
        return cls(
            schema_version=str(value["schema_version"]),
            state_id=str(value["state_id"]),
            suite=str(value["suite"]),
            task_id=int(value["task_id"]),
            task_name=str(value["task_name"]),
            language=str(value["language"]),
            source_episode_id=str(value["source_episode_id"]),
            lerobot_episode_index=int(value["lerobot_episode_index"]),
            frame_index=int(value["frame_index"]),
            simulator_seed=(
                None if value.get("simulator_seed") is None else int(value["simulator_seed"])
            ),
            observation=ObservationReference.from_dict(value["observation"]),
            replay=ReplayReference.from_dict(value["replay"]),
            labels=InteractionLabels.from_dict(value["labels"]),
            source_revision=str(value["source_revision"]),
            annotator_sha256=str(value["annotator_sha256"]),
        )
