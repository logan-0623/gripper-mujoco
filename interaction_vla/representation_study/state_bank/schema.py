from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import re
from typing import Final

from interaction_vla.lerobot_bridge.interaction_phase import PHASE_NAMES

from ..schemas.stages import ArtifactBinding


STATE_BANK_RECORD_SCHEMA_VERSION: Final[str] = "interaction_state_bank_record_v1"
STATE_BANK_MANIFEST_SCHEMA_VERSION: Final[str] = "interaction_state_bank_manifest_v1"
STATE_BANK_STRATA: Final[tuple[str, ...]] = (
    "nominal",
    "perturbation",
    "recovery",
    "terminal",
)
STATE_BANK_DOMAINS: Final[tuple[str, ...]] = ("expert_support", "policy_shift")
STATE_BANK_PARTITIONS: Final[tuple[str, ...]] = ("train", "validation", "test")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("canonical JSON numeric values must be finite")
        return result
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = _text(key, "canonical JSON key")
            if name in result:
                raise ValueError("canonical JSON keys must be unique")
            result[name] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class CanonicalJson:
    text: str

    def __post_init__(self) -> None:
        try:
            value = json.loads(str(self.text))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("canonical JSON text is invalid") from error
        canonical = json.dumps(
            _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        object.__setattr__(self, "text", canonical)

    @classmethod
    def from_value(cls, value: object) -> CanonicalJson:
        normalized = _json_value(value)
        return cls(
            json.dumps(
                normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )

    def to_value(self) -> object:
        return json.loads(self.text)


@dataclass(frozen=True)
class ObservationReference:
    source_uri: str
    source_index: int
    agent_rgb_key: str
    wrist_rgb_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_uri", _text(self.source_uri, "source_uri"))
        object.__setattr__(
            self,
            "source_index",
            _nonnegative_integer(self.source_index, "source_index"),
        )
        object.__setattr__(
            self, "agent_rgb_key", _text(self.agent_rgb_key, "agent_rgb_key")
        )
        object.__setattr__(
            self, "wrist_rgb_key", _text(self.wrist_rgb_key, "wrist_rgb_key")
        )
        if self.agent_rgb_key == self.wrist_rgb_key:
            raise ValueError("agent and wrist RGB keys must differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_uri": self.source_uri,
            "source_index": self.source_index,
            "agent_rgb_key": self.agent_rgb_key,
            "wrist_rgb_key": self.wrist_rgb_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> ObservationReference:
        required = {"source_uri", "source_index", "agent_rgb_key", "wrist_rgb_key"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("observation reference fields are incompatible")
        return cls(
            source_uri=str(value["source_uri"]),
            source_index=value["source_index"],
            agent_rgb_key=str(value["agent_rgb_key"]),
            wrist_rgb_key=str(value["wrist_rgb_key"]),
        )


@dataclass(frozen=True)
class StateBankRecord:
    state_id: str
    source_episode_id: str
    split_group_id: str
    frame_index: int
    task_id: str
    stratum: str
    domain: str
    phase: str
    seed: int
    instruction: str
    observation: ObservationReference
    robot_state: tuple[float, ...]
    privileged_state: CanonicalJson
    ontology_labels: CanonicalJson
    provenance: CanonicalJson
    schema_version: str = STATE_BANK_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_BANK_RECORD_SCHEMA_VERSION:
            raise ValueError("State Bank record schema_version is incompatible")
        for name in (
            "state_id",
            "source_episode_id",
            "split_group_id",
            "task_id",
            "instruction",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "frame_index", _nonnegative_integer(self.frame_index, "frame_index")
        )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        stratum = str(self.stratum).strip()
        domain = str(self.domain).strip()
        phase = str(self.phase).strip()
        if stratum not in STATE_BANK_STRATA:
            raise ValueError(f"unknown State Bank stratum: {stratum}")
        if domain not in STATE_BANK_DOMAINS:
            raise ValueError(f"unknown State Bank domain: {domain}")
        if phase not in PHASE_NAMES:
            raise ValueError(f"unknown interaction phase: {phase}")
        if not isinstance(self.observation, ObservationReference):
            raise ValueError("observation must be an ObservationReference")
        if not isinstance(self.robot_state, Sequence) or isinstance(
            self.robot_state, (str, bytes)
        ):
            raise ValueError("robot_state must be a numeric sequence")
        robot_state = tuple(float(value) for value in self.robot_state)
        if not robot_state or not all(math.isfinite(value) for value in robot_state):
            raise ValueError("robot_state must be non-empty and finite")
        for name in ("privileged_state", "ontology_labels", "provenance"):
            payload = getattr(self, name)
            if not isinstance(payload, CanonicalJson) or not isinstance(
                payload.to_value(), dict
            ):
                raise ValueError(f"{name} must be a canonical JSON object")
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "robot_state", robot_state)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "state_id": self.state_id,
            "source_episode_id": self.source_episode_id,
            "split_group_id": self.split_group_id,
            "frame_index": self.frame_index,
            "task_id": self.task_id,
            "stratum": self.stratum,
            "domain": self.domain,
            "phase": self.phase,
            "seed": self.seed,
            "instruction": self.instruction,
            "observation": self.observation.to_dict(),
            "robot_state": list(self.robot_state),
            "privileged_state": self.privileged_state.to_value(),
            "ontology_labels": self.ontology_labels.to_value(),
            "provenance": self.provenance.to_value(),
        }

    @classmethod
    def from_dict(cls, value: object) -> StateBankRecord:
        required = {
            "schema_version",
            "state_id",
            "source_episode_id",
            "split_group_id",
            "frame_index",
            "task_id",
            "stratum",
            "domain",
            "phase",
            "seed",
            "instruction",
            "observation",
            "robot_state",
            "privileged_state",
            "ontology_labels",
            "provenance",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("State Bank record fields are incompatible")
        return cls(
            schema_version=str(value["schema_version"]),
            state_id=str(value["state_id"]),
            source_episode_id=str(value["source_episode_id"]),
            split_group_id=str(value["split_group_id"]),
            frame_index=value["frame_index"],
            task_id=str(value["task_id"]),
            stratum=str(value["stratum"]),
            domain=str(value["domain"]),
            phase=str(value["phase"]),
            seed=value["seed"],
            instruction=str(value["instruction"]),
            observation=ObservationReference.from_dict(value["observation"]),
            robot_state=tuple(value["robot_state"]),
            privileged_state=CanonicalJson.from_value(value["privileged_state"]),
            ontology_labels=CanonicalJson.from_value(value["ontology_labels"]),
            provenance=CanonicalJson.from_value(value["provenance"]),
        )


def _state_ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} State Bank split must be a sequence")
    result = tuple(_text(item, f"{name} state ID") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} State Bank split must be non-empty and unique")
    return result


@dataclass(frozen=True)
class StateBankSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in STATE_BANK_PARTITIONS:
            object.__setattr__(self, name, _state_ids(getattr(self, name), name))
        all_ids = (*self.train, *self.validation, *self.test)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("State Bank split state IDs must not overlap")

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in STATE_BANK_PARTITIONS}

    @classmethod
    def from_dict(cls, value: object) -> StateBankSplit:
        if not isinstance(value, Mapping) or set(value) != set(STATE_BANK_PARTITIONS):
            raise ValueError("State Bank split fields are incompatible")
        return cls(**{name: tuple(value[name]) for name in STATE_BANK_PARTITIONS})


@dataclass(frozen=True)
class StateBankManifest:
    bank_id: str
    dataset: ArtifactBinding
    ontology: ArtifactBinding
    source: ArtifactBinding
    selection_config: ArtifactBinding
    records_sha256: str
    record_count: int
    state_dim: int
    split: StateBankSplit
    schema_version: str = STATE_BANK_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_BANK_MANIFEST_SCHEMA_VERSION:
            raise ValueError("State Bank manifest schema_version is incompatible")
        object.__setattr__(self, "bank_id", _text(self.bank_id, "bank_id"))
        for name in ("dataset", "ontology", "source", "selection_config"):
            if not isinstance(getattr(self, name), ArtifactBinding):
                raise ValueError(f"{name} must be an ArtifactBinding")
        digest = str(self.records_sha256).strip().lower()
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("records_sha256 must be 64 hexadecimal digits")
        record_count = _nonnegative_integer(self.record_count, "record_count")
        state_dim = _nonnegative_integer(self.state_dim, "state_dim")
        if record_count < 1 or state_dim < 1:
            raise ValueError("record_count and state_dim must be positive")
        if not isinstance(self.split, StateBankSplit):
            raise ValueError("split must be a StateBankSplit")
        if sum(len(getattr(self.split, name)) for name in STATE_BANK_PARTITIONS) != record_count:
            raise ValueError("record_count must match State Bank split entries")
        object.__setattr__(self, "records_sha256", digest)
        object.__setattr__(self, "record_count", record_count)
        object.__setattr__(self, "state_dim", state_dim)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "dataset": self.dataset.to_dict(),
            "ontology": self.ontology.to_dict(),
            "source": self.source.to_dict(),
            "selection_config": self.selection_config.to_dict(),
            "records_sha256": self.records_sha256,
            "record_count": self.record_count,
            "state_dim": self.state_dim,
            "split": self.split.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> StateBankManifest:
        required = {
            "schema_version",
            "bank_id",
            "dataset",
            "ontology",
            "source",
            "selection_config",
            "records_sha256",
            "record_count",
            "state_dim",
            "split",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("State Bank manifest fields are incompatible")
        return cls(
            schema_version=str(value["schema_version"]),
            bank_id=str(value["bank_id"]),
            dataset=ArtifactBinding.from_dict(value["dataset"]),
            ontology=ArtifactBinding.from_dict(value["ontology"]),
            source=ArtifactBinding.from_dict(value["source"]),
            selection_config=ArtifactBinding.from_dict(value["selection_config"]),
            records_sha256=str(value["records_sha256"]),
            record_count=value["record_count"],
            state_dim=value["state_dim"],
            split=StateBankSplit.from_dict(value["split"]),
        )
