from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from interaction_vla.env import TerminationReason
from interaction_vla.lerobot_bridge.interaction_phase import PHASE_NAMES

from .schema import ALL_CONDITIONS, TOKEN_DIM, TOKEN_SLICES, validate_token
from .sensitivity import CATEGORICAL_GROUPS


TRACE_SCHEMA_VERSION: Final[str] = "graph_control_step_trace_v1"

_TRACE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "trace_schema_version",
        "episode_id",
        "case_id",
        "environment_seed",
        "condition",
        "policy_seed",
        "layout",
        "object_count",
        "training_distribution",
        "step",
        "phase",
        "policy_token",
        "teacher_token",
        "graph_error_by_group",
        "raw_action",
        "clipped_action",
        "executed_world_action",
        "action_was_clipped",
        "ik_projection_scale",
        "gripper_command",
        "end_effector_position",
        "end_effector_orientation",
        "target_relative_position",
        "receptacle_relative_position",
        "minimum_distractor_clearance",
        "target_contact",
        "stable_target_grasp",
        "wrong_object_contact",
        "stable_wrong_object_grasp",
        "events",
        "done",
        "termination_reason",
        "success",
        "target_drop",
        "timeout",
        "gripper_switch_count",
        "checkpoint",
    }
)


def _categorical_label(values: np.ndarray) -> int | None:
    if float(np.sum(np.clip(values, 0.0, None))) <= 1.0e-12:
        return None
    return int(np.argmax(values))


def graph_group_errors(
    policy_token: object, teacher_token: object, *, condition: str
) -> dict[str, dict[str, object]]:
    if condition not in ALL_CONDITIONS:
        raise ValueError("trace condition is invalid")
    policy = validate_token(policy_token).astype(np.float64)
    teacher = validate_token(teacher_token).astype(np.float64)
    result: dict[str, dict[str, object]] = {}
    for group, bounds in TOKEN_SLICES.items():
        if group in CATEGORICAL_GROUPS:
            policy_label = _categorical_label(policy[bounds])
            teacher_label = _categorical_label(teacher[bounds])
            result[group] = {
                "kind": "categorical",
                "agreement": (
                    None
                    if policy_label is None and teacher_label is None
                    else policy_label == teacher_label
                ),
                "policy_label": policy_label,
                "teacher_label": teacher_label,
            }
            continue
        difference = policy[bounds] - teacher[bounds]
        result[group] = {
            "kind": "continuous",
            "l1": float(np.sum(np.abs(difference))),
            "l2": float(np.linalg.norm(difference)),
        }
    return result


def _finite_vector(value: object, *, shape: tuple[int, ...], name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.tolist()


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _event(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "substep",
        "bilateral_objects",
        "stable_objects",
        "dropped_target",
    }:
        raise ValueError("trace event schema is incompatible")
    bilateral = value["bilateral_objects"]
    stable = value["stable_objects"]
    if not isinstance(bilateral, list) or not all(isinstance(item, str) for item in bilateral):
        raise ValueError("trace event bilateral_objects must be a string list")
    if not isinstance(stable, list) or not all(isinstance(item, str) for item in stable):
        raise ValueError("trace event stable_objects must be a string list")
    if not isinstance(value["dropped_target"], bool):
        raise ValueError("trace event dropped_target must be boolean")
    return {
        "substep": _nonnegative_integer(value["substep"], "event substep"),
        "bilateral_objects": list(bilateral),
        "stable_objects": list(stable),
        "dropped_target": bool(value["dropped_target"]),
    }


def validate_trace_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _TRACE_FIELDS:
        raise ValueError("trace record fields are incompatible")
    record = dict(value)
    if record["trace_schema_version"] != TRACE_SCHEMA_VERSION:
        raise ValueError("trace schema version is incompatible")
    for name in ("episode_id", "case_id", "checkpoint"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError(f"trace {name} must be a non-empty string")
    condition = str(record["condition"])
    if condition not in ALL_CONDITIONS:
        raise ValueError("trace condition is invalid")
    layout = str(record["layout"])
    if layout not in {"normal", "crowded"}:
        raise ValueError("trace layout is invalid")
    expected_distribution = "id" if layout == "normal" else "ood"
    if record["training_distribution"] != expected_distribution:
        raise ValueError("trace training distribution is inconsistent with layout")
    for name in ("environment_seed", "policy_seed", "step", "gripper_switch_count"):
        record[name] = _nonnegative_integer(record[name], name)
    object_count = _nonnegative_integer(record["object_count"], "object_count")
    if object_count < 2:
        raise ValueError("trace object_count must be at least two")
    record["object_count"] = object_count
    if record["phase"] not in PHASE_NAMES:
        raise ValueError("trace phase is invalid")

    policy = validate_token(record["policy_token"])
    teacher = validate_token(record["teacher_token"])
    record["policy_token"] = policy.tolist()
    record["teacher_token"] = teacher.tolist()
    expected_errors = graph_group_errors(policy, teacher, condition=condition)
    if record["graph_error_by_group"] != expected_errors:
        raise ValueError("trace graph_error_by_group is inconsistent with tokens")
    record["graph_error_by_group"] = expected_errors

    for name in ("raw_action", "clipped_action", "executed_world_action"):
        record[name] = _finite_vector(record[name], shape=(7,), name=name)
    for name in (
        "end_effector_position",
        "target_relative_position",
        "receptacle_relative_position",
    ):
        record[name] = _finite_vector(record[name], shape=(3,), name=name)
    record["end_effector_orientation"] = _finite_vector(
        record["end_effector_orientation"],
        shape=(4,),
        name="end_effector_orientation",
    )
    for name in (
        "ik_projection_scale",
        "gripper_command",
        "minimum_distractor_clearance",
    ):
        scalar = float(record[name])
        if not np.isfinite(scalar):
            raise ValueError(f"trace {name} must be finite")
        record[name] = scalar
    if not 0.0 <= float(record["ik_projection_scale"]) <= 1.0:
        raise ValueError("trace ik_projection_scale must lie within [0, 1]")
    if not 0.0 <= float(record["gripper_command"]) <= 1.0:
        raise ValueError("trace gripper_command must lie within [0, 1]")

    boolean_fields = (
        "action_was_clipped",
        "target_contact",
        "stable_target_grasp",
        "wrong_object_contact",
        "stable_wrong_object_grasp",
        "done",
        "success",
        "target_drop",
        "timeout",
    )
    for name in boolean_fields:
        if not isinstance(record[name], bool):
            raise ValueError(f"trace {name} must be boolean")
    events = record["events"]
    if not isinstance(events, list):
        raise ValueError("trace events must be a list")
    record["events"] = [_event(event) for event in events]

    reasons = {reason.value for reason in TerminationReason}
    reason = str(record["termination_reason"])
    if reason not in reasons:
        raise ValueError("trace termination reason is invalid")
    if bool(record["done"]) != (reason != TerminationReason.RUNNING.value):
        raise ValueError("trace done and termination reason are inconsistent")
    if bool(record["success"]) != (reason == TerminationReason.SUCCESS.value):
        raise ValueError("trace success and termination reason are inconsistent")
    if bool(record["timeout"]) != (reason == TerminationReason.TIMEOUT.value):
        raise ValueError("trace timeout and termination reason are inconsistent")
    return record


def _validate_episode(records: object) -> list[dict[str, object]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("trace episode must contain records")
    values = [validate_trace_record(record) for record in records]
    identity_fields = (
        "episode_id",
        "case_id",
        "environment_seed",
        "condition",
        "policy_seed",
        "layout",
        "object_count",
        "checkpoint",
    )
    first = values[0]
    for record in values[1:]:
        if any(record[name] != first[name] for name in identity_fields):
            raise ValueError("trace episode identity changes between steps")
    if [record["step"] for record in values] != list(range(len(values))):
        raise ValueError("trace episode steps must be contiguous from zero")
    if any(record["done"] for record in values[:-1]) or not values[-1]["done"]:
        raise ValueError("trace episode must end with one complete terminal record")
    return values


def write_trace_episode_atomic(path: str | Path, records: object) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"trace episode already exists: {destination}")
    values = _validate_episode(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in values:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_trace_episode(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"trace episode is invalid: {source}") from error
    return _validate_episode(records)


def trace_episode_summary(records: object) -> dict[str, object]:
    values = _validate_episode(records)
    first = values[0]
    final = values[-1]
    projection = np.asarray(
        [record["ik_projection_scale"] for record in values], dtype=np.float64
    )
    return {
        "condition": first["condition"],
        "policy_seed": first["policy_seed"],
        "case_id": first["case_id"],
        "environment_seed": first["environment_seed"],
        "layout": first["layout"],
        "training_distribution": first["training_distribution"],
        "object_count": first["object_count"],
        "success": final["success"],
        "wrong_object_interaction": any(
            bool(record["wrong_object_contact"]) for record in values
        ),
        "wrong_object_stable_grasp": any(
            bool(record["stable_wrong_object_grasp"]) for record in values
        ),
        "target_drop": any(bool(record["target_drop"]) for record in values),
        "timeout": final["timeout"],
        "termination_reason": final["termination_reason"],
        "steps": len(values),
        "mean_ik_projection_scale": float(np.mean(projection)),
        "action_clipping_rate": float(
            np.mean([bool(record["action_was_clipped"]) for record in values])
        ),
        "gripper_switch_count": final["gripper_switch_count"],
        "checkpoint": first["checkpoint"],
    }

