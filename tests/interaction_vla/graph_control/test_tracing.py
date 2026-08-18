from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES
from interaction_vla.graph_control.tracing import (
    TRACE_SCHEMA_VERSION,
    graph_group_errors,
    load_trace_episode,
    trace_episode_summary,
    validate_trace_record,
    write_trace_episode_atomic,
)


def test_graph_group_errors_distinguish_continuous_and_categorical_groups() -> None:
    teacher = np.zeros(TOKEN_DIM, dtype=np.float32)
    policy = teacher.copy()
    geometry = TOKEN_SLICES["gripper_target_geometry"]
    policy[geometry.start : geometry.start + 2] = (3.0, 4.0)
    phase = TOKEN_SLICES["phase"]
    teacher[phase.start] = 1.0
    policy[phase.start + 1] = 1.0

    errors = graph_group_errors(policy, teacher, condition="predicted_random_v2")

    assert errors["gripper_target_geometry"] == {
        "kind": "continuous",
        "l1": 7.0,
        "l2": 5.0,
    }
    assert errors["phase"] == {
        "kind": "categorical",
        "agreement": False,
        "policy_label": 1,
        "teacher_label": 0,
    }
    assert set(errors) == set(TOKEN_SLICES)

    missing = graph_group_errors(
        np.zeros(TOKEN_DIM), np.zeros(TOKEN_DIM), condition="flat"
    )
    assert missing["phase"]["agreement"] is None


def _record(step: int, *, done: bool) -> dict[str, object]:
    policy = np.zeros(TOKEN_DIM, dtype=np.float32)
    teacher = np.zeros(TOKEN_DIM, dtype=np.float32)
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "episode_id": "seed_0/flat/normal_n2_000",
        "case_id": "normal_n2_000",
        "environment_seed": 17,
        "condition": "flat",
        "policy_seed": 0,
        "layout": "normal",
        "object_count": 2,
        "training_distribution": "id",
        "step": step,
        "phase": "approach",
        "policy_token": policy.tolist(),
        "teacher_token": teacher.tolist(),
        "graph_error_by_group": graph_group_errors(
            policy, teacher, condition="flat"
        ),
        "raw_action": [0.0] * 7,
        "clipped_action": [0.0] * 7,
        "executed_world_action": [0.0] * 7,
        "action_was_clipped": step == 0,
        "ik_projection_scale": 0.5 + 0.5 * step,
        "gripper_command": 1.0,
        "end_effector_position": [0.4, 0.0, 0.4],
        "end_effector_orientation": [1.0, 0.0, 0.0, 0.0],
        "target_relative_position": [0.1, 0.0, -0.1],
        "receptacle_relative_position": [0.2, 0.0, 0.0],
        "minimum_distractor_clearance": 0.05,
        "target_contact": step > 0,
        "stable_target_grasp": False,
        "wrong_object_contact": False,
        "stable_wrong_object_grasp": False,
        "events": [],
        "done": done,
        "termination_reason": "success" if done else "running",
        "success": done,
        "target_drop": False,
        "timeout": False,
        "gripper_switch_count": step,
        "checkpoint": "checkpoint/seed_0/flat",
    }


def test_trace_record_validation_rejects_shapes_nonfinite_and_terminal_mismatch() -> None:
    record = validate_trace_record(_record(0, done=False))
    assert record["step"] == 0

    wrong_shape = _record(0, done=False)
    wrong_shape["raw_action"] = [0.0] * 6
    with pytest.raises(ValueError, match="raw_action"):
        validate_trace_record(wrong_shape)

    nonfinite = _record(0, done=False)
    nonfinite["minimum_distractor_clearance"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_trace_record(nonfinite)

    mismatch = _record(0, done=False)
    mismatch["termination_reason"] = "success"
    with pytest.raises(ValueError, match="termination"):
        validate_trace_record(mismatch)


def test_trace_episode_write_is_atomic_complete_and_reloadable(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "seed_0" / "flat" / "case.jsonl"
    records = [_record(0, done=False), _record(1, done=True)]

    write_trace_episode_atomic(path, records)

    assert load_trace_episode(path) == records
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        write_trace_episode_atomic(path, records)

    path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete terminal"):
        load_trace_episode(path)


def test_trace_episode_summary_recovers_existing_rollout_metrics() -> None:
    records = [_record(0, done=False), _record(1, done=True)]

    summary = trace_episode_summary(records)

    assert summary["condition"] == "flat"
    assert summary["steps"] == 2
    assert summary["success"] is True
    assert summary["target_drop"] is False
    assert summary["mean_ik_projection_scale"] == pytest.approx(0.75)
    assert summary["action_clipping_rate"] == pytest.approx(0.5)
    assert summary["gripper_switch_count"] == 1

