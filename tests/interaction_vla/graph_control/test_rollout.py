from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interaction_vla.graph_finetune.schema import GraphV2Normalization
from interaction_vla.graph_control.rollout import (
    FlatTokenProvider,
    OracleGraphV2TokenProvider,
    PredictedTokenProvider,
    _next_queued_action,
    aggregate_rollouts,
    augment_policy_observation,
    paired_evaluation_cases,
)
from interaction_vla.lerobot_bridge.rollout import ActionChunkQueue
from interaction_vla.graph_control.schema import (
    ORACLE_CONDITIONS,
    TOKEN_DIM,
    TOKEN_SLICES,
)
from interaction_vla.lerobot_bridge.teacher_schema import TeacherFrame


def _camera() -> SimpleNamespace:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    return SimpleNamespace(
        policy_step=4,
        views={"agent": SimpleNamespace(rgb=rgb), "wrist": SimpleNamespace(rgb=rgb)},
    )


class _Runtime:
    def __init__(self) -> None:
        self.calls = []
        self.reset_calls = 0
        self.normalization = GraphV2Normalization(
            state_mean=np.zeros(10, dtype=np.float32),
            state_std=np.ones(10, dtype=np.float32),
            workspace_scale=1.0,
            velocity_scale=1.0,
        )

    def reset(self):
        self.reset_calls += 1

    def predict_token(self, **kwargs):
        self.calls.append(kwargs)
        token = np.zeros(TOKEN_DIM, dtype=np.float32)
        token[TOKEN_SLICES["next_relation"]] = 1.0 / 8.0
        token[TOKEN_SLICES["relation_operator"]] = 1.0 / 5.0
        token[TOKEN_SLICES["predicate"]] = 1.0 / 7.0
        token[TOKEN_SLICES["goal_residual"]] = 0.5
        return token


class _Teacher:
    def __init__(self) -> None:
        self.calls = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def extract(self, snapshot, camera_frame, *, state):
        self.calls.append((snapshot, camera_frame, state.copy()))
        frame = TeacherFrame.zeros(
            frame_index=len(self.calls) - 1,
            timestamp=float(len(self.calls) - 1),
            state_hash=f"frame-{len(self.calls)}",
        )
        frame.entity_pose[:, 3:] = np.asarray((1, 0, 0, 0, 1, 0), dtype=np.float32)
        frame.entity_mask[:] = True
        frame.entity_visibility[:] = 0.75
        frame.relation_mask[:] = True
        frame.relation_values[:, 16:20] = 0.5
        frame.relation_values[:, 20:22] = 0.25
        frame.relation_values[:, 22:24] = 1.0
        return frame


def test_online_token_providers_are_causal_and_oracle_needs_no_graph_runtime() -> None:
    camera = _camera()
    state = np.zeros(10, dtype=np.float32)
    snapshot = object()

    flat = FlatTokenProvider()
    assert np.array_equal(
        flat.token(snapshot=snapshot, camera_frame=camera, state=state, task="place"),
        np.zeros(TOKEN_DIM, dtype=np.float32),
    )

    runtime = _Runtime()
    predicted = PredictedTokenProvider(runtime)
    predicted.reset()
    predicted_token = predicted.token(
        snapshot=snapshot, camera_frame=camera, state=state, task="place"
    )
    assert len(runtime.calls) == 1
    assert runtime.reset_calls == 1

    teacher = _Teacher()
    oracle = OracleGraphV2TokenProvider(
        teacher=teacher,
        normalization=runtime.normalization,
    )
    oracle.reset()
    oracle_token = oracle.token(
        snapshot=snapshot, camera_frame=camera, state=state, task="place"
    )

    assert teacher.reset_calls == 1
    assert len(teacher.calls) == 1
    assert teacher.calls[0][0] is snapshot
    assert teacher.calls[0][1] is camera
    np.testing.assert_array_equal(teacher.calls[0][2], state)
    assert len(runtime.calls) == 1
    np.testing.assert_array_equal(
        oracle_token[TOKEN_SLICES["entity_presence"]], np.ones(6)
    )
    assert oracle_token[TOKEN_SLICES["phase"]].sum() == 1.0
    assert oracle_token[TOKEN_SLICES["next_relation"]].sum() == 1.0


def test_policy_observation_keeps_graph_separate_from_10d_state() -> None:
    camera = _camera()
    state = np.arange(10, dtype=np.float32)
    token = np.arange(TOKEN_DIM, dtype=np.float32)
    observation = augment_policy_observation(
        agent_rgb=camera.views["agent"].rgb,
        wrist_rgb=camera.views["wrist"].rgb,
        state=state,
        token=token,
    )
    assert set(observation) == {
        "observation.images.agent",
        "observation.images.wrist",
        "observation.state",
        "observation.environment_state",
    }
    assert observation["observation.state"].shape == (10,)
    assert torch.equal(
        observation["observation.environment_state"], torch.from_numpy(token)
    )


def test_paired_case_schedule_crosses_cells_and_is_deterministic() -> None:
    first = paired_evaluation_cases(
        layouts=("normal", "crowded"),
        object_counts=(2, 3),
        cases_per_cell=2,
        master_seed=17,
    )
    second = paired_evaluation_cases(
        layouts=("normal", "crowded"),
        object_counts=(2, 3),
        cases_per_cell=2,
        master_seed=17,
    )
    assert first == second
    assert len(first) == 8
    assert len({case.case_id for case in first}) == 8
    assert {(case.layout, case.object_count) for case in first} == {
        ("normal", 2),
        ("normal", 3),
        ("crowded", 2),
        ("crowded", 3),
    }

    oracle = paired_evaluation_cases(
        layouts=("normal",),
        object_counts=(2,),
        cases_per_cell=20,
        master_seed=17,
    )
    assert len(oracle) == 20


def _record(condition: str, seed: int, case: str, success: bool) -> dict[str, object]:
    return {
        "condition": condition,
        "policy_seed": seed,
        "case_id": case,
        "environment_seed": 100 if case == "a" else 200,
        "layout": "normal" if case == "a" else "crowded",
        "object_count": 2 if case == "a" else 3,
        "success": success,
        "wrong_object_interaction": not success,
        "wrong_object_stable_grasp": False,
        "target_drop": False,
        "timeout": not success,
        "steps": 10 if success else 20,
        "mean_ik_projection_scale": 0.9,
        "action_clipping_rate": 0.1,
        "gripper_switch_count": 2,
    }


def test_rollout_aggregation_keeps_policy_seed_as_replication_unit() -> None:
    records = []
    for seed in (0, 1, 2):
        for case in ("a", "b"):
            records.extend(
                [
                    _record("flat", seed, case, False),
                    _record("oracle_graph_v2", seed, case, True),
                ]
            )
    report = aggregate_rollouts(records, conditions=ORACLE_CONDITIONS)

    assert report["by_condition"]["flat"]["success_rate"] == 0.0
    assert report["by_condition"]["oracle_graph_v2"]["success_rate"] == 1.0
    primary = report["contrasts"]["oracle_graph_v2-flat"]
    assert primary["success_rate"]["per_seed"] == {"0": 1.0, "1": 1.0, "2": 1.0}
    assert primary["success_rate"]["mean"] == 1.0
    assert report["replication_unit"] == "policy_seed"
    case_deltas = report["paired_case_deltas"]
    assert len(case_deltas) == 3 * 2 * 1 * 9
    assert {
        "policy_seed": 0,
        "case_id": "a",
        "environment_seed": 100,
        "layout": "normal",
        "object_count": 2,
        "contrast": "oracle_graph_v2-flat",
        "metric": "success_rate",
        "delta": 1.0,
    } in case_deltas


def test_rollout_aggregation_rejects_unpaired_cases() -> None:
    records = [_record("flat", 0, "a", True)]
    with pytest.raises(ValueError, match="paired"):
        aggregate_rollouts(records, conditions=ORACLE_CONDITIONS)


@pytest.mark.parametrize("field", ["environment_seed", "layout", "object_count"])
def test_rollout_aggregation_rejects_mismatched_case_identity(field: str) -> None:
    records = [_record(condition, 0, "a", True) for condition in ORACLE_CONDITIONS]
    records[-1][field] = {
        "environment_seed": 999,
        "layout": "crowded",
        "object_count": 3,
    }[field]

    with pytest.raises(ValueError, match=field):
        aggregate_rollouts(records, conditions=ORACLE_CONDITIONS)


def test_graph_observation_is_rebuilt_at_receding_horizon_queue_index_zero(monkeypatch) -> None:
    calls = []

    def observation_factory():
        calls.append(len(calls))
        return {"observation.state": torch.zeros(10)}

    monkeypatch.setattr(
        "interaction_vla.graph_control.rollout._predict_chunk",
        lambda runtime, observation: np.zeros((8, 7), dtype=np.float32),
    )
    queue = ActionChunkQueue(chunk_size=8, n_action_steps=1)

    for _ in range(9):
        _next_queued_action(queue, object(), observation_factory)

    assert len(calls) == 9


def test_oracle_gate_requires_ten_point_gain_without_more_wrong_grasps() -> None:
    records = []
    for index in range(20):
        records.append(_record("flat", 0, str(index), index < 3))
        records.append(_record("oracle_graph_v2", 0, str(index), index < 5))
        records[-2]["case_id"] = records[-1]["case_id"] = str(index)
        records[-2]["environment_seed"] = records[-1]["environment_seed"] = index
        records[-2]["layout"] = records[-1]["layout"] = "normal"
        records[-2]["object_count"] = records[-1]["object_count"] = 2

    report = aggregate_rollouts(records, conditions=ORACLE_CONDITIONS)

    assert report["oracle_gate"] == {
        "passed": True,
        "success_delta": 0.1,
        "wrong_object_stable_grasp_delta": 0.0,
        "required_success_delta": 0.1,
    }

    for record in records:
        if record["condition"] == "oracle_graph_v2" and record["case_id"] == "4":
            record["success"] = False
    assert aggregate_rollouts(records, conditions=ORACLE_CONDITIONS)["oracle_gate"][
        "passed"
    ] is False
