from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.features import GraphNormalization
from interaction_vla.graph_control.rollout import (
    FlatTokenProvider,
    OracleCurrentTokenProvider,
    PredictedTokenProvider,
    aggregate_rollouts,
    augment_policy_observation,
    paired_evaluation_cases,
)
from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES


def _camera() -> SimpleNamespace:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    return SimpleNamespace(
        policy_step=4,
        views={"agent": SimpleNamespace(rgb=rgb), "wrist": SimpleNamespace(rgb=rgb)},
    )


class _Runtime:
    def __init__(self) -> None:
        self.calls = []
        self.normalization = GraphNormalization(
            state_mean=np.zeros(10, dtype=np.float32),
            state_std=np.ones(10, dtype=np.float32),
            relation_mean=np.zeros((8, 10), dtype=np.float32),
            relation_std=np.ones((8, 10), dtype=np.float32),
            residual_mean=0.0,
            residual_std=1.0,
        )

    def predict_tokens(self, **kwargs):
        self.calls.append(kwargs)
        token = np.zeros((1, TOKEN_DIM), dtype=np.float32)
        token[:, TOKEN_SLICES["next_relation"]] = 1.0 / 8.0
        token[:, TOKEN_SLICES["relation_operator"]] = 1.0 / 5.0
        token[:, TOKEN_SLICES["predicate"]] = 1.0 / 7.0
        token[:, TOKEN_SLICES["goal_residual"]] = 0.5
        return token


class _Teacher:
    def __init__(self) -> None:
        self.calls = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def extract(self, snapshot, camera_frame, *, state):
        self.calls.append((snapshot, camera_frame, state.copy()))
        relation_values = np.zeros((8, 24), dtype=np.float32)
        relation_values[:, 12:22] = np.arange(80, dtype=np.float32).reshape(8, 10)
        return SimpleNamespace(
            entity_mask=np.ones(6, dtype=np.bool_),
            entity_visibility=np.full((6, 2), 0.75, dtype=np.float32),
            relation_mask=np.ones(8, dtype=np.bool_),
            relation_values=relation_values,
        )


def test_online_token_providers_are_causal_and_keep_goals_predicted() -> None:
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
    predicted_token = predicted.token(
        snapshot=snapshot, camera_frame=camera, state=state, task="place"
    )
    assert len(runtime.calls) == 1

    teacher = _Teacher()
    oracle = OracleCurrentTokenProvider(runtime, teacher)
    oracle.reset()
    oracle_token = oracle.token(
        snapshot=snapshot, camera_frame=camera, state=state, task="place"
    )

    assert teacher.reset_calls == 1
    assert len(teacher.calls) == 1
    assert teacher.calls[0][0] is snapshot
    assert teacher.calls[0][1] is camera
    np.testing.assert_array_equal(teacher.calls[0][2], state)
    assert len(runtime.calls) == 2
    np.testing.assert_array_equal(
        oracle_token[TOKEN_SLICES["entity_presence"]], np.ones(6)
    )
    for name in ("next_relation", "relation_operator", "predicate", "goal_residual"):
        np.testing.assert_array_equal(
            oracle_token[TOKEN_SLICES[name]], predicted_token[TOKEN_SLICES[name]]
        )
    assert not hasattr(teacher.calls[0], "relation_goal")


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


def _record(condition: str, seed: int, case: str, success: bool) -> dict[str, object]:
    return {
        "condition": condition,
        "policy_seed": seed,
        "case_id": case,
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
                    _record("predicted_random", seed, case, seed > 0),
                    _record("predicted_reflect", seed, case, True),
                    _record("oracle_current", seed, case, True),
                ]
            )
    report = aggregate_rollouts(records)

    assert report["by_condition"]["flat"]["success_rate"] == 0.0
    assert report["by_condition"]["predicted_reflect"]["success_rate"] == 1.0
    primary = report["contrasts"]["predicted_reflect-flat"]
    assert primary["success_rate"]["per_seed"] == {"0": 1.0, "1": 1.0, "2": 1.0}
    assert primary["success_rate"]["mean"] == 1.0
    assert report["replication_unit"] == "policy_seed"


def test_rollout_aggregation_rejects_unpaired_cases() -> None:
    records = [_record(condition, 0, "a", True) for condition in (
        "flat", "predicted_random", "predicted_reflect"
    )]
    with pytest.raises(ValueError, match="paired"):
        aggregate_rollouts(records)
