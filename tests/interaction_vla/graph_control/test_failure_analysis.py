from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph_control.failure_analysis import (
    build_failure_analysis_report,
    episode_error_exposure,
    failure_association,
    training_error_thresholds,
)
from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES


def test_training_thresholds_use_only_supplied_train_tokens_per_seed_condition() -> None:
    teacher = np.zeros((4, TOKEN_DIM), dtype=np.float64)
    teacher[:, TOKEN_SLICES["phase"].start] = 1.0
    predicted = teacher.copy()
    predicted[:, TOKEN_SLICES["goal_residual"]] = np.asarray(
        [0.0, 1.0, 2.0, 3.0]
    )[:, None]
    predicted[:, TOKEN_SLICES["phase"]] = 0.0
    predicted[[0, 2], TOKEN_SLICES["phase"].start] = 1.0
    predicted[[1, 3], TOKEN_SLICES["phase"].start + 1] = 1.0

    thresholds = training_error_thresholds(
        condition_tokens={(0, "predicted_random_v2"): predicted},
        teacher_tokens_by_seed={0: teacher},
        quantile=0.75,
    )

    selected = thresholds["seed_0/predicted_random_v2"]
    assert selected["goal_residual"] == pytest.approx(2.25)
    assert selected["phase"] == pytest.approx(1.0)
    assert set(selected) == set(TOKEN_SLICES)


def _exposure_record(step: int, error: float) -> dict[str, object]:
    phases = ("approach", "approach", "grasp", "lift", "release")
    done = step == 4
    return {
        "policy_seed": 0,
        "condition": "predicted_random_v2",
        "case_id": "case",
        "environment_seed": 17,
        "layout": "normal",
        "object_count": 2,
        "step": step,
        "phase": phases[step],
        "target_contact": step >= 2,
        "stable_target_grasp": step in {2, 3},
        "graph_error_by_group": {
            "goal_residual": {"kind": "continuous", "l1": error, "l2": error}
        },
        "policy_token": [0.0] * TOKEN_DIM,
        "teacher_token": [0.0] * TOKEN_DIM,
        "done": done,
        "success": done,
        "timeout": False,
        "target_drop": False,
        "stable_wrong_object_grasp": False,
    }


def test_episode_exposure_builds_event_windows_and_above_threshold_duration() -> None:
    records = [_exposure_record(step, float(step)) for step in range(5)]

    exposure = episode_error_exposure(
        records, thresholds={"goal_residual": 1.5}
    )

    group = exposure["groups"]["goal_residual"]
    assert group["mean_error"] == pytest.approx(2.0)
    assert group["max_error"] == pytest.approx(4.0)
    assert group["duration_above_threshold"] == 3
    assert group["windows"]["pre_target_contact"]["mean"] == pytest.approx(1.0)
    assert group["windows"]["grasp_to_lift"]["mean"] == pytest.approx(2.5)
    assert group["windows"]["transport"] is None
    assert group["windows"]["pre_release"]["mean"] == pytest.approx(2.0)
    assert group["windows"]["terminal"]["mean"] == pytest.approx(2.0)
    assert exposure["outcomes"]["success"] is True


def test_failure_association_reports_counts_risk_and_underpowered_flag() -> None:
    episodes = []
    for index in range(10):
        high = index >= 5
        timeout = index in {4, 5, 6, 7, 8}
        episodes.append(
            {
                "policy_seed": 0,
                "case_id": f"case_{index}",
                "condition": "predicted_random_v2",
                "outcomes": {"timeout": timeout},
                "groups": {
                    "goal_residual": {
                        "windows": {
                            "terminal": {"mean": 2.0 if high else 0.0}
                        }
                    }
                },
            }
        )

    first = failure_association(
        episodes,
        outcome="timeout",
        group="goal_residual",
        window="terminal",
        threshold=1.0,
        bootstrap_samples=500,
        bootstrap_seed=17,
    )
    second = failure_association(
        episodes,
        outcome="timeout",
        group="goal_residual",
        window="terminal",
        threshold=1.0,
        bootstrap_samples=500,
        bootstrap_seed=17,
    )

    assert first == second
    assert first["high"] == {"episodes": 5, "positive": 4, "risk": 0.8}
    assert first["low"] == {"episodes": 5, "positive": 1, "risk": 0.2}
    assert first["failure_association_score"] == pytest.approx(0.6)
    assert first["risk_ratio"] == pytest.approx(4.0)
    assert first["positive_outcomes"] == 5
    assert first["underpowered"] is False
    assert first["cluster_unit"] == "policy_seed,case_id"
    assert first["bootstrap_valid_replicates"] > 0


def test_failure_report_keeps_policy_seed_condition_and_marks_missing_windows() -> None:
    episodes = []
    for index in range(6):
        episodes.append(
            {
                "policy_seed": 0,
                "case_id": f"case_{index}",
                "condition": "predicted_random_v2",
                "outcomes": {
                    "success": index < 3,
                    "timeout": index >= 3,
                    "target_drop": False,
                    "wrong_object_stable_grasp": False,
                },
                "groups": {
                    "goal_residual": {
                        "windows": {
                            "pre_target_contact": None,
                            "grasp_to_lift": None,
                            "transport": None,
                            "pre_release": None,
                            "terminal": {"mean": float(index)},
                        }
                    }
                },
            }
        )

    report = build_failure_analysis_report(
        episodes,
        thresholds={
            "seed_0/predicted_random_v2": {"goal_residual": 2.5}
        },
        bootstrap_samples=100,
        bootstrap_seed=17,
    )

    assert report["passed"] is True
    assert report["episodes"] == 6
    selected = report["by_seed_condition"]["seed_0/predicted_random_v2"]
    assert selected["goal_residual"]["pre_target_contact"]["success"] == {
        "available": False,
        "reason": "no_episode_exposure",
    }
    assert selected["goal_residual"]["terminal"]["timeout"]["available"] is True
