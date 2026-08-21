import pytest

from interaction_vla.representation_study.evaluation import aggregate_episode_outcomes


def test_stage_outcomes_keep_failure_modes_separate() -> None:
    result = aggregate_episode_outcomes(
        [
            {
                "termination_reason": "success",
                "steps": 10,
                "action_clipping_rate": 0.1,
                "mean_ik_projection_scale": 1.0,
            },
            {
                "termination_reason": "timeout",
                "steps": 20,
                "action_clipping_rate": 0.3,
                "mean_ik_projection_scale": 0.5,
            },
            {
                "termination_reason": "dropped",
                "steps": 5,
                "action_clipping_rate": 0.2,
                "mean_ik_projection_scale": 0.75,
            },
            {
                "termination_reason": "wrong_object",
                "steps": 6,
                "action_clipping_rate": 0.0,
                "mean_ik_projection_scale": 1.0,
            },
        ]
    )
    assert result["success_rate"] == 0.25
    assert result["timeout_rate"] == 0.25
    assert result["drop_rate"] == 0.25
    assert result["wrong_object_rate"] == 0.25
    assert result["action_clipping_rate"] == pytest.approx(0.15)
    assert result["mean_ik_projection_scale"] == pytest.approx(0.8125)
