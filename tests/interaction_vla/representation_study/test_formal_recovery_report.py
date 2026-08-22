from __future__ import annotations

from interaction_vla.representation_study.rl.formal_report import (
    _paired_effects,
    assemble_formal_evidence,
    expansion_gate,
)


def test_report_keeps_evidence_axes_separate() -> None:
    rows = assemble_formal_evidence(
        accessible=[
            {
                "axis": "accessible",
                "unit": "state",
                "metric": "balanced_accuracy",
                "value": 0.7,
            }
        ],
        useful=[
            {
                "axis": "useful",
                "unit": "episode",
                "metric": "success_rate",
                "value": 0.5,
            }
        ],
        plasticity=[
            {
                "axis": "plasticity",
                "unit": "training_run",
                "metric": "recovery_auc",
                "value": 0.4,
            }
        ],
    )
    assert {row["axis"] for row in rows} >= {
        "accessible",
        "useful",
        "plasticity",
    }
    assert all(
        row["unit"] == "episode"
        for row in rows
        if row["axis"] == "useful"
    )


def test_expansion_gate_requires_two_of_three_seed_directions() -> None:
    assert expansion_gate([0.10, 0.05, -0.02]).passed is True
    assert expansion_gate([0.10, -0.01, -0.04]).passed is False


def test_paired_effects_runs_sign_flip_over_seed_directions() -> None:
    reports = {}
    for seed_index in range(3):
        reports[("sft", seed_index)] = {
            "rows": [
                {"case_id": "n", "family": "nominal", "success": False},
                {"case_id": "r", "family": "recovery", "success": False},
            ]
        }
        reports[("rl_head", seed_index)] = {
            "rows": [
                {"case_id": "n", "family": "nominal", "success": True},
                {"case_id": "r", "family": "recovery", "success": True},
            ]
        }

    effects = _paired_effects(reports, seed=7)
    assert len(effects) == 2
    assert all(row["seed_directions"] == [1.0, 1.0, 1.0] for row in effects)
