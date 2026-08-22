from __future__ import annotations

from interaction_vla.representation_study.rl.formal_report import (
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
