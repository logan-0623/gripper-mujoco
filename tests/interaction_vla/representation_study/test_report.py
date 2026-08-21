from interaction_vla.representation_study.report import (
    closed_loop_intervention_rows,
    intervention_result_rows,
    probe_result_rows,
    rl_result_rows,
    representation_relationships,
    utility_result_rows,
)


def test_report_rows_keep_encoded_used_useful_and_plasticity_distinct() -> None:
    probes = probe_result_rows(
        {
            "rows": [{
                "backend": "act", "stage": "sft", "tap": "pre_action",
                "target": "phase", "model_kind": "linear",
                "metrics": {"test": {"balanced_accuracy": 0.7}},
            }]
        },
        source="probe.json",
    )
    used = intervention_result_rows(
        {
            "backend": "act", "stage": "sft",
            "aggregate": {"pre_action/zero": {"first_action_l2": 0.2}},
        },
        source="use.json",
    )
    rl = rl_result_rows(
        {
            "backend": "act", "stage": "rl_head",
            "normalized_learning_curve_auc": 0.4,
            "initial_success_rate": 0.2,
            "final_success_rate": 0.5,
            "final_return_gain": 0.3,
            "steps_to_fixed_threshold": 100,
        },
        source="rl.json",
    )
    assert probes[0]["axis"] == "accessible"
    assert used[0]["axis"] == "used"
    assert {row["axis"] for row in rl} == {"useful", "plasticity"}
    utility = utility_result_rows(
        {
            "backend": "act", "stage": "sft", "episodes": 20,
            "success_rate": 0.3, "timeout_rate": 0.6, "drop_rate": 0.1,
            "wrong_object_rate": 0.0, "mean_steps": 150,
        },
        source="utility.json",
    )
    assert {row["axis"] for row in utility} == {"useful"}
    causal = closed_loop_intervention_rows(
        {
            "backend": "act", "stage": "sft",
            "summaries": [{
                "tap": "pre_action", "mode": "zero", "episodes": 20,
                "success_rate": 0.2, "paired_success_delta": -0.1,
                "ci_low": -0.2, "ci_high": 0.0,
                "paired_sign_flip_p": 0.04, "bh_adjusted_p": 0.08,
            }],
        },
        source="causal.json",
    )
    assert {row["axis"] for row in causal} == {"used_to_useful"}


def test_relationship_analysis_joins_probe_and_behavior_by_stage() -> None:
    rows = []
    for index, stage in enumerate(("pretrained", "sft", "continued_sft")):
        rows.extend(
            [
                {
                    "axis": "accessible",
                    "backend": "smolvla",
                    "stage": stage,
                    "tap": "pre_action",
                    "target": "phase",
                    "partition": "test",
                    "metric": "balanced_accuracy",
                    "value": 0.2 + 0.2 * index,
                },
                {
                    "axis": "useful",
                    "backend": "smolvla",
                    "stage": stage,
                    "metric": "success_rate",
                    "value": 0.1 + 0.3 * index,
                },
            ]
        )

    result = representation_relationships(rows)

    assert result[0]["relationship"] == "accessible_vs_utility"
    assert result[0]["spearman"] == 1.0
    assert result[0]["stages"] == ["continued_sft", "pretrained", "sft"]
