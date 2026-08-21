from __future__ import annotations

from pathlib import Path

import yaml


def test_icra_base_config_preserves_act_and_assigns_modern_vla_roles() -> None:
    root = Path(__file__).resolve().parents[3]
    payload = yaml.safe_load(
        (root / "configs/representation_study/base.yaml").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "interaction_representation_study_v1"
    assert payload["backends"]["act"]["role"] == "controlled_mechanism_study"
    assert payload["backends"]["smolvla"]["role"] == "modern_vla_validation"
    assert payload["backends"]["pi0"]["required"] is False
    assert payload["backends"]["act"]["stages"] == [
        "pretrained",
        "sft",
        "continued_sft",
        "rl_head",
        "rl_representation",
    ]
    assert payload["backends"]["smolvla"]["inputs"] == [
        "agent_rgb",
        "wrist_rgb",
        "end_effector_state",
        "language",
    ]


def test_icra_base_config_predeclares_measurement_and_scale_contracts() -> None:
    root = Path(__file__).resolve().parents[3]
    payload = yaml.safe_load(
        (root / "configs/representation_study/base.yaml").read_text(encoding="utf-8")
    )

    assert payload["graph_role"] == "measurement_ontology"
    assert payload["sensitivity"]["schema_version"] == "graph_policy_sensitivity_v3"
    assert payload["sensitivity"]["normalization"] == "training_action_iqr"
    assert payload["sensitivity"]["required_controls"] == [
        "zero",
        "temporally_matched_random",
    ]
    assert payload["state_bank"]["primary_strata"] == [
        "nominal",
        "perturbation",
        "recovery",
    ]
    assert payload["statistics"]["primary_unit"] == "episode"
    assert payload["gates"]["smolvla"]["requires"] == "act_go_no_go_pass"
