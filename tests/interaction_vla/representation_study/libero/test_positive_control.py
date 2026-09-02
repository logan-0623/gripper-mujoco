import json
from dataclasses import replace
from pathlib import Path

import pytest

from interaction_vla.representation_study.libero.latents import validate_requested_taps
from interaction_vla.representation_study.libero.positive_control import (
    load_positive_control_plan,
    official_success_rate,
    plan_positive_control,
    positive_control_decision,
)
from interaction_vla.representation_study.libero.config import load_libero_study_config
from interaction_vla.representation_study.libero.taps import SEMANTIC_TAPS


def test_requested_taps_default_to_all_semantic_taps() -> None:
    assert validate_requested_taps(None) == tuple(SEMANTIC_TAPS)


def test_requested_taps_are_unique_known_and_ordered() -> None:
    assert validate_requested_taps(("action_expert_input",)) == (
        "action_expert_input",
    )
    with pytest.raises(ValueError, match="unknown semantic tap"):
        validate_requested_taps(("missing",))
    with pytest.raises(ValueError, match="duplicate semantic tap"):
        validate_requested_taps(("pre_action", "pre_action"))


def test_official_success_rate_reads_lerobot_percent(tmp_path: Path) -> None:
    report = tmp_path / "eval_info.json"
    report.write_text(
        json.dumps({"overall": {"pc_success": 70.0, "n_episodes": 10}}),
        encoding="utf-8",
    )
    assert official_success_rate(report) == (0.70, 10)


def test_official_success_rate_rejects_invalid_values(tmp_path: Path) -> None:
    report = tmp_path / "eval_info.json"
    report.write_text(
        json.dumps({"overall": {"pc_success": float("nan"), "n_episodes": 10}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        official_success_rate(report)


def test_floor_policy_stops_before_probe_claims() -> None:
    result = positive_control_decision(
        success_rate=0.20,
        accessible=True,
        specificity_passed=True,
        usage_ci=(0.1, 0.2),
        factor="stable_grasp",
    )
    assert result["decision"] == "failed_policy_floor"


def test_positive_control_plan_rejects_changed_evaluation(tmp_path: Path) -> None:
    config = replace(
        load_libero_study_config(
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
        ),
        output_dir=tmp_path,
    )
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    evaluation = tmp_path / "eval"
    evaluation.mkdir()
    eval_info = evaluation / "eval_info.json"
    eval_info.write_text(
        json.dumps({"overall": {"pc_success": 70.0, "n_episodes": 10}}),
        encoding="utf-8",
    )
    state_bank = tmp_path / "state_bank" / "manifest.json"
    state_bank.parent.mkdir()
    state_bank.write_text('{"audit_passed":true}', encoding="utf-8")

    plan_positive_control(config, checkpoint=checkpoint, eval_dir=evaluation)
    eval_info.write_text(
        json.dumps({"overall": {"pc_success": 80.0, "n_episodes": 10}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation report hash changed"):
        load_positive_control_plan(config)
