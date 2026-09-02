import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction_vla.representation_study.libero.latents import validate_requested_taps
from interaction_vla.representation_study.libero.positive_control import (
    extract_positive_control,
    _select_factor_records,
    factor_specificity_gate,
    load_positive_control_plan,
    official_success_rate,
    plan_positive_control,
    positive_control_decision,
    positive_control_root,
    summarize_positive_control_probe_results,
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


def test_positive_usage_authorizes_official_longitudinal_training() -> None:
    result = positive_control_decision(
        success_rate=0.70,
        accessible=True,
        specificity_passed=True,
        usage_ci=(0.01, 0.03),
        factor="stable_grasp",
    )
    assert result == {
        "decision": "continue_official_longitudinal",
        "authorize_longitudinal_training": True,
    }


def test_failed_factors_stop_after_one_contact_replication() -> None:
    stable = positive_control_decision(
        success_rate=0.70,
        accessible=True,
        specificity_passed=True,
        usage_ci=(-0.01, 0.01),
        factor="stable_grasp",
    )
    contact = positive_control_decision(
        success_rate=0.70,
        accessible=True,
        specificity_passed=True,
        usage_ci=(-0.01, 0.01),
        factor="contact",
    )
    assert stable["decision"] == "replicate_contact_once"
    assert contact["decision"] == "pivot_interaction_supervised_sft"


def test_factor_specificity_gate_keeps_stablegrasp_place_control() -> None:
    common = dict(
        target_minus_random={"ci_low": 0.1},
        target_effect=1.0,
        non_target_effects={
            "stable_grasp": 0.2,
            "contact": 0.2,
            "phase": 0.3,
            "geometry": 0.1,
        },
        activation_norm_ratio=1.0,
    )
    assert factor_specificity_gate(
        factor="stable_grasp",
        place_target_minus_random={"ci_low": 0.1},
        **common,
    )["passed"]
    assert not factor_specificity_gate(
        factor="stable_grasp",
        place_target_minus_random=None,
        **common,
    )["passed"]
    assert factor_specificity_gate(
        factor="contact",
        place_target_minus_random=None,
        **common,
    )["passed"]


def test_contact_replication_uses_stablegrasp_state_support() -> None:
    def record(state_id: str, *, stable: bool) -> SimpleNamespace:
        applicability = SimpleNamespace(contact=True, stable_grasp=stable)
        labels = SimpleNamespace(applicability=applicability, phase="approach")
        return SimpleNamespace(
            suite="libero_spatial", task_id=0, state_id=state_id, labels=labels
        )

    records = (record("contact-only", stable=False), record("shared", stable=True))
    assert _select_factor_records(records, factor="contact", max_states=2) == (1,)


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


def test_positive_control_uses_protocol_v4_without_touching_v3(tmp_path: Path) -> None:
    root = positive_control_root(tmp_path)
    assert root == tmp_path / "protocol_v4" / "positive_control"
    assert "protocol_v3" not in str(root)


def test_positive_control_extracts_only_action_expert_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_libero_study_config(
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
        ),
        output_dir=tmp_path,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "interaction_vla.representation_study.libero.positive_control.load_positive_control_plan",
        lambda _config: {
            "checkpoint": "/tmp/official",
            "checkpoint_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        "interaction_vla.representation_study.libero.positive_control._file_sha256",
        lambda _path: "b" * 64,
    )
    monkeypatch.setattr(
        "interaction_vla.representation_study.libero.positive_control.extract_smolvla_latents_from_checkpoint",
        lambda _config, **kwargs: calls.append(kwargs) or {"passed": True},
    )

    assert extract_positive_control(config, batch_size=32)["passed"]
    assert calls[0]["taps"] == ("action_expert_input",)
    assert calls[0]["output_dir"] == positive_control_root(tmp_path) / "latents"


def test_probe_summary_requires_complete_stablegrasp_accessibility() -> None:
    cells = {
        factor: {"status": "complete", "accessible": factor == "stable_grasp"}
        for factor in ("stable_grasp", "contact", "phase", "geometry")
    }
    summary = summarize_positive_control_probe_results(cells)
    assert summary["passed"]
    assert summary["stable_grasp_accessible"]

    cells["stable_grasp"] = {"status": "failed_gate", "accessible": None}
    failed = summarize_positive_control_probe_results(cells)
    assert not failed["passed"]
    assert not failed["stable_grasp_accessible"]
