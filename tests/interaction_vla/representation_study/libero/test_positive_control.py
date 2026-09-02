import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction_vla.representation_study.libero.latents import validate_requested_taps
from interaction_vla.representation_study.libero.positive_control import (
    _factor_intervention_root,
    _seal_positive_control_evaluation,
    _select_factor_records,
    extract_positive_control,
    evaluate_positive_control,
    factor_specificity_gate,
    load_positive_control_plan,
    official_success_rate,
    plan_positive_control,
    positive_control_decision,
    positive_control_root,
    report_positive_control,
    run_positive_control_intervention,
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
    (checkpoint / "config.json").write_text(
        json.dumps({"n_action_steps": 10, "empty_cameras": 1}), encoding="utf-8"
    )
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

    _seal_positive_control_evaluation(
        checkpoint=checkpoint,
        eval_dir=evaluation,
        command=(
            f"--policy.path={checkpoint}",
            f"--output_dir={evaluation}",
        ),
    )
    plan_positive_control(config, checkpoint=checkpoint, eval_dir=evaluation)
    eval_info.write_text(
        json.dumps({"overall": {"pc_success": 80.0, "n_episodes": 10}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation report hash changed"):
        load_positive_control_plan(config)


def test_evaluation_contract_rejects_another_checkpoint(tmp_path: Path) -> None:
    evaluation = tmp_path / "eval"
    evaluation.mkdir()
    (evaluation / "eval_info.json").write_text(
        json.dumps({"overall": {"pc_success": 70.0, "n_episodes": 10}})
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    for checkpoint, marker in ((first, b"a"), (second, b"b")):
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"n_action_steps": 10, "empty_cameras": 1})
        )
        (checkpoint / "model.safetensors").write_bytes(marker)
    _seal_positive_control_evaluation(
        checkpoint=first,
        eval_dir=evaluation,
        command=(f"--policy.path={first}", f"--output_dir={evaluation}"),
    )

    config = replace(
        load_libero_study_config(
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
        ),
        output_dir=tmp_path,
    )
    bank = tmp_path / "state_bank" / "manifest.json"
    bank.parent.mkdir()
    bank.write_text("{}")
    with pytest.raises(ValueError, match="different checkpoint"):
        plan_positive_control(config, checkpoint=second, eval_dir=evaluation)


def test_official_evaluator_seals_the_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"n_action_steps": 10, "empty_cameras": 1})
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    evaluation = tmp_path / "new-eval"

    def run(command, *, check):
        assert check
        evaluation.mkdir()
        (evaluation / "eval_info.json").write_text(
            json.dumps({"overall": {"pc_success": 70.0, "n_episodes": 10}})
        )

    monkeypatch.setattr(
        "interaction_vla.representation_study.libero.positive_control.subprocess.run",
        run,
    )
    report = evaluate_positive_control(checkpoint=checkpoint, eval_dir=evaluation)
    assert f"--policy.path={checkpoint}" in report["command"]
    assert "--policy.n_action_steps=10" in report["command"]


def test_positive_control_uses_protocol_v4_without_touching_v3(tmp_path: Path) -> None:
    root = positive_control_root(tmp_path)
    assert root == tmp_path / "protocol_v4" / "positive_control"
    assert "protocol_v3" not in str(root)
    assert _factor_intervention_root(root, "stable_grasp", 64) != (
        _factor_intervention_root(root, "stable_grasp", 1600)
    )


def test_report_refuses_to_freeze_missing_action_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_libero_study_config(
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
        ),
        output_dir=tmp_path,
    )
    root = positive_control_root(tmp_path)
    probe = root / "probe" / "report.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        json.dumps({"cells": {"stable_grasp": {"accessible": True}}})
    )
    monkeypatch.setattr(
        "interaction_vla.representation_study.libero.positive_control.load_positive_control_plan",
        lambda _config: {"baseline_success_rate": 0.7},
    )
    with pytest.raises(FileNotFoundError, match="specificity"):
        report_positive_control(config, factor="stable_grasp", max_states=1600)
    assert not (root / "intervention" / "stable_grasp" / "n_1600" / "report.json").exists()


def test_contact_requires_stablegrasp_replication_decision(tmp_path: Path) -> None:
    config = replace(
        load_libero_study_config(
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
        ),
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="StableGrasp decision"):
        run_positive_control_intervention(
            config, factor="contact", max_states=1600, batch_size=32
        )


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
