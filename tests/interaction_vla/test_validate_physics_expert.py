from __future__ import annotations

import json
from pathlib import Path
import sys

import interaction_vla.validate_physics_expert as validation_module
import pytest
from interaction_vla.validate_physics_expert import (
    ExpertValidationCase,
    ExpertValidationResult,
    build_gate_report,
    make_validation_cases,
    no_attachment_audit,
)
from interaction_vla.physics_provenance import (
    controller_source_hash,
    learned_rollout_module_names,
    physics_control_module_names,
    scene_asset_hash,
    training_pipeline_module_names,
)


class ProgressSpy:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfixes: list[dict[str, object]] = []
        self.closed = False

    def update(self, amount: int) -> None:
        self.updates.append(amount)

    def set_postfix(self, **kwargs: object) -> None:
        self.postfixes.append(kwargs)

    def close(self) -> None:
        self.closed = True


def result(condition: str, index: int, *, success: bool) -> ExpertValidationResult:
    return ExpertValidationResult(
        condition=condition,
        seed=700_000 + index,
        object_count=2 + index % 2,
        success=success,
        reason="success" if success else "timeout",
        steps=80,
        stable_lift=success,
        physics_failure=False,
    )


def test_validation_cases_are_deterministic_and_use_a_distinct_namespace() -> None:
    first = make_validation_cases(config_seed=42, cases_per_condition=4)
    second = make_validation_cases(config_seed=42, cases_per_condition=4)

    assert first == second
    assert {case.condition for case in first} == {"normal", "crowded"}
    assert len(first) == 8
    assert len({case.seed for case in first}) == 8
    assert all(case.seed not in {42, 1000, 1001, 1100, 1101} for case in first)


def test_gate_requires_at_least_90_percent_in_each_condition() -> None:
    passing = tuple(
        result(condition, index, success=index < 18)
        for condition in ("normal", "crowded")
        for index in range(20)
    )
    report = build_gate_report(
        passing,
        threshold=0.9,
        controller_hash="a" * 64,
        scene_hash="b" * 64,
        config_hash="c" * 64,
        attachment_audit_passed=True,
    )

    assert report["passed"] is True
    assert report["success_rate"] == 0.9
    assert report["conditions"]["normal"] == {"success": 18, "total": 20, "rate": 0.9}

    failing = tuple(
        result(
            condition,
            index,
            success=index < (17 if condition == "crowded" else 20),
        )
        for condition in ("normal", "crowded")
        for index in range(20)
    )
    failed_report = build_gate_report(
        failing,
        threshold=0.9,
        controller_hash="a" * 64,
        scene_hash="b" * 64,
        config_hash="c" * 64,
        attachment_audit_passed=True,
    )
    assert failed_report["passed"] is False


def test_gate_fails_on_physics_or_attachment_audit_even_with_success_reasons() -> None:
    values = tuple(
        result(condition, index, success=True)
        for condition in ("normal", "crowded")
        for index in range(2)
    )
    values = values[:-1] + (
        ExpertValidationResult(
            condition="crowded",
            seed=999,
            object_count=3,
            success=True,
            reason="success",
            steps=80,
            stable_lift=True,
            physics_failure=True,
        ),
    )

    report = build_gate_report(
        values,
        threshold=0.9,
        controller_hash="a" * 64,
        scene_hash="b" * 64,
        config_hash="c" * 64,
        attachment_audit_passed=False,
    )

    assert report["passed"] is False


def test_gate_hashes_cover_environment_contact_code_and_included_robot_assets() -> None:
    controller = controller_source_hash()
    scene = scene_asset_hash()

    assert len(controller) == 64
    assert len(scene) == 64
    assert controller_source_hash() == controller
    assert scene_asset_hash() == scene
    assert no_attachment_audit()


def test_provenance_module_sets_cover_strict_temporal_pipeline() -> None:
    assert "placement.py" in physics_control_module_names()
    assert "chunked_controller.py" in learned_rollout_module_names()
    assert "models/policy.py" in learned_rollout_module_names()
    assert "sequence_training.py" in training_pipeline_module_names()
    assert "source_split.py" in training_pipeline_module_names()


def test_validate_expert_reports_case_progress(monkeypatch, tmp_path: Path) -> None:
    cases = (
        ExpertValidationCase(condition="normal", seed=101, object_count=2),
        ExpertValidationCase(condition="crowded", seed=202, object_count=3),
    )
    progress_instances: list[ProgressSpy] = []

    def fake_tqdm(**kwargs: object) -> ProgressSpy:
        progress = ProgressSpy(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(validation_module, "make_validation_cases", lambda **kwargs: cases)
    monkeypatch.setattr(
        validation_module,
        "run_validation_case",
        lambda config, case: result(
            case.condition,
            0,
            success=case.condition == "normal",
        ),
    )
    monkeypatch.setattr(validation_module, "controller_source_hash", lambda: "a" * 64)
    monkeypatch.setattr(validation_module, "scene_asset_hash", lambda: "b" * 64)
    monkeypatch.setattr(validation_module, "config_file_hash", lambda path: "c" * 64)
    monkeypatch.setattr(validation_module, "no_attachment_audit", lambda: True)
    monkeypatch.setattr(validation_module, "tqdm", fake_tqdm)

    output = validation_module.validate_expert_from_config(
        "configs/physics_smoke_macos.yaml",
        output=tmp_path / "gate.json",
        show_progress=True,
    )

    assert output == tmp_path / "gate.json"
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.kwargs == {
        "total": 2,
        "desc": "expert gate",
        "unit": "case",
        "dynamic_ncols": True,
    }
    assert progress.updates == [1, 1]
    assert progress.postfixes == [
        {
            "condition": "normal",
            "objects": 2,
            "seed": 101,
            "success": 1,
            "passed": 1,
            "failed": 0,
        },
        {
            "condition": "crowded",
            "objects": 3,
            "seed": 202,
            "success": 0,
            "passed": 1,
            "failed": 1,
        },
    ]
    assert progress.closed

    validation_module.validate_expert_from_config(
        "configs/physics_smoke_macos.yaml",
        output=tmp_path / "quiet_gate.json",
    )
    assert len(progress_instances) == 1


def test_validation_cli_enables_progress(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    output.write_text(
        json.dumps(
            {
                "passed": True,
                "success_rate": 1.0,
                "conditions": {"normal": {"rate": 1.0}, "crowded": {"rate": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_validate(config_path: str, **kwargs: object) -> Path:
        captured.update({"config_path": config_path, **kwargs})
        return output

    monkeypatch.setattr(validation_module, "validate_expert_from_config", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_physics_expert",
            "--config",
            "configs/physics_recovery_smoke_macos.yaml",
            "--output",
            str(output),
        ],
    )

    validation_module.main()

    assert captured == {
        "config_path": "configs/physics_recovery_smoke_macos.yaml",
        "output": str(output),
        "show_progress": True,
    }


def test_validation_progress_closes_on_unexpected_exception(
    monkeypatch, tmp_path: Path
) -> None:
    progress = ProgressSpy(total=1)
    case = ExpertValidationCase(condition="normal", seed=101, object_count=2)
    monkeypatch.setattr(
        validation_module,
        "make_validation_cases",
        lambda **kwargs: (case,),
    )
    monkeypatch.setattr(
        validation_module,
        "run_validation_case",
        lambda config, current_case: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(validation_module, "tqdm", lambda **kwargs: progress)

    with pytest.raises(RuntimeError, match="boom"):
        validation_module.validate_expert_from_config(
            "configs/physics_smoke_macos.yaml",
            output=tmp_path / "gate.json",
            show_progress=True,
        )

    assert progress.closed
