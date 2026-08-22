from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_vla.representation_study.rl.protocol import (
    require_passing_gate,
    run_recovery_command,
    write_gate_atomic,
)
from interaction_vla.representation_study.rl.distributions import build_case_manifest
from interaction_vla.representation_study.rl.foundation import paired_evaluation_cases


class _Config:
    def __init__(self, root: Path) -> None:
        self.output_dir = root


def _gate(path: Path, name: str, *, passed: bool = True) -> None:
    write_gate_atomic(
        path,
        {
            "schema_version": "recovery_rl_gate_v2",
            "gate": name,
            "passed": passed,
            "reasons": [],
            "inputs": {},
        },
    )


def test_gate_writer_is_immutable_and_loader_requires_pass(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    _gate(path, "distribution")
    assert require_passing_gate(path, expected_gate="distribution")["passed"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        write_gate_atomic(path, {"passed": False})


def test_gate_loader_rejects_stale_binding(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    write_gate_atomic(
        path,
        {
            "schema_version": "recovery_rl_gate_v2",
            "gate": "distribution",
            "passed": True,
            "reasons": [],
            "inputs": {"binding": "old"},
        },
    )
    with pytest.raises(ValueError, match="binding"):
        require_passing_gate(
            path,
            expected_gate="distribution",
            expected_binding="current",
        )


def test_screen_requires_distribution_gate(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="distribution gate"):
        run_recovery_command(_Config(tmp_path), "screen", resume=False)


def test_recovery_commands_follow_gate_order(tmp_path: Path, monkeypatch) -> None:
    import interaction_vla.representation_study.rl.protocol as protocol

    calls: list[str] = []
    monkeypatch.setattr(
        protocol,
        "run_algorithm_screen",
        lambda config, resume: calls.append("screen") or {"passed": True},
    )
    _gate(tmp_path / "gates" / "distribution.json", "distribution")
    result = run_recovery_command(_Config(tmp_path), "screen", resume=True)
    assert result["passed"] is True
    assert calls == ["screen"]

    with pytest.raises(FileNotFoundError, match="backend gate"):
        run_recovery_command(_Config(tmp_path), "oracle-gate", resume=False)


def test_gate_json_is_canonical(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    _gate(path, "distribution")
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == "recovery_rl_gate_v2"
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_paired_evaluation_cases_use_one_nominal_and_recovery_per_source() -> None:
    manifest = build_case_manifest(
        seed=7,
        calibration=2,
        training=3,
        curve=4,
        final=5,
        severity=0.75,
    )
    cases = paired_evaluation_cases(manifest, partition="curve", count=4)
    nominal = [case for case in cases if case.family == "nominal"]
    recovery = [case for case in cases if case.family == "recovery"]
    assert len(nominal) == len(recovery) == 4
    assert {case.source_seed for case in nominal} == {
        case.source_seed for case in recovery
    }
