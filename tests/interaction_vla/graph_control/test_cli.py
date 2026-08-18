from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_vla.graph_control.cli import build_parser, main
from interaction_vla.graph_control.config import load_graph_control_config
from interaction_vla.graph_control.schema import ALL_CONDITIONS, ORACLE_CONDITIONS


def test_cli_exposes_complete_graph_control_workflow() -> None:
    parser = build_parser()
    for command in (
        "inspect",
        "cache",
        "smoke",
        "compare",
        "evaluate",
        "diagnose",
        "sensitivity",
        "trace",
    ):
        parsed = parser.parse_args([command, "--config", "config.yaml"])
        assert parsed.command == command
        assert parsed.config == Path("config.yaml")

    parsed = parser.parse_args(
        ["diagnose", "--config", "config.yaml", "--partition", "validation"]
    )
    assert parsed.partition == "validation"

    parsed = parser.parse_args(
        ["sensitivity", "--config", "config.yaml", "--partition", "test"]
    )
    assert parsed.partition == "test"

    parsed = parser.parse_args(
        [
            "failure-analysis",
            "--config",
            "config.yaml",
            "--traces",
            "traced_evaluation",
        ]
    )
    assert parsed.traces == Path("traced_evaluation")


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("inspect", "inspect_from_config"),
        ("cache", "cache_from_config"),
        ("smoke", "smoke_from_config"),
        ("compare", "compare_from_config"),
        ("evaluate", "evaluate_from_config"),
        ("trace", "trace_from_config"),
    ],
)
def test_cli_dispatches_and_prints_json(
    monkeypatch, capsys, command: str, target: str
) -> None:
    received = []
    monkeypatch.setattr(
        f"interaction_vla.graph_control.cli.{target}",
        lambda config: received.append(config) or {"passed": True, "command": command},
    )
    main([command, "--config", "config.yaml"])
    assert received == [Path("config.yaml")]
    assert json.loads(capsys.readouterr().out) == {
        "passed": True,
        "command": command,
    }


def test_cli_failures_are_one_json_object(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "interaction_vla.graph_control.cli.inspect_from_config",
        lambda config: (_ for _ in ()).throw(ValueError("bad split")),
    )
    with pytest.raises(SystemExit) as raised:
        main(["inspect", "--config", "config.yaml"])
    assert raised.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "passed": False,
        "error": "ValueError",
        "message": "bad split",
    }


def test_cli_scientific_gate_failure_prints_report_and_exits_one(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "interaction_vla.graph_control.cli.evaluate_from_config",
        lambda config: {
            "passed": False,
            "oracle_gate": {"passed": False, "success_delta": 0.05},
            "report_path": Path("outputs/report.json"),
        },
    )
    with pytest.raises(SystemExit) as raised:
        main(["evaluate", "--config", "config.yaml"])

    assert raised.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "passed": False,
        "oracle_gate": {"passed": False, "success_delta": 0.05},
        "report_path": "outputs/report.json",
    }


def test_cli_usage_failure_is_json(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cache"])
    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUsageError"


def test_cli_dispatches_diagnostics_partition(monkeypatch, capsys) -> None:
    received = []
    monkeypatch.setattr(
        "interaction_vla.graph_control.cli.diagnose_from_config",
        lambda config, *, partition: received.append((config, partition))
        or {"passed": True, "partition": partition or "test"},
    )

    main(
        [
            "diagnose",
            "--config",
            "config.yaml",
            "--partition",
            "validation",
        ]
    )

    assert received == [(Path("config.yaml"), "validation")]
    assert json.loads(capsys.readouterr().out) == {
        "passed": True,
        "partition": "validation",
    }


def test_cli_dispatches_sensitivity_partition(monkeypatch, capsys) -> None:
    received = []
    monkeypatch.setattr(
        "interaction_vla.graph_control.cli.sensitivity_from_config",
        lambda config, *, partition: received.append((config, partition))
        or {"passed": True, "partition": partition or "test"},
    )

    main(
        [
            "sensitivity",
            "--config",
            "config.yaml",
            "--partition",
            "validation",
        ]
    )

    assert received == [(Path("config.yaml"), "validation")]
    assert json.loads(capsys.readouterr().out) == {
        "passed": True,
        "partition": "validation",
    }


def test_cli_dispatches_failure_analysis_trace_root(monkeypatch, capsys) -> None:
    received = []
    monkeypatch.setattr(
        "interaction_vla.graph_control.cli.failure_analysis_from_config",
        lambda config, *, traces: received.append((config, traces))
        or {"passed": True, "episodes": 4},
    )

    main(
        [
            "failure-analysis",
            "--config",
            "config.yaml",
            "--traces",
            "traced_evaluation",
        ]
    )

    assert received == [
        (Path("config.yaml"), Path("traced_evaluation"))
    ]
    assert json.loads(capsys.readouterr().out) == {
        "passed": True,
        "episodes": 4,
    }


def test_cli_rejects_invalid_diagnostics_partition_as_json(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "diagnose",
                "--config",
                "config.yaml",
                "--partition",
                "heldout",
            ]
        )

    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUsageError"


def test_repository_configs_lock_smoke_and_formal_matrices() -> None:
    oracle = load_graph_control_config("configs/graph_v2_act_oracle_macos.yaml")
    pilot = load_graph_control_config("configs/graph_v2_act_pilot_macos.yaml")

    assert oracle.conditions == ORACLE_CONDITIONS
    assert pilot.conditions == ALL_CONDITIONS
    assert oracle.seeds == (0,)
    assert pilot.seeds == (0, 1, 2)
    assert oracle.training.smoke_steps == pilot.training.smoke_steps == 1
    assert oracle.split_manifest == pilot.split_manifest == Path(
        "outputs/graph_finetune/mujoco_graph_v2/split_manifest.json"
    )
    assert oracle.evaluation.cases_per_cell == 20
    assert pilot.evaluation.cases_per_cell == 5
    for seed in pilot.seeds:
        assert "fraction_1" in str(
            pilot.graph_checkpoint("predicted_reflect_v2", seed)
        )
        assert f"seed_{seed}" in str(
            pilot.graph_checkpoint("predicted_random_v2", seed)
        )


@pytest.mark.parametrize(
    "path",
    [
        "configs/graph_control_act_smoke_macos.yaml",
        "configs/graph_control_act_pilot_macos.yaml",
    ],
)
def test_legacy_graph_control_configs_are_explicitly_incompatible(path: str) -> None:
    with pytest.raises(ValueError, match="conditions.*Graph v2|missing.*oracle"):
        load_graph_control_config(path)
