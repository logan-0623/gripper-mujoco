from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_vla.graph_control.cli import build_parser, main
from interaction_vla.graph_control.config import load_graph_control_config
from interaction_vla.graph_control.schema import CONDITIONS


def test_cli_exposes_complete_graph_control_workflow() -> None:
    parser = build_parser()
    for command in ("inspect", "cache", "smoke", "compare", "evaluate"):
        parsed = parser.parse_args([command, "--config", "config.yaml"])
        assert parsed.command == command
        assert parsed.config == Path("config.yaml")


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("inspect", "inspect_from_config"),
        ("cache", "cache_from_config"),
        ("smoke", "smoke_from_config"),
        ("compare", "compare_from_config"),
        ("evaluate", "evaluate_from_config"),
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


def test_cli_usage_failure_is_json(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cache"])
    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUsageError"


def test_repository_configs_lock_smoke_and_formal_matrices() -> None:
    smoke = load_graph_control_config("configs/graph_control_act_smoke_macos.yaml")
    pilot = load_graph_control_config("configs/graph_control_act_pilot_macos.yaml")

    assert smoke.conditions == pilot.conditions == CONDITIONS
    assert smoke.seeds == (0,)
    assert pilot.seeds == (0, 1, 2)
    assert smoke.training.smoke_steps == pilot.training.smoke_steps == 1
    assert smoke.split_manifest == pilot.split_manifest == Path(
        "outputs/graph_finetune/mujoco_pilot/split_manifest.json"
    )
    assert smoke.evaluation.cases_per_cell == 1
    assert pilot.evaluation.cases_per_cell == 5
    for seed in pilot.seeds:
        assert "fraction_1" in str(pilot.graph_checkpoint("predicted_reflect", seed))
        assert f"seed_{seed}" in str(pilot.graph_checkpoint("predicted_random", seed))

