import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction_vla.lerobot_bridge.cli import (
    build_parser,
    main,
    write_smoke_report,
)


def test_cli_exposes_only_local_explicit_commands() -> None:
    parser = build_parser()
    for command in ("collect", "validate", "act-check", "act-train", "smoke"):
        args = parser.parse_args(
            [command, "--config", "configs/lerobot_act_smoke_macos.yaml"]
        )
        assert args.command == command
    rollout = parser.parse_args(
        [
            "rollout",
            "--config",
            "configs/lerobot_act_smoke_macos.yaml",
            "--checkpoint",
            "outputs/lerobot/act_smoke/checkpoint",
        ]
    )
    assert rollout.command == "rollout"
    help_text = parser.format_help().lower()
    assert "push" not in help_text
    assert "upload" not in help_text


def test_cli_exposes_act_diagnostics_and_recovery_commands() -> None:
    diagnose = build_parser().parse_args(
        [
            "act-diagnose",
            "--config",
            "recovery.yaml",
            "--checkpoint",
            "checkpoint",
        ]
    )
    recovery = build_parser().parse_args(
        [
            "act-recovery",
            "--config",
            "recovery.yaml",
            "--checkpoint",
            "checkpoint",
        ]
    )

    assert diagnose.command == "act-diagnose"
    assert diagnose.checkpoint == Path("checkpoint")
    assert recovery.command == "act-recovery"
    assert recovery.checkpoint == Path("checkpoint")


@pytest.mark.parametrize(
    ("command", "target"),
    (
        ("act-diagnose", "evaluate_checkpoint_actions"),
        ("act-recovery", "evaluate_recovery"),
    ),
)
def test_act_evaluation_commands_dispatch_checkpoint(
    command: str, target: str, monkeypatch, capsys
) -> None:
    received = {}

    def fake_evaluate(*args):
        received["args"] = args
        return {"passed": True}

    monkeypatch.setattr(f"interaction_vla.lerobot_bridge.cli.{target}", fake_evaluate)

    main(
        [
            command,
            "--config",
            "recovery.yaml",
            "--checkpoint",
            "checkpoint",
        ]
    )

    assert received["args"] == (Path("recovery.yaml"), Path("checkpoint"))
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_smoke_dispatch_order(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.validate_from_config",
        lambda *a, **k: calls.append("validate") or {},
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.check_from_config",
        lambda *a, **k: calls.append("act-check") or {},
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.train_from_config",
        lambda *a, **k: calls.append("act-train")
        or {"checkpoint": "checkpoint"},
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.rollout_from_config",
        lambda *a, **k: calls.append("rollout") or {},
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.write_smoke_report",
        lambda *a, **k: calls.append("report") or {},
    )

    main(["smoke", "--config", "configs/lerobot_act_smoke_macos.yaml"])

    assert calls == ["validate", "act-check", "act-train", "rollout", "report"]


def test_smoke_pass_does_not_require_task_success(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "act"
    dataset.mkdir()
    checkpoint.mkdir()
    config = SimpleNamespace(
        dataset=SimpleNamespace(root=dataset, episodes=5),
        act=SimpleNamespace(output_dir=output, steps=500),
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.load_bridge_config", lambda path: config
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.fingerprint_tree",
        lambda path: "a" * 64 if Path(path) == dataset else "b" * 64,
    )
    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.source_fingerprint", lambda: "c" * 64
    )

    report = write_smoke_report(
        "unused.yaml",
        {"passed": True, "episodes": 5},
        SimpleNamespace(loss=1.0, gradient_norm=2.0, reload_max_abs_error=0.0),
        {"steps": 500, "losses": [1.0] * 500, "checkpoint": checkpoint},
        {"finite_rollout": True, "task_success": False},
    )

    assert report["passed"] is True
    assert report["task_success"] is False
    assert (output / "smoke_report.json").is_file()
    assert (output / "training_summary.json").is_file()


def test_cli_usage_failure_is_one_json_object(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["validate"])

    payload = json.loads(capsys.readouterr().out)
    assert raised.value.code == 2
    assert payload["passed"] is False
    assert payload["error"] == "CLIUsageError"


def test_rollout_cli_forwards_gif_destination(monkeypatch, capsys) -> None:
    received = {}

    def fake_rollout(*args, **kwargs):
        received.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.rollout_from_config", fake_rollout
    )

    main(
        [
            "rollout",
            "--config",
            "configs/lerobot_act_smoke_macos.yaml",
            "--checkpoint",
            "outputs/lerobot/act_smoke/checkpoint",
            "--gif",
            "outputs/lerobot/act_smoke/rollout.gif",
        ]
    )

    assert received["gif_path"] == Path("outputs/lerobot/act_smoke/rollout.gif")
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_rollout_cli_defaults_to_no_gif() -> None:
    args = build_parser().parse_args(
        [
            "rollout",
            "--config",
            "configs/lerobot_act_smoke_macos.yaml",
            "--checkpoint",
            "outputs/lerobot/act_smoke/checkpoint",
        ]
    )

    assert args.gif_path is None
