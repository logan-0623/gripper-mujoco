from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_vla.graph_finetune.cli import build_parser, main


def test_cli_exposes_inspect_compare_and_evaluate() -> None:
    parser = build_parser()

    inspect = parser.parse_args(["inspect", "--config", "config.yaml"])
    compare = parser.parse_args(["compare", "--config", "config.yaml"])
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--checkpoint",
            "run/checkpoint.pt",
            "--partition",
            "validation",
        ]
    )

    assert inspect.command == "inspect"
    assert compare.command == "compare"
    assert evaluate.command == "evaluate"
    assert evaluate.config == Path("config.yaml")
    assert evaluate.checkpoint == Path("run/checkpoint.pt")
    assert evaluate.partition == "validation"


def test_cli_dispatches_compare_and_prints_json(monkeypatch, capsys) -> None:
    received: list[Path] = []
    monkeypatch.setattr(
        "interaction_vla.graph_finetune.cli.compare_from_config",
        lambda config: received.append(config) or {"passed": True},
    )

    main(["compare", "--config", "config.yaml"])

    assert received == [Path("config.yaml")]
    assert json.loads(capsys.readouterr().out) == {"passed": True}


def test_cli_forwards_evaluation_partition(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    def fake_evaluate(config, checkpoint, *, partition):
        received.update(
            config=config,
            checkpoint=checkpoint,
            partition=partition,
        )
        return {"passed": True}

    monkeypatch.setattr(
        "interaction_vla.graph_finetune.cli.evaluate_from_config", fake_evaluate
    )

    main(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--partition",
            "test",
        ]
    )

    assert received == {
        "config": Path("config.yaml"),
        "checkpoint": Path("checkpoint.pt"),
        "partition": "test",
    }
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_cli_usage_error_is_one_json_object(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["evaluate", "--config", "config.yaml"])

    payload = json.loads(capsys.readouterr().out)
    assert raised.value.code == 2
    assert payload["passed"] is False
    assert payload["error"] == "CLIUsageError"


def test_cli_runtime_error_is_one_json_object(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "interaction_vla.graph_finetune.cli.inspect_from_config",
        lambda config: (_ for _ in ()).throw(RuntimeError("invalid local dataset")),
    )

    with pytest.raises(SystemExit) as raised:
        main(["inspect", "--config", "config.yaml"])

    payload = json.loads(capsys.readouterr().out)
    assert raised.value.code == 1
    assert payload == {
        "passed": False,
        "error": "RuntimeError",
        "message": "invalid local dataset",
    }
