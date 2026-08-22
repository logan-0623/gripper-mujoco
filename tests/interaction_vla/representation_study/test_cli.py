from interaction_vla.representation_study.cli import build_parser
import pytest


def test_state_bank_cli_commands_are_parseable() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["state-bank", "collect", "--config", "configs/representation_study/icra_act_macos.yaml"]
    )
    assert args.family == "state-bank"
    assert args.command == "collect"

    args = parser.parse_args(
        [
            "latents",
            "extract",
            "--config",
            "configs/representation_study/icra_act_macos.yaml",
            "--backend",
            "act",
            "--stage",
            "sft",
            "--limit",
            "4",
        ]
    )
    assert args.family == "latents"
    assert args.limit == 4

    args = parser.parse_args(
        [
            "sft", "train", "--config", "study.yaml", "--backend", "smolvla",
            "--stage", "sft",
        ]
    )
    assert args.family == "sft"
    assert args.stage == "sft"

    args = parser.parse_args(
        [
            "rl", "train", "--config", "study.yaml", "--backend", "act",
            "--stage", "rl_representation", "--resume",
        ]
    )
    assert args.family == "rl"
    assert args.resume is True

    args = parser.parse_args(
        [
            "policy", "evaluate", "--config", "study.yaml", "--backend", "act",
            "--stage", "sft", "--force",
        ]
    )
    assert args.family == "policy"
    assert args.force is True

    args = parser.parse_args(
        [
            "measure", "run", "--config", "study.yaml", "--backend", "smolvla",
            "--stage", "sft", "--secondary-probe", "--closed-loop-intervention",
        ]
    )
    assert args.family == "measure"
    assert args.secondary_probe and args.closed_loop_intervention

    args = parser.parse_args(["report", "build", "--config", "study.yaml"])
    assert args.family == "report"


@pytest.mark.parametrize(
    "command",
    ["calibrate", "screen", "oracle-gate", "anchor-screen"],
)
def test_recovery_rl_commands_parse(command: str) -> None:
    args = build_parser().parse_args(
        ["recovery-rl", command, "--config", "v2.yaml"]
    )
    assert args.family == "recovery-rl"
    assert args.command == command
    assert args.resume is False


def test_recovery_training_commands_accept_resume() -> None:
    parser = build_parser()
    for command in ("screen", "anchor-screen"):
        args = parser.parse_args(
            ["recovery-rl", command, "--config", "v2.yaml", "--resume"]
        )
        assert args.resume is True


@pytest.mark.parametrize(
    "command",
    ["state-bank", "train", "measure", "evaluate", "report"],
)
def test_formal_commands_parse(command: str) -> None:
    arguments = ["recovery-rl", "formal", command, "--config", "v2.yaml"]
    if command in {"train", "measure", "evaluate"}:
        arguments.extend(("--condition", "rl_head", "--seed-index", "0"))
    args = build_parser().parse_args(arguments)
    assert args.family == "recovery-rl"
    assert args.command == "formal"
    assert args.formal_command == command


def test_formal_train_accepts_only_training_conditions_and_resume() -> None:
    args = build_parser().parse_args(
        [
            "recovery-rl",
            "formal",
            "train",
            "--config",
            "v2.yaml",
            "--condition",
            "rl_representation",
            "--seed-index",
            "2",
            "--resume",
        ]
    )
    assert args.condition == "rl_representation"
    assert args.seed_index == 2
    assert args.resume is True
