from interaction_vla.representation_study.cli import build_parser


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
