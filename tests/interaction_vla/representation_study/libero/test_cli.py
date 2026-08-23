from interaction_vla.representation_study.cli import build_parser


def test_libero_cli_has_required_ordered_families() -> None:
    parser = build_parser()
    collect = parser.parse_args(
        [
            "libero",
            "state-bank",
            "collect",
            "--config",
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml",
        ]
    )
    assert collect.family == "libero"
    assert collect.libero_family == "state-bank"
    assert collect.libero_command == "collect"

    approve = parser.parse_args(
        [
            "libero",
            "state-bank",
            "approve-timelines",
            "--config",
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml",
        ]
    )
    assert approve.libero_command == "approve-timelines"

    for family, command in (
        ("stages", "plan"),
        ("latents", "extract"),
        ("probes", "run"),
        ("interventions", "run"),
        ("evaluate", "paired"),
    ):
        argv = [
                "libero",
                family,
                command,
                "--config",
                "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml",
            ]
        if family == "latents":
            argv.extend(("--stage", "pretrained"))
        args = parser.parse_args(argv)
        assert args.libero_family == family
        assert args.libero_command == command

    train = parser.parse_args(
        [
            "libero", "stages", "train", "--stage", "sft_25", "--dry-run", "--resume",
            "--config", "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml",
        ]
    )
    assert train.stage == "sft_25"
    assert train.dry_run
    assert train.resume
    snapshot = parser.parse_args(
        [
            "libero", "stages", "snapshot", "--config",
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml",
        ]
    )
    assert snapshot.libero_command == "snapshot"


def test_libero_cli_does_not_offer_rl_commands() -> None:
    parser = build_parser()
    try:
        parser.parse_args(
            [
                "libero",
                "rl",
                "train",
                "--config",
                "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml",
            ]
        )
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("LIBERO representation CLI must stop before RL")
