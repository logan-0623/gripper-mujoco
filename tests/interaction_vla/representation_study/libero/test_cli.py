
from interaction_vla.representation_study.cli import build_parser


def test_libero_cli_exposes_state_bank_commands() -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    audit = parser.parse_args(["libero", "audit", "--config", config])
    assert audit.libero_family == "audit"
    for command in ("collect", "inspect", "visualize", "approve-timelines"):
        args = parser.parse_args(["libero", "state-bank", command, "--config", config])
        assert args.family == "libero"
        assert args.libero_family == "state-bank"
        assert args.libero_command == command


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
