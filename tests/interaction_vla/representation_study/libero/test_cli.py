import interaction_vla.representation_study.libero.probe_runner as probe_runner
import interaction_vla.representation_study.libero.probe_transitions as probe_transitions
import interaction_vla.representation_study.libero.crossfit_probes as crossfit_probes
from interaction_vla.representation_study.cli import build_parser
from interaction_vla.representation_study.libero.cli import dispatch


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
        ("longitudinal", "plan"),
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

    longitudinal_extract = parser.parse_args(
        [
            "libero", "longitudinal", "extract", "--condition", "d25_u16070",
            "--config", "configs/representation_study/libero_smolvla_linux_cuda.yaml",
        ]
    )
    assert longitudinal_extract.condition == "d25_u16070"
    for command in ("probes", "probe-report"):
        longitudinal_probe = parser.parse_args(
            [
                "libero", "longitudinal", command, "--config",
                "configs/representation_study/libero_smolvla_linux_cuda.yaml",
            ]
        )
        assert longitudinal_probe.libero_command == command

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


def test_probe_commands_automatically_enrich_adjacent_stage_deltas(
    monkeypatch,
) -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    calls: list[str] = []

    monkeypatch.setattr(
        probe_runner,
        "run_probe_study",
        lambda _config: calls.append("run") or {"probe": "raw"},
    )
    monkeypatch.setattr(
        probe_transitions,
        "enrich_adjacent_stage_deltas",
        lambda _config: calls.append("enrich") or {"probe": "enriched"},
    )

    run_args = parser.parse_args(
        ["libero", "probes", "run", "--config", config]
    )
    assert dispatch(run_args) == {"probe": "enriched"}
    assert calls == ["run", "enrich"]

    calls.clear()
    report_args = parser.parse_args(
        ["libero", "probes", "report", "--config", config]
    )
    assert dispatch(report_args) == {"probe": "enriched"}
    assert calls == ["enrich"]


def test_longitudinal_crossfit_probe_cli_routes_without_loading_models(monkeypatch) -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_linux_cuda.yaml"
    calls: list[str] = []
    monkeypatch.setattr(
        crossfit_probes,
        "run_crossfit_probe_study",
        lambda _config: calls.append("run") or {"passed": True},
    )
    monkeypatch.setattr(
        crossfit_probes,
        "inspect_crossfit_probe_report",
        lambda _config: calls.append("inspect") or {"passed": True},
    )

    run_args = parser.parse_args(
        ["libero", "longitudinal", "probes", "--config", config]
    )
    assert dispatch(run_args) == {"passed": True}
    report_args = parser.parse_args(
        ["libero", "longitudinal", "probe-report", "--config", config]
    )
    assert dispatch(report_args) == {"passed": True}
    assert calls == ["run", "inspect"]
