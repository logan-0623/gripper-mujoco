import interaction_vla.representation_study.libero.probe_runner as probe_runner
import subprocess
import sys
import interaction_vla.representation_study.libero.probe_transitions as probe_transitions
import interaction_vla.representation_study.libero.crossfit_probes as crossfit_probes
import interaction_vla.representation_study.libero.positive_control as positive_control
import interaction_vla.representation_study.libero.feature_discovery as feature_discovery
from interaction_vla.representation_study.cli import build_parser
from interaction_vla.representation_study.libero.cli import (
    _required_intervention_batch_size,
    dispatch,
)


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
        ("positive-control", "probe"),
        ("features", "discover"),
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


def test_importing_cli_does_not_eagerly_import_rl_training() -> None:
    code = (
        "import sys; import interaction_vla.representation_study.cli; "
        "assert 'interaction_vla.representation_study.rl.training' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sparse_feature_cli_routes(monkeypatch) -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_linux_cuda.yaml"
    calls: list[object] = []
    monkeypatch.setattr(
        feature_discovery,
        "discover_sparse_features",
        lambda _config: calls.append("discover") or {"passed": True},
    )
    monkeypatch.setattr(
        feature_discovery,
        "intervene_sparse_features",
        lambda _config, **kwargs: calls.append(kwargs) or {"passed": True},
    )
    monkeypatch.setattr(
        feature_discovery,
        "report_sparse_features",
        lambda _config, **kwargs: calls.append(kwargs) or {"passed": True},
    )
    discover = parser.parse_args(["libero", "features", "discover", "--config", config])
    assert dispatch(discover) == {"passed": True}
    intervene = parser.parse_args(
        [
            "libero", "features", "intervene", "--config", config,
            "--max-states", "64", "--batch-size", "32",
        ]
    )
    assert dispatch(intervene) == {"passed": True}
    report = parser.parse_args(
        ["libero", "features", "report", "--config", config, "--max-states", "64"]
    )
    assert dispatch(report) == {"passed": True}
    assert calls == ["discover", {"max_states": 64, "batch_size": 32}, {"max_states": 64}]


def test_positive_control_cli_binds_checkpoint_and_routes(monkeypatch) -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_linux_cuda.yaml"
    plan = parser.parse_args(
        [
            "libero",
            "positive-control",
            "plan",
            "--config",
            config,
            "--checkpoint",
            "/tmp/model",
            "--eval-dir",
            "/tmp/eval",
        ]
    )
    calls: list[object] = []
    monkeypatch.setattr(
        positive_control,
        "plan_positive_control",
        lambda _config, **kwargs: calls.append(kwargs) or {"passed": True},
    )
    assert dispatch(plan) == {"passed": True}
    assert calls == [{"checkpoint": plan.checkpoint, "eval_dir": plan.eval_dir}]

    intervene = parser.parse_args(
        [
            "libero",
            "positive-control",
            "intervene",
            "--config",
            config,
            "--factor",
            "contact",
            "--max-states",
            "64",
            "--batch-size",
            "32",
        ]
    )
    assert intervene.factor == "contact"


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


def test_longitudinal_recruitment_cli_has_audit_and_gated_run() -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_linux_cuda.yaml"
    audit = parser.parse_args(
        ["libero", "interventions", "audit", "--config", config]
    )
    assert audit.libero_command == "audit"
    run = parser.parse_args(
        [
            "libero", "interventions", "run", "--config", config,
            "--max-states", "64", "--batch-size", "8", "--specificity-only",
        ]
    )
    assert run.max_states == 64
    assert run.batch_size == 8
    assert run.specificity_only


def test_intervention_batch_size_matches_frozen_latent_runtime(tmp_path) -> None:
    for condition in ("pretrained", "d25_u16070", "d100_u16617", "d100_u66470"):
        report = tmp_path / "latents" / condition / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text('{"runtime":{"batch_size":32}}')
    assert _required_intervention_batch_size(tmp_path) == 32

    changed = tmp_path / "latents" / "d100_u66470" / "report.json"
    changed.write_text('{"runtime":{"batch_size":16}}')
    try:
        _required_intervention_batch_size(tmp_path)
    except ValueError as error:
        assert "batch sizes differ" in str(error)
    else:
        raise AssertionError("mixed latent runtime batch sizes must fail")
