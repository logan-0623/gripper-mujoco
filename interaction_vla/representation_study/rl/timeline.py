from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file

from ..config import ProbeConfig
from ..extraction import extract_formal_snapshot_latents
from ..probes.training import (
    FORMAL_PRIMARY_TARGETS,
    FORMAL_SECONDARY_TARGETS,
    train_v2_probe_suite,
)
from ..state_bank.io import write_json_atomic
from .formal import (
    CONSTANT_CONTROL_CONDITIONS,
    FORMAL_SCHEMA,
    FormalRun,
    build_constant_control_timeline,
    prepare_formal_run,
)
from .snapshots import SNAPSHOT_SCHEMA, SNAPSHOT_STEPS
from .v2_config import RecoveryRLV2Config


TIMELINE_SCHEMA = "recovery_representation_timeline_v2"


@dataclass(frozen=True)
class MeasurementTimeline:
    linear_steps: tuple[int, ...] = SNAPSHOT_STEPS
    mlp_steps: tuple[int, ...] = (0, 20480)

    def __post_init__(self) -> None:
        if self.linear_steps != SNAPSHOT_STEPS:
            raise ValueError("formal linear timeline differs from snapshot schedule")
        if self.mlp_steps != (self.linear_steps[0], self.linear_steps[-1]):
            raise ValueError("formal MLP timeline must contain endpoints")


def measurement_timeline() -> MeasurementTimeline:
    return MeasurementTimeline()


@dataclass(frozen=True)
class TimelinePoint:
    condition: str
    seed_index: int
    environment_steps: int
    snapshot: Path
    snapshot_hash: str


@dataclass(frozen=True)
class SnapshotMeasurementContext:
    condition: str
    seed_index: int
    environment_steps: int
    expected_binding: str
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.condition.strip() or self.seed_index < 0:
            raise ValueError("snapshot measurement run identity is invalid")
        if self.environment_steps not in SNAPSHOT_STEPS:
            raise ValueError("snapshot measurement step is not registered")
        if len(self.expected_binding) != 64:
            raise ValueError("snapshot measurement binding must be SHA-256")


def measure_snapshot(
    snapshot: str | Path,
    context: SnapshotMeasurementContext,
) -> TimelinePoint:
    source = Path(snapshot)
    if not (source / "COMPLETED").is_file():
        raise ValueError(f"formal snapshot has no COMPLETED marker: {source}")
    manifest_path = source / "manifest.json"
    payload_path = source / "training_state.pt"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise ValueError("formal snapshot is missing its manifest or payload")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("formal snapshot schema is incompatible")
    if int(manifest.get("environment_steps", -1)) != context.environment_steps:
        raise ValueError("formal snapshot environment step differs")
    if manifest.get("binding") != context.expected_binding:
        raise ValueError("formal snapshot binding differs")
    if manifest.get("payload_sha256") != sha256_file(payload_path):
        raise ValueError("formal snapshot payload SHA-256 differs")
    return TimelinePoint(
        condition=context.condition,
        seed_index=context.seed_index,
        environment_steps=context.environment_steps,
        snapshot=source,
        snapshot_hash=fingerprint_tree(source),
    )


def _probe_config(config: RecoveryRLV2Config) -> ProbeConfig:
    return ProbeConfig(
        output_dir=config.output_dir / "formal" / "measurements",
        epochs=30,
        batch_size=64,
        learning_rate=1.0e-3,
        weight_decays=(0.0, 1.0e-4, 1.0e-3),
        seed=config.seed + 700_000,
    )


def _measurement_root(config: RecoveryRLV2Config, run: FormalRun) -> Path:
    return (
        config.output_dir
        / "formal"
        / "measurements"
        / run.condition
        / f"seed_{run.seed_index}"
    )


def _measure_point(
    config: RecoveryRLV2Config,
    run: FormalRun,
    *,
    environment_steps: int,
    snapshot: Path | None,
) -> dict[str, object]:
    root = _measurement_root(config, run) / f"step_{environment_steps:06d}"
    if snapshot is not None:
        timeline_point = measure_snapshot(
            snapshot,
            SnapshotMeasurementContext(
                condition=run.condition,
                seed_index=run.seed_index,
                environment_steps=environment_steps,
                expected_binding=run.binding,
                output_dir=root,
            ),
        )
        snapshot_hash: str | None = timeline_point.snapshot_hash
    else:
        snapshot_hash = None
    latent_root = root / "latents"
    latent = extract_formal_snapshot_latents(
        config,
        run=run,
        environment_steps=environment_steps,
        snapshot=snapshot,
        destination=latent_root,
        batch_size=16 if config.device != "cuda" else 32,
    )
    probe_config = _probe_config(config)
    model_kinds = (
        ("linear", "shallow_mlp")
        if environment_steps in measurement_timeline().mlp_steps
        else ("linear",)
    )
    probes: dict[str, dict[str, object]] = {}
    for model_kind in model_kinds:
        targets = (
            (*FORMAL_PRIMARY_TARGETS, *FORMAL_SECONDARY_TARGETS)
            if environment_steps == config.formal_steps or run.constant_control
            else FORMAL_PRIMARY_TARGETS
        )
        probes[model_kind] = train_v2_probe_suite(
            latent_path=latent_root / "latents.npz",
            records_path=config.output_dir / "state_bank_v2" / "records.jsonl",
            split_path=config.output_dir / "state_bank_v2" / "split.json",
            state_bank_manifest=config.output_dir
            / "state_bank_v2"
            / "manifest.json",
            backend="act",
            condition=run.condition,
            seed_index=run.seed_index,
            environment_steps=environment_steps,
            model_kind=model_kind,
            config=probe_config,
            destination=root / "probes" / model_kind,
            targets=targets,
        )
    return {
        "condition": run.condition,
        "seed_index": run.seed_index,
        "environment_steps": environment_steps,
        "snapshot": None if snapshot is None else snapshot.as_posix(),
        "snapshot_sha256": snapshot_hash,
        "latent_report": (latent_root / "report.json").as_posix(),
        "latent_sha256": latent["latent_sha256"],
        "probe_reports": {
            kind: (root / "probes" / kind / "report.json").as_posix()
            for kind in probes
        },
        "probe_report_hashes": {
            kind: sha256_file(root / "probes" / kind / "report.json")
            for kind in probes
        },
        "referenced_measurement_step": environment_steps,
    }


def _write_ledger(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"formal measurement ledger is immutable: {path}")
        return
    write_json_atomic(path, dict(value))


def measure_formal_timeline(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> dict[str, object]:
    run = prepare_formal_run(config, condition=condition, seed_index=seed_index)
    root = _measurement_root(config, run)
    ledger_path = root / "ledger.json"
    if ledger_path.is_file():
        report = json.loads(ledger_path.read_text(encoding="utf-8"))
        if report.get("formal_binding") != run.binding:
            raise ValueError("formal measurement ledger binding differs")
        return report
    points: list[dict[str, object]] = []
    if condition in CONSTANT_CONTROL_CONDITIONS:
        control = build_constant_control_timeline(config, run)
        measured = _measure_point(
            config,
            run,
            environment_steps=0,
            snapshot=None,
        )
        for step in config.snapshot_steps:
            point = dict(measured)
            if step not in measurement_timeline().mlp_steps:
                point["probe_reports"] = {
                    "linear": measured["probe_reports"]["linear"]
                }
                point["probe_report_hashes"] = {
                    "linear": measured["probe_report_hashes"]["linear"]
                }
            point.update(
                {
                    "environment_steps": step,
                    "constant_control": True,
                    "referenced_measurement_step": 0,
                }
            )
            points.append(point)
        control_hash = sha256_file(run.output_dir / "timeline.json")
    else:
        training_report = run.output_dir / "training_report.json"
        if not training_report.is_file():
            raise FileNotFoundError(
                f"formal training report not found: {training_report}"
            )
        training = json.loads(training_report.read_text(encoding="utf-8"))
        if training.get("binding") != run.binding or training.get("passed") is not True:
            raise ValueError("formal training report is incompatible")
        for step in config.snapshot_steps:
            snapshot = run.output_dir / "snapshots" / f"step_{step:06d}"
            points.append(
                _measure_point(
                    config,
                    run,
                    environment_steps=step,
                    snapshot=snapshot,
                )
            )
        control_hash = None
    report = {
        "schema_version": TIMELINE_SCHEMA,
        "passed": True,
        "condition": condition,
        "seed_index": seed_index,
        "training_seed": run.seed,
        "formal_binding": run.binding,
        "constant_control": run.constant_control,
        "independent_representation_runs": 1,
        "schedule": asdict(measurement_timeline()),
        "control_timeline_sha256": control_hash,
        "points": points,
    }
    _write_ledger(ledger_path, report)
    return report
