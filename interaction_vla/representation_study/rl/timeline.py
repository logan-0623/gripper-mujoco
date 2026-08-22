from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file

from ..config import ProbeConfig
from ..extraction import extract_formal_snapshot_latents
from ..probes.training import (
    FORMAL_PRIMARY_TARGETS,
    FORMAL_SECONDARY_TARGETS,
    train_v2_probe_suite,
)
from ..state_bank.io import write_json_atomic
from ..taps.registry import registered_taps
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


def validate_probe_report(
    report: Mapping[str, object],
    *,
    report_path: Path,
    run: FormalRun,
    environment_steps: int,
    model_kind: str,
    expected_targets: tuple[str, ...],
    latent_sha256: str,
    state_bank_manifest_sha256: str,
) -> None:
    expected_header = {
        "passed": True,
        "schema_version": "recovery_frozen_probe_v2",
        "backend": "act",
        "condition": run.condition,
        "seed_index": run.seed_index,
        "environment_steps": environment_steps,
        "model_kind": model_kind,
        "targets": list(expected_targets),
        "latent_sha256": latent_sha256,
        "state_bank_manifest_sha256": state_bank_manifest_sha256,
        "primary_split": "test",
        "selection_split": "validation",
    }
    if any(report.get(name) != value for name, value in expected_header.items()):
        raise ValueError("formal probe report metadata differs")
    rows = report.get("rows")
    taps = tuple(tap.tap_id for tap in registered_taps("act"))
    expected_pairs = {(tap, target) for tap in taps for target in expected_targets}
    if not isinstance(rows, list) or len(rows) != len(expected_pairs):
        raise ValueError("formal probe report has incomplete tap-target coverage")
    observed_pairs: set[tuple[str, str]] = set()
    target_specs = {
        "geometry": ("continuous", 16, []),
        "phase": ("categorical", 6, []),
        "recovery_state": ("categorical", 3, []),
        "recovery_type": ("categorical", 7, []),
        "next_relation": ("structured", 20, [8, 5, 7]),
        "contact": ("binary", 2, []),
        "stable_grasp": ("binary", 2, []),
    }
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("formal probe row must be a mapping")
        tap = str(raw.get("tap", ""))
        target = str(raw.get("target", ""))
        pair = (tap, target)
        if pair not in expected_pairs or pair in observed_pairs:
            raise ValueError("formal probe tap-target row is unknown or duplicated")
        observed_pairs.add(pair)
        expected_role = "primary" if target in FORMAL_PRIMARY_TARGETS else "secondary"
        row_expected = {
            "schema_version": "recovery_frozen_probe_v2",
            "backend": "act",
            "condition": run.condition,
            "seed_index": run.seed_index,
            "environment_steps": environment_steps,
            "tap": tap,
            "target": target,
            "target_role": expected_role,
            "model_kind": model_kind,
            "state_bank_manifest_sha256": state_bank_manifest_sha256,
            "latent_sha256": latent_sha256,
        }
        if any(raw.get(name) != value for name, value in row_expected.items()):
            raise ValueError("formal probe row binding differs")
        target_kind, output_dim, head_widths = target_specs[target]
        if (
            raw.get("target_kind") != target_kind
            or raw.get("output_dim") != output_dim
            or raw.get("head_widths") != head_widths
            or type(raw.get("input_dim")) is not int
            or int(raw["input_dim"]) < 1
        ):
            raise ValueError("formal probe target head metadata differs")
        metrics = raw.get("metrics")
        baseline = raw.get("baseline_metrics")
        sample_counts = raw.get("sample_counts")
        if not all(isinstance(value, Mapping) for value in (metrics, baseline, sample_counts)):
            raise ValueError("formal probe row metrics are incomplete")
        if set(metrics) != {"train", "validation", "test"} or set(baseline) != set(metrics):
            raise ValueError("formal probe metric partitions differ")
        for partitions in (metrics, baseline):
            for values in partitions.values():
                if (
                    not isinstance(values, Mapping)
                    or not values
                    or any(not np.isfinite(float(value)) for value in values.values())
                ):
                    raise ValueError("formal probe metrics must be finite mappings")
        primary_metric = "r2" if target == "geometry" else "balanced_accuracy"
        if primary_metric not in metrics["test"]:
            raise ValueError("formal probe row has no registered primary metric")
        if set(sample_counts) != {"train", "validation", "test"} or any(
            type(value) is not int or value < 1 for value in sample_counts.values()
        ):
            raise ValueError("formal probe sample counts are incompatible")
        checkpoint = report_path.parent / tap / f"{target}.pt"
        if raw.get("checkpoint") != checkpoint.as_posix():
            raise ValueError("formal probe checkpoint path differs")
        if not checkpoint.is_file() or raw.get("checkpoint_sha256") != sha256_file(
            checkpoint
        ):
            raise ValueError("formal probe checkpoint hash differs")
    if observed_pairs != expected_pairs:
        raise ValueError("formal probe report tap-target coverage differs")


def validate_measurement_ledger(
    config: RecoveryRLV2Config,
    run: FormalRun,
    report: Mapping[str, object],
) -> None:
    control_timeline = run.output_dir / "timeline.json"
    expected_control_hash = (
        sha256_file(control_timeline) if run.constant_control else None
    )
    expected_header = {
        "schema_version": TIMELINE_SCHEMA,
        "passed": True,
        "condition": run.condition,
        "seed_index": run.seed_index,
        "training_seed": run.seed,
        "formal_binding": run.binding,
        "constant_control": run.constant_control,
        "independent_representation_runs": 1,
        "schedule": asdict(measurement_timeline()),
        "control_timeline_sha256": expected_control_hash,
    }
    differing = [
        name for name, value in expected_header.items() if report.get(name) != value
    ]
    if differing:
        raise ValueError(
            "formal measurement ledger metadata differs: " + ", ".join(differing)
        )
    points = report.get("points")
    if not isinstance(points, list) or len(points) != len(config.snapshot_steps):
        raise ValueError("formal measurement ledger does not contain six points")
    if [int(point.get("environment_steps", -1)) for point in points] != list(
        config.snapshot_steps
    ):
        raise ValueError("formal measurement ledger schedule differs")
    bank_hash = sha256_file(config.output_dir / "state_bank_v2" / "manifest.json")
    state_bank_records = [
        json.loads(line)
        for line in (
            config.output_dir / "state_bank_v2" / "records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_state_ids = [str(record["state_id"]) for record in state_bank_records]
    tap_ids = tuple(tap.tap_id for tap in registered_taps("act"))
    for point in points:
        step = int(point["environment_steps"])
        source_step = int(point.get("referenced_measurement_step", step))
        if run.constant_control:
            if source_step != 0 or point.get("snapshot_sha256") is not None:
                raise ValueError("constant measurement must reference the step-zero policy")
        else:
            snapshot = run.output_dir / "snapshots" / f"step_{step:06d}"
            measured = measure_snapshot(
                snapshot,
                SnapshotMeasurementContext(
                    condition=run.condition,
                    seed_index=run.seed_index,
                    environment_steps=step,
                    expected_binding=run.binding,
                    output_dir=snapshot,
                ),
            )
            if point.get("snapshot_sha256") != measured.snapshot_hash:
                raise ValueError("formal measurement snapshot hash differs")
        latent_report_path = Path(str(point.get("latent_report", "")))
        if not latent_report_path.is_file():
            raise FileNotFoundError(
                f"formal latent report not found: {latent_report_path}"
            )
        latent_report = json.loads(latent_report_path.read_text(encoding="utf-8"))
        latent_path = latent_report_path.parent / "latents.npz"
        latent_hash = sha256_file(latent_path)
        checkpoint = Path(
            run.parent_checkpoint if run.constant_control else config.sft_checkpoint
        )
        checkpoint_hash = (
            fingerprint_tree(checkpoint)
            if checkpoint.is_dir()
            else sha256_file(checkpoint)
        )
        latent_expected = {
            "passed": True,
            "schema_version": "recovery_stage_latents_v2",
            "condition": run.condition,
            "seed_index": run.seed_index,
            "environment_steps": source_step,
            "formal_binding": run.binding,
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_hash,
            "latent_sha256": latent_hash,
            "snapshot_sha256": point.get("snapshot_sha256"),
            "state_bank_manifest_sha256": bank_hash,
            "records": len(expected_state_ids),
        }
        if any(latent_report.get(name) != value for name, value in latent_expected.items()):
            raise ValueError("formal latent report binding or hash differs")
        with np.load(latent_path, allow_pickle=False) as archive:
            expected_arrays = {"state_id", "__action__", *tap_ids}
            if set(archive.files) != expected_arrays:
                raise ValueError("formal latent artifact tap set differs")
            if archive["state_id"].astype(str).tolist() != expected_state_ids:
                raise ValueError("formal latent artifact State Bank rows differ")
            for name in ("__action__", *tap_ids):
                values = archive[name]
                if (
                    values.shape[0] != len(expected_state_ids)
                    or not np.issubdtype(values.dtype, np.floating)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("formal latent values are incomplete or non-finite")
            expected_tap_shapes = {
                name: list(archive[name].shape[1:]) for name in tap_ids
            }
        if latent_report.get("taps") != expected_tap_shapes:
            raise ValueError("formal latent report tap shapes differ")
        if point.get("latent_sha256") != latent_hash:
            raise ValueError("formal measurement latent hash differs")
        probe_reports = point.get("probe_reports")
        probe_hashes = point.get("probe_report_hashes")
        if not isinstance(probe_reports, Mapping) or not isinstance(probe_hashes, Mapping):
            raise ValueError("formal measurement probe report mapping is missing")
        expected_models = (
            {"linear", "shallow_mlp"}
            if step in measurement_timeline().mlp_steps
            else {"linear"}
        )
        if set(probe_reports) != expected_models or set(probe_hashes) != expected_models:
            raise ValueError("formal measurement probe schedule differs")
        for model_kind, raw_path in probe_reports.items():
            probe_path = Path(str(raw_path))
            if not probe_path.is_file() or probe_hashes[model_kind] != sha256_file(probe_path):
                raise ValueError("formal measurement probe report hash differs")
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            targets = (
                (*FORMAL_PRIMARY_TARGETS, *FORMAL_SECONDARY_TARGETS)
                if run.constant_control or source_step == config.formal_steps
                else FORMAL_PRIMARY_TARGETS
            )
            validate_probe_report(
                probe,
                report_path=probe_path,
                run=run,
                environment_steps=source_step,
                model_kind=str(model_kind),
                expected_targets=tuple(targets),
                latent_sha256=latent_hash,
                state_bank_manifest_sha256=bank_hash,
            )


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
        validate_measurement_ledger(config, run, report)
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
    validate_measurement_ledger(config, run, report)
    return report
