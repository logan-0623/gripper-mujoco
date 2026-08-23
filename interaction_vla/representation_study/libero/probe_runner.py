from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .latents import load_latent_cache
from .probes import (
    FACTOR_NAMES,
    STUDY_STAGES,
    STUDY_TAPS,
    build_stage_tap_factor_grid,
    classification_metrics,
    geometry_metrics,
    run_linear_probe,
    run_shallow_mlp_probe,
)
from .schema import StateRecord
from .splits import PARTITIONS, SplitManifest
from .state_bank import load_state_bank
from .visualize import validate_annotation_timeline_report


PROBE_REPORT_SCHEMA = "libero_stage_tap_factor_probe_report_v1"


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factor_target(record: StateRecord, factor: str) -> object:
    if factor not in FACTOR_NAMES:
        raise ValueError(f"unknown interaction factor: {factor}")
    if not bool(getattr(record.labels.applicability, factor)):
        raise ValueError(f"factor {factor} is not applicable to {record.state_id}")
    if factor == "entity":
        assert record.labels.entity is not None
        return record.labels.entity.target
    if factor == "geometry":
        assert record.labels.geometry is not None
        label = record.labels.geometry
        return np.asarray(
            (*label.gripper_to_target, *label.target_to_goal,
             label.gripper_target_distance, label.target_goal_distance),
            dtype=np.float32,
        )
    if factor == "contact":
        assert record.labels.contact is not None
        return bool(record.labels.contact.gripper_target)
    if factor == "stable_grasp":
        assert record.labels.stable_grasp is not None
        return bool(record.labels.stable_grasp)
    if factor == "phase":
        assert record.labels.phase is not None
        return record.labels.phase
    assert record.labels.next_relation is not None
    relation = record.labels.next_relation
    return (
        relation.subject_role,
        relation.predicate,
        relation.object_role,
        relation.operator,
    )


def partition_indices(
    records: Sequence[StateRecord],
    assignments: Mapping[str, str],
    *,
    applicable: np.ndarray,
) -> dict[str, list[int]]:
    if applicable.shape != (len(records),):
        raise ValueError("applicability mask must match State Bank rows")
    result = {partition: [] for partition in PARTITIONS}
    for index, record in enumerate(records):
        if not applicable[index]:
            continue
        partition = assignments.get(record.state_id)
        if partition not in result:
            raise ValueError(f"missing or invalid split assignment for {record.state_id}")
        result[partition].append(index)
    return result


def _cluster_key(record: StateRecord, group_unit: str) -> str:
    if group_unit == "task":
        return f"{record.suite}:{record.task_id}"
    if group_unit == "episode":
        return f"{record.suite}:{record.task_id}:{record.source_episode_id}"
    raise ValueError(f"unsupported bootstrap group unit: {group_unit}")


def _primary_metric(factor: str) -> tuple[str, bool]:
    if factor == "geometry":
        return "normalized_mae", False
    if factor in {"contact", "stable_grasp"}:
        return "auprc", True
    return "macro_f1", True


def _shortcut_baseline_metrics(
    *,
    records: Sequence[StateRecord],
    source_indices: np.ndarray,
    targets: np.ndarray,
    local_parts: Mapping[str, Sequence[int]],
    factor: str,
) -> dict[str, dict[str, float]]:
    train = np.asarray(local_parts["train"], dtype=np.int64)
    test = np.asarray(local_parts["test"], dtype=np.int64)
    task_keys = np.asarray(
        [f"{records[index].suite}:{records[index].task_id}" for index in source_indices]
    )
    instruction_keys = np.asarray(
        [records[index].language for index in source_indices]
    )
    episode_max: dict[tuple[str, int, str], int] = {}
    for record in records:
        key = (record.suite, record.task_id, record.source_episode_id)
        episode_max[key] = max(episode_max.get(key, 0), record.frame_index)
    time_bins = np.asarray(
        [
            min(
                9,
                int(
                    10
                    * records[index].frame_index
                    / max(
                        episode_max[
                            (
                                records[index].suite,
                                records[index].task_id,
                                records[index].source_episode_id,
                            )
                        ]
                        + 1,
                        1,
                    )
                ),
            )
            for index in source_indices
        ],
        dtype=np.int64,
    )

    def predict_by_group(groups: np.ndarray) -> np.ndarray:
        fallback = (
            targets[train].mean(axis=0)
            if factor == "geometry"
            else min(
                set(targets[train].tolist()),
                key=lambda value: (-int(np.sum(targets[train] == value)), str(value)),
            )
        )
        predictions: list[object] = []
        for index in test:
            candidates = train[groups[train] == groups[index]]
            if not len(candidates):
                predictions.append(fallback)
            elif factor == "geometry":
                predictions.append(targets[candidates].mean(axis=0))
            else:
                values = targets[candidates]
                predictions.append(
                    min(
                        set(values.tolist()),
                        key=lambda value: (-int(np.sum(values == value)), str(value)),
                    )
                )
        if factor == "geometry":
            return np.stack(predictions).astype(np.float64)
        return np.asarray(predictions, dtype=targets.dtype)

    result: dict[str, dict[str, float]] = {}
    for name, groups in (
        ("task_id", task_keys),
        ("instruction", instruction_keys),
        ("normalized_time_bin", time_bins),
    ):
        prediction = predict_by_group(groups)
        if factor == "geometry":
            scale = np.ptp(targets[train], axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            result[name] = geometry_metrics(
                targets[test], prediction, normalization_scale=scale
            )
        else:
            binary = factor in {"contact", "stable_grasp"}
            if binary:
                train_classes = sorted(set(targets[train].tolist()), key=str)
                positive = train_classes[-1]
                encoded_target = (targets[test] == positive).astype(np.int64)
                encoded_prediction = (prediction == positive).astype(np.int64)
                result[name] = classification_metrics(
                    encoded_target,
                    encoded_prediction,
                    score=encoded_prediction.astype(np.float64),
                    binary=True,
                )
            else:
                result[name] = classification_metrics(targets[test], prediction)
    return result


def _accessibility_decision(
    *,
    confidence_interval: Mapping[str, object],
    threshold: float,
    higher_is_better: bool,
) -> bool | None:
    low = confidence_interval.get("low")
    high = confidence_interval.get("high")
    if low is None or high is None:
        return None
    lower = float(low)
    upper = float(high)
    return lower > threshold if higher_is_better else upper < threshold


def _bootstrap_ci(
    *,
    factor: str,
    target: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray | None,
    clusters: Sequence[str],
    samples: int,
    confidence: float,
    seed: int,
    normalization_scale: np.ndarray | None = None,
) -> dict[str, object]:
    unique = tuple(sorted(set(clusters)))
    if len(unique) < 2:
        return {"unit": "group", "groups": len(unique), "low": None, "high": None}
    by_cluster = {
        cluster: np.asarray([i for i, value in enumerate(clusters) if value == cluster])
        for cluster in unique
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    metric_name, _ = _primary_metric(factor)
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[str(cluster)] for cluster in chosen])
        if factor == "geometry":
            assert normalization_scale is not None
            value = geometry_metrics(
                target[indices], prediction[indices], normalization_scale=normalization_scale
            )[metric_name]
        else:
            binary = factor in {"contact", "stable_grasp"}
            sampled_score = None if score is None else score[indices]
            value = classification_metrics(
                target[indices],
                prediction[indices],
                score=sampled_score,
                binary=binary,
                labels=np.unique(target),
            )[metric_name]
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"unit": "group", "groups": len(unique), "low": None, "high": None}
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "group",
        "groups": len(unique),
        "samples": len(values),
        "confidence_level": confidence,
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
    }


def _run_cell(
    *,
    config: LiberoStudyConfig,
    records: Sequence[StateRecord],
    features: np.ndarray,
    split: SplitManifest,
    factor: str,
    seed_offset: int,
    run_capacity_check: bool,
) -> dict[str, object]:
    applicable = np.asarray(
        [bool(getattr(record.labels.applicability, factor)) for record in records],
        dtype=bool,
    )
    partitions = partition_indices(records, split.assignments, applicable=applicable)
    targets_all = [factor_target(record, factor) for record, use in zip(records, applicable, strict=True) if use]
    source_indices = np.flatnonzero(applicable)
    source_to_local = {int(source): local for local, source in enumerate(source_indices)}
    local_parts = {
        name: [source_to_local[index] for index in indices]
        for name, indices in partitions.items()
    }
    selected_features = features[applicable]
    if factor == "geometry":
        targets = np.stack(targets_all).astype(np.float32)
        task = "regression"
    else:
        targets = np.asarray(
            ["|".join(item) if isinstance(item, tuple) else item for item in targets_all]
        )
        task = "classification"
        if factor in {"contact", "stable_grasp"}:
            for partition, indices in local_parts.items():
                if len(set(targets[np.asarray(indices, dtype=np.int64)].tolist())) < 2:
                    raise ValueError(
                        f"{factor} {partition} partition lacks both binary classes; AUPRC is undefined"
                    )
    linear = run_linear_probe(
        selected_features,
        targets,
        train_indices=local_parts["train"],
        validation_indices=local_parts["validation"],
        test_indices=local_parts["test"],
        task=task,
        seed=config.seed + seed_offset,
        l2_grid=config.probes.linear_l2,
        epochs=config.probes.linear_epochs,
    )
    mlp = (
        run_shallow_mlp_probe(
            selected_features,
            targets,
            train_indices=local_parts["train"],
            validation_indices=local_parts["validation"],
            test_indices=local_parts["test"],
            task=task,
            seed=config.seed + seed_offset,
            hidden_dim=config.probes.mlp_hidden_dim,
            l2=float(linear["selected_l2"]),
            epochs=config.probes.mlp_epochs,
        )
        if run_capacity_check
        else {"status": "not_run", "reason": "capacity check is preregistered on the primary task-group split"}
    )
    test_source_indices = partitions["test"]
    clusters = [_cluster_key(records[index], split.group_unit) for index in test_source_indices]
    target = np.asarray(linear["test_target"])
    prediction = np.asarray(linear["test_prediction"])
    score_value = linear.get("test_score")
    score = None if score_value is None else np.asarray(score_value, dtype=np.float64)
    normalization_scale = None
    if factor == "geometry":
        train_targets = targets[np.asarray(local_parts["train"], dtype=np.int64)]
        normalization_scale = np.ptp(train_targets, axis=0)
        normalization_scale = np.where(normalization_scale > 1e-8, normalization_scale, 1.0)
    confidence_interval = _bootstrap_ci(
        factor=factor,
        target=target,
        prediction=prediction,
        score=score,
        clusters=clusters,
        samples=config.probes.bootstrap_samples,
        confidence=config.probes.confidence_level,
        seed=config.seed + seed_offset,
        normalization_scale=normalization_scale,
    )
    metric_name, higher_is_better = _primary_metric(factor)
    metric = float(linear["test_metrics"][metric_name])
    baseline = float(linear["baseline_metrics"][metric_name])
    shortcut_baselines = _shortcut_baseline_metrics(
        records=records,
        source_indices=source_indices,
        targets=targets,
        local_parts=local_parts,
        factor=factor,
    )
    shortcut_values = [float(value[metric_name]) for value in shortcut_baselines.values()]
    threshold = max([baseline, *shortcut_values]) if higher_is_better else min([baseline, *shortcut_values])
    accessible = _accessibility_decision(
        confidence_interval=confidence_interval,
        threshold=threshold,
        higher_is_better=higher_is_better,
    )
    return {
        "status": "complete",
        "accessible": accessible,
        "primary_metric_name": metric_name,
        "primary_metric": metric,
        "baseline_metric": baseline,
        "shortcut_baselines": shortcut_baselines,
        "accessibility_threshold": threshold,
        "confidence_interval": confidence_interval,
        "applicable_states": int(applicable.sum()),
        "states_by_partition": {name: len(values) for name, values in partitions.items()},
        "linear": linear,
        "capacity_check": mlp,
    }


def run_probe_study(config: LiberoStudyConfig) -> dict[str, object]:
    bank_root = config.output_dir / "state_bank"
    records, bank_manifest, task_split, episode_split = load_state_bank(bank_root)
    timeline_path = config.output_dir / "timelines" / "report.json"
    if not timeline_path.is_file():
        raise ValueError("annotation timeline gate has not run; run state-bank visualize")
    bank_hash = _file_sha256(bank_root / "manifest.json")
    validate_annotation_timeline_report(
        timeline_path,
        state_bank_manifest=bank_root / "manifest.json",
        require_approved=True,
    )
    completed: list[dict[str, object]] = []
    latent_bindings: dict[str, str] = {}
    split_reports: dict[str, list[dict[str, object]]] = {
        "task_group": [],
        "episode_group": [],
    }
    config_hash = _file_sha256(config.source_path)
    implementation_hash = hashlib.sha256(
        (
            _file_sha256(Path(__file__))
            + _file_sha256(Path(__file__).with_name("probes.py"))
        ).encode("ascii")
    ).hexdigest()
    for stage_index, stage in enumerate(STUDY_STAGES):
        for tap_index, tap in enumerate(STUDY_TAPS):
            cache_root = config.output_dir / "latents" / stage / tap
            if not (cache_root / "manifest.json").is_file():
                continue
            state_ids, features, latent_manifest = load_latent_cache(cache_root)
            latent_bindings[f"{stage}/{tap}"] = _file_sha256(
                cache_root / "manifest.json"
            )
            latent_hash = latent_bindings[f"{stage}/{tap}"]
            expected = tuple(record.state_id for record in records)
            if state_ids != expected:
                raise ValueError(f"latent State Bank ordering mismatch: {stage}/{tap}")
            if latent_manifest.get("state_bank_sha256") != bank_hash:
                raise ValueError(f"latent State Bank binding mismatch: {stage}/{tap}")
            for factor_index, factor in enumerate(
                tqdm(
                    FACTOR_NAMES,
                    desc=f"probes {stage}/{tap}",
                    unit="factor",
                    leave=False,
                )
            ):
                for split_name, split in (("task_group", task_split), ("episode_group", episode_split)):
                    identity = {"stage": stage, "tap": tap, "factor": factor, "split": split_name}
                    cell_binding = {
                        **identity,
                        "state_bank_manifest_sha256": bank_hash,
                        "latent_manifest_sha256": latent_hash,
                        "config_sha256": config_hash,
                        "implementation_sha256": implementation_hash,
                    }
                    cell_path = (
                        config.output_dir
                        / "probes"
                        / ".cells"
                        / stage
                        / tap
                        / split_name
                        / f"{factor}.json"
                    )
                    if cell_path.is_file():
                        cell = json.loads(cell_path.read_text(encoding="utf-8"))
                        if cell.get("binding") != cell_binding:
                            raise ValueError(f"stale probe cell cache: {cell_path}")
                        row = cell.get("row")
                        if not isinstance(row, dict):
                            raise ValueError(f"probe cell cache has an invalid row: {cell_path}")
                    else:
                        try:
                            result = _run_cell(
                                config=config,
                                records=records,
                                features=features,
                                split=split,
                                factor=factor,
                                seed_offset=stage_index * 100 + tap_index * 10 + factor_index,
                                run_capacity_check=split_name == "task_group",
                            )
                            row = {**identity, **result}
                        except ValueError as error:
                            row = {
                                **identity,
                                "status": "failed_gate",
                                "accessible": None,
                                "primary_metric": None,
                                "reason": str(error),
                            }
                        safe_cell = _json_safe({"binding": cell_binding, "row": row})
                        write_json_atomic(cell_path, safe_cell)
                    split_reports[split_name].append(row)
                    if split_name == "task_group":
                        completed.append(row)
    grid = build_stage_tap_factor_grid(completed)
    secondary_grid = build_stage_tap_factor_grid(split_reports["episode_group"])
    report = {
        "schema_version": PROBE_REPORT_SCHEMA,
        "passed": any(row.get("status") == "complete" for row in completed),
        "complete": all(row.get("status") == "complete" for row in grid),
        "state_bank_manifest_sha256": bank_hash,
        "latent_cache_manifest_sha256": dict(sorted(latent_bindings.items())),
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "primary_split": "task_group",
        "secondary_split": "episode_group",
        "bootstrap_unit": {"task_group": "task", "episode_group": "episode"},
        "rows": grid,
        "secondary_rows": secondary_grid,
        "missing_cells_are_not_zero": True,
        "rl_in_scope": False,
    }
    report = _json_safe(report)  # type: ignore[assignment]
    assert isinstance(report, dict)
    write_json_atomic(config.output_dir / "probes" / "report.json", report)
    return report


def inspect_probe_report(config: LiberoStudyConfig) -> dict[str, object]:
    path = config.output_dir / "probes" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PROBE_REPORT_SCHEMA:
        raise ValueError("probe report schema is incompatible")
    bank_manifest = config.output_dir / "state_bank" / "manifest.json"
    if report.get("state_bank_manifest_sha256") != _file_sha256(bank_manifest):
        raise ValueError("probe report State Bank binding is stale")
    if report.get("config_sha256") != _file_sha256(config.source_path):
        raise ValueError("probe report config binding is stale")
    implementation_hash = hashlib.sha256(
        (
            _file_sha256(Path(__file__))
            + _file_sha256(Path(__file__).with_name("probes.py"))
        ).encode("ascii")
    ).hexdigest()
    if report.get("implementation_sha256") != implementation_hash:
        raise ValueError("probe report implementation binding is stale")
    latent_bindings = report.get("latent_cache_manifest_sha256")
    if not isinstance(latent_bindings, dict):
        raise ValueError("probe report latent-cache bindings are missing")
    for key, expected in latent_bindings.items():
        stage, tap = str(key).split("/", maxsplit=1)
        path = config.output_dir / "latents" / stage / tap / "manifest.json"
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"probe report latent-cache binding is stale: {key}")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(STUDY_STAGES) * len(STUDY_TAPS) * len(FACTOR_NAMES):
        raise ValueError("probe report does not contain the exact preregistered grid")
    identities = {
        (str(row.get("stage")), str(row.get("tap")), str(row.get("factor")))
        for row in rows
        if isinstance(row, dict)
    }
    expected_identities = {
        (stage, tap, factor)
        for stage in STUDY_STAGES
        for tap in STUDY_TAPS
        for factor in FACTOR_NAMES
    }
    if identities != expected_identities:
        raise ValueError("probe report preregistered cell identities are incomplete")
    secondary_rows = report.get("secondary_rows")
    if not isinstance(secondary_rows, list) or len(secondary_rows) != len(rows):
        raise ValueError("probe report secondary grid is incomplete")
    secondary_identities = {
        (str(row.get("stage")), str(row.get("tap")), str(row.get("factor")))
        for row in secondary_rows
        if isinstance(row, dict)
    }
    if secondary_identities != expected_identities:
        raise ValueError("probe report secondary cell identities are incomplete")
    return report
