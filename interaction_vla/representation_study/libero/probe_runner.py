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


PROBE_PROTOCOL = "protocol_v2"
PROBE_REPORT_SCHEMA = "libero_stage_tap_factor_probe_report_v2"


def _probe_artifact_root(output_dir: Path) -> Path:
    return output_dir / "probes" / PROBE_PROTOCOL


def _matched_probe_seed(
    *,
    base_seed: int,
    tap: str,
    factor: str,
    split_name: str,
    replicate_offset: int,
) -> int:
    if base_seed < 0 or replicate_offset < 0:
        raise ValueError("probe seeds and offsets must be non-negative")
    if tap not in STUDY_TAPS or factor not in FACTOR_NAMES:
        raise ValueError("probe seed identity contains an unknown tap or factor")
    if split_name not in {"task_group", "episode_group"}:
        raise ValueError("probe seed identity contains an unknown split")
    payload = json.dumps(
        [base_seed, tap, factor, split_name, replicate_offset],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


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
                    labels=np.asarray([0, 1], dtype=np.int64),
                )
            else:
                result[name] = classification_metrics(
                    targets[test],
                    prediction,
                    labels=np.unique(targets[train]),
                )
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
    labels: np.ndarray | None = None,
    minimum_valid_rate: float = 0.0,
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
    label_universe = labels
    if factor != "geometry" and label_universe is None:
        label_universe = np.unique(np.concatenate((target, prediction)))
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
                labels=label_universe,
            )[metric_name]
        if np.isfinite(value):
            values.append(float(value))
    valid_rate = len(values) / samples
    if not values or valid_rate < minimum_valid_rate:
        return {
            "unit": "group",
            "groups": len(unique),
            "requested_samples": samples,
            "samples": len(values),
            "valid_rate": valid_rate,
            "minimum_valid_rate": minimum_valid_rate,
            "gate_reason": "bootstrap valid rate is below threshold",
            "low": None,
            "high": None,
        }
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "group",
        "groups": len(unique),
        "requested_samples": samples,
        "samples": len(values),
        "valid_rate": valid_rate,
        "minimum_valid_rate": minimum_valid_rate,
        "confidence_level": confidence,
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
    }


def _paired_metric_value(
    *,
    factor: str,
    target: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray | None,
    indices: np.ndarray,
    normalization_scale: np.ndarray | None,
    labels: np.ndarray | None,
) -> float:
    metric_name, _ = _primary_metric(factor)
    if factor == "geometry":
        if normalization_scale is None:
            raise ValueError("paired geometry delta requires normalization scale")
        return float(
            geometry_metrics(
                target[indices],
                prediction[indices],
                normalization_scale=normalization_scale,
            )[metric_name]
        )
    binary = factor in {"contact", "stable_grasp"}
    sampled_score = None if score is None else score[indices]
    return float(
        classification_metrics(
            target[indices],
            prediction[indices],
            score=sampled_score,
            binary=binary,
            labels=labels,
        )[metric_name]
    )


def _replicated_bootstrap_ci(
    *,
    factor: str,
    target: np.ndarray,
    predictions: Sequence[np.ndarray],
    scores: Sequence[np.ndarray | None],
    clusters: Sequence[str],
    samples: int,
    confidence: float,
    seed: int,
    normalization_scale: np.ndarray | None,
    labels: np.ndarray | None,
    minimum_valid_rate: float,
) -> dict[str, object]:
    if not predictions or len(predictions) != len(scores):
        raise ValueError("replicated bootstrap requires aligned probe predictions and scores")
    unique = tuple(sorted(set(clusters)))
    if len(unique) < 2:
        return {"unit": "group", "groups": len(unique), "low": None, "high": None}
    label_universe = labels
    if factor != "geometry" and label_universe is None:
        label_universe = np.unique(np.concatenate([target, *predictions]))
    cluster_array = np.asarray(clusters)
    by_cluster = {
        cluster: np.flatnonzero(cluster_array == cluster) for cluster in unique
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[str(cluster)] for cluster in chosen])
        replicate_values = [
            _paired_metric_value(
                factor=factor,
                target=target,
                prediction=prediction,
                score=score,
                indices=indices,
                normalization_scale=normalization_scale,
                labels=label_universe,
            )
            for prediction, score in zip(predictions, scores, strict=True)
        ]
        if all(np.isfinite(value) for value in replicate_values):
            values.append(float(np.mean(replicate_values)))
    valid_rate = len(values) / samples
    if not values or valid_rate < minimum_valid_rate:
        return {
            "unit": "group",
            "groups": len(unique),
            "requested_samples": samples,
            "samples": len(values),
            "valid_rate": valid_rate,
            "minimum_valid_rate": minimum_valid_rate,
            "gate_reason": "bootstrap valid rate is below threshold",
            "low": None,
            "high": None,
        }
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "group",
        "groups": len(unique),
        "requested_samples": samples,
        "samples": len(values),
        "valid_rate": valid_rate,
        "minimum_valid_rate": minimum_valid_rate,
        "confidence_level": confidence,
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
    }


def _paired_stage_delta(
    *,
    factor: str,
    reference: Mapping[str, object],
    destination: Mapping[str, object],
    samples: int,
    confidence: float,
    seed: int,
    minimum_valid_rate: float = 0.9,
) -> dict[str, object]:
    if reference.get("status") != "complete" or destination.get("status") != "complete":
        return {"status": "not_available", "reason": "one or both probe cells did not complete"}
    reference_payload = reference.get("paired_payload")
    destination_payload = destination.get("paired_payload")
    if not isinstance(reference_payload, Mapping) or not isinstance(destination_payload, Mapping):
        return {"status": "not_available", "reason": "paired prediction payload is missing"}
    state_ids = tuple(str(item) for item in reference_payload.get("state_ids", ()))
    destination_state_ids = tuple(
        str(item) for item in destination_payload.get("state_ids", ())
    )
    clusters = tuple(str(item) for item in reference_payload.get("clusters", ()))
    destination_clusters = tuple(
        str(item) for item in destination_payload.get("clusters", ())
    )
    if not state_ids or state_ids != destination_state_ids or clusters != destination_clusters:
        return {"status": "not_available", "reason": "paired state/group identities differ"}
    target = np.asarray(reference_payload.get("target"))
    destination_target = np.asarray(destination_payload.get("target"))
    if target.shape[0] != len(state_ids) or not np.array_equal(target, destination_target):
        return {"status": "not_available", "reason": "paired factor targets differ"}
    reference_replicates = reference_payload.get("replicates")
    destination_replicates = destination_payload.get("replicates")
    if not isinstance(reference_replicates, list) or not isinstance(destination_replicates, list):
        return {"status": "not_available", "reason": "paired probe replicates are missing"}
    reference_by_seed = {
        int(item["seed"]): item for item in reference_replicates if isinstance(item, Mapping)
    }
    destination_by_seed = {
        int(item["seed"]): item for item in destination_replicates if isinstance(item, Mapping)
    }
    matched_seeds = tuple(sorted(set(reference_by_seed).intersection(destination_by_seed)))
    if not matched_seeds or set(reference_by_seed) != set(destination_by_seed):
        return {"status": "not_available", "reason": "paired probe seeds differ"}
    normalization_value = reference_payload.get("normalization_scale")
    destination_normalization = destination_payload.get("normalization_scale")
    normalization_scale = (
        None if normalization_value is None else np.asarray(normalization_value, dtype=np.float64)
    )
    if normalization_value != destination_normalization:
        return {"status": "not_available", "reason": "paired normalization scales differ"}
    labels = None
    if factor != "geometry":
        reference_labels = reference_payload.get("labels")
        destination_labels = destination_payload.get("labels")
        if reference_labels != destination_labels:
            return {"status": "not_available", "reason": "paired label universes differ"}
        if reference_labels is not None:
            labels = np.asarray(reference_labels)
        else:
            predictions = [
                np.asarray(reference_by_seed[value]["prediction"])
                for value in matched_seeds
            ] + [
                np.asarray(destination_by_seed[value]["prediction"])
                for value in matched_seeds
            ]
            labels = np.unique(np.concatenate([target, *predictions]))
    all_indices = np.arange(len(state_ids), dtype=np.int64)

    def replicate_delta(probe_seed: int, indices: np.ndarray) -> float:
        reference_item = reference_by_seed[probe_seed]
        destination_item = destination_by_seed[probe_seed]
        reference_score_value = reference_item.get("score")
        destination_score_value = destination_item.get("score")
        reference_metric = _paired_metric_value(
            factor=factor,
            target=target,
            prediction=np.asarray(reference_item["prediction"]),
            score=(
                None
                if reference_score_value is None
                else np.asarray(reference_score_value, dtype=np.float64)
            ),
            indices=indices,
            normalization_scale=normalization_scale,
            labels=labels,
        )
        destination_metric = _paired_metric_value(
            factor=factor,
            target=target,
            prediction=np.asarray(destination_item["prediction"]),
            score=(
                None
                if destination_score_value is None
                else np.asarray(destination_score_value, dtype=np.float64)
            ),
            indices=indices,
            normalization_scale=normalization_scale,
            labels=labels,
        )
        return destination_metric - reference_metric

    seed_deltas = {
        str(probe_seed): replicate_delta(probe_seed, all_indices)
        for probe_seed in matched_seeds
    }
    point = float(np.mean(tuple(seed_deltas.values())))
    unique_clusters = tuple(sorted(set(clusters)))
    if len(unique_clusters) < 2:
        delta_low = delta_high = None
        valid_samples = 0
        valid_rate = 0.0
        gate_reason = "paired bootstrap requires at least two held-out groups"
    else:
        by_cluster = {
            cluster: np.flatnonzero(np.asarray(clusters) == cluster)
            for cluster in unique_clusters
        }
        rng = np.random.default_rng(seed)
        bootstrapped: list[float] = []
        for _ in range(samples):
            chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            indices = np.concatenate([by_cluster[str(cluster)] for cluster in chosen])
            values = [replicate_delta(probe_seed, indices) for probe_seed in matched_seeds]
            if all(np.isfinite(value) for value in values):
                bootstrapped.append(float(np.mean(values)))
        valid_samples = len(bootstrapped)
        valid_rate = valid_samples / samples
        if bootstrapped and valid_rate >= minimum_valid_rate:
            alpha = (1.0 - confidence) / 2.0
            delta_low = float(np.quantile(bootstrapped, alpha))
            delta_high = float(np.quantile(bootstrapped, 1.0 - alpha))
            gate_reason = None
        else:
            delta_low = delta_high = None
            gate_reason = "paired bootstrap valid rate is below threshold"
    metric_name, higher_is_better = _primary_metric(factor)
    improvement_low = delta_low
    improvement_high = delta_high
    if not higher_is_better and delta_low is not None and delta_high is not None:
        improvement_low = -delta_high
        improvement_high = -delta_low
    return {
        "status": "complete" if gate_reason is None else "failed_gate",
        "gate_reason": gate_reason,
        "metric": metric_name,
        "higher_is_better": higher_is_better,
        "destination_minus_reference": point,
        "improvement": point if higher_is_better else -point,
        "seed_deltas": seed_deltas,
        "matched_probe_seeds": list(matched_seeds),
        "groups": len(unique_clusters),
        "requested_bootstrap_samples": samples,
        "bootstrap_samples": valid_samples,
        "bootstrap_valid_rate": valid_rate,
        "minimum_bootstrap_valid_rate": minimum_valid_rate,
        "confidence_level": confidence,
        "delta_low": delta_low,
        "delta_high": delta_high,
        "improvement_low": improvement_low,
        "improvement_high": improvement_high,
    }


def _assert_stage_invariant_probe_results(
    reference: Mapping[str, object], destination: Mapping[str, object]
) -> None:
    excluded = {"stage", "cell_artifact"}
    reference_value = {
        key: value for key, value in reference.items() if key not in excluded
    }
    destination_value = {
        key: value for key, value in destination.items() if key not in excluded
    }
    if _json_safe(reference_value) != _json_safe(destination_value):
        raise ValueError("identical latent matrices produced different probe results")


def _run_cell(
    *,
    config: LiberoStudyConfig,
    records: Sequence[StateRecord],
    features: np.ndarray,
    split: SplitManifest,
    split_name: str,
    tap: str,
    factor: str,
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
    probe_seeds = [
        _matched_probe_seed(
            base_seed=config.seed,
            tap=tap,
            factor=factor,
            split_name=split_name,
            replicate_offset=offset,
        )
        for offset in config.probes.matched_seed_offsets
    ]
    linears = [
        run_linear_probe(
            selected_features,
            targets,
            train_indices=local_parts["train"],
            validation_indices=local_parts["validation"],
            test_indices=local_parts["test"],
            task=task,
            seed=probe_seed,
            l2_grid=config.probes.linear_l2,
            epochs=config.probes.linear_epochs,
        )
        for probe_seed in probe_seeds
    ]
    linear = linears[0]
    mlp = (
        run_shallow_mlp_probe(
            selected_features,
            targets,
            train_indices=local_parts["train"],
            validation_indices=local_parts["validation"],
            test_indices=local_parts["test"],
            task=task,
            seed=probe_seeds[0],
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
    classification_labels = None
    if factor != "geometry":
        classification_labels = np.arange(len(linear["classes"]), dtype=np.int64)
        if any(item.get("classes") != linear.get("classes") for item in linears[1:]):
            raise ValueError("matched probe replicates produced different class universes")
    predictions = [np.asarray(item["test_prediction"]) for item in linears]
    scores = [
        (
            None
            if item.get("test_score") is None
            else np.asarray(item["test_score"], dtype=np.float64)
        )
        for item in linears
    ]
    normalization_scale = None
    if factor == "geometry":
        train_targets = targets[np.asarray(local_parts["train"], dtype=np.int64)]
        normalization_scale = np.ptp(train_targets, axis=0)
        normalization_scale = np.where(normalization_scale > 1e-8, normalization_scale, 1.0)
    confidence_interval = _replicated_bootstrap_ci(
        factor=factor,
        target=target,
        predictions=predictions,
        scores=scores,
        clusters=clusters,
        samples=config.probes.bootstrap_samples,
        confidence=config.probes.confidence_level,
        seed=_matched_probe_seed(
            base_seed=config.seed,
            tap=tap,
            factor=factor,
            split_name=split_name,
            replicate_offset=10_000,
        ),
        normalization_scale=normalization_scale,
        labels=classification_labels,
        minimum_valid_rate=config.probes.minimum_bootstrap_valid_rate,
    )
    metric_name, higher_is_better = _primary_metric(factor)
    seed_metrics = [float(item["test_metrics"][metric_name]) for item in linears]
    metric = float(np.mean(seed_metrics))
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
    confidence_gate_reason = confidence_interval.get("gate_reason")
    return {
        "status": "failed_gate" if confidence_gate_reason is not None else "complete",
        "reason": confidence_gate_reason,
        "accessible": accessible,
        "primary_metric_name": metric_name,
        "primary_metric": metric,
        "probe_metric_std": float(np.std(seed_metrics)),
        "probe_seeds": probe_seeds,
        "probe_seed_metrics": {
            str(probe_seed): value
            for probe_seed, value in zip(probe_seeds, seed_metrics, strict=True)
        },
        "baseline_metric": baseline,
        "shortcut_baselines": shortcut_baselines,
        "accessibility_threshold": threshold,
        "confidence_interval": confidence_interval,
        "applicable_states": int(applicable.sum()),
        "states_by_partition": {name: len(values) for name, values in partitions.items()},
        "linear": linear,
        "linear_replicates": [
            {
                "seed": probe_seed,
                "selected_l2": item["selected_l2"],
                "test_metrics": item["test_metrics"],
                "baseline_metrics": item["baseline_metrics"],
            }
            for probe_seed, item in zip(probe_seeds, linears, strict=True)
        ],
        "paired_payload": {
            "state_ids": [records[index].state_id for index in test_source_indices],
            "clusters": clusters,
            "target": linear["test_target"],
            "normalization_scale": (
                None if normalization_scale is None else normalization_scale.tolist()
            ),
            "labels": (
                None if classification_labels is None else classification_labels.tolist()
            ),
            "replicates": [
                {
                    "seed": probe_seed,
                    "prediction": item["test_prediction"],
                    "score": item.get("test_score"),
                }
                for probe_seed, item in zip(probe_seeds, linears, strict=True)
            ],
        },
        "capacity_check": mlp,
        "capacity_check_seed": probe_seeds[0] if run_capacity_check else None,
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _compact_probe_row(row: Mapping[str, object], *, cell_path: Path) -> dict[str, object]:
    keys = (
        "stage",
        "tap",
        "factor",
        "split",
        "status",
        "accessible",
        "primary_metric_name",
        "primary_metric",
        "probe_metric_std",
        "probe_seeds",
        "probe_seed_metrics",
        "baseline_metric",
        "shortcut_baselines",
        "accessibility_threshold",
        "confidence_interval",
        "applicable_states",
        "states_by_partition",
        "linear_replicates",
        "capacity_check_seed",
        "reason",
    )
    compact = {key: row[key] for key in keys if key in row}
    capacity = row.get("capacity_check")
    if isinstance(capacity, Mapping):
        compact["capacity_check"] = {
            key: capacity[key]
            for key in ("status", "model", "task", "test_metrics", "hidden_dim", "reason")
            if key in capacity
        }
    linear = row.get("linear")
    if isinstance(linear, Mapping):
        compact["primary_linear_probe"] = {
            key: linear[key]
            for key in ("model", "task", "selected_l2", "test_metrics", "baseline_metrics")
            if key in linear
        }
    compact["cell_artifact"] = str(cell_path)
    return compact


def _stage_delta_grid(
    rows: Sequence[Mapping[str, object]],
    *,
    split_name: str,
    config: LiberoStudyConfig,
) -> list[dict[str, object]]:
    lookup = {
        (str(row.get("stage")), str(row.get("tap")), str(row.get("factor"))): row
        for row in rows
    }
    result: list[dict[str, object]] = []
    for stage_index, destination_stage in enumerate(STUDY_STAGES[1:], start=1):
        for tap in STUDY_TAPS:
            for factor in FACTOR_NAMES:
                reference = lookup.get(("pretrained", tap, factor), {"status": "not_run"})
                destination = lookup.get(
                    (destination_stage, tap, factor), {"status": "not_run"}
                )
                delta = _paired_stage_delta(
                    factor=factor,
                    reference=reference,
                    destination=destination,
                    samples=config.probes.bootstrap_samples,
                    confidence=config.probes.confidence_level,
                    minimum_valid_rate=config.probes.minimum_bootstrap_valid_rate,
                    seed=_matched_probe_seed(
                        base_seed=config.seed,
                        tap=tap,
                        factor=factor,
                        split_name=split_name,
                        replicate_offset=20_000 + stage_index,
                    ),
                )
                result.append(
                    {
                        "reference_stage": "pretrained",
                        "destination_stage": destination_stage,
                        "tap": tap,
                        "factor": factor,
                        "split": split_name,
                        **delta,
                    }
                )
    return result


def _identical_latent_sanity(
    split_reports: Mapping[str, Sequence[Mapping[str, object]]],
    latent_content_hashes: Mapping[str, str],
) -> dict[str, object]:
    lookup = {
        (
            split_name,
            str(row.get("stage")),
            str(row.get("tap")),
            str(row.get("factor")),
        ): row
        for split_name, rows in split_reports.items()
        for row in rows
    }
    checked: list[dict[str, str]] = []
    for tap in STUDY_TAPS:
        available = [
            stage for stage in STUDY_STAGES if f"{stage}/{tap}" in latent_content_hashes
        ]
        for left_index, reference_stage in enumerate(available):
            for destination_stage in available[left_index + 1 :]:
                if (
                    latent_content_hashes[f"{reference_stage}/{tap}"]
                    != latent_content_hashes[f"{destination_stage}/{tap}"]
                ):
                    continue
                for split_name in split_reports:
                    for factor in FACTOR_NAMES:
                        _assert_stage_invariant_probe_results(
                            lookup[(split_name, reference_stage, tap, factor)],
                            lookup[(split_name, destination_stage, tap, factor)],
                        )
                checked.append(
                    {
                        "tap": tap,
                        "reference_stage": reference_stage,
                        "destination_stage": destination_stage,
                    }
                )
    return {"passed": True, "identical_stage_tap_pairs_checked": checked}


def run_probe_study(config: LiberoStudyConfig) -> dict[str, object]:
    bank_root = config.output_dir / "state_bank"
    artifact_root = _probe_artifact_root(config.output_dir)
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
    latent_content_hashes: dict[str, str] = {}
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
    split_hashes = {
        "task_group": _file_sha256(bank_root / "splits" / "task_group.json"),
        "episode_group": _file_sha256(bank_root / "splits" / "episode_group.json"),
    }
    for stage in STUDY_STAGES:
        for tap in STUDY_TAPS:
            cache_root = config.output_dir / "latents" / stage / tap
            if not (cache_root / "manifest.json").is_file():
                continue
            state_ids, features, latent_manifest = load_latent_cache(cache_root)
            latent_bindings[f"{stage}/{tap}"] = _file_sha256(
                cache_root / "manifest.json"
            )
            latent_hash = latent_bindings[f"{stage}/{tap}"]
            latent_content_hashes[f"{stage}/{tap}"] = _array_sha256(features)
            expected = tuple(record.state_id for record in records)
            if state_ids != expected:
                raise ValueError(f"latent State Bank ordering mismatch: {stage}/{tap}")
            if latent_manifest.get("state_bank_sha256") != bank_hash:
                raise ValueError(f"latent State Bank binding mismatch: {stage}/{tap}")
            for factor in (
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
                        "probe_protocol": PROBE_PROTOCOL,
                        "state_bank_manifest_sha256": bank_hash,
                        "split_manifest_sha256": split_hashes[split_name],
                        "latent_manifest_sha256": latent_hash,
                        "latent_content_sha256": latent_content_hashes[f"{stage}/{tap}"],
                        "config_sha256": config_hash,
                        "implementation_sha256": implementation_hash,
                    }
                    cell_path = (
                        artifact_root
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
                                split_name=split_name,
                                tap=tap,
                                factor=factor,
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
                    row = dict(row)
                    row["cell_artifact"] = str(cell_path)
                    split_reports[split_name].append(row)
                    if split_name == "task_group":
                        completed.append(row)
    identical_latent_sanity = _identical_latent_sanity(
        split_reports, latent_content_hashes
    )
    primary_stage_deltas = _stage_delta_grid(
        split_reports["task_group"], split_name="task_group", config=config
    )
    secondary_stage_deltas = _stage_delta_grid(
        split_reports["episode_group"], split_name="episode_group", config=config
    )
    compact_primary = [
        _compact_probe_row(row, cell_path=Path(str(row["cell_artifact"])))
        for row in completed
    ]
    compact_secondary = [
        _compact_probe_row(row, cell_path=Path(str(row["cell_artifact"])))
        for row in split_reports["episode_group"]
    ]
    grid = build_stage_tap_factor_grid(compact_primary)
    secondary_grid = build_stage_tap_factor_grid(compact_secondary)
    report = {
        "schema_version": PROBE_REPORT_SCHEMA,
        "probe_protocol": PROBE_PROTOCOL,
        "passed": any(row.get("status") == "complete" for row in completed),
        "complete": all(row.get("status") == "complete" for row in grid),
        "state_bank_manifest_sha256": bank_hash,
        "split_manifest_sha256": split_hashes,
        "latent_cache_manifest_sha256": dict(sorted(latent_bindings.items())),
        "latent_content_sha256": dict(sorted(latent_content_hashes.items())),
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "primary_split": "task_group",
        "secondary_split": "episode_group",
        "bootstrap_unit": {"task_group": "task", "episode_group": "episode"},
        "probe_seed_offsets": list(config.probes.matched_seed_offsets),
        "minimum_bootstrap_valid_rate": config.probes.minimum_bootstrap_valid_rate,
        "probe_seed_matching": "matched across training stages by tap/factor/split/replicate",
        "identical_latent_sanity": identical_latent_sanity,
        "rows": grid,
        "secondary_rows": secondary_grid,
        "stage_deltas": primary_stage_deltas,
        "secondary_stage_deltas": secondary_stage_deltas,
        "stage_delta_reference": "pretrained",
        "stage_delta_interpretation": (
            "destination_minus_reference; improvement flips sign for lower-is-better metrics"
        ),
        "missing_cells_are_not_zero": True,
        "rl_in_scope": False,
    }
    report = _json_safe(report)  # type: ignore[assignment]
    assert isinstance(report, dict)
    write_json_atomic(artifact_root / "report.json", report)
    return report


def inspect_probe_report(config: LiberoStudyConfig) -> dict[str, object]:
    path = _probe_artifact_root(config.output_dir) / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PROBE_REPORT_SCHEMA:
        raise ValueError("probe report schema is incompatible")
    bank_manifest = config.output_dir / "state_bank" / "manifest.json"
    if report.get("state_bank_manifest_sha256") != _file_sha256(bank_manifest):
        raise ValueError("probe report State Bank binding is stale")
    split_bindings = report.get("split_manifest_sha256")
    expected_split_bindings = {
        "task_group": _file_sha256(bank_manifest.parent / "splits" / "task_group.json"),
        "episode_group": _file_sha256(
            bank_manifest.parent / "splits" / "episode_group.json"
        ),
    }
    if split_bindings != expected_split_bindings:
        raise ValueError("probe report split-manifest binding is stale")
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
