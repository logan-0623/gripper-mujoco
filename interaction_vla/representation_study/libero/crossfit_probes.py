from __future__ import annotations

import hashlib
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .config import LiberoStudyConfig
from .latents import load_latent_cache
from .longitudinal import CONDITION_SPECS, inspect_longitudinal_latents, load_longitudinal_plan
from .probe_runner import factor_target
from .probes import FACTOR_NAMES, STUDY_TAPS, classification_metrics, run_linear_probe
from .schema import StateRecord
from .splits import PARTITIONS
from .state_bank import load_state_bank
from .visualize import validate_annotation_timeline_report


CROSSFIT_PROTOCOL = "crossfit_v1"
CROSSFIT_REPORT_SCHEMA = "libero_longitudinal_crossfit_probe_report_v3"
SPLIT_NAMES = ("task_group", "episode_group")
CONDITION_CONTRASTS = (
    ("pretrained", "d100_u16617", "early_d100"),
    ("d25_u16070", "d50_u16324", "data_25_to_50_at_16k"),
    ("d50_u16324", "d100_u16617", "data_50_to_100_at_16k"),
    ("d50_u32650", "d100_u33234", "data_50_to_100_at_32k"),
    ("d100_u16617", "d100_u33234", "d100_16_to_32k"),
    ("d100_u33234", "d100_u49851", "d100_32_to_50k"),
    ("d100_u49851", "d100_u66470", "d100_50_to_66k"),
)


@dataclass(frozen=True)
class CrossFitManifest:
    split_name: str
    group_unit: str
    folds: int
    seed: int
    group_folds: Mapping[str, int]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("probe_runner.py"),
        Path(__file__).with_name("probes.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _probe_runtime() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_type": "cuda" if cuda else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
    }


def _write_immutable_json(path: Path, value: object) -> None:
    safe = _json_safe(value)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != safe:
            raise FileExistsError(f"scientific binding changed: {path}")
        return
    write_json_atomic(path, safe)


def _write_immutable_gzip_json(path: Path, value: object) -> None:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if path.is_file():
        if gzip.decompress(path.read_bytes()) != payload:
            raise FileExistsError(f"scientific binding changed: {path}")
        return
    write_bytes_atomic(path, gzip.compress(payload, compresslevel=6))


def _read_gzip_json(path: Path) -> dict[str, object]:
    value = json.loads(gzip.decompress(path.read_bytes()))
    if not isinstance(value, dict):
        raise ValueError(f"cross-fit cell is not a JSON object: {path}")
    return value


def build_crossfit_report_grid(
    rows: Sequence[Mapping[str, object]], *, split_name: str
) -> list[dict[str, object]]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown cross-fit split: {split_name}")
    lookup = {
        (str(row["condition"]), str(row["tap"]), str(row["factor"])): row
        for row in rows
        if row.get("split") == split_name
    }
    return [
        dict(
            lookup.get(
                (spec.condition, tap, factor),
                {
                    "condition": spec.condition,
                    "tap": tap,
                    "factor": factor,
                    "split": split_name,
                    "status": "not_run",
                    "accessible": None,
                    "primary_metric": None,
                },
            )
        )
        for spec in CONDITION_SPECS
        for tap in STUDY_TAPS
        for factor in FACTOR_NAMES
    ]


def _task_key(record: StateRecord) -> str:
    return f"{record.suite}:{record.task_id}"


def _group_key(record: StateRecord, split_name: str) -> str:
    task = _task_key(record)
    if split_name == "task_group":
        return task
    if split_name == "episode_group":
        return f"{task}:{record.source_episode_id}"
    raise ValueError(f"unknown cross-fit split: {split_name}")


def _hash_order(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def build_crossfit_manifest(
    records: Sequence[StateRecord],
    *,
    split_name: str,
    folds: int,
    seed: int,
) -> CrossFitManifest:
    if not records:
        raise ValueError("cross-fit requires State Bank records")
    if folds < 3 or seed < 0:
        raise ValueError("cross-fit requires at least three folds and a non-negative seed")
    group_unit = "task" if split_name == "task_group" else "episode"
    group_folds: dict[str, int] = {}
    if split_name == "task_group":
        groups = sorted({_task_key(record) for record in records}, key=lambda x: _hash_order(x, seed))
        if len(groups) < folds:
            raise ValueError("task cross-fit has fewer groups than folds")
        group_folds.update({group: index % folds for index, group in enumerate(groups)})
    elif split_name == "episode_group":
        by_task: dict[str, set[str]] = {}
        for record in records:
            by_task.setdefault(_task_key(record), set()).add(
                _group_key(record, split_name)
            )
        for task, task_groups in sorted(by_task.items()):
            if len(task_groups) < folds:
                raise ValueError(f"episode cross-fit task has fewer groups than folds: {task}")
            ordered = sorted(task_groups, key=lambda x: _hash_order(x, seed))
            group_folds.update(
                {group: index % folds for index, group in enumerate(ordered)}
            )
    else:
        raise ValueError(f"unknown cross-fit split: {split_name}")
    return CrossFitManifest(
        split_name=split_name,
        group_unit=group_unit,
        folds=folds,
        seed=seed,
        group_folds=dict(sorted(group_folds.items())),
    )


def crossfit_partition_indices(
    records: Sequence[StateRecord],
    manifest: CrossFitManifest,
    *,
    fold: int,
    applicable: np.ndarray,
) -> dict[str, list[int]]:
    if applicable.shape != (len(records),):
        raise ValueError("cross-fit applicability mask must match State Bank rows")
    if not 0 <= fold < manifest.folds:
        raise ValueError("cross-fit fold index is out of range")
    validation_fold = (fold + 1) % manifest.folds
    result = {partition: [] for partition in PARTITIONS}
    for index, (record, use) in enumerate(zip(records, applicable, strict=True)):
        if not use:
            continue
        group = _group_key(record, manifest.split_name)
        group_fold = manifest.group_folds.get(group)
        if group_fold is None:
            raise ValueError(f"cross-fit group is missing from manifest: {group}")
        partition = (
            "test"
            if group_fold == fold
            else "validation"
            if group_fold == validation_fold
            else "train"
        )
        result[partition].append(index)
    return result


def _primary_metric(factor: str) -> tuple[str, bool]:
    if factor == "geometry":
        return "normalized_mae", False
    if factor in {"contact", "stable_grasp"}:
        return "auprc", True
    return "macro_f1", True


def _crossfit_seed(
    *,
    base_seed: int,
    tap: str,
    factor: str,
    split_name: str,
    fold: int,
    seed_offset: int,
) -> int:
    payload = f"{base_seed}:{tap}:{factor}:{split_name}:{fold}:{seed_offset}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**31 - 1
    )


def _mode(values: np.ndarray) -> object:
    unique, counts = np.unique(values, return_counts=True)
    return unique[int(np.argmax(counts))]


def _time_bins(
    records: Sequence[StateRecord],
    source_indices: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    train_max = max(records[int(source_indices[index])].frame_index for index in train)
    return np.asarray(
        [
            min(
                9,
                int(10 * records[index].frame_index / max(train_max + 1, 1)),
            )
            for index in source_indices
        ],
        dtype=np.int64,
    )


def _fold_baseline_predictions(
    *,
    records: Sequence[StateRecord],
    source_indices: np.ndarray,
    targets: np.ndarray,
    local_parts: Mapping[str, Sequence[int]],
    factor: str,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    train = np.asarray(local_parts["train"], dtype=np.int64)
    test = np.asarray(local_parts["test"], dtype=np.int64)
    regression = factor == "geometry"
    fallback = targets[train].mean(axis=0) if regression else _mode(targets[train])
    binary = factor in {"contact", "stable_grasp"}
    positive = None if not binary else np.unique(targets)[-1]
    negative = None if not binary else np.unique(targets)[0]

    def predict(groups: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        predictions: list[object] = []
        scores: list[float] = []
        for index in test:
            candidates = train if groups is None else train[groups[train] == groups[index]]
            if not len(candidates):
                predictions.append(fallback)
                if binary:
                    scores.append(float(np.mean(targets[train] == positive)))
            elif regression:
                predictions.append(targets[candidates].mean(axis=0))
            elif binary:
                probability = float(np.mean(targets[candidates] == positive))
                predictions.append(positive if probability >= 0.5 else negative)
                scores.append(probability)
            else:
                predictions.append(_mode(targets[candidates]))
        prediction = (
            np.stack(predictions).astype(np.float64)
            if regression
            else np.asarray(predictions, dtype=targets.dtype)
        )
        return prediction, np.asarray(scores, dtype=np.float64) if binary else None

    task_keys = np.asarray(
        [f"{records[index].suite}:{records[index].task_id}" for index in source_indices]
    )
    instruction_keys = np.asarray([records[index].language for index in source_indices])
    return {
        "majority_or_mean": predict(None),
        "task_id": predict(task_keys),
        "instruction": predict(instruction_keys),
        "normalized_time_bin": predict(_time_bins(records, source_indices, train)),
    }


def _geometry_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    row_scale: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    row_scale = np.asarray(row_scale, dtype=np.float64)
    if target.shape != prediction.shape or row_scale.shape != target.shape:
        raise ValueError("cross-fit geometry arrays do not share shape")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all() or np.any(row_scale <= 0):
        raise ValueError("cross-fit geometry arrays must be finite with positive scales")
    residual = float(np.sum((target - prediction) ** 2))
    centered = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    return {
        "normalized_mae": float(np.mean(np.abs(target - prediction) / row_scale)),
        "r2": 1.0 - residual / centered if centered > 0 else float("nan"),
    }


def _metrics(
    *,
    factor: str,
    target: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray | None,
    labels: np.ndarray | None,
    row_scale: np.ndarray | None,
) -> dict[str, float]:
    if factor == "geometry":
        if row_scale is None:
            raise ValueError("cross-fit geometry requires per-row normalization scales")
        return _geometry_metrics(target, prediction, row_scale)
    return classification_metrics(
        target,
        prediction,
        score=score,
        binary=factor in {"contact", "stable_grasp"},
        labels=labels,
    )


def _bootstrap_accessibility(
    *,
    factor: str,
    target: np.ndarray,
    predictions: Sequence[np.ndarray],
    scores: Sequence[np.ndarray | None],
    baselines: Mapping[str, tuple[np.ndarray, np.ndarray | None]],
    clusters: Sequence[str],
    labels: np.ndarray | None,
    row_scale: np.ndarray | None,
    samples: int,
    confidence: float,
    minimum_valid_rate: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    unique = tuple(sorted(set(clusters)))
    if len(unique) < 2:
        missing = {"low": None, "high": None, "groups": len(unique), "valid_rate": 0.0}
        return missing, dict(missing)
    cluster_array = np.asarray(clusters)
    by_cluster = {group: np.flatnonzero(cluster_array == group) for group in unique}
    metric_name, higher_is_better = _primary_metric(factor)
    probe_by_group: list[list[float]] = []
    baseline_by_group: list[list[float]] = []
    for group in unique:
        indices = by_cluster[group]
        probe_values = [
            _metrics(
                factor=factor,
                target=target[indices],
                prediction=prediction[indices],
                score=None if score is None else score[indices],
                labels=labels,
                row_scale=None if row_scale is None else row_scale[indices],
            )[metric_name]
            for prediction, score in zip(predictions, scores, strict=True)
        ]
        baseline_values = [
            _metrics(
                factor=factor,
                target=target[indices],
                prediction=prediction[indices],
                score=None if score is None else score[indices],
                labels=labels,
                row_scale=None if row_scale is None else row_scale[indices],
            )[metric_name]
            for prediction, score in baselines.values()
        ]
        if all(np.isfinite(value) for value in (*probe_values, *baseline_values)):
            probe_by_group.append(probe_values)
            baseline_by_group.append(baseline_values)
    valid_rate = len(probe_by_group) / len(unique)
    common: dict[str, object] = {
        "unit": "group",
        "groups": len(unique),
        "requested_samples": samples,
        "valid_groups": len(probe_by_group),
        "valid_rate": valid_rate,
        "minimum_valid_rate": minimum_valid_rate,
        "confidence_level": confidence,
    }
    if not probe_by_group or valid_rate < minimum_valid_rate:
        failed = {**common, "low": None, "high": None, "gate_reason": "bootstrap valid rate is below threshold"}
        return failed, dict(failed)
    probe_matrix = np.asarray(probe_by_group, dtype=np.float64)
    baseline_matrix = np.asarray(baseline_by_group, dtype=np.float64)
    primary_groups = probe_matrix.mean(axis=1)
    baseline_points = baseline_matrix.mean(axis=0)
    strongest_index = int(
        np.argmax(baseline_points) if higher_is_better else np.argmin(baseline_points)
    )
    strongest_groups = baseline_matrix[:, strongest_index]
    utility_groups = (
        primary_groups - strongest_groups
        if higher_is_better
        else strongest_groups - primary_groups
    )
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(primary_groups), size=(samples, len(primary_groups)))
    primary_values = primary_groups[choices].mean(axis=1)
    sampled_baselines = baseline_matrix[choices].mean(axis=1)
    sampled_strongest = (
        sampled_baselines.max(axis=1)
        if higher_is_better
        else sampled_baselines.min(axis=1)
    )
    utility_values = (
        primary_values - sampled_strongest
        if higher_is_better
        else sampled_strongest - primary_values
    )
    alpha = (1.0 - confidence) / 2.0
    return (
        {
            **common,
            "samples": samples,
            "point": float(primary_groups.mean()),
            "replicate_points": probe_matrix.mean(axis=0).tolist(),
            "low": float(np.quantile(primary_values, alpha)),
            "high": float(np.quantile(primary_values, 1.0 - alpha)),
        },
        {
            **common,
            "samples": samples,
            "point": float(utility_groups.mean()),
            "baseline_points": {
                name: float(baseline_points[index])
                for index, name in enumerate(baselines)
            },
            "low": float(np.quantile(utility_values, alpha)),
            "high": float(np.quantile(utility_values, 1.0 - alpha)),
        },
    )


def run_crossfit_cell(
    *,
    config: LiberoStudyConfig,
    records: Sequence[StateRecord],
    features: np.ndarray,
    manifest: CrossFitManifest,
    tap: str,
    factor: str,
) -> dict[str, object]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(records) or not np.isfinite(features).all():
        raise ValueError("cross-fit features must be a finite State Bank matrix")
    applicable = np.asarray(
        [bool(getattr(record.labels.applicability, factor)) for record in records],
        dtype=bool,
    )
    source_indices = np.flatnonzero(applicable)
    if not len(source_indices):
        return {"status": "not_estimable", "reason": "factor has no applicable states", "primary_metric": None}
    applicable_features = features[applicable]
    raw_targets = [factor_target(records[index], factor) for index in source_indices]
    targets = (
        np.stack(raw_targets).astype(np.float32)
        if factor == "geometry"
        else np.asarray(["|".join(item) if isinstance(item, tuple) else item for item in raw_targets])
    )
    task = "regression" if factor == "geometry" else "classification"
    probe_device = "cpu" if factor == "geometry" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    classes = None if task == "regression" else np.unique(targets)
    source_to_local = {int(source): local for local, source in enumerate(source_indices)}
    offsets = tuple(config.probes.matched_seed_offsets)
    prediction_parts: dict[int, list[np.ndarray]] = {offset: [] for offset in offsets}
    score_parts: dict[int, list[np.ndarray | None]] = {offset: [] for offset in offsets}
    fold_seeds: dict[int, list[int]] = {offset: [] for offset in offsets}
    baseline_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("majority_or_mean", "task_id", "instruction", "normalized_time_bin")
    }
    baseline_score_parts: dict[str, list[np.ndarray | None]] = {
        name: [] for name in baseline_parts
    }
    ordered_sources: list[int] = []
    target_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    cluster_parts: list[str] = []

    for fold in range(manifest.folds):
        source_parts = crossfit_partition_indices(
            records, manifest, fold=fold, applicable=applicable
        )
        local_parts = {
            name: [source_to_local[index] for index in indices]
            for name, indices in source_parts.items()
        }
        if any(not indices for indices in local_parts.values()):
            return {"status": "not_estimable", "reason": f"fold {fold} has an empty partition", "primary_metric": None}
        if task == "classification":
            assert classes is not None
            for partition, indices in local_parts.items():
                observed = set(targets[np.asarray(indices, dtype=np.int64)].tolist())
                if partition == "train" and observed != set(classes.tolist()):
                    return {
                        "status": "not_estimable",
                        "reason": f"fold {fold} contains classes absent from training",
                        "primary_metric": None,
                    }
                if factor in {"contact", "stable_grasp"} and len(observed) < 2:
                    return {
                        "status": "not_estimable",
                        "reason": f"fold {fold} {partition} lacks both binary classes",
                        "primary_metric": None,
                    }
        test_local = np.asarray(local_parts["test"], dtype=np.int64)
        fold_results: dict[int, Mapping[str, object]] = {}
        fit_offsets = offsets[:1] if factor == "geometry" else offsets
        for offset in fit_offsets:
            seed = _crossfit_seed(
                base_seed=config.seed,
                tap=tap,
                factor=factor,
                split_name=manifest.split_name,
                fold=fold,
                seed_offset=offset,
            )
            fold_seeds[offset].append(seed)
            fold_results[offset] = run_linear_probe(
                applicable_features,
                targets,
                train_indices=local_parts["train"],
                validation_indices=local_parts["validation"],
                test_indices=local_parts["test"],
                task=task,
                seed=seed,
                l2_grid=config.probes.linear_l2,
                epochs=config.probes.linear_epochs,
                selection_metric=_primary_metric(factor)[0],
                device=probe_device,
            )
        if factor == "geometry":
            first = fold_results[fit_offsets[0]]
            for offset in offsets:
                fold_results[offset] = first
                if offset not in fit_offsets:
                    fold_seeds[offset].append(
                        _crossfit_seed(
                            base_seed=config.seed,
                            tap=tap,
                            factor=factor,
                            split_name=manifest.split_name,
                            fold=fold,
                            seed_offset=offset,
                        )
                    )
        for offset in offsets:
            result = fold_results[offset]
            if task == "classification" and result.get("classes") != classes.tolist():
                raise ValueError("cross-fit fold class universe changed")
            prediction_parts[offset].append(np.asarray(result["test_prediction"]))
            score = result.get("test_score")
            score_parts[offset].append(None if score is None else np.asarray(score, dtype=np.float64))
        fold_baselines = _fold_baseline_predictions(
            records=records,
            source_indices=source_indices,
            targets=targets,
            local_parts=local_parts,
            factor=factor,
        )
        for name, (prediction, score) in fold_baselines.items():
            baseline_parts[name].append(prediction)
            baseline_score_parts[name].append(score)
        ordered_sources.extend(source_parts["test"])
        if task == "regression":
            target_parts.append(np.asarray(targets[test_local], dtype=np.float64))
            train_targets = np.asarray(targets[np.asarray(local_parts["train"], dtype=np.int64)], dtype=np.float64)
            scale = np.ptp(train_targets, axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            scale_parts.append(np.broadcast_to(scale, (len(test_local), len(scale))).copy())
        else:
            assert classes is not None
            class_to_index = {value: index for index, value in enumerate(classes.tolist())}
            target_parts.append(np.asarray([class_to_index[value] for value in targets[test_local]], dtype=np.int64))
        cluster_parts.extend(_group_key(records[index], manifest.split_name) for index in source_parts["test"])

    if len(ordered_sources) != len(source_indices) or set(ordered_sources) != set(source_indices.tolist()):
        raise ValueError("cross-fit out-of-fold coverage is incomplete")
    target = np.concatenate(target_parts)
    row_scale = np.concatenate(scale_parts) if scale_parts else None
    labels = None if classes is None else np.arange(len(classes), dtype=np.int64)
    predictions = [np.concatenate(prediction_parts[offset]) for offset in offsets]
    scores = [
        None
        if any(item is None for item in score_parts[offset])
        else np.concatenate([item for item in score_parts[offset] if item is not None])
        for offset in offsets
    ]
    encoded_baselines: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for name in baseline_parts:
        prediction = np.concatenate(baseline_parts[name])
        if classes is not None:
            class_to_index = {value: index for index, value in enumerate(classes.tolist())}
            prediction = np.asarray([class_to_index[value] for value in prediction], dtype=np.int64)
        score = (
            None
            if any(item is None for item in baseline_score_parts[name])
            else np.concatenate([item for item in baseline_score_parts[name] if item is not None])
        )
        encoded_baselines[name] = (prediction, score)
    metric_name, higher_is_better = _primary_metric(factor)
    pooled_seed_metrics = [
        _metrics(
            factor=factor,
            target=target,
            prediction=prediction,
            score=score,
            labels=labels,
            row_scale=row_scale,
        )[metric_name]
        for prediction, score in zip(predictions, scores, strict=True)
    ]
    pooled_baseline_metrics = {
        name: _metrics(
            factor=factor,
            target=target,
            prediction=prediction,
            score=score,
            labels=labels,
            row_scale=row_scale,
        )
        for name, (prediction, score) in encoded_baselines.items()
    }
    primary_ci, utility_ci = _bootstrap_accessibility(
        factor=factor,
        target=target,
        predictions=predictions,
        scores=scores,
        baselines=encoded_baselines,
        clusters=cluster_parts,
        labels=labels,
        row_scale=row_scale,
        samples=config.probes.bootstrap_samples,
        confidence=config.probes.confidence_level,
        minimum_valid_rate=config.probes.minimum_bootstrap_valid_rate,
        seed=_crossfit_seed(
            base_seed=config.seed,
            tap=tap,
            factor=factor,
            split_name=manifest.split_name,
            fold=manifest.folds,
            seed_offset=10_000,
        ),
    )
    gate_reason = primary_ci.get("gate_reason")
    seed_metrics = [float(value) for value in primary_ci.get("replicate_points", pooled_seed_metrics)]
    baseline_values = {
        name: float(value)
        for name, value in utility_ci.get(
            "baseline_points",
            {name: metrics[metric_name] for name, metrics in pooled_baseline_metrics.items()},
        ).items()
    }
    strongest = max(baseline_values.values()) if higher_is_better else min(baseline_values.values())
    utility_point = utility_ci.get("point")
    return {
        "status": "failed_gate" if gate_reason else "complete",
        "reason": gate_reason,
        "accessible": None if gate_reason else bool(float(utility_ci["low"]) > 0.0),
        "primary_metric_name": metric_name,
        "selection_metric": metric_name,
        "probe_device": probe_device,
        "primary_metric": None if gate_reason else float(primary_ci["point"]),
        "pooled_primary_metric": float(np.mean(pooled_seed_metrics)),
        "probe_metric_std": float(np.std(seed_metrics)),
        "probe_seed_offsets": list(offsets),
        "probe_seed_metrics": {str(offset): value for offset, value in zip(offsets, seed_metrics, strict=True)},
        "fold_seeds": {str(offset): values for offset, values in fold_seeds.items()},
        "baseline_metric": baseline_values["majority_or_mean"],
        "shortcut_baselines": {
            name: {**pooled_baseline_metrics[name], "cluster_macro_primary_metric": baseline_values[name]}
            for name in pooled_baseline_metrics
            if name != "majority_or_mean"
        },
        "accessibility_threshold": strongest,
        "accessibility_utility": utility_point,
        "accessibility_utility_ci": utility_ci,
        "confidence_interval": primary_ci,
        "folds_completed": manifest.folds,
        "oof_states": len(ordered_sources),
        "applicable_states": int(applicable.sum()),
        "paired_payload": {
            "state_ids": [records[index].state_id for index in ordered_sources],
            "clusters": cluster_parts,
            "target": target.tolist(),
            "row_normalization_scale": None if row_scale is None else row_scale.tolist(),
            "labels": None if labels is None else labels.tolist(),
            "replicates": [
                {
                    "seed_offset": offset,
                    "prediction": prediction.tolist(),
                    "score": None if score is None else score.tolist(),
                }
                for offset, prediction, score in zip(offsets, predictions, scores, strict=True)
            ],
        },
        "capacity_check": {"status": "not_run", "reason": "protocol-v3 cross-fit primary is linear"},
    }


def paired_crossfit_delta(
    *,
    factor: str,
    reference: Mapping[str, object],
    destination: Mapping[str, object],
    samples: int,
    confidence: float,
    minimum_valid_rate: float,
    seed: int,
) -> dict[str, object]:
    if reference.get("status") != "complete" or destination.get("status") != "complete":
        return {"status": "not_available", "reason": "one or both cross-fit cells did not complete"}
    left = reference.get("paired_payload")
    right = destination.get("paired_payload")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {"status": "not_available", "reason": "paired cross-fit payload is missing"}
    for key in ("state_ids", "clusters", "target", "row_normalization_scale", "labels"):
        if left.get(key) != right.get(key):
            return {"status": "not_available", "reason": f"paired cross-fit {key} differs"}
    left_replicates = {int(row["seed_offset"]): row for row in left.get("replicates", [])}
    right_replicates = {int(row["seed_offset"]): row for row in right.get("replicates", [])}
    if not left_replicates or set(left_replicates) != set(right_replicates):
        return {"status": "not_available", "reason": "paired cross-fit replicate offsets differ"}
    offsets = tuple(sorted(left_replicates))
    target = np.asarray(left["target"])
    clusters = tuple(str(value) for value in left["clusters"])
    labels = None if left.get("labels") is None else np.asarray(left["labels"])
    row_scale = None if left.get("row_normalization_scale") is None else np.asarray(left["row_normalization_scale"], dtype=np.float64)
    metric_name, higher_is_better = _primary_metric(factor)

    def delta(offset: int, indices: np.ndarray) -> float:
        left_row = left_replicates[offset]
        right_row = right_replicates[offset]
        left_score = left_row.get("score")
        right_score = right_row.get("score")
        left_value = _metrics(
            factor=factor,
            target=target[indices],
            prediction=np.asarray(left_row["prediction"])[indices],
            score=None if left_score is None else np.asarray(left_score)[indices],
            labels=labels,
            row_scale=None if row_scale is None else row_scale[indices],
        )[metric_name]
        right_value = _metrics(
            factor=factor,
            target=target[indices],
            prediction=np.asarray(right_row["prediction"])[indices],
            score=None if right_score is None else np.asarray(right_score)[indices],
            labels=labels,
            row_scale=None if row_scale is None else row_scale[indices],
        )[metric_name]
        return float(right_value - left_value)

    unique = tuple(sorted(set(clusters)))
    cluster_array = np.asarray(clusters)
    by_cluster = {group: np.flatnonzero(cluster_array == group) for group in unique}
    group_deltas = np.asarray(
        [[delta(offset, by_cluster[group]) for offset in offsets] for group in unique],
        dtype=np.float64,
    )
    valid_groups = np.isfinite(group_deltas).all(axis=1)
    valid_rate = float(valid_groups.mean()) if len(valid_groups) else 0.0
    valid_matrix = group_deltas[valid_groups]
    seed_deltas = {
        str(offset): float(valid_matrix[:, index].mean())
        for index, offset in enumerate(offsets)
    } if len(valid_matrix) else {str(offset): float("nan") for offset in offsets}
    point = float(valid_matrix.mean()) if len(valid_matrix) else float("nan")
    gate_reason = None
    low = high = None
    bootstrap_values = np.asarray([], dtype=np.float64)
    if len(valid_matrix) and valid_rate >= minimum_valid_rate:
        rng = np.random.default_rng(seed)
        choices = rng.integers(
            0, len(valid_matrix), size=(samples, len(valid_matrix))
        )
        bootstrap_values = valid_matrix[choices].mean(axis=(1, 2))
        alpha = (1.0 - confidence) / 2.0
        low = float(np.quantile(bootstrap_values, alpha))
        high = float(np.quantile(bootstrap_values, 1.0 - alpha))
    else:
        gate_reason = "paired cross-fit bootstrap valid rate is below threshold"
    improvement_low, improvement_high = low, high
    if not higher_is_better and low is not None and high is not None:
        improvement_low, improvement_high = -high, -low
    return {
        "status": "failed_gate" if gate_reason else "complete",
        "gate_reason": gate_reason,
        "metric": metric_name,
        "higher_is_better": higher_is_better,
        "destination_minus_reference": point,
        "improvement": point if higher_is_better else -point,
        "improvement_low": improvement_low,
        "improvement_high": improvement_high,
        "delta_low": low,
        "delta_high": high,
        "seed_deltas": seed_deltas,
        "groups": len(unique),
        "valid_groups": int(valid_groups.sum()),
        "requested_bootstrap_samples": samples,
        "bootstrap_samples": len(bootstrap_values),
        "bootstrap_valid_rate": valid_rate,
        "minimum_bootstrap_valid_rate": minimum_valid_rate,
        "confidence_level": confidence,
    }


def _crossfit_root(config: LiberoStudyConfig) -> Path:
    return Path(config.output_dir) / "protocol_v3" / "probes" / CROSSFIT_PROTOCOL


def _cell_summary(cell: Mapping[str, object], artifact: Path) -> dict[str, object]:
    result = cell.get("result")
    binding = cell.get("binding")
    if not isinstance(result, Mapping) or not isinstance(binding, Mapping):
        raise ValueError(f"cross-fit cell is malformed: {artifact}")
    return {
        "condition": binding["condition"],
        "stage": binding["stage"],
        "data_fraction": binding["data_fraction"],
        "training_step": binding["training_step"],
        "tap": binding["tap"],
        "factor": binding["factor"],
        "split": binding["split"],
        **{key: value for key, value in result.items() if key != "paired_payload"},
        "cell_artifact": str(artifact),
        "cell_artifact_sha256": _file_sha256(artifact),
    }


def _current_protocol_binding(config: LiberoStudyConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    bank_path = output / "state_bank" / "manifest.json"
    plan_path = output / "protocol_v3" / "conditions" / "manifest.json"
    gate_path = output / "protocol_v3" / "latent_gate" / "report.json"
    return {
        "state_bank_sha256": _file_sha256(bank_path),
        "condition_plan_sha256": _file_sha256(plan_path),
        "latent_gate_sha256": _file_sha256(gate_path),
        "timeline_report_sha256": _file_sha256(output / "timelines" / "report.json"),
        "config_sha256": _file_sha256(config.source_path),
        "implementation_sha256": _implementation_sha256(),
        "probe_runtime_fingerprint_sha256": _canonical_sha256(_probe_runtime()),
    }


def _paired_delta_gate(
    deltas: Sequence[Mapping[str, object]],
    cells: Mapping[tuple[str, str, str, str], Mapping[str, object]],
) -> dict[str, object]:
    failures = []
    for row in deltas:
        key = (str(row["tap"]), str(row["split"]), str(row["factor"]))
        statuses = (
            cells[(str(row["reference"]), *key)]["result"]["status"],
            cells[(str(row["destination"]), *key)]["result"]["status"],
        )
        allowed_unavailable = row.get("status") == "not_available" and "not_estimable" in statuses
        if row.get("status") != "complete" and not allowed_unavailable:
            failures.append(
                {
                    "contrast": row["contrast"],
                    "tap": row["tap"],
                    "split": row["split"],
                    "factor": row["factor"],
                    "delta_status": row.get("status"),
                    "cell_statuses": list(statuses),
                }
            )
    return {"passed": not failures, "failures": failures}


def run_crossfit_probe_study(config: LiberoStudyConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    bank_root = output / "state_bank"
    bank_manifest_path = bank_root / "manifest.json"
    validate_annotation_timeline_report(
        output / "timelines" / "report.json",
        state_bank_manifest=bank_manifest_path,
        require_approved=True,
    )
    records, _, _, _ = load_state_bank(bank_root)
    plan = load_longitudinal_plan(config)
    latent_gate = inspect_longitudinal_latents(config)
    if not latent_gate.get("passed"):
        raise ValueError("protocol-v3 latent gate did not pass")
    protocol_binding = _current_protocol_binding(config)
    root = _crossfit_root(config)
    manifests = {
        split: build_crossfit_manifest(
            records,
            split_name=split,
            folds=config.probes.crossfit_folds,
            seed=config.seed,
        )
        for split in SPLIT_NAMES
    }
    fold_report = {
        "schema_version": "libero_longitudinal_crossfit_folds_v1",
        "passed": True,
        "protocol_binding": protocol_binding,
        "manifests": {name: asdict(value) for name, value in manifests.items()},
    }
    folds_path = root / "folds.json"
    _write_immutable_json(folds_path, fold_report)
    folds_sha256 = _file_sha256(folds_path)
    plan_rows = {str(row["condition"]): row for row in plan["conditions"]}
    expected_state_ids = tuple(record.state_id for record in records)
    cells: dict[tuple[str, str, str, str], dict[str, object]] = {}
    result_cache: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    total = len(CONDITION_SPECS) * len(STUDY_TAPS) * len(SPLIT_NAMES) * len(FACTOR_NAMES)
    progress = tqdm(total=total, desc="Protocol-v3 cross-fit probes", unit="cell")
    try:
        for spec in CONDITION_SPECS:
            condition_row = plan_rows[spec.condition]
            latent_root = output / "protocol_v3" / "latents" / spec.condition
            for tap in STUDY_TAPS:
                state_ids, features, latent_manifest = load_latent_cache(latent_root / tap)
                if state_ids != expected_state_ids:
                    raise ValueError(
                        f"protocol-v3 latent order differs from State Bank: {spec.condition}/{tap}"
                    )
                if latent_manifest.get("state_bank_sha256") != protocol_binding[
                    "state_bank_sha256"
                ]:
                    raise ValueError(
                        f"protocol-v3 latent State Bank binding is stale: {spec.condition}/{tap}"
                    )
                latent_manifest_path = latent_root / tap / "manifest.json"
                for split in SPLIT_NAMES:
                    for factor in FACTOR_NAMES:
                        binding = {
                            "schema_version": "libero_longitudinal_crossfit_cell_v1",
                            "condition": spec.condition,
                            "stage": spec.stage,
                            "data_fraction": spec.data_fraction,
                            "training_step": spec.step,
                            "checkpoint_sha256": condition_row["checkpoint_sha256"],
                            "tap": tap,
                            "factor": factor,
                            "split": split,
                            "folds_sha256": folds_sha256,
                            "latent_manifest_sha256": _file_sha256(latent_manifest_path),
                            "latent_values_sha256": latent_manifest["values_sha256"],
                            **protocol_binding,
                        }
                        artifact = root / "cells" / spec.condition / tap / split / f"{factor}.json.gz"
                        cache_key = (
                            tap,
                            str(latent_manifest["values_sha256"]),
                            split,
                            factor,
                        )
                        if artifact.is_file():
                            cell = _read_gzip_json(artifact)
                            if cell.get("binding") != binding:
                                raise ValueError(f"cross-fit cell has a stale binding: {artifact}")
                        else:
                            result = result_cache.get(cache_key)
                            if result is None:
                                result = run_crossfit_cell(
                                    config=config,
                                    records=records,
                                    features=features,
                                    manifest=manifests[split],
                                    tap=tap,
                                    factor=factor,
                                )
                            cell = {"binding": binding, "result": result}
                            _write_immutable_gzip_json(artifact, cell)
                            cell = _read_gzip_json(artifact)
                        result = cell.get("result")
                        if not isinstance(result, Mapping):
                            raise ValueError(f"cross-fit cell result is malformed: {artifact}")
                        result_cache[cache_key] = result
                        cells[(spec.condition, tap, split, factor)] = cell
                        progress.update()
    finally:
        progress.close()

    rows = [
        _cell_summary(cells[(spec.condition, tap, split, factor)], root / "cells" / spec.condition / tap / split / f"{factor}.json.gz")
        for spec in CONDITION_SPECS
        for tap in STUDY_TAPS
        for split in SPLIT_NAMES
        for factor in FACTOR_NAMES
    ]
    grids = {
        split: build_crossfit_report_grid(rows, split_name=split)
        for split in SPLIT_NAMES
    }
    deltas: list[dict[str, object]] = []
    for reference, destination, contrast in CONDITION_CONTRASTS:
        for tap in STUDY_TAPS:
            for split in SPLIT_NAMES:
                for factor in FACTOR_NAMES:
                    delta = paired_crossfit_delta(
                        factor=factor,
                        reference=cells[(reference, tap, split, factor)]["result"],
                        destination=cells[(destination, tap, split, factor)]["result"],
                        samples=config.probes.bootstrap_samples,
                        confidence=config.probes.confidence_level,
                        minimum_valid_rate=config.probes.minimum_bootstrap_valid_rate,
                        seed=_crossfit_seed(
                            base_seed=config.seed,
                            tap=tap,
                            factor=factor,
                            split_name=split,
                            fold=config.probes.crossfit_folds,
                            seed_offset=int.from_bytes(
                                hashlib.sha256(contrast.encode("utf-8")).digest()[:4],
                                "big",
                            ),
                        ),
                    )
                    deltas.append(
                        {
                            "contrast": contrast,
                            "reference": reference,
                            "destination": destination,
                            "tap": tap,
                            "split": split,
                            "factor": factor,
                            **delta,
                        }
                    )

    identical_failures: list[dict[str, object]] = []
    for pair in latent_gate.get("identical_cache_pairs", []):
        if not isinstance(pair, Mapping):
            continue
        left, right = str(pair["left"]), str(pair["right"])
        for tap in pair.get("taps", []):
            for split in SPLIT_NAMES:
                for factor in FACTOR_NAMES:
                    left_result = cells[(left, str(tap), split, factor)]["result"]
                    right_result = cells[(right, str(tap), split, factor)]["result"]
                    if _canonical_sha256(left_result) != _canonical_sha256(right_result):
                        identical_failures.append(
                            {"left": left, "right": right, "tap": tap, "split": split, "factor": factor}
                        )
    status_counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in ("complete", "not_estimable", "failed_gate", "not_run")
    }
    paired_delta_gate = _paired_delta_gate(deltas, cells)
    report = {
        "schema_version": CROSSFIT_REPORT_SCHEMA,
        "passed": status_counts["complete"] > 0
        and status_counts["failed_gate"] == 0
        and status_counts["not_run"] == 0
        and not identical_failures
        and paired_delta_gate["passed"],
        "protocol": CROSSFIT_PROTOCOL,
        "protocol_binding": {**protocol_binding, "folds_sha256": folds_sha256},
        "conditions": [spec.condition for spec in CONDITION_SPECS],
        "taps": list(STUDY_TAPS),
        "factors": list(FACTOR_NAMES),
        "splits": list(SPLIT_NAMES),
        "status_counts": status_counts,
        "probe_runtime": _probe_runtime(),
        "grids": grids,
        "condition_contrasts": [
            {"reference": left, "destination": right, "name": name}
            for left, right, name in CONDITION_CONTRASTS
        ],
        "paired_deltas": deltas,
        "paired_delta_gate": paired_delta_gate,
        "identical_latent_sanity": {
            "passed": not identical_failures,
            "failures": identical_failures,
        },
        "interpretation_boundary": {
            "accessible": "cross-fitted probe utility exceeds the strongest preregistered shortcut baseline",
            "functionally_used": "not measured by this report",
            "closed_loop_useful": "not measured by this report",
        },
    }
    report_path = root / "report.json"
    _write_immutable_json(report_path, report)
    return inspect_crossfit_probe_report(config)


def inspect_crossfit_probe_report(config: LiberoStudyConfig) -> dict[str, object]:
    root = _crossfit_root(config)
    report_path = root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"protocol-v3 cross-fit report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != CROSSFIT_REPORT_SCHEMA:
        raise ValueError("protocol-v3 cross-fit report schema is incompatible")
    expected_binding = _current_protocol_binding(config)
    binding = report.get("protocol_binding")
    if not isinstance(binding, Mapping) or any(
        binding.get(key) != value for key, value in expected_binding.items()
    ):
        raise ValueError("protocol-v3 cross-fit report has a stale binding")
    folds_path = root / "folds.json"
    if not folds_path.is_file() or _file_sha256(folds_path) != binding.get(
        "folds_sha256"
    ):
        raise ValueError("protocol-v3 cross-fit fold manifest is missing or stale")
    expected_cells = len(CONDITION_SPECS) * len(STUDY_TAPS) * len(FACTOR_NAMES)
    grids = report.get("grids")
    if not isinstance(grids, Mapping) or any(
        not isinstance(grids.get(split), list) or len(grids[split]) != expected_cells
        for split in SPLIT_NAMES
    ):
        raise ValueError("protocol-v3 cross-fit report grid is incomplete")
    missing_artifacts = [
        row.get("cell_artifact")
        for split in SPLIT_NAMES
        for row in grids[split]
        if not Path(str(row.get("cell_artifact", ""))).is_file()
        or _file_sha256(Path(str(row.get("cell_artifact", ""))))
        != row.get("cell_artifact_sha256")
    ]
    if missing_artifacts:
        raise ValueError("protocol-v3 cross-fit report references missing or stale cells")
    return {
        "schema_version": "libero_longitudinal_crossfit_probe_inspection_v3",
        "passed": bool(report.get("passed")),
        "report": str(report_path),
        "report_sha256": _file_sha256(report_path),
        "conditions": len(report["conditions"]),
        "taps": len(report["taps"]),
        "factors": len(report["factors"]),
        "splits": len(report["splits"]),
        "cells": sum(report["status_counts"].values()),
        "status_counts": report["status_counts"],
        "paired_deltas": len(report["paired_deltas"]),
        "paired_delta_gate": report["paired_delta_gate"],
        "identical_latent_sanity": report["identical_latent_sanity"],
        "interpretation_boundary": report["interpretation_boundary"],
    }
