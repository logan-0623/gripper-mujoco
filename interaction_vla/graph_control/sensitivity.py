from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Final

import numpy as np

from .schema import TOKEN_DIM, TOKEN_SLICES
from .diagnostics import EpisodeLayout, cluster_bootstrap_mean


SENSITIVITY_SCHEMA_VERSION: Final[str] = "graph_policy_sensitivity_v1"
CATEGORICAL_GROUPS: Final[frozenset[str]] = frozenset(
    {"phase", "next_relation", "relation_operator", "predicate"}
)
SENSITIVITY_METRICS: Final[tuple[str, ...]] = (
    "action_l1",
    "action_l2",
    "translation_l2",
    "rotation_l2",
    "gripper_absolute_change",
    "action_direction_cosine_change",
    "translation_sign_changed",
    "standardized_perturbation_magnitude",
    "normalized_action_l2",
)


def _tokens(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != TOKEN_DIM:
        raise ValueError(f"{name} must have shape [rows, {TOKEN_DIM}]")
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result.copy()


def _group(name: str) -> slice:
    try:
        return TOKEN_SLICES[name]
    except KeyError as error:
        raise ValueError(f"unknown Graph token group: {name}") from error


def mask_token_group(tokens: object, group: str) -> np.ndarray:
    values = _tokens(tokens, "tokens")
    values[:, _group(group)] = 0.0
    return values


def training_feature_statistics(tokens: object) -> dict[str, np.ndarray]:
    values = _tokens(tokens, "training tokens")
    return {
        "std": np.std(values, axis=0, ddof=0),
        "p01": np.quantile(values, 0.01, axis=0, method="linear"),
        "p99": np.quantile(values, 0.99, axis=0, method="linear"),
    }


def _statistics(value: Mapping[str, object]) -> dict[str, np.ndarray]:
    if set(value) != {"std", "p01", "p99"}:
        raise ValueError("training statistics must contain std, p01, and p99")
    result: dict[str, np.ndarray] = {}
    for name in ("std", "p01", "p99"):
        array = np.asarray(value[name], dtype=np.float64)
        if array.shape != (TOKEN_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"training statistic {name} must be finite [{TOKEN_DIM}]")
        result[name] = array.copy()
    if np.any(result["std"] < 0.0) or np.any(result["p01"] > result["p99"]):
        raise ValueError("training statistics contain invalid bounds")
    return result


def _categorical_interventions(
    values: np.ndarray, bounds: slice, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    toward = values.copy()
    away = values.copy()
    source = np.clip(values[:, bounds], 0.0, None)
    totals = source.sum(axis=1)
    active = totals > 1.0e-12
    if not np.any(active):
        return toward, away
    probability = source[active] / totals[active, None]
    uniform = np.full_like(probability, 1.0 / probability.shape[1])
    less_confident = (1.0 - scale) * probability + scale * uniform
    more_confident = np.clip(
        probability + scale * (probability - uniform), 0.0, None
    )
    more_confident /= more_confident.sum(axis=1, keepdims=True)
    toward[np.ix_(active, np.arange(bounds.start, bounds.stop))] = less_confident
    away[np.ix_(active, np.arange(bounds.start, bounds.stop))] = more_confident
    return toward, away


def finite_difference_interventions(
    tokens: object,
    group: str,
    *,
    statistics: Mapping[str, object],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = _tokens(tokens, "tokens")
    bounds = _group(group)
    amount = float(scale)
    if not np.isfinite(amount) or not 0.0 < amount <= 1.0:
        raise ValueError("finite-difference scale must lie within (0, 1]")
    if group in CATEGORICAL_GROUPS:
        return _categorical_interventions(values, bounds, amount)

    stats = _statistics(statistics)
    delta = amount * stats["std"][bounds]
    minus = values.copy()
    plus = values.copy()
    minus[:, bounds] = np.clip(
        values[:, bounds] - delta,
        stats["p01"][bounds],
        stats["p99"][bounds],
    )
    plus[:, bounds] = np.clip(
        values[:, bounds] + delta,
        stats["p01"][bounds],
        stats["p99"][bounds],
    )
    return minus, plus


def standardized_perturbation_magnitude(
    baseline: object,
    changed: object,
    group: str,
    *,
    training_std: object,
) -> np.ndarray:
    source = _tokens(baseline, "baseline tokens")
    target = _tokens(changed, "changed tokens")
    if target.shape != source.shape:
        raise ValueError("baseline and changed tokens must share shape")
    bounds = _group(group)
    delta = target[:, bounds] - source[:, bounds]
    if group in CATEGORICAL_GROUPS:
        return np.linalg.norm(delta, axis=1)
    std = np.asarray(training_std, dtype=np.float64)
    if std.shape != (TOKEN_DIM,) or not np.isfinite(std).all() or np.any(std < 0.0):
        raise ValueError(f"training_std must be finite and non-negative [{TOKEN_DIM}]")
    scale = std[bounds]
    standardized = np.divide(
        delta,
        scale,
        out=np.zeros_like(delta),
        where=scale > 1.0e-12,
    )
    return np.linalg.norm(standardized, axis=1)


def _actions(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 7 or not len(result):
        raise ValueError(f"{name} must have shape [rows, 7]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def action_change_metrics(
    baseline: object,
    changed: object,
    *,
    perturbation_magnitude: object,
) -> dict[str, np.ndarray]:
    source = _actions(baseline, "baseline actions")
    target = _actions(changed, "changed actions")
    if target.shape != source.shape:
        raise ValueError("baseline and changed actions must share shape")
    magnitude = np.asarray(perturbation_magnitude, dtype=np.float64)
    if magnitude.shape != (len(source),) or not np.isfinite(magnitude).all():
        raise ValueError("perturbation magnitude must be finite with one value per row")
    if np.any(magnitude < 0.0):
        raise ValueError("perturbation magnitude must be non-negative")

    source_motion = source[:, :6]
    target_motion = target[:, :6]
    difference = target_motion - source_motion
    source_norm = np.linalg.norm(source_motion, axis=1)
    target_norm = np.linalg.norm(target_motion, axis=1)
    cosine = np.divide(
        np.sum(source_motion * target_motion, axis=1),
        source_norm * target_norm,
        out=np.ones(len(source), dtype=np.float64),
        where=(source_norm * target_norm) > 1.0e-12,
    )
    action_l2 = np.linalg.norm(difference, axis=1)
    normalized = np.divide(
        action_l2,
        magnitude,
        out=np.full(len(source), np.nan, dtype=np.float64),
        where=magnitude > 1.0e-12,
    )
    return {
        "action_l1": np.sum(np.abs(difference), axis=1),
        "action_l2": action_l2,
        "translation_l2": np.linalg.norm(difference[:, :3], axis=1),
        "rotation_l2": np.linalg.norm(difference[:, 3:6], axis=1),
        "gripper_absolute_change": np.abs(target[:, 6] - source[:, 6]),
        "action_direction_cosine_change": 1.0 - np.clip(cosine, -1.0, 1.0),
        "translation_sign_changed": np.any(
            np.sign(source[:, :3]) != np.sign(target[:, :3]), axis=1
        ),
        "standardized_perturbation_magnitude": magnitude.copy(),
        "normalized_action_l2": normalized,
    }


def select_episode_balanced_positions(
    layout: EpisodeLayout, *, rows_per_episode: int
) -> np.ndarray:
    limit = int(rows_per_episode)
    if isinstance(rows_per_episode, bool) or limit != rows_per_episode or limit < 1:
        raise ValueError("rows_per_episode must be a positive integer")
    selected: list[int] = []
    for bounds in layout.episode_slices:
        length = bounds.stop - bounds.start
        count = min(limit, length)
        if count == 1:
            local = np.asarray([(length - 1) // 2], dtype=np.int64)
        else:
            local = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
        if len(set(local.tolist())) != count:
            raise ValueError("episode-balanced row selection produced duplicates")
        selected.extend((bounds.start + local).tolist())
    result = np.asarray(selected, dtype=np.int64)
    if not len(result) or np.any(result < 0) or np.any(result >= len(layout.row_indices)):
        raise ValueError("episode-balanced row selection is invalid")
    return result


def make_sensitivity_records(
    *,
    policy_seed: int,
    condition: str,
    group: str,
    intervention: str,
    row_indices: object,
    episode_indices: object,
    frame_indices: object,
    metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    if int(policy_seed) != policy_seed or int(policy_seed) < 0:
        raise ValueError("policy_seed must be a non-negative integer")
    _group(group)
    if not condition or not intervention:
        raise ValueError("condition and intervention must be non-empty")
    coordinates: dict[str, np.ndarray] = {}
    for name, value in (
        ("row_index", row_indices),
        ("episode_index", episode_indices),
        ("frame_index", frame_indices),
    ):
        array = np.asarray(value)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        coordinates[name] = array.astype(np.int64, copy=True)
    rows = len(coordinates["row_index"])
    if not rows or any(len(value) != rows for value in coordinates.values()):
        raise ValueError("sensitivity record coordinates must share non-empty length")
    if any(np.any(value < 0) for value in coordinates.values()):
        raise ValueError("sensitivity record coordinates must be non-negative")
    if set(metrics) != set(SENSITIVITY_METRICS):
        raise ValueError("sensitivity metrics are incomplete")
    metric_arrays: dict[str, np.ndarray] = {}
    for name in SENSITIVITY_METRICS:
        array = np.asarray(metrics[name])
        if array.shape != (rows,):
            raise ValueError(f"sensitivity metric {name} must have one value per row")
        if name == "translation_sign_changed":
            metric_arrays[name] = array.astype(bool, copy=True)
            continue
        numeric = array.astype(np.float64, copy=True)
        if np.isinf(numeric).any():
            raise ValueError(f"sensitivity metric {name} must not contain infinity")
        metric_arrays[name] = numeric

    records: list[dict[str, object]] = []
    for index in range(rows):
        record: dict[str, object] = {
            "policy_seed": int(policy_seed),
            "condition": condition,
            "group": group,
            "intervention": intervention,
            **{name: int(values[index]) for name, values in coordinates.items()},
        }
        for name, values in metric_arrays.items():
            if name == "translation_sign_changed":
                record[name] = bool(values[index])
            else:
                value = float(values[index])
                record[name] = value if np.isfinite(value) else None
        records.append(record)
    return records


def _derived_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def build_sensitivity_report(
    records: object,
    *,
    partition: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not isinstance(records, list) or not records:
        raise ValueError("sensitivity records must be a non-empty list")
    if partition not in {"train", "validation", "test"}:
        raise ValueError("sensitivity partition is invalid")
    typed: list[Mapping[str, object]] = []
    required = {
        "policy_seed",
        "condition",
        "group",
        "intervention",
        "row_index",
        "episode_index",
        "frame_index",
        *SENSITIVITY_METRICS,
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("sensitivity record schema is incompatible")
        typed.append(record)

    keys = sorted(
        {
            (
                int(record["policy_seed"]),
                str(record["condition"]),
                str(record["group"]),
                str(record["intervention"]),
            )
            for record in typed
        }
    )
    by_seed_condition: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    estimates: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for policy_seed, condition, group, intervention in keys:
        selected = [
            record
            for record in typed
            if (
                int(record["policy_seed"]),
                str(record["condition"]),
                str(record["group"]),
                str(record["intervention"]),
            )
            == (policy_seed, condition, group, intervention)
        ]
        metric_report: dict[str, object] = {}
        for metric in SENSITIVITY_METRICS:
            episode_values: dict[int, list[float]] = {}
            for record in selected:
                value = record[metric]
                if value is None:
                    continue
                episode_values.setdefault(int(record["episode_index"]), []).append(
                    float(value)
                )
            if not episode_values:
                metric_report[metric] = None
                continue
            clustered = cluster_bootstrap_mean(
                {
                    episode: float(np.mean(values))
                    for episode, values in episode_values.items()
                },
                samples=bootstrap_samples,
                seed=_derived_seed(
                    bootstrap_seed,
                    policy_seed,
                    condition,
                    group,
                    intervention,
                    metric,
                ),
            )
            metric_report[metric] = clustered
            estimates.setdefault(
                (condition, group, intervention, metric), {}
            )[policy_seed] = float(clustered["estimate"])
        by_seed_condition.setdefault(
            f"seed_{policy_seed}/{condition}", {}
        ).setdefault(group, {})[intervention] = metric_report

    across: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for (condition, group, intervention, metric), seed_values in sorted(
        estimates.items()
    ):
        values = np.asarray(
            [seed_values[seed] for seed in sorted(seed_values)], dtype=np.float64
        )
        across.setdefault(condition, {}).setdefault(group, {}).setdefault(
            intervention, {}
        )[metric] = {
            "estimate": float(np.mean(values)),
            "policy_seed_std": float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            ),
            "policy_seeds": int(len(values)),
        }

    return {
        "passed": True,
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "partition": partition,
        "rows": len(typed),
        "policy_seeds": sorted({int(record["policy_seed"]) for record in typed}),
        "conditions": sorted({str(record["condition"]) for record in typed}),
        "groups": [name for name in TOKEN_SLICES if any(key[2] == name for key in keys)],
        "interventions": sorted({str(record["intervention"]) for record in typed}),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "by_seed_condition": by_seed_condition,
        "across_policy_seeds": across,
    }
