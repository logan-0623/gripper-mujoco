from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Final

import numpy as np

from .schema import (
    ALL_CONDITIONS,
    ORACLE_CONDITIONS,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SLICES,
)


DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "graph_representation_diagnostics_v1"
CATEGORICAL_GROUPS: Final[tuple[str, ...]] = (
    "phase",
    "next_relation",
    "relation_operator",
    "predicate",
)
HARD_BINARY_GROUPS: Final[tuple[str, ...]] = (
    "entity_presence",
    "relation_presence",
)


def _integer_vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=True)
    if not len(result) or np.any(result < 0):
        raise ValueError(f"{name} must be non-empty and non-negative")
    return result


@dataclass(frozen=True)
class EpisodeLayout:
    row_indices: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    episode_ids: tuple[int, ...]
    episode_slices: tuple[slice, ...]

    def __post_init__(self) -> None:
        for name in ("row_indices", "episode_indices", "frame_indices"):
            value = np.asarray(getattr(self, name), dtype=np.int64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def validate_episode_layout(
    *,
    row_indices: object,
    episode_indices: object,
    frame_indices: object,
) -> EpisodeLayout:
    rows = _integer_vector(row_indices, "row_indices")
    episodes = _integer_vector(episode_indices, "episode_indices")
    frames = _integer_vector(frame_indices, "frame_indices")
    if not (len(rows) == len(episodes) == len(frames)):
        raise ValueError("row, episode, and frame indices must share length")
    if len(set(rows.tolist())) != len(rows):
        raise ValueError("row_indices must be unique")

    episode_ids: list[int] = []
    episode_slices: list[slice] = []
    seen: set[int] = set()
    start = 0
    while start < len(episodes):
        episode = int(episodes[start])
        if episode in seen:
            raise ValueError("episode rows must form one contiguous block")
        seen.add(episode)
        stop = start + 1
        while stop < len(episodes) and int(episodes[stop]) == episode:
            stop += 1
        actual_frames = frames[start:stop]
        if int(actual_frames[0]) != 0:
            raise ValueError(f"episode {episode} must start at frame 0")
        expected_frames = np.arange(stop - start, dtype=np.int64)
        if not np.array_equal(actual_frames, expected_frames):
            raise ValueError(f"episode {episode} frame indices must be contiguous")
        episode_ids.append(episode)
        episode_slices.append(slice(start, stop))
        start = stop

    return EpisodeLayout(
        row_indices=rows,
        episode_indices=episodes,
        frame_indices=frames,
        episode_ids=tuple(episode_ids),
        episode_slices=tuple(episode_slices),
    )


def validate_tokens(tokens: object, *, rows: int) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float64)
    if values.shape != (rows, TOKEN_DIM):
        raise ValueError(f"tokens must have shape [{rows}, {TOKEN_DIM}]")
    if not np.isfinite(values).all():
        raise ValueError("tokens must be finite")
    result = values.copy()
    result.setflags(write=False)
    return result


def feature_distribution(
    values: object, *, active_epsilon: float
) -> dict[str, float | int]:
    epsilon = float(active_epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("active_epsilon must be finite and positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("feature values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("feature values must be finite")
    p05, p25, median, p75, p95 = np.quantile(
        array, (0.05, 0.25, 0.5, 0.75, 0.95), method="linear"
    )
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(p05),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p95": float(p95),
        "robust_range": float(p95 - p05),
        "active_fraction": float(np.mean(np.abs(array) > epsilon)),
        "negative_saturation_fraction": float(np.mean(array <= -0.99)),
        "positive_saturation_fraction": float(np.mean(array >= 0.99)),
    }


def temporal_feature_metrics(
    values: object, layout: EpisodeLayout
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(layout.row_indices),):
        raise ValueError("temporal feature values must match episode layout rows")
    if not np.isfinite(array).all():
        raise ValueError("temporal feature values must be finite")
    first = [
        np.diff(array[bounds])
        for bounds in layout.episode_slices
        if bounds.stop - bounds.start >= 2
    ]
    second = [
        np.diff(array[bounds], n=2)
        for bounds in layout.episode_slices
        if bounds.stop - bounds.start >= 3
    ]
    first_values = np.concatenate(first) if first else np.empty(0, dtype=np.float64)
    second_values = (
        np.concatenate(second) if second else np.empty(0, dtype=np.float64)
    )
    return {
        "first_difference_count": int(len(first_values)),
        "first_difference_mae": (
            None if not len(first_values) else float(np.mean(np.abs(first_values)))
        ),
        "first_difference_rms": (
            None
            if not len(first_values)
            else float(np.sqrt(np.mean(np.square(first_values))))
        ),
        "second_difference_count": int(len(second_values)),
        "second_difference_mae": (
            None if not len(second_values) else float(np.mean(np.abs(second_values)))
        ),
    }


def _probabilities(value: object, layout: EpisodeLayout, name: str) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(layout.row_indices) or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [rows, categories]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    mass = array.sum(axis=1)
    valid = mass > 0.0
    normalized = np.zeros_like(array)
    normalized[valid] = array[valid] / mass[valid, None]
    return normalized, valid


def categorical_sequence_metrics(
    probabilities: object,
    layout: EpisodeLayout,
    *,
    teacher_probabilities: object | None = None,
) -> dict[str, float | int | None]:
    normalized, valid = _probabilities(probabilities, layout, "probabilities")
    labels = np.argmax(normalized, axis=1)
    safe_probabilities = np.clip(normalized, np.finfo(np.float64).tiny, None)
    entropies = -np.sum(
        np.where(
            normalized > 0.0,
            normalized * np.log(safe_probabilities),
            0.0,
        ),
        axis=1,
    )

    teacher_labels: np.ndarray | None = None
    teacher_valid: np.ndarray | None = None
    if teacher_probabilities is not None:
        teacher, teacher_valid = _probabilities(
            teacher_probabilities, layout, "teacher probabilities"
        )
        if teacher.shape != normalized.shape:
            raise ValueError("teacher probabilities must share categorical shape")
        teacher_labels = np.argmax(teacher, axis=1)

    flips = 0
    transitions = 0
    false_flips = 0
    false_flip_transitions = 0
    dwell_lengths: list[int] = []
    for bounds in layout.episode_slices:
        segment_start: int | None = None
        previous_label: int | None = None
        for index in range(bounds.start, bounds.stop):
            if not valid[index]:
                if segment_start is not None:
                    dwell_lengths.append(index - segment_start)
                segment_start = None
                previous_label = None
                continue
            label = int(labels[index])
            if previous_label is None:
                segment_start = index
            elif label != previous_label:
                assert segment_start is not None
                dwell_lengths.append(index - segment_start)
                segment_start = index
            previous_label = label
        if segment_start is not None:
            dwell_lengths.append(bounds.stop - segment_start)

        for index in range(bounds.start + 1, bounds.stop):
            if valid[index - 1] and valid[index]:
                transitions += 1
                changed = bool(labels[index] != labels[index - 1])
                flips += int(changed)
                if (
                    teacher_labels is not None
                    and teacher_valid is not None
                    and teacher_valid[index - 1]
                    and teacher_valid[index]
                ):
                    false_flip_transitions += 1
                    false_flips += int(
                        changed
                        and teacher_labels[index] == teacher_labels[index - 1]
                    )

    valid_count = int(valid.sum())
    return {
        "valid_frames": valid_count,
        "missing_frames": int(len(valid) - valid_count),
        "mean_entropy": (
            None if not valid_count else float(np.mean(entropies[valid]))
        ),
        "flip_numerator": int(flips),
        "flip_denominator": int(transitions),
        "flip_rate": None if not transitions else float(flips / transitions),
        "mean_dwell_length": (
            None if not dwell_lengths else float(np.mean(dwell_lengths))
        ),
        "false_flip_numerator": int(false_flips),
        "false_flip_denominator": int(false_flip_transitions),
        "false_flip_rate": (
            None
            if not false_flip_transitions
            else float(false_flips / false_flip_transitions)
        ),
    }


def covariance_effective_rank(
    values: object, *, epsilon: float = 1.0e-12
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or array.shape[1] < 1:
        raise ValueError("effective-rank values must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("effective-rank values must be finite")
    threshold = float(epsilon)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("effective-rank epsilon must be finite and positive")
    centered = array - np.mean(array, axis=0, keepdims=True)
    covariance = centered.T @ centered / len(centered)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= threshold:
        return 0.0
    weights = eigenvalues[eigenvalues > threshold] / total
    return float(np.exp(-np.sum(weights * np.log(weights))))


def teacher_distance_metrics(
    predicted: object, teacher: object
) -> dict[str, float | int | None]:
    predicted_values = np.asarray(predicted, dtype=np.float64)
    teacher_values = np.asarray(teacher, dtype=np.float64)
    if (
        predicted_values.ndim != 2
        or not len(predicted_values)
        or predicted_values.shape != teacher_values.shape
    ):
        raise ValueError("predicted and teacher values must share non-empty 2D shape")
    if not (
        np.isfinite(predicted_values).all()
        and np.isfinite(teacher_values).all()
    ):
        raise ValueError("predicted and teacher values must be finite")
    difference = predicted_values - teacher_values
    l1 = np.sum(np.abs(difference), axis=1)
    l2 = np.linalg.norm(difference, axis=1)
    predicted_norm = np.linalg.norm(predicted_values, axis=1)
    teacher_norm = np.linalg.norm(teacher_values, axis=1)
    cosine_valid = (predicted_norm > 0.0) & (teacher_norm > 0.0)
    cosine_count = int(cosine_valid.sum())
    cosine_distance: np.ndarray | None = None
    if cosine_count:
        similarity = np.sum(
            predicted_values[cosine_valid] * teacher_values[cosine_valid], axis=1
        ) / (predicted_norm[cosine_valid] * teacher_norm[cosine_valid])
        cosine_distance = 1.0 - np.clip(similarity, -1.0, 1.0)
    return {
        "frames": int(len(predicted_values)),
        "mean_l1": float(np.mean(l1)),
        "mean_l2": float(np.mean(l2)),
        "cosine_frames": cosine_count,
        "mean_cosine_distance": (
            None if cosine_distance is None else float(np.mean(cosine_distance))
        ),
    }


def _lagged_pairs(
    predicted: np.ndarray,
    teacher: np.ndarray,
    layout: EpisodeLayout,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    for bounds in layout.episode_slices:
        predicted_episode = predicted[bounds]
        teacher_episode = teacher[bounds]
        if abs(lag) >= len(predicted_episode):
            continue
        if lag > 0:
            predicted_parts.append(predicted_episode[lag:])
            teacher_parts.append(teacher_episode[:-lag])
        elif lag < 0:
            predicted_parts.append(predicted_episode[:lag])
            teacher_parts.append(teacher_episode[-lag:])
        else:
            predicted_parts.append(predicted_episode)
            teacher_parts.append(teacher_episode)
    if not predicted_parts:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(predicted_parts), np.concatenate(teacher_parts)


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    correlation = float(np.corrcoef(first, second)[0, 1])
    return correlation if np.isfinite(correlation) else None


def lagged_feature_correlation(
    predicted: object,
    teacher: object,
    layout: EpisodeLayout,
    *,
    max_lag: int,
) -> dict[str, float | int | None]:
    if isinstance(max_lag, bool) or int(max_lag) != max_lag or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    predicted_values = np.asarray(predicted, dtype=np.float64)
    teacher_values = np.asarray(teacher, dtype=np.float64)
    expected_shape = (len(layout.row_indices),)
    if predicted_values.shape != expected_shape or teacher_values.shape != expected_shape:
        raise ValueError("lagged feature values must match episode layout rows")
    if not (
        np.isfinite(predicted_values).all()
        and np.isfinite(teacher_values).all()
    ):
        raise ValueError("lagged feature values must be finite")

    values: list[tuple[int, int, float | None]] = []
    for lag in range(-int(max_lag), int(max_lag) + 1):
        predicted_pairs, teacher_pairs = _lagged_pairs(
            predicted_values, teacher_values, layout, lag
        )
        values.append((lag, len(predicted_pairs), _pearson(predicted_pairs, teacher_pairs)))
    lag_zero = next(value for value in values if value[0] == 0)
    defined = [value for value in values if value[2] is not None]
    best = (
        None
        if not defined
        else max(defined, key=lambda value: (float(value[2]), -abs(value[0]), -value[0]))
    )
    return {
        "lag_zero_pairs": int(lag_zero[1]),
        "lag_zero_correlation": lag_zero[2],
        "best_lag": None if best is None else int(best[0]),
        "best_pairs": 0 if best is None else int(best[1]),
        "best_correlation": None if best is None else float(best[2]),
    }


def cluster_bootstrap_mean(
    episode_values: Mapping[int, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if not isinstance(episode_values, Mapping) or not episode_values:
        raise ValueError("episode_values must be a non-empty mapping")
    if isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    values = np.asarray(
        [float(episode_values[key]) for key in sorted(episode_values)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("episode_values must be finite")
    generator = np.random.default_rng(int(seed))
    sampled_indices = generator.integers(
        0, len(values), size=(int(samples), len(values))
    )
    replicate_means = np.mean(values[sampled_indices], axis=1)
    ci_low, ci_high = np.quantile(
        replicate_means, (0.025, 0.975), method="linear"
    )
    return {
        "estimate": float(np.mean(values)),
        "episodes": int(len(values)),
        "bootstrap_samples": int(samples),
        "confidence": 0.95,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _hard_state_sequence_metrics(
    values: np.ndarray, layout: EpisodeLayout
) -> dict[str, float | int | None]:
    if values.ndim != 2 or values.shape[0] != len(layout.row_indices):
        raise ValueError("hard-state values must have shape [rows, features]")
    states = values >= 0.5
    flips = 0
    transitions = 0
    for bounds in layout.episode_slices:
        episode = states[bounds]
        if len(episode) < 2:
            continue
        changed = episode[1:] != episode[:-1]
        flips += int(changed.sum())
        transitions += int(changed.size)
    return {
        "flip_numerator": flips,
        "flip_denominator": transitions,
        "flip_rate": None if not transitions else float(flips / transitions),
    }


def _feature_metrics(
    values: np.ndarray,
    layout: EpisodeLayout,
    *,
    active_epsilon: float,
    teacher: np.ndarray | None,
    max_lag: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "distribution": feature_distribution(
            values, active_epsilon=active_epsilon
        ),
        "temporal": temporal_feature_metrics(values, layout),
    }
    if teacher is not None:
        result["teacher_lag"] = lagged_feature_correlation(
            values, teacher, layout, max_lag=max_lag
        )
    return result


def _group_metrics(
    name: str,
    values: np.ndarray,
    layout: EpisodeLayout,
    *,
    teacher: np.ndarray | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "width": int(values.shape[1]),
        "effective_rank": covariance_effective_rank(values),
    }
    if name in CATEGORICAL_GROUPS:
        result["categorical"] = categorical_sequence_metrics(
            values,
            layout,
            teacher_probabilities=teacher,
        )
    if name in HARD_BINARY_GROUPS:
        result["hard_state"] = _hard_state_sequence_metrics(values, layout)
    if teacher is not None:
        result["teacher_distance"] = teacher_distance_metrics(values, teacher)
    return result


def _entry_metrics(
    tokens: np.ndarray,
    layout: EpisodeLayout,
    *,
    teacher: np.ndarray | None,
    active_epsilon: float,
    max_lag: int,
) -> dict[str, object]:
    features = {
        feature_name: _feature_metrics(
            tokens[:, feature_index],
            layout,
            active_epsilon=active_epsilon,
            teacher=(
                None if teacher is None else teacher[:, feature_index]
            ),
            max_lag=max_lag,
        )
        for feature_index, feature_name in enumerate(TOKEN_FEATURE_NAMES)
    }
    groups = {
        group_name: _group_metrics(
            group_name,
            tokens[:, bounds],
            layout,
            teacher=None if teacher is None else teacher[:, bounds],
        )
        for group_name, bounds in TOKEN_SLICES.items()
    }
    return {"features": features, "groups": groups}


def _numeric_leaves(
    value: Mapping[str, object],
    *,
    prefix: str = "",
    skip: frozenset[str] = frozenset({"episode_clustered"}),
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, item in value.items():
        if name in skip:
            continue
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(item, Mapping):
            result.update(_numeric_leaves(item, prefix=path, skip=skip))
        elif (
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
            and np.isfinite(float(item))
        ):
            result[path] = float(item)
    return result


def _derived_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") % (2**32)


def _attach_episode_intervals(
    aggregate: dict[str, object],
    episode_metrics: Mapping[int, Mapping[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    metric_path: str,
) -> None:
    leaves_by_episode = {
        episode: _numeric_leaves(metrics)
        for episode, metrics in episode_metrics.items()
    }
    leaf_paths = sorted(
        {path for leaves in leaves_by_episode.values() for path in leaves}
    )
    intervals: dict[str, object] = {}
    for leaf_path in leaf_paths:
        values = {
            episode: leaves[leaf_path]
            for episode, leaves in leaves_by_episode.items()
            if leaf_path in leaves
        }
        intervals[leaf_path] = cluster_bootstrap_mean(
            values,
            samples=bootstrap_samples,
            seed=_derived_seed(bootstrap_seed, metric_path, leaf_path),
        )
    aggregate["episode_clustered"] = intervals


def _sub_layout(layout: EpisodeLayout, bounds: slice) -> EpisodeLayout:
    return validate_episode_layout(
        row_indices=layout.row_indices[bounds],
        episode_indices=layout.episode_indices[bounds],
        frame_indices=layout.frame_indices[bounds],
    )


def _condition_seed_summary(entries: list[Mapping[str, object]]) -> dict[str, object]:
    leaves = [_numeric_leaves(entry["groups"]) for entry in entries]
    paths = sorted({path for item in leaves for path in item})
    summary: dict[str, object] = {}
    for path in paths:
        values = np.asarray(
            [item[path] for item in leaves if path in item], dtype=np.float64
        )
        summary[path] = {
            "mean": float(np.mean(values)),
            "sample_std": (
                0.0 if len(values) == 1 else float(np.std(values, ddof=1))
            ),
            "seeds": int(len(values)),
        }
    return summary


def build_representation_diagnostics(
    *,
    condition_tokens: Mapping[tuple[int, str], np.ndarray],
    teacher_tokens: np.ndarray,
    layout: EpisodeLayout,
    partition: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    max_lag: int,
    active_epsilon: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if partition not in {"train", "validation", "test"}:
        raise ValueError("diagnostics partition must be train, validation, or test")
    if not condition_tokens:
        raise ValueError("diagnostics condition matrix must be non-empty")
    keys = tuple(condition_tokens)
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or int(key[0]) < 0
        for key in keys
    ):
        raise ValueError("diagnostics condition matrix keys are invalid")
    seeds = sorted({int(seed) for seed, _ in keys})
    conditions_by_seed = {
        seed: {str(condition) for actual_seed, condition in keys if int(actual_seed) == seed}
        for seed in seeds
    }
    first_conditions = tuple(
        condition
        for condition in ALL_CONDITIONS
        if condition in conditions_by_seed[seeds[0]]
    )
    if first_conditions not in {ORACLE_CONDITIONS, ALL_CONDITIONS} or any(
        conditions_by_seed[seed] != set(first_conditions) for seed in seeds
    ):
        raise ValueError("diagnostics condition matrix must be complete for every seed")

    rows = len(layout.row_indices)
    teacher = validate_tokens(teacher_tokens, rows=rows)
    validated = {
        (int(seed), str(condition)): validate_tokens(tokens, rows=rows)
        for (seed, condition), tokens in condition_tokens.items()
    }
    for seed in seeds:
        if not np.array_equal(validated[(seed, "oracle_graph_v2")], teacher):
            raise ValueError("Privileged Teacher cache differs from teacher tokens")
        if np.any(validated[(seed, "flat")] != 0.0):
            raise ValueError("flat diagnostic cache must contain only zeros")

    entries: list[tuple[str, str, int | None, np.ndarray, bool]] = [
        ("shared/flat", "flat", None, validated[(seeds[0], "flat")], False),
        ("shared/oracle_graph_v2", "oracle_graph_v2", None, teacher, False),
    ]
    for seed in seeds:
        for condition in first_conditions:
            if condition in ORACLE_CONDITIONS:
                continue
            entries.append(
                (
                    f"seed_{seed}/{condition}",
                    condition,
                    seed,
                    validated[(seed, condition)],
                    True,
                )
            )

    aggregate_entries: dict[str, dict[str, object]] = {}
    episode_records: list[dict[str, object]] = []
    for entry_key, condition, estimator_seed, tokens, compare_teacher in entries:
        aggregate = _entry_metrics(
            tokens,
            layout,
            teacher=teacher if compare_teacher else None,
            active_epsilon=active_epsilon,
            max_lag=max_lag,
        )
        aggregate.update(
            {
                "condition": condition,
                "estimator_seed": estimator_seed,
                "rows": rows,
                "episodes": len(layout.episode_ids),
            }
        )
        per_episode: dict[int, dict[str, object]] = {}
        for episode_id, bounds in zip(layout.episode_ids, layout.episode_slices):
            episode_layout = _sub_layout(layout, bounds)
            episode_metrics = _entry_metrics(
                tokens[bounds],
                episode_layout,
                teacher=teacher[bounds] if compare_teacher else None,
                active_epsilon=active_epsilon,
                max_lag=max_lag,
            )
            per_episode[episode_id] = episode_metrics
            episode_records.append(
                {
                    "condition": condition,
                    "estimator_seed": estimator_seed,
                    "episode_id": episode_id,
                    "frames": int(bounds.stop - bounds.start),
                    **episode_metrics,
                }
            )
        for feature_name in TOKEN_FEATURE_NAMES:
            _attach_episode_intervals(
                aggregate["features"][feature_name],
                {
                    episode: metrics["features"][feature_name]
                    for episode, metrics in per_episode.items()
                },
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                metric_path=f"{entry_key}.features.{feature_name}",
            )
        for group_name in TOKEN_SLICES:
            _attach_episode_intervals(
                aggregate["groups"][group_name],
                {
                    episode: metrics["groups"][group_name]
                    for episode, metrics in per_episode.items()
                },
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                metric_path=f"{entry_key}.groups.{group_name}",
            )
        aggregate_entries[entry_key] = aggregate

    by_condition: dict[str, object] = {}
    for condition in first_conditions:
        matching = [
            entry
            for entry in aggregate_entries.values()
            if entry["condition"] == condition
        ]
        by_condition[condition] = {
            "estimator_seeds": [
                int(entry["estimator_seed"])
                for entry in matching
                if entry["estimator_seed"] is not None
            ],
            "group_metric_seed_summary": _condition_seed_summary(matching),
        }

    report: dict[str, object] = {
        "passed": True,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "partition": partition,
        "rows": rows,
        "episodes": len(layout.episode_ids),
        "conditions": list(first_conditions),
        "estimator_seeds": seeds,
        "teacher_deduplicated": True,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "max_lag": int(max_lag),
        "active_epsilon": float(active_epsilon),
        "by_seed_condition": aggregate_entries,
        "by_condition": by_condition,
    }
    episode_records.sort(
        key=lambda record: (
            str(record["condition"]),
            -1 if record["estimator_seed"] is None else int(record["estimator_seed"]),
            int(record["episode_id"]),
        )
    )
    return report, episode_records
