from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .schema import TOKEN_DIM


DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "graph_representation_diagnostics_v1"


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
