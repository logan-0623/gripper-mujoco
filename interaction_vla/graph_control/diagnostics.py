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
