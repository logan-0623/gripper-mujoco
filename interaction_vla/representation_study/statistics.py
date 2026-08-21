from __future__ import annotations

from typing import Sequence

import numpy as np


def clustered_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[object],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    group = np.asarray([str(value) for value in clusters])
    if x.ndim != 1 or group.shape != x.shape or not len(x) or not np.isfinite(x).all():
        raise ValueError("cluster bootstrap inputs must be aligned finite non-empty vectors")
    if samples < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("cluster bootstrap settings are invalid")
    unique = np.unique(group)
    rng = np.random.default_rng(seed)
    replicates = np.empty(samples, dtype=np.float64)
    indices = {key: np.flatnonzero(group == key) for key in unique}
    for sample in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[key] for key in selected])
        replicates[sample] = float(np.mean(x[rows]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(x)),
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "clusters": int(len(unique)),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a finite vector within [0, 1]")
    if not len(values):
        return []
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(ranked)
    result[order] = ranked
    return result.tolist()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        result[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return result


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman correlation requires >=3 aligned finite values")
    ranked_x = _average_ranks(x)
    ranked_y = _average_ranks(y)
    if np.std(ranked_x) == 0.0 or np.std(ranked_y) == 0.0:
        return float("nan")
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def paired_sign_flip_pvalue(
    differences: Sequence[float], *, samples: int, seed: int
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or samples < 100:
        raise ValueError("paired sign-flip test requires finite differences and >=100 samples")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(samples, len(values)))
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float((1 + np.sum(null >= observed)) / (samples + 1))
