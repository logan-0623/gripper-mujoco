from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from .schema import TOKEN_DIM, TOKEN_SLICES


SENSITIVITY_SCHEMA_VERSION: Final[str] = "graph_policy_sensitivity_v1"
CATEGORICAL_GROUPS: Final[frozenset[str]] = frozenset(
    {"phase", "next_relation", "relation_operator", "predicate"}
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

