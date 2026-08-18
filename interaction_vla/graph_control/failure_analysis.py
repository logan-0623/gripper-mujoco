from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Final

import numpy as np

from .schema import TOKEN_DIM, TOKEN_SLICES
from .sensitivity import CATEGORICAL_GROUPS


FAILURE_ANALYSIS_SCHEMA_VERSION: Final[str] = "graph_failure_association_v1"
EXPOSURE_WINDOWS: Final[tuple[str, ...]] = (
    "pre_target_contact",
    "grasp_to_lift",
    "transport",
    "pre_release",
    "terminal",
)
OUTCOMES: Final[tuple[str, ...]] = (
    "success",
    "timeout",
    "target_drop",
    "wrong_object_stable_grasp",
)


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != TOKEN_DIM or not len(result):
        raise ValueError(f"{name} must have shape [rows, {TOKEN_DIM}]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _categorical_errors(policy: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    policy_total = np.clip(policy, 0.0, None).sum(axis=1)
    teacher_total = np.clip(teacher, 0.0, None).sum(axis=1)
    policy_active = policy_total > 1.0e-12
    teacher_active = teacher_total > 1.0e-12
    errors = (policy_active != teacher_active).astype(np.float64)
    both = policy_active & teacher_active
    errors[both] = (
        np.argmax(policy[both], axis=1) != np.argmax(teacher[both], axis=1)
    ).astype(np.float64)
    return errors


def training_error_thresholds(
    *,
    condition_tokens: Mapping[tuple[int, str], object],
    teacher_tokens_by_seed: Mapping[int, object],
    quantile: float = 0.75,
) -> dict[str, dict[str, float]]:
    level = float(quantile)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("training error quantile must lie within (0, 1)")
    if not condition_tokens or not teacher_tokens_by_seed:
        raise ValueError("training error token mappings must be non-empty")
    teachers = {
        int(seed): _matrix(tokens, f"Teacher train tokens for seed {seed}")
        for seed, tokens in teacher_tokens_by_seed.items()
    }
    result: dict[str, dict[str, float]] = {}
    for (seed_value, condition), token_values in sorted(condition_tokens.items()):
        seed = int(seed_value)
        if seed not in teachers:
            raise ValueError(f"Teacher train tokens are missing seed {seed}")
        tokens = _matrix(
            token_values, f"train tokens for seed {seed}/{condition}"
        )
        teacher = teachers[seed]
        if tokens.shape != teacher.shape:
            raise ValueError("condition and Teacher train tokens must share shape")
        thresholds: dict[str, float] = {}
        for group, bounds in TOKEN_SLICES.items():
            if group in CATEGORICAL_GROUPS:
                errors = _categorical_errors(tokens[:, bounds], teacher[:, bounds])
            else:
                errors = np.linalg.norm(
                    tokens[:, bounds] - teacher[:, bounds], axis=1
                )
            thresholds[group] = float(
                np.quantile(errors, level, method="linear")
            )
        result[f"seed_{seed}/{condition}"] = thresholds
    return result


def _frame_error(record: Mapping[str, object], group: str) -> float:
    errors = record.get("graph_error_by_group")
    if not isinstance(errors, Mapping) or group not in errors:
        raise ValueError(f"trace Graph errors are missing group {group}")
    value = errors[group]
    if not isinstance(value, Mapping):
        raise ValueError(f"trace Graph error group {group} is invalid")
    if value.get("kind") == "continuous":
        result = float(value.get("l2", np.nan))
    elif value.get("kind") == "categorical":
        agreement = value.get("agreement")
        result = 0.0 if agreement is True or agreement is None else 1.0
    else:
        raise ValueError(f"trace Graph error kind for {group} is invalid")
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("trace Graph error must be finite and non-negative")
    return result


def _window_summary(
    errors: np.ndarray, indices: Sequence[int], *, threshold: float
) -> dict[str, float | int] | None:
    if not indices:
        return None
    values = errors[np.asarray(indices, dtype=np.int64)]
    return {
        "steps": int(len(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "duration_above_threshold": int(np.sum(values > threshold)),
    }


def _false_flip_count(records: Sequence[Mapping[str, object]], group: str) -> int | None:
    if group not in CATEGORICAL_GROUPS:
        return None
    labels: list[tuple[int | None, int | None]] = []
    for record in records:
        value = record["graph_error_by_group"][group]
        labels.append((value.get("policy_label"), value.get("teacher_label")))
    false_flips = 0
    for previous, current in zip(labels, labels[1:]):
        policy_previous, teacher_previous = previous
        policy_current, teacher_current = current
        if None in previous or None in current:
            continue
        false_flips += int(
            policy_current != policy_previous
            and teacher_current == teacher_previous
        )
    return false_flips


def episode_error_exposure(
    records: object, *, thresholds: Mapping[str, float]
) -> dict[str, object]:
    if not isinstance(records, list) or not records:
        raise ValueError("failure analysis requires a non-empty trace episode")
    if not thresholds or any(group not in TOKEN_SLICES for group in thresholds):
        raise ValueError("failure analysis thresholds contain invalid groups")
    typed: list[Mapping[str, object]] = []
    required = {
        "policy_seed",
        "condition",
        "case_id",
        "environment_seed",
        "layout",
        "object_count",
        "step",
        "phase",
        "target_contact",
        "stable_target_grasp",
        "graph_error_by_group",
        "done",
        "success",
        "timeout",
        "target_drop",
        "stable_wrong_object_grasp",
    }
    for record in records:
        if not isinstance(record, Mapping) or not required <= set(record):
            raise ValueError("failure analysis trace record is incomplete")
        typed.append(record)
    if [int(record["step"]) for record in typed] != list(range(len(typed))):
        raise ValueError("failure analysis trace steps must be contiguous")
    if any(bool(record["done"]) for record in typed[:-1]) or not bool(typed[-1]["done"]):
        raise ValueError("failure analysis trace episode must be terminal")
    identity = (
        int(typed[0]["policy_seed"]),
        str(typed[0]["condition"]),
        str(typed[0]["case_id"]),
    )
    if any(
        (
            int(record["policy_seed"]),
            str(record["condition"]),
            str(record["case_id"]),
        )
        != identity
        for record in typed[1:]
    ):
        raise ValueError("failure analysis trace identity changes within episode")

    first_contact = next(
        (index for index, record in enumerate(typed) if record["target_contact"]),
        None,
    )
    first_release = next(
        (index for index, record in enumerate(typed) if record["phase"] == "release"),
        None,
    )
    first_stable = next(
        (
            index
            for index, record in enumerate(typed)
            if record["stable_target_grasp"]
        ),
        None,
    )
    windows: dict[str, list[int]] = {
        "pre_target_contact": (
            []
            if first_contact is None
            else list(range(max(0, first_contact - 9), first_contact + 1))
        ),
        "grasp_to_lift": (
            []
            if first_stable is None
            else [
                index
                for index in range(first_stable, len(typed))
                if typed[index]["phase"] in {"grasp", "lift"}
            ]
        ),
        "transport": [
            index for index, record in enumerate(typed) if record["phase"] == "transport"
        ],
        "pre_release": (
            []
            if first_release is None
            else list(range(max(0, first_release - 9), first_release + 1))
        ),
        "terminal": list(range(max(0, len(typed) - 20), len(typed))),
    }
    group_reports: dict[str, object] = {}
    for group, threshold_value in thresholds.items():
        threshold = float(threshold_value)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("failure analysis threshold must be finite and non-negative")
        errors = np.asarray(
            [_frame_error(record, group) for record in typed], dtype=np.float64
        )
        group_reports[group] = {
            "threshold": threshold,
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "duration_above_threshold": int(np.sum(errors > threshold)),
            "false_flips": _false_flip_count(typed, group),
            "windows": {
                name: _window_summary(errors, indices, threshold=threshold)
                for name, indices in windows.items()
            },
        }
    final = typed[-1]
    return {
        "policy_seed": identity[0],
        "condition": identity[1],
        "case_id": identity[2],
        "environment_seed": int(typed[0]["environment_seed"]),
        "layout": str(typed[0]["layout"]),
        "object_count": int(typed[0]["object_count"]),
        "steps": len(typed),
        "outcomes": {
            "success": bool(final["success"]),
            "timeout": bool(final["timeout"]),
            "target_drop": any(bool(record["target_drop"]) for record in typed),
            "wrong_object_stable_grasp": any(
                bool(record["stable_wrong_object_grasp"]) for record in typed
            ),
        },
        "groups": group_reports,
    }


def failure_association(
    episodes: object,
    *,
    outcome: str,
    group: str,
    window: str,
    threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if outcome not in OUTCOMES or group not in TOKEN_SLICES or window not in EXPOSURE_WINDOWS:
        raise ValueError("failure association selector is invalid")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("failure association episodes must be non-empty")
    level = float(threshold)
    if not np.isfinite(level) or level < 0.0:
        raise ValueError("failure association threshold must be finite and non-negative")
    if bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("failure association bootstrap settings are invalid")
    selected: list[tuple[tuple[int, str], float, bool]] = []
    for episode in episodes:
        try:
            window_value = episode["groups"][group]["windows"][window]
            positive = bool(episode["outcomes"][outcome])
            cluster = (int(episode["policy_seed"]), str(episode["case_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("failure association episode schema is incompatible") from error
        if window_value is None:
            continue
        exposure = float(window_value["mean"])
        if not np.isfinite(exposure):
            raise ValueError("failure association exposure must be finite")
        selected.append((cluster, exposure, positive))
    if not selected:
        raise ValueError("failure association window has no episode exposure")
    clusters = [value[0] for value in selected]
    if len(set(clusters)) != len(clusters):
        raise ValueError("failure association cluster keys must be unique")
    exposure = np.asarray([value[1] for value in selected], dtype=np.float64)
    positive = np.asarray([value[2] for value in selected], dtype=np.bool_)
    high = exposure > level

    def risk(mask: np.ndarray) -> float | None:
        return None if not np.any(mask) else float(np.mean(positive[mask]))

    high_risk = risk(high)
    low_risk = risk(~high)
    score = (
        None
        if high_risk is None or low_risk is None
        else float(high_risk - low_risk)
    )
    ratio = (
        None
        if high_risk is None or low_risk is None or low_risk == 0.0
        else float(high_risk / low_risk)
    )
    generator = np.random.default_rng(int(bootstrap_seed))
    replicate_scores: list[float] = []
    for _ in range(int(bootstrap_samples)):
        indices = generator.integers(0, len(selected), size=len(selected))
        sampled_high = high[indices]
        sampled_positive = positive[indices]
        if not np.any(sampled_high) or not np.any(~sampled_high):
            continue
        replicate_scores.append(
            float(
                np.mean(sampled_positive[sampled_high])
                - np.mean(sampled_positive[~sampled_high])
            )
        )
    interval = (
        (None, None)
        if not replicate_scores
        else tuple(
            float(value)
            for value in np.quantile(
                np.asarray(replicate_scores), (0.025, 0.975), method="linear"
            )
        )
    )
    positives = int(np.sum(positive))
    return {
        "outcome": outcome,
        "group": group,
        "window": window,
        "threshold": level,
        "episodes": len(selected),
        "positive_outcomes": positives,
        "underpowered": positives < 5,
        "high": {
            "episodes": int(np.sum(high)),
            "positive": int(np.sum(positive[high])),
            "risk": high_risk,
        },
        "low": {
            "episodes": int(np.sum(~high)),
            "positive": int(np.sum(positive[~high])),
            "risk": low_risk,
        },
        "failure_association_score": score,
        "risk_ratio": ratio,
        "ci_low": interval[0],
        "ci_high": interval[1],
        "confidence": 0.95,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_valid_replicates": len(replicate_scores),
        "cluster_unit": "policy_seed,case_id",
        "causal": False,
    }


def derived_bootstrap_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(str(int(seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def build_failure_analysis_report(
    episodes: object,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("failure analysis report requires episode exposures")
    if not thresholds:
        raise ValueError("failure analysis report requires training thresholds")
    typed: list[Mapping[str, object]] = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("failure analysis episode exposure must be a mapping")
        typed.append(episode)
    by_seed_condition: dict[str, object] = {}
    underpowered = 0
    unavailable = 0
    for key, group_thresholds in sorted(thresholds.items()):
        try:
            seed_label, condition = key.split("/", 1)
            seed = int(seed_label.removeprefix("seed_"))
        except (ValueError, AttributeError) as error:
            raise ValueError("failure analysis threshold key is invalid") from error
        selected = [
            dict(episode)
            for episode in typed
            if int(episode.get("policy_seed", -1)) == seed
            and str(episode.get("condition", "")) == condition
        ]
        if not selected:
            raise ValueError(f"failure analysis episodes are missing {key}")
        condition_report: dict[str, object] = {}
        for group, threshold in group_thresholds.items():
            if group not in TOKEN_SLICES:
                raise ValueError("failure analysis threshold group is invalid")
            window_report: dict[str, object] = {}
            for window in EXPOSURE_WINDOWS:
                outcome_report: dict[str, object] = {}
                for outcome in OUTCOMES:
                    try:
                        association = failure_association(
                            selected,
                            outcome=outcome,
                            group=group,
                            window=window,
                            threshold=float(threshold),
                            bootstrap_samples=bootstrap_samples,
                            bootstrap_seed=derived_bootstrap_seed(
                                bootstrap_seed,
                                key,
                                group,
                                window,
                                outcome,
                            ),
                        )
                    except ValueError as error:
                        if "no episode exposure" not in str(error):
                            raise
                        outcome_report[outcome] = {
                            "available": False,
                            "reason": "no_episode_exposure",
                        }
                        unavailable += 1
                        continue
                    outcome_report[outcome] = {
                        "available": True,
                        **association,
                    }
                    underpowered += int(bool(association["underpowered"]))
                window_report[window] = outcome_report
            condition_report[group] = window_report
        by_seed_condition[key] = condition_report
    return {
        "passed": True,
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "episodes": len(typed),
        "policy_seeds": sorted(
            {int(episode["policy_seed"]) for episode in typed}
        ),
        "conditions": sorted({str(episode["condition"]) for episode in typed}),
        "outcomes": list(OUTCOMES),
        "windows": list(EXPOSURE_WINDOWS),
        "threshold_source": "train_split_p75",
        "threshold_quantile": 0.75,
        "thresholds": {
            key: dict(value) for key, value in sorted(thresholds.items())
        },
        "cluster_unit": "policy_seed,case_id",
        "causal": False,
        "underpowered_comparisons": underpowered,
        "unavailable_comparisons": unavailable,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "by_seed_condition": by_seed_condition,
    }
