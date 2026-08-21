from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import numpy as np


def trace_is_perturbed(record: Mapping[str, object]) -> bool:
    scale = float(record.get("ik_projection_scale", 1.0))
    clearance = float(record.get("minimum_distractor_clearance", np.inf))
    if not np.isfinite(scale) or not np.isfinite(clearance):
        raise ValueError("trace perturbation signals must be finite")
    return bool(
        record.get("action_was_clipped", False)
        or scale < 0.95
        or record.get("wrong_object_contact", False)
        or record.get("target_drop", False)
        or clearance < 0.02
    )


def classify_trace_strata(
    records: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if not records:
        raise ValueError("trace episode must contain records")
    recovering = False
    result: list[str] = []
    expected_step = 0
    for record in records:
        step = int(record.get("step", -1))
        if step != expected_step:
            raise ValueError("trace episode steps must be contiguous and start at zero")
        expected_step += 1
        if bool(record.get("done", False)):
            result.append("terminal")
        elif trace_is_perturbed(record):
            result.append("perturbation")
            recovering = True
        elif recovering:
            result.append("recovery")
        else:
            result.append("nominal")
    return tuple(result)


def select_stratified_indices(
    strata: Sequence[str], *, per_stratum: int
) -> tuple[int, ...]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    selected: list[int] = []
    for name in ("nominal", "perturbation", "recovery", "terminal"):
        candidates = np.asarray(
            [index for index, value in enumerate(strata) if value == name],
            dtype=np.int64,
        )
        if len(candidates) <= per_stratum:
            selected.extend(int(value) for value in candidates)
            continue
        offsets = np.linspace(0, len(candidates) - 1, per_stratum)
        chosen = np.rint(offsets).astype(np.int64)
        selected.extend(int(candidates[value]) for value in chosen)
    return tuple(sorted(set(selected)))


def assign_groups_to_partitions(
    groups: Sequence[str],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    unique = tuple(sorted({str(value).strip() for value in groups}))
    if len(unique) < 3 or any(not value for value in unique):
        raise ValueError("State Bank grouping requires at least three non-empty groups")
    weights = np.asarray(ratios, dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("split ratios must be positive and sum to one")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"{int(seed)}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    raw = weights * len(ordered)
    counts = np.floor(raw).astype(np.int64)
    counts = np.maximum(counts, 1)
    while int(counts.sum()) > len(ordered):
        candidates = np.flatnonzero(counts > 1)
        index = int(candidates[np.argmin(raw[candidates] - counts[candidates])])
        counts[index] -= 1
    while int(counts.sum()) < len(ordered):
        index = int(np.argmax(raw - counts))
        counts[index] += 1
    names = ("train", "validation", "test")
    result: dict[str, str] = {}
    cursor = 0
    for name, count in zip(names, counts, strict=True):
        for group in ordered[cursor : cursor + int(count)]:
            result[group] = name
        cursor += int(count)
    return result

