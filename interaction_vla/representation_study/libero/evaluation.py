from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


FAILURE_MODES = {
    "contact_failure",
    "grasp_failure",
    "object_selection_error",
    "placement_failure",
    "timeout",
    "drop",
    "task_specific_failure",
}


@dataclass(frozen=True)
class PairedRollout:
    case_id: str
    suite: str
    task_id: int
    source_episode_id: str
    initial_state_sha256: str
    inference_seed: int
    original_success: bool
    intervened_success: bool
    original_steps: int
    intervened_steps: int
    original_failure: str | None
    intervened_failure: str | None
    action_delta_translation: float
    action_delta_rotation: float
    action_delta_gripper: float

    def __post_init__(self) -> None:
        if not self.case_id or len(self.initial_state_sha256) != 64:
            raise ValueError("paired rollout case identity is invalid")
        if min(self.task_id, self.inference_seed, self.original_steps, self.intervened_steps) < 0:
            raise ValueError("paired rollout integer fields must be non-negative")
        values = np.asarray(
            (
                self.action_delta_translation,
                self.action_delta_rotation,
                self.action_delta_gripper,
            )
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("action deltas must be finite and non-negative")
        for success, failure, label in (
            (self.original_success, self.original_failure, "original"),
            (self.intervened_success, self.intervened_failure, "intervened"),
        ):
            if success and failure is not None:
                raise ValueError(f"successful {label} rollout must not have a failure mode")
            if not success and failure not in FAILURE_MODES:
                raise ValueError(f"failed {label} rollout requires a registered failure mode")


def _cluster_bootstrap(
    rows: Sequence[PairedRollout], *, samples: int, seed: int
) -> tuple[float, float]:
    groups: dict[tuple[str, int], list[PairedRollout]] = {}
    for row in rows:
        groups.setdefault((row.suite, row.task_id), []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = rng.choice(len(keys), size=len(keys), replace=True)
        sample_rows = [row for index in selected for row in groups[keys[int(index)]]]
        values.append(
            float(
                np.mean([row.intervened_success for row in sample_rows])
                - np.mean([row.original_success for row in sample_rows])
            )
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_outcome_report(
    rows: Sequence[PairedRollout], *, bootstrap_samples: int, seed: int
) -> dict[str, object]:
    if not rows or bootstrap_samples <= 0:
        raise ValueError("paired evaluation requires rows and positive bootstrap samples")
    case_ids = [row.case_id for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("paired evaluation case IDs must be unique")
    original = np.asarray([row.original_success for row in rows], dtype=np.float64)
    intervened = np.asarray([row.intervened_success for row in rows], dtype=np.float64)
    delta = float(intervened.mean() - original.mean())
    interval = _cluster_bootstrap(rows, samples=bootstrap_samples, seed=seed)
    mean_action = {
        "translation": float(np.mean([row.action_delta_translation for row in rows])),
        "rotation": float(np.mean([row.action_delta_rotation for row in rows])),
        "gripper": float(np.mean([row.action_delta_gripper for row in rows])),
    }
    action_sensitive = any(value > 1e-8 for value in mean_action.values())
    useful = bool(delta != 0.0 and (interval[0] > 0.0 or interval[1] < 0.0))
    return {
        "schema_version": "libero_paired_closed_loop_intervention_v1",
        "rows": [asdict(row) for row in rows],
        "pairs": len(rows),
        "bootstrap_unit": "task",
        "bootstrap_samples": bootstrap_samples,
        "original_success_rate": float(original.mean()),
        "intervened_success_rate": float(intervened.mean()),
        "delta_success": delta,
        "delta_success_ci95": list(interval),
        "mean_action_delta": mean_action,
        "action_sensitive": action_sensitive,
        "useful_for_closed_loop": useful,
        "original_failures": dict(
            sorted(Counter(row.original_failure for row in rows if row.original_failure).items())
        ),
        "intervened_failures": dict(
            sorted(Counter(row.intervened_failure for row in rows if row.intervened_failure).items())
        ),
    }
