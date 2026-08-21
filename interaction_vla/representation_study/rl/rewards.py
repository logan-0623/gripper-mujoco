from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardTerms:
    terminal: float
    progress: float
    residual: float

    @property
    def total(self) -> float:
        return self.terminal + self.progress + self.residual


def terminal_reward(reason: str) -> float:
    if reason == "success":
        return 1.0
    if reason in {"dropped", "wrong_object"}:
        return -1.0
    if reason in {"running", "timeout"}:
        return 0.0
    if reason == "physics_failure":
        raise ValueError("physics_failure cannot be converted into an RL reward")
    raise ValueError(f"unknown terminal reason: {reason}")


def recovery_reward(
    *,
    reason: str,
    previous_potential: float,
    next_potential: float,
    residual: object,
    residual_scale: object,
    gamma: float,
    progress_coefficient: float = 0.10,
    residual_coefficient: float = 0.01,
) -> RewardTerms:
    previous = float(previous_potential)
    following = float(next_potential)
    discount = float(gamma)
    progress_scale = float(progress_coefficient)
    residual_scale_value = float(residual_coefficient)
    action = np.asarray(residual, dtype=np.float64)
    scale = np.asarray(residual_scale, dtype=np.float64)
    numeric = np.asarray(
        (previous, following, discount, progress_scale, residual_scale_value),
        dtype=np.float64,
    )
    if (
        action.shape != (7,)
        or scale.shape != (7,)
        or not np.isfinite(action).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(numeric).all()
    ):
        raise ValueError("recovery reward contains a non-finite or incompatible input")
    if not 0.0 < discount <= 1.0:
        raise ValueError("reward gamma must lie within (0, 1]")
    if progress_scale < 0.0 or residual_scale_value < 0.0 or np.any(scale < 0.0):
        raise ValueError("reward coefficients and residual scale must be non-negative")
    progress = progress_scale * (discount * following - previous)
    scaled_action = action * scale
    cost = -residual_scale_value * float(np.dot(scaled_action, scaled_action))
    result = RewardTerms(terminal_reward(reason), progress, cost)
    if not np.isfinite(result.total):
        raise ValueError("recovery reward is non-finite")
    return result
