from __future__ import annotations

"""Evaluation-only IK safety for learned Cartesian policies."""

from dataclasses import dataclass

import numpy as np

from .franka_controller import ControllerDiagnostics, FrankaCartesianController


DEFAULT_IK_PROJECTION_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)


@dataclass(frozen=True)
class IKProjectionResult:
    raw_action: np.ndarray
    action: np.ndarray
    scale: float
    raw_diagnostics: ControllerDiagnostics
    projected_diagnostics: ControllerDiagnostics


def project_cartesian_action(
    controller: FrankaCartesianController,
    action: np.ndarray,
    *,
    scales: tuple[float, ...] = DEFAULT_IK_PROJECTION_SCALES,
) -> IKProjectionResult:
    raw_action = np.asarray(action, dtype=np.float32).copy()
    if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
        raise ValueError("Cartesian action must be a finite vector with shape (7,)")
    resolved_scales = tuple(float(scale) for scale in scales)
    if (
        len(resolved_scales) < 2
        or resolved_scales[0] != 1.0
        or resolved_scales[-1] != 0.0
        or not all(np.isfinite(scale) and 0.0 <= scale <= 1.0 for scale in resolved_scales)
        or not all(
            left > right
            for left, right in zip(resolved_scales, resolved_scales[1:])
        )
    ):
        raise ValueError(
            "IK scale schedule must strictly decrease from 1.0 to 0.0 within [0, 1]"
        )
    raw_diagnostics: ControllerDiagnostics | None = None
    for scale in resolved_scales:
        candidate = raw_action.copy()
        candidate[:6] *= float(scale)
        diagnostics = controller.apply_action(candidate)
        if raw_diagnostics is None:
            raw_diagnostics = diagnostics
        if not diagnostics.ik_limited:
            return IKProjectionResult(
                raw_action=raw_action,
                action=candidate,
                scale=float(scale),
                raw_diagnostics=raw_diagnostics,
                projected_diagnostics=diagnostics,
            )
    raise RuntimeError("zero-pose Cartesian action is unexpectedly IK-limited")
