from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class PhysicsRecoveryKind(str, Enum):
    WRONG_WAY_TRANSPORT = "wrong_way_transport"
    PREMATURE_OPEN = "premature_open"
    RECEPTACLE_MISALIGNMENT = "receptacle_misalignment"
    POST_PLACEMENT_RECLOSE = "post_placement_reclose"


@dataclass(frozen=True)
class PhysicsRecoverySpec:
    source_seed: int
    variant_id: int
    kind: PhysicsRecoveryKind
    trigger_phase: str
    direction_sign: float
    translation_steps: int
    translation_distance: float
    open_substeps: int
    close_descent_steps: int = 0

    def __post_init__(self) -> None:
        if self.source_seed < 0 or self.variant_id < 0:
            raise ValueError("source_seed and variant_id must be non-negative")
        if self.direction_sign not in {-1.0, 1.0}:
            raise ValueError("direction_sign must be -1 or 1")
        if (
            self.translation_steps < 0
            or self.open_substeps < 0
            or self.close_descent_steps < 0
        ):
            raise ValueError("recovery step counts must be non-negative")
        if not math.isfinite(self.translation_distance) or self.translation_distance < 0.0:
            raise ValueError("translation_distance must be finite and non-negative")
        if self.kind is PhysicsRecoveryKind.WRONG_WAY_TRANSPORT:
            expected_phase, expected = "transport", (3, 0.06, 0, 0)
        elif self.kind is PhysicsRecoveryKind.PREMATURE_OPEN:
            expected_phase, expected = "transport", (0, 0.0, 1, 0)
        elif self.kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
            expected_phase, expected = "transport", (2, 0.04, 0, 0)
        else:
            expected_phase, expected = "retreat", (0, 0.0, 0, 5)
        actual = (
            self.translation_steps,
            self.translation_distance,
            self.open_substeps,
            self.close_descent_steps,
        )
        if self.trigger_phase != expected_phase or actual != expected:
            raise ValueError(
                f"{self.kind.value} requires phase={expected_phase} and "
                "steps/distance/open_substeps/close_descent_steps="
                f"{expected}"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "source_seed": self.source_seed,
            "variant_id": self.variant_id,
            "kind": self.kind.value,
            "trigger_phase": self.trigger_phase,
            "direction_sign": self.direction_sign,
            "translation_steps": self.translation_steps,
            "translation_distance": self.translation_distance,
            "open_substeps": self.open_substeps,
            "close_descent_steps": self.close_descent_steps,
        }


def make_physics_recovery_spec(
    source_seed: int,
    variant_id: int,
    *,
    kind_index: int | None = None,
) -> PhysicsRecoverySpec:
    if source_seed < 0 or variant_id < 0:
        raise ValueError("source_seed and variant_id must be non-negative")
    resolved_kind_index = variant_id if kind_index is None else int(kind_index)
    if resolved_kind_index < 0:
        raise ValueError("kind_index must be non-negative")
    kinds = tuple(PhysicsRecoveryKind)
    kind = kinds[resolved_kind_index % len(kinds)]
    rng = np.random.default_rng(
        np.random.SeedSequence((source_seed, variant_id, 0x50524752))
    )
    direction_sign = float(rng.choice(np.asarray((-1.0, 1.0))))
    if kind is PhysicsRecoveryKind.WRONG_WAY_TRANSPORT:
        trigger_phase = "transport"
        translation_steps, translation_distance, open_substeps, close_steps = (
            3,
            0.06,
            0,
            0,
        )
    elif kind is PhysicsRecoveryKind.PREMATURE_OPEN:
        trigger_phase = "transport"
        translation_steps, translation_distance, open_substeps, close_steps = (
            0,
            0.0,
            1,
            0,
        )
    elif kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
        trigger_phase = "transport"
        translation_steps, translation_distance, open_substeps, close_steps = (
            2,
            0.04,
            0,
            0,
        )
    else:
        trigger_phase = "retreat"
        translation_steps, translation_distance, open_substeps, close_steps = (
            0,
            0.0,
            0,
            5,
        )
    return PhysicsRecoverySpec(
        source_seed=source_seed,
        variant_id=variant_id,
        kind=kind,
        trigger_phase=trigger_phase,
        direction_sign=direction_sign,
        translation_steps=translation_steps,
        translation_distance=translation_distance,
        open_substeps=open_substeps,
        close_descent_steps=close_steps,
    )


def recovery_trigger_ready(
    spec: PhysicsRecoverySpec,
    *,
    phase: str,
    stable_target: bool,
    distance: float,
    supported_target: bool = False,
) -> bool:
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("recovery trigger distance must be finite and non-negative")
    if phase != spec.trigger_phase:
        return False
    if spec.kind is PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE:
        return supported_target and distance <= 0.065
    if not stable_target:
        return False
    if spec.kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
        return distance <= 0.10
    return distance > 0.15


def recovery_translation_direction(
    spec: PhysicsRecoverySpec,
    target_xy: np.ndarray,
    receptacle_xy: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_xy, dtype=np.float64)
    receptacle = np.asarray(receptacle_xy, dtype=np.float64)
    if (
        target.shape != (2,)
        or receptacle.shape != (2,)
        or not np.isfinite(target).all()
        or not np.isfinite(receptacle).all()
    ):
        raise ValueError("target_xy and receptacle_xy must be finite 2D vectors")
    radial = target - receptacle
    norm = float(np.linalg.norm(radial))
    if norm < 1e-8:
        radial = np.asarray((1.0, 0.0), dtype=np.float64)
    else:
        radial /= norm
    if spec.kind is PhysicsRecoveryKind.WRONG_WAY_TRANSPORT:
        direction = radial
    elif spec.kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
        direction = spec.direction_sign * np.asarray((-radial[1], radial[0]))
    else:
        raise ValueError(f"{spec.kind.value} does not define a translation direction")
    return direction.astype(np.float32)
