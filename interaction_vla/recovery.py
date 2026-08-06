from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .env import KinematicTabletopEnv
from .expert import ExpertPhase
from .graph.schema import SceneSnapshot


class PerturbationKind(str, Enum):
    ALIGN_OFFSET = "align_offset"
    FAILED_CLOSE = "failed_close"
    LIFT_OFFSET = "lift_offset"
    TRANSPORT_OFFSET = "transport_offset"


@dataclass(frozen=True)
class RecoverySpec:
    source_seed: int
    variant_id: int
    kind: PerturbationKind
    injection_phase: ExpertPhase
    delta: np.ndarray
    gripper_open: float | None


_KINDS = (
    PerturbationKind.ALIGN_OFFSET,
    PerturbationKind.FAILED_CLOSE,
    PerturbationKind.LIFT_OFFSET,
    PerturbationKind.TRANSPORT_OFFSET,
)

_PHASE_BY_KIND = {
    PerturbationKind.ALIGN_OFFSET: ExpertPhase.ALIGN,
    PerturbationKind.FAILED_CLOSE: ExpertPhase.CLOSE,
    PerturbationKind.LIFT_OFFSET: ExpertPhase.LIFT,
    PerturbationKind.TRANSPORT_OFFSET: ExpertPhase.TRANSPORT,
}


def make_recovery_spec(source_seed: int, variant_id: int) -> RecoverySpec:
    if source_seed < 0 or variant_id < 0:
        raise ValueError("source_seed and variant_id must be non-negative")
    kind = _KINDS[(source_seed + variant_id) % len(_KINDS)]
    rng = np.random.default_rng(
        np.random.SeedSequence((source_seed, variant_id, 0x5245434F))
    )
    angle = float(rng.uniform(-np.pi, np.pi))

    if kind is PerturbationKind.ALIGN_OFFSET:
        lateral = float(rng.uniform(0.04, 0.06))
        vertical = float(rng.uniform(0.02, 0.04))
        gripper_open: float | None = 1.0
    elif kind is PerturbationKind.FAILED_CLOSE:
        lateral = float(rng.uniform(0.075, 0.09))
        vertical = 0.0
        gripper_open = 0.0
    elif kind is PerturbationKind.LIFT_OFFSET:
        lateral = float(rng.uniform(0.03, 0.05))
        vertical = -0.02
        gripper_open = None
    else:
        lateral = float(rng.uniform(0.05, 0.07))
        vertical = 0.0
        gripper_open = None

    delta = np.asarray(
        (lateral * np.cos(angle), lateral * np.sin(angle), vertical),
        dtype=np.float32,
    )
    return RecoverySpec(
        source_seed=source_seed,
        variant_id=variant_id,
        kind=kind,
        injection_phase=_PHASE_BY_KIND[kind],
        delta=delta,
        gripper_open=gripper_open,
    )


def apply_recovery_spec(
    env: KinematicTabletopEnv,
    spec: RecoverySpec,
) -> SceneSnapshot:
    return env.perturb_gripper_state(spec.delta, gripper_open=spec.gripper_open)
