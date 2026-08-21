from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np


GATE_SCHEMA = "recovery_rl_gate_v2"


@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    reasons: tuple[str, ...]
    inputs: dict[str, object]
    selected_backend: str | None = None
    input_hashes: dict[str, str] | None = None
    schema_version: str = GATE_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSelection:
    passed: bool
    severity: float | None
    recovery_success: float | None
    target: tuple[float, float]
    reasons: tuple[str, ...]
    candidates: dict[float, float]
    rejected_candidates: dict[float, tuple[str, ...]]


@dataclass(frozen=True)
class CalibrationCandidate:
    severity: float
    recovery_success: float
    accepted_by_kind: dict[str, int]

    def __post_init__(self) -> None:
        if not math.isfinite(self.severity) or not 0.0 < self.severity <= 1.0:
            raise ValueError("calibration severity must lie within (0, 1]")
        if (
            not math.isfinite(self.recovery_success)
            or not 0.0 <= self.recovery_success <= 1.0
        ):
            raise ValueError("calibration recovery success must lie within [0, 1]")
        if not self.accepted_by_kind or any(
            not str(kind).strip() or int(count) < 0
            for kind, count in self.accepted_by_kind.items()
        ):
            raise ValueError("calibration accepted counts must be non-negative by kind")


@dataclass(frozen=True)
class BackendScreen:
    backend: str
    recovery_auc: tuple[float, ...]
    final_nominal_success: tuple[float, ...]
    sft_nominal_success: float
    finite: bool
    resume_valid: bool
    simulator_integrity_failures: int

    def __post_init__(self) -> None:
        if self.backend not in {"ppo", "sac"}:
            raise ValueError("backend screen must identify PPO or SAC")
        if len(self.recovery_auc) < 2 or len(self.final_nominal_success) < 2:
            raise ValueError("backend screen requires two development seeds")
        if len(self.recovery_auc) != len(self.final_nominal_success):
            raise ValueError("backend screen seed counts must agree")
        values = (*self.recovery_auc, *self.final_nominal_success, self.sft_nominal_success)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("backend screen metrics must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("backend screen rates must lie within [0, 1]")
        if self.simulator_integrity_failures < 0:
            raise ValueError("simulator failure count must be non-negative")


@dataclass(frozen=True)
class AnchoringScreen:
    variant: str
    recovery_auc: tuple[float, ...]
    final_nominal_success: tuple[float, ...]
    sft_nominal_success: float

    def __post_init__(self) -> None:
        if self.variant not in {"no_anchor", "residual_only", "full_anchoring"}:
            raise ValueError(f"unknown anchoring variant: {self.variant}")
        if len(self.recovery_auc) < 2 or len(self.final_nominal_success) < 2:
            raise ValueError("anchoring screen requires two development seeds")
        values = (*self.recovery_auc, *self.final_nominal_success, self.sft_nominal_success)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("anchoring screen metrics must be rates within [0, 1]")


def _rates(value: float | Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a rate or sequence of rates")
    if np.isscalar(value):
        result = (float(value),)
    else:
        result = tuple(float(item) for item in value)
    if not result or not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result):
        raise ValueError(f"{name} must contain rates within [0, 1]")
    return result


def select_calibrated_severity(
    candidates: Mapping[float, float | CalibrationCandidate],
    *,
    target: tuple[float, float] = (0.30, 0.50),
    minimum_accepted_per_kind: int = 0,
) -> CalibrationSelection:
    lower, upper = (float(target[0]), float(target[1]))
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("calibration target must lie within [0, 1]")
    if minimum_accepted_per_kind < 0:
        raise ValueError("minimum accepted count must be non-negative")
    normalized: dict[float, float] = {}
    rejected: dict[float, tuple[str, ...]] = {}
    for severity, candidate in candidates.items():
        severity_value = float(severity)
        if isinstance(candidate, CalibrationCandidate):
            if not math.isclose(
                severity_value,
                candidate.severity,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("calibration candidate key and severity differ")
            rate_value = candidate.recovery_success
            failures = tuple(
                sorted(
                    kind
                    for kind, count in candidate.accepted_by_kind.items()
                    if count < minimum_accepted_per_kind
                )
            )
            if failures:
                rejected[severity_value] = tuple(
                    f"{kind} accepted below {minimum_accepted_per_kind}"
                    for kind in failures
                )
        else:
            rate_value = float(candidate)
        if (
            not math.isfinite(severity_value)
            or not 0.0 < severity_value <= 1.0
            or not math.isfinite(rate_value)
            or not 0.0 <= rate_value <= 1.0
        ):
            raise ValueError("calibration candidates must contain valid severities and rates")
        normalized[severity_value] = rate_value
    if not normalized:
        raise ValueError("calibration requires at least one candidate")
    eligible = [
        (severity, rate)
        for severity, rate in normalized.items()
        if lower <= rate <= upper and severity not in rejected
    ]
    if not eligible:
        return CalibrationSelection(
            passed=False,
            severity=None,
            recovery_success=None,
            target=(lower, upper),
            reasons=("no severity places frozen-SFT recovery success in band",),
            candidates=dict(sorted(normalized.items())),
            rejected_candidates=dict(sorted(rejected.items())),
        )
    center = 0.5 * (lower + upper)
    severity, rate = min(
        eligible,
        key=lambda item: (round(abs(item[1] - center), 12), item[0]),
    )
    return CalibrationSelection(
        passed=True,
        severity=severity,
        recovery_success=rate,
        target=(lower, upper),
        reasons=(),
        candidates=dict(sorted(normalized.items())),
        rejected_candidates=dict(sorted(rejected.items())),
    )


def _backend_reasons(screen: BackendScreen) -> tuple[str, ...]:
    reasons: list[str] = []
    if not screen.finite:
        reasons.append(f"{screen.backend} has non-finite loss or action")
    if not screen.resume_valid:
        reasons.append(f"{screen.backend} exact resume is invalid")
    if screen.simulator_integrity_failures > 0:
        reasons.append(f"{screen.backend} has simulator-integrity failures")
    nominal = float(np.median(screen.final_nominal_success))
    if nominal < screen.sft_nominal_success - 0.10 - 1.0e-12:
        reasons.append(f"{screen.backend} nominal success drops by more than 0.10")
    return tuple(reasons)


def select_backend(*, ppo: BackendScreen, sac: BackendScreen) -> GateDecision:
    if ppo.backend != "ppo" or sac.backend != "sac":
        raise ValueError("backend selection requires named PPO and SAC screens")
    screens = {"ppo": ppo, "sac": sac}
    rejection = {name: _backend_reasons(value) for name, value in screens.items()}
    eligible = [name for name in ("ppo", "sac") if not rejection[name]]
    inputs = {
        "screens": {name: asdict(value) for name, value in screens.items()},
        "rejections": {name: list(value) for name, value in rejection.items()},
        "auc_tie_threshold": 0.02,
    }
    if not eligible:
        return GateDecision(
            gate="backend",
            passed=False,
            reasons=tuple(reason for name in ("ppo", "sac") for reason in rejection[name]),
            inputs=inputs,
        )
    if len(eligible) == 1:
        selected = eligible[0]
    else:
        medians = {
            name: float(np.median(screens[name].recovery_auc)) for name in eligible
        }
        difference = abs(medians["ppo"] - medians["sac"])
        if difference >= 0.02 - 1.0e-12:
            selected = max(("ppo", "sac"), key=lambda name: medians[name])
        else:
            variances = {
                name: float(np.var(screens[name].recovery_auc)) for name in eligible
            }
            if math.isclose(
                variances["ppo"], variances["sac"], rel_tol=0.0, abs_tol=1.0e-12
            ):
                selected = "ppo"
            else:
                selected = min(("ppo", "sac"), key=lambda name: variances[name])
        inputs["median_recovery_auc"] = medians
        inputs["recovery_auc_variance"] = {
            name: float(np.var(screens[name].recovery_auc)) for name in eligible
        }
    inputs["selected_backend"] = selected
    return GateDecision(
        gate="backend",
        passed=True,
        reasons=(),
        inputs=inputs,
        selected_backend=selected,
    )


def oracle_gate(
    *,
    sft_recovery: float | Sequence[float],
    rl_recovery: float | Sequence[float],
    sft_nominal: float | Sequence[float],
    rl_nominal: float | Sequence[float],
    sft_recovery_auc: float | Sequence[float] | None = None,
    rl_recovery_auc: float | Sequence[float] | None = None,
) -> GateDecision:
    sft_r = _rates(sft_recovery, "sft_recovery")
    rl_r = _rates(rl_recovery, "rl_recovery")
    sft_n = _rates(sft_nominal, "sft_nominal")
    rl_n = _rates(rl_nominal, "rl_nominal")
    lengths = {len(sft_r), len(rl_r), len(sft_n), len(rl_n)}
    if len(lengths) != 1:
        raise ValueError("Oracle gate development seed counts must agree")
    sft_auc = _rates(
        sft_r if sft_recovery_auc is None else sft_recovery_auc,
        "sft_recovery_auc",
    )
    rl_auc = _rates(
        rl_r if rl_recovery_auc is None else rl_recovery_auc,
        "rl_recovery_auc",
    )
    if len(sft_auc) != len(rl_auc) or len(sft_auc) != len(sft_r):
        raise ValueError("Oracle gate AUC seed counts must agree")
    reasons: list[str] = []
    for index, values in enumerate(zip(sft_r, rl_r, sft_n, rl_n, strict=True)):
        base_recovery, final_recovery, base_nominal, final_nominal = values
        if final_recovery < base_recovery + 0.10 - 1.0e-12:
            reasons.append(
                f"development seed {index} recovery gain is below 0.10"
            )
        if final_nominal < base_nominal - 0.10 - 1.0e-12:
            reasons.append(
                f"development seed {index} nominal retention is below -0.10"
            )
    if float(np.median(rl_auc)) <= float(np.median(sft_auc)) + 1.0e-12:
        reasons.append("median recovery AUC does not improve over constant SFT")
    inputs = {
        "sft_recovery": list(sft_r),
        "rl_recovery": list(rl_r),
        "sft_nominal": list(sft_n),
        "rl_nominal": list(rl_n),
        "sft_recovery_auc": list(sft_auc),
        "rl_recovery_auc": list(rl_auc),
    }
    return GateDecision(
        gate="oracle",
        passed=not reasons,
        reasons=tuple(reasons),
        inputs=inputs,
    )


def select_anchoring(screens: Sequence[AnchoringScreen]) -> GateDecision:
    by_name = {screen.variant: screen for screen in screens}
    if len(by_name) != len(screens):
        raise ValueError("anchoring variants must be unique")
    required = ("no_anchor", "residual_only", "full_anchoring")
    if set(by_name) != set(required):
        raise ValueError("anchoring screen must contain the three registered variants")
    median_auc = {
        name: float(np.median(by_name[name].recovery_auc)) for name in required
    }
    median_nominal = {
        name: float(np.median(by_name[name].final_nominal_success)) for name in required
    }
    best_auc = max(median_auc.values())
    eligible = [
        name
        for name in required
        if median_nominal[name]
        >= by_name[name].sft_nominal_success - 0.10 - 1.0e-12
        and median_auc[name] >= best_auc - 0.02 - 1.0e-12
    ]
    inputs: dict[str, object] = {
        "screens": {name: asdict(by_name[name]) for name in required},
        "median_recovery_auc": median_auc,
        "median_nominal_success": median_nominal,
        "eligible_variants": eligible,
    }
    if not eligible:
        return GateDecision(
            gate="anchoring",
            passed=False,
            reasons=("no anchoring variant satisfies AUC and retention constraints",),
            inputs=inputs,
        )
    selected = eligible[0]
    inputs["selected_variant"] = selected
    return GateDecision(
        gate="anchoring",
        passed=True,
        reasons=(),
        inputs=inputs,
    )
