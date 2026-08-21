from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from interaction_vla.representation_study.rl.distributions import build_case_manifest
from interaction_vla.representation_study.rl.evaluation_v2 import (
    evaluate_case_manifest,
)
from interaction_vla.representation_study.rl.gates import (
    AnchoringScreen,
    BackendScreen,
    CalibrationCandidate,
    oracle_gate,
    select_anchoring,
    select_backend,
    select_calibrated_severity,
)


def test_oracle_gate_requires_recovery_gain_and_retention() -> None:
    passed = oracle_gate(
        sft_recovery=0.40,
        rl_recovery=0.52,
        sft_nominal=0.70,
        rl_nominal=0.61,
    )
    failed = oracle_gate(
        sft_recovery=0.40,
        rl_recovery=0.55,
        sft_nominal=0.70,
        rl_nominal=0.59,
    )
    assert passed.passed is True
    assert failed.passed is False


def test_oracle_gate_requires_each_development_seed() -> None:
    result = oracle_gate(
        sft_recovery=(0.35, 0.35),
        rl_recovery=(0.48, 0.43),
        sft_nominal=(0.70, 0.70),
        rl_nominal=(0.62, 0.65),
        sft_recovery_auc=(0.35, 0.35),
        rl_recovery_auc=(0.44, 0.42),
    )
    assert result.passed is False
    assert any("development seed 1" in reason for reason in result.reasons)


def _screen(
    name: str,
    *,
    auc: tuple[float, float],
    nominal: tuple[float, float] = (0.65, 0.65),
    finite: bool = True,
) -> BackendScreen:
    return BackendScreen(
        backend=name,
        recovery_auc=auc,
        final_nominal_success=nominal,
        sft_nominal_success=0.70,
        finite=finite,
        resume_valid=True,
        simulator_integrity_failures=0,
    )


def test_backend_selection_uses_variance_when_auc_is_tied() -> None:
    result = select_backend(
        ppo=_screen("ppo", auc=(0.50, 0.54)),
        sac=_screen("sac", auc=(0.515, 0.516)),
    )
    assert result.selected_backend == "sac"


def test_backend_selection_rejects_unstable_or_forgetting_backend() -> None:
    result = select_backend(
        ppo=_screen("ppo", auc=(0.60, 0.61), finite=False),
        sac=_screen("sac", auc=(0.50, 0.51), nominal=(0.58, 0.59)),
    )
    assert result.passed is False
    assert result.selected_backend is None


def test_calibration_selects_rate_nearest_band_center() -> None:
    selected = select_calibrated_severity(
        {0.50: 0.62, 0.75: 0.47, 1.00: 0.32},
        target=(0.30, 0.50),
    )
    assert selected.passed is True
    assert selected.severity == 0.75


def test_calibration_uses_lower_severity_on_exact_tie() -> None:
    selected = select_calibrated_severity(
        {0.50: 0.35, 0.75: 0.45},
        target=(0.30, 0.50),
    )
    assert selected.severity == 0.50


def test_calibration_rejects_candidate_with_insufficient_kind_coverage() -> None:
    selected = select_calibrated_severity(
        {
            0.50: CalibrationCandidate(
                severity=0.50,
                recovery_success=0.40,
                accepted_by_kind={"premature_open": 9, "wrong_way_transport": 12},
            ),
            0.75: CalibrationCandidate(
                severity=0.75,
                recovery_success=0.45,
                accepted_by_kind={"premature_open": 10, "wrong_way_transport": 12},
            ),
        },
        target=(0.30, 0.50),
        minimum_accepted_per_kind=10,
    )
    assert selected.severity == 0.75


def test_anchoring_selects_simplest_eligible_variant() -> None:
    decision = select_anchoring(
        (
            AnchoringScreen("no_anchor", (0.48, 0.50), (0.57, 0.58), 0.70),
            AnchoringScreen("residual_only", (0.51, 0.52), (0.62, 0.63), 0.70),
            AnchoringScreen("full_anchoring", (0.52, 0.53), (0.66, 0.67), 0.70),
        )
    )
    assert decision.passed is True
    assert decision.inputs["selected_variant"] == "residual_only"


@dataclass(frozen=True)
class _Transition:
    done: bool
    reason: str
    episode_length: int
    episode_return: float
    reward_terminal: float
    reward_progress: float
    reward_residual: float
    residual: np.ndarray
    executed_local_action: np.ndarray
    action_was_clipped: bool
    ik_projection_scale: float
    episode_action_clipping_rate: float
    episode_mean_ik_projection_scale: float
    next_oracle_state: np.ndarray


class _Policy:
    def act(
        self,
        *,
        latent: torch.Tensor,
        oracle_state: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        assert deterministic is True
        return np.zeros(7, dtype=np.float32)


class _Runtime:
    def __init__(self) -> None:
        self.case = None
        self.steps = 0

    def reset_case(self, case):
        self.case = case
        self.steps = 0
        return type("Reset", (), {"oracle_state": np.zeros(36, dtype=np.float32)})()

    def policy_features(self):
        return np.zeros(7, dtype=np.float32), torch.zeros(1, 8)

    def step(self, *, base_action, latent, residual):
        self.steps += 1
        done = self.steps == 2
        return _Transition(
            done=done,
            reason="success" if done else "running",
            episode_length=self.steps,
            episode_return=float(done),
            reward_terminal=float(done),
            reward_progress=0.01,
            reward_residual=0.0,
            residual=np.asarray(residual),
            executed_local_action=np.full(7, self.steps, dtype=np.float32),
            action_was_clipped=False,
            ik_projection_scale=1.0,
            episode_action_clipping_rate=0.0,
            episode_mean_ik_projection_scale=1.0,
            next_oracle_state=np.full(36, self.steps, dtype=np.float32),
        )


def test_paired_evaluator_preserves_case_order_and_episode_rows() -> None:
    manifest = build_case_manifest(
        seed=5,
        calibration=1,
        training=1,
        curve=1,
        final=1,
    )
    cases = manifest.partition("curve")[:3]
    report = evaluate_case_manifest(
        _Policy(),
        _Runtime(),
        cases,
        policy_seed=17,
    )
    assert tuple(row.case_id for row in report.rows) == tuple(
        case.case_id for case in cases
    )
    assert report.all.episodes == 3
    assert report.all.success_rate == 1.0
    assert report.all.mean_steps == 2.0
