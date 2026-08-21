from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Protocol, Sequence

import numpy as np
import torch

from interaction_vla.env import TerminationReason

from .distributions import RecoveryCase


EVALUATION_V2_SCHEMA = "recovery_rl_evaluation_v2"


class EvaluationPolicy(Protocol):
    def act(
        self,
        *,
        latent: torch.Tensor,
        oracle_state: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray: ...


class EvaluationRuntime(Protocol):
    def reset_case(self, case: RecoveryCase) -> object: ...

    def policy_features(self) -> tuple[np.ndarray, torch.Tensor]: ...

    def step(
        self,
        *,
        base_action: np.ndarray,
        latent: torch.Tensor,
        residual: np.ndarray,
    ) -> object: ...


@dataclass(frozen=True)
class EpisodeOutcome:
    case_id: str
    source_seed: int
    variant_id: int
    family: str
    intervention_kind: str
    policy_seed: int
    success: bool
    termination_reason: str
    steps: int
    episode_return: float
    reward_terminal: float
    reward_progress: float
    reward_residual: float
    mean_residual_norm: float
    action_clipping_rate: float
    action_smoothness: float
    mean_ik_projection_scale: float


@dataclass(frozen=True)
class DistributionOutcome:
    episodes: int
    success_rate: float
    timeout_rate: float
    drop_rate: float
    wrong_object_rate: float
    mean_steps: float
    mean_return: float
    mean_residual_norm: float
    action_clipping_rate: float
    action_smoothness: float
    mean_ik_projection_scale: float


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    policy_seed: int
    case_ids: tuple[str, ...]
    rows: tuple[EpisodeOutcome, ...]
    all: DistributionOutcome
    nominal: DistributionOutcome | None
    perturbation: DistributionOutcome | None
    recovery: DistributionOutcome | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_seed": self.policy_seed,
            "case_ids": list(self.case_ids),
            "rows": [asdict(row) for row in self.rows],
            "all": asdict(self.all),
            "nominal": None if self.nominal is None else asdict(self.nominal),
            "perturbation": (
                None if self.perturbation is None else asdict(self.perturbation)
            ),
            "recovery": None if self.recovery is None else asdict(self.recovery),
        }


def _episode_seed(policy_seed: int, case: RecoveryCase) -> int:
    return int(
        np.random.SeedSequence(
            (policy_seed, case.source_seed, case.variant_id, 0x45564132)
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _aggregate(rows: Sequence[EpisodeOutcome]) -> DistributionOutcome:
    if not rows:
        raise ValueError("evaluation aggregation requires episode rows")
    count = len(rows)
    reasons = [row.termination_reason for row in rows]
    return DistributionOutcome(
        episodes=count,
        success_rate=float(np.mean([row.success for row in rows])),
        timeout_rate=reasons.count(TerminationReason.TIMEOUT.value) / count,
        drop_rate=reasons.count(TerminationReason.DROPPED.value) / count,
        wrong_object_rate=(
            reasons.count(TerminationReason.WRONG_OBJECT.value) / count
        ),
        mean_steps=float(np.mean([row.steps for row in rows])),
        mean_return=float(np.mean([row.episode_return for row in rows])),
        mean_residual_norm=float(
            np.mean([row.mean_residual_norm for row in rows])
        ),
        action_clipping_rate=float(
            np.mean([row.action_clipping_rate for row in rows])
        ),
        action_smoothness=float(np.mean([row.action_smoothness for row in rows])),
        mean_ik_projection_scale=float(
            np.mean([row.mean_ik_projection_scale for row in rows])
        ),
    )


def evaluate_case_manifest(
    policy: EvaluationPolicy,
    runtime: EvaluationRuntime,
    cases: Sequence[RecoveryCase],
    *,
    policy_seed: int,
) -> EvaluationReport:
    case_ids = tuple(case.case_id for case in cases)
    if not case_ids:
        raise ValueError("paired evaluation requires at least one case")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("paired evaluation case ids must be unique")
    if policy_seed < 0:
        raise ValueError("paired evaluation policy_seed must be non-negative")
    rows: list[EpisodeOutcome] = []
    for case in cases:
        seed = _episode_seed(policy_seed, case)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        reset = runtime.reset_case(case)
        oracle_state = np.asarray(
            getattr(reset, "oracle_state"),
            dtype=np.float32,
        )
        if oracle_state.shape != (36,) or not np.isfinite(oracle_state).all():
            raise ValueError("evaluation reset Oracle-State must be finite and 36D")
        residual_norms: list[float] = []
        executed_actions: list[np.ndarray] = []
        reward_terminal = 0.0
        reward_progress = 0.0
        reward_residual = 0.0
        while True:
            base_action, latent = runtime.policy_features()
            residual = np.asarray(
                policy.act(
                    latent=latent,
                    oracle_state=oracle_state,
                    deterministic=True,
                ),
                dtype=np.float32,
            )
            if (
                residual.shape != (7,)
                or not np.isfinite(residual).all()
                or np.any(np.abs(residual) > 1.0 + 1.0e-6)
            ):
                raise ValueError("evaluation residual must be finite and bounded 7D")
            transition = runtime.step(
                base_action=np.asarray(base_action, dtype=np.float32),
                latent=latent,
                residual=residual,
            )
            residual_norms.append(float(np.linalg.norm(residual)))
            executed = np.asarray(
                getattr(transition, "executed_local_action"),
                dtype=np.float32,
            )
            if executed.shape != (7,) or not np.isfinite(executed).all():
                raise ValueError("evaluation executed action must be finite and 7D")
            executed_actions.append(executed)
            reward_terminal += float(getattr(transition, "reward_terminal"))
            reward_progress += float(getattr(transition, "reward_progress"))
            reward_residual += float(getattr(transition, "reward_residual"))
            next_state = getattr(transition, "next_oracle_state")
            if not bool(getattr(transition, "done")):
                oracle_state = np.asarray(next_state, dtype=np.float32)
                if (
                    oracle_state.shape != (36,)
                    or not np.isfinite(oracle_state).all()
                ):
                    raise ValueError(
                        "evaluation next Oracle-State must be finite and 36D"
                    )
                continue
            reason = str(getattr(transition, "reason"))
            smoothness = 0.0
            if len(executed_actions) > 1:
                smoothness = float(
                    np.mean(
                        [
                            np.linalg.norm(current - previous)
                            for previous, current in zip(
                                executed_actions[:-1],
                                executed_actions[1:],
                                strict=True,
                            )
                        ]
                    )
                )
            rows.append(
                EpisodeOutcome(
                    case_id=case.case_id,
                    source_seed=case.source_seed,
                    variant_id=case.variant_id,
                    family=case.family,
                    intervention_kind=case.intervention_kind,
                    policy_seed=seed,
                    success=reason == TerminationReason.SUCCESS.value,
                    termination_reason=reason,
                    steps=int(getattr(transition, "episode_length")),
                    episode_return=float(getattr(transition, "episode_return")),
                    reward_terminal=reward_terminal,
                    reward_progress=reward_progress,
                    reward_residual=reward_residual,
                    mean_residual_norm=float(np.mean(residual_norms)),
                    action_clipping_rate=float(
                        getattr(transition, "episode_action_clipping_rate")
                    ),
                    action_smoothness=smoothness,
                    mean_ik_projection_scale=float(
                        getattr(transition, "episode_mean_ik_projection_scale")
                    ),
                )
            )
            break
    all_rows = tuple(rows)

    def family(name: str) -> DistributionOutcome | None:
        selected = tuple(row for row in all_rows if row.family == name)
        return None if not selected else _aggregate(selected)

    return EvaluationReport(
        schema_version=EVALUATION_V2_SCHEMA,
        policy_seed=policy_seed,
        case_ids=case_ids,
        rows=all_rows,
        all=_aggregate(all_rows),
        nominal=family("nominal"),
        perturbation=family("perturbation"),
        recovery=family("recovery"),
    )
