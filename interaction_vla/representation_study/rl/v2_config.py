from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


RECOVERY_RL_V2_SCHEMA = "recovery_rl_v2"
RECOVERY_RL_V2_SNAPSHOTS = (0, 4096, 8192, 12288, 16384, 20480)
RECOVERY_RL_V2_PROBABILITIES = (0.50, 0.30, 0.20)
IMMUTABLE_V1_ROOT = Path("outputs/representation_study/icra")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _only(raw: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ", ".join(sorted(unknown)))


def _finite_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_nonnegative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RecoveryDistributionConfig:
    recovery_probability: float
    perturbation_probability: float
    nominal_probability: float
    calibration_seed_count: int
    training_seed_count: int
    curve_case_count: int
    final_case_count: int
    severity_candidates: tuple[float, ...]
    minimum_accepted_per_kind: int

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (
            self.recovery_probability,
            self.perturbation_probability,
            self.nominal_probability,
        )

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float64)
        if (
            values.shape != (3,)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.isclose(values.sum(), 1.0)
        ):
            raise ValueError("distribution probabilities must be finite, non-negative, and sum to one")
        if not np.allclose(values, RECOVERY_RL_V2_PROBABILITIES):
            raise ValueError("Recovery RL v2 distribution must be recovery=0.50, perturbation=0.30, nominal=0.20")
        counts = (
            self.calibration_seed_count,
            self.training_seed_count,
            self.curve_case_count,
            self.final_case_count,
            self.minimum_accepted_per_kind,
        )
        if min(counts) < 1:
            raise ValueError("distribution counts must be positive")
        severities = np.asarray(self.severity_candidates, dtype=np.float64)
        if (
            severities.ndim != 1
            or severities.size == 0
            or not np.isfinite(severities).all()
            or np.any(severities <= 0.0)
            or np.any(severities > 1.0)
            or len(set(self.severity_candidates)) != len(self.severity_candidates)
            or tuple(sorted(self.severity_candidates)) != self.severity_candidates
        ):
            raise ValueError("severity_candidates must be unique increasing values within (0, 1]")


@dataclass(frozen=True)
class PPOV2Config:
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    actor_learning_rate: float
    value_learning_rate: float
    gae_lambda: float
    clip_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        if min(self.rollout_steps, self.update_epochs, self.minibatch_size) < 1:
            raise ValueError("PPO counts must be positive")
        if self.rollout_steps % self.minibatch_size:
            raise ValueError("ppo.rollout_steps must be divisible by minibatch_size")
        _finite_positive(self.actor_learning_rate, "ppo.actor_learning_rate")
        _finite_positive(self.value_learning_rate, "ppo.value_learning_rate")
        _finite_positive(self.max_grad_norm, "ppo.max_grad_norm")
        if not np.isfinite(self.gae_lambda) or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("ppo.gae_lambda must lie within [0, 1]")
        if not np.isfinite(self.clip_coefficient) or not 0.0 < self.clip_coefficient <= 1.0:
            raise ValueError("ppo.clip_coefficient must lie within (0, 1]")
        _finite_nonnegative(self.entropy_coefficient, "ppo.entropy_coefficient")


@dataclass(frozen=True)
class SACConfig:
    replay_capacity: int
    warmup_steps: int
    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    temperature_learning_rate: float
    polyak: float
    updates_per_environment_step: int
    target_entropy: float

    def __post_init__(self) -> None:
        if min(
            self.replay_capacity,
            self.warmup_steps,
            self.batch_size,
            self.updates_per_environment_step,
        ) < 1:
            raise ValueError("SAC counts must be positive")
        if self.warmup_steps >= self.replay_capacity:
            raise ValueError("sac.warmup_steps must be smaller than replay_capacity")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "temperature_learning_rate",
        ):
            _finite_positive(float(getattr(self, name)), f"sac.{name}")
        if not np.isfinite(self.polyak) or not 0.0 < self.polyak <= 1.0:
            raise ValueError("sac.polyak must lie within (0, 1]")
        if not np.isfinite(self.target_entropy) or self.target_entropy >= 0.0:
            raise ValueError("sac.target_entropy must be finite and negative")


@dataclass(frozen=True)
class RecoveryRLV2Config:
    config_path: Path
    output_dir: Path
    bridge_config: Path
    sft_checkpoint: str
    continued_sft_checkpoint: str
    device: str
    distribution: RecoveryDistributionConfig
    screen_steps: int
    formal_steps: int
    snapshot_steps: tuple[int, ...]
    oracle_state_dim: int
    residual_scale: tuple[float, ...]
    gamma: float
    progress_coefficient: float
    residual_coefficient: float
    nominal_anchor_coefficient: float
    latent_anchor_coefficient: float
    ppo: PPOV2Config
    sac: SACConfig
    seed: int
    schema_version: str = RECOVERY_RL_V2_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_RL_V2_SCHEMA:
            raise ValueError("Recovery RL v2 schema_version is incompatible")
        if self.output_dir.as_posix().rstrip("/") == IMMUTABLE_V1_ROOT.as_posix():
            raise ValueError("Recovery RL v2 cannot target the immutable v1 root")
        if not self.sft_checkpoint.strip() or not self.continued_sft_checkpoint.strip():
            raise ValueError("v2 parent checkpoints must be non-empty")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")
        if min(self.screen_steps, self.formal_steps) < 1 or self.seed < 0:
            raise ValueError("training steps must be positive and seed non-negative")
        if self.snapshot_steps != RECOVERY_RL_V2_SNAPSHOTS:
            raise ValueError("formal snapshot schedule is incompatible")
        if self.formal_steps != self.snapshot_steps[-1]:
            raise ValueError("formal_steps must equal the final snapshot step")
        if self.oracle_state_dim != 36:
            raise ValueError("Compact Oracle-State must have width 36")
        if len(self.residual_scale) != 7 or any(
            not np.isfinite(value) or value < 0.0 for value in self.residual_scale
        ):
            raise ValueError("residual_scale must contain seven finite non-negative values")
        if not np.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie within (0, 1]")
        for name in (
            "progress_coefficient",
            "residual_coefficient",
            "nominal_anchor_coefficient",
            "latent_anchor_coefficient",
        ):
            _finite_nonnegative(float(getattr(self, name)), name)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        config_path: str | Path,
    ) -> "RecoveryRLV2Config":
        raw = _mapping(value, "Recovery RL v2 config")
        _only(
            raw,
            {
                "schema_version",
                "output_dir",
                "bridge_config",
                "sft_checkpoint",
                "continued_sft_checkpoint",
                "device",
                "distribution",
                "screen_steps",
                "formal_steps",
                "snapshot_steps",
                "oracle_state_dim",
                "residual_scale",
                "gamma",
                "progress_coefficient",
                "residual_coefficient",
                "nominal_anchor_coefficient",
                "latent_anchor_coefficient",
                "ppo",
                "sac",
                "seed",
            },
            "config",
        )
        distribution = _mapping(raw.get("distribution"), "distribution")
        ppo = _mapping(raw.get("ppo"), "ppo")
        sac = _mapping(raw.get("sac"), "sac")
        _only(
            distribution,
            {
                "recovery_probability",
                "perturbation_probability",
                "nominal_probability",
                "calibration_seed_count",
                "training_seed_count",
                "curve_case_count",
                "final_case_count",
                "severity_candidates",
                "minimum_accepted_per_kind",
            },
            "distribution",
        )
        _only(
            ppo,
            {
                "rollout_steps",
                "update_epochs",
                "minibatch_size",
                "actor_learning_rate",
                "value_learning_rate",
                "gae_lambda",
                "clip_coefficient",
                "entropy_coefficient",
                "max_grad_norm",
            },
            "ppo",
        )
        _only(
            sac,
            {
                "replay_capacity",
                "warmup_steps",
                "batch_size",
                "actor_learning_rate",
                "critic_learning_rate",
                "temperature_learning_rate",
                "polyak",
                "updates_per_environment_step",
                "target_entropy",
            },
            "sac",
        )
        return cls(
            config_path=Path(config_path),
            schema_version=str(raw.get("schema_version", "")),
            output_dir=Path(raw["output_dir"]),
            bridge_config=Path(raw["bridge_config"]),
            sft_checkpoint=str(raw["sft_checkpoint"]),
            continued_sft_checkpoint=str(raw["continued_sft_checkpoint"]),
            device=str(raw["device"]),
            distribution=RecoveryDistributionConfig(
                recovery_probability=float(distribution["recovery_probability"]),
                perturbation_probability=float(distribution["perturbation_probability"]),
                nominal_probability=float(distribution["nominal_probability"]),
                calibration_seed_count=int(distribution["calibration_seed_count"]),
                training_seed_count=int(distribution["training_seed_count"]),
                curve_case_count=int(distribution["curve_case_count"]),
                final_case_count=int(distribution["final_case_count"]),
                severity_candidates=tuple(
                    float(item) for item in distribution["severity_candidates"]
                ),
                minimum_accepted_per_kind=int(
                    distribution["minimum_accepted_per_kind"]
                ),
            ),
            screen_steps=int(raw["screen_steps"]),
            formal_steps=int(raw["formal_steps"]),
            snapshot_steps=tuple(int(item) for item in raw["snapshot_steps"]),
            oracle_state_dim=int(raw["oracle_state_dim"]),
            residual_scale=tuple(float(item) for item in raw["residual_scale"]),
            gamma=float(raw["gamma"]),
            progress_coefficient=float(raw["progress_coefficient"]),
            residual_coefficient=float(raw["residual_coefficient"]),
            nominal_anchor_coefficient=float(raw["nominal_anchor_coefficient"]),
            latent_anchor_coefficient=float(raw["latent_anchor_coefficient"]),
            ppo=PPOV2Config(
                rollout_steps=int(ppo["rollout_steps"]),
                update_epochs=int(ppo["update_epochs"]),
                minibatch_size=int(ppo["minibatch_size"]),
                actor_learning_rate=float(ppo["actor_learning_rate"]),
                value_learning_rate=float(ppo["value_learning_rate"]),
                gae_lambda=float(ppo["gae_lambda"]),
                clip_coefficient=float(ppo["clip_coefficient"]),
                entropy_coefficient=float(ppo["entropy_coefficient"]),
                max_grad_norm=float(ppo["max_grad_norm"]),
            ),
            sac=SACConfig(
                replay_capacity=int(sac["replay_capacity"]),
                warmup_steps=int(sac["warmup_steps"]),
                batch_size=int(sac["batch_size"]),
                actor_learning_rate=float(sac["actor_learning_rate"]),
                critic_learning_rate=float(sac["critic_learning_rate"]),
                temperature_learning_rate=float(sac["temperature_learning_rate"]),
                polyak=float(sac["polyak"]),
                updates_per_environment_step=int(
                    sac["updates_per_environment_step"]
                ),
                target_entropy=float(sac["target_entropy"]),
            ),
            seed=int(raw["seed"]),
        )


def load_recovery_rl_v2_config(path: str | Path) -> RecoveryRLV2Config:
    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return RecoveryRLV2Config.from_mapping(loaded, config_path=config_path)
