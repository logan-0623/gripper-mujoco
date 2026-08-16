from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .device import DEVICE_REQUESTS


def _as_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _validate_positive_range(name: str, values: tuple[float, float]) -> None:
    if (
        len(values) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in values)
        or values[0] > values[1]
    ):
        raise ValueError(f"{name} must be a finite positive (low, high) range")


@dataclass(frozen=True)
class TrainConfig:
    object_counts: tuple[int, ...] = (2, 3)
    episodes: int = 50
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    model_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not self.object_counts or min(self.object_counts) < 2:
            raise ValueError("train.object_counts must contain values >= 2")
        if self.episodes < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("training sizes must be positive")


@dataclass(frozen=True)
class EvalConfig:
    object_counts: tuple[int, ...] = (2, 3, 4, 5)
    ood_object_counts: tuple[int, ...] = (4, 5)
    crowded_object_counts: tuple[int, ...] = (4, 5)
    episodes_per_count: int = 20
    max_steps: int = 120

    def __post_init__(self) -> None:
        if not set(self.ood_object_counts).issubset(self.object_counts):
            raise ValueError("eval.ood_object_counts must be included in eval.object_counts")
        if not set(self.crowded_object_counts).issubset(self.object_counts):
            raise ValueError(
                "eval.crowded_object_counts must be included in eval.object_counts"
            )
        if self.episodes_per_count < 1 or self.max_steps < 1:
            raise ValueError("evaluation sizes must be positive")


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    message_rounds: int = 2
    action_dim: int = 4

    def __post_init__(self) -> None:
        if min(self.embedding_dim, self.hidden_dim, self.message_rounds, self.action_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if self.action_dim not in {4, 7}:
            raise ValueError("model.action_dim must be either 4 or 7")


@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int = 120
    workspace_low: tuple[float, float, float] = (-0.45, -0.35, 0.02)
    workspace_high: tuple[float, float, float] = (0.45, 0.35, 0.55)
    min_object_distance: float = 0.12
    crowded_anchor_min_distance: float = 0.085
    crowded_anchor_max_distance: float = 0.105

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.min_object_distance <= 0:
            raise ValueError("environment limits must be positive")
        if not (
            0.0 < self.crowded_anchor_min_distance
            < self.crowded_anchor_max_distance
            < self.min_object_distance
        ):
            raise ValueError(
                "environment crowded anchor distances must satisfy "
                "0 < min < max < min_object_distance"
            )
        if (
            len(self.workspace_low) != 3
            or len(self.workspace_high) != 3
            or not all(math.isfinite(value) for value in self.workspace_low + self.workspace_high)
            or not all(low < high for low, high in zip(self.workspace_low, self.workspace_high, strict=True))
        ):
            raise ValueError("environment workspace must be finite 3D bounds with low < high")


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    variants_per_episode: int = 0
    min_acceptance_rate: float = 0.0
    training_source_fraction: float = 1.0
    benchmark_enabled: bool = False

    def __post_init__(self) -> None:
        if self.variants_per_episode < 0:
            raise ValueError("recovery.variants_per_episode must not be negative")
        if self.enabled and self.variants_per_episode < 1:
            raise ValueError(
                "recovery.variants_per_episode must be positive when recovery is enabled"
            )
        if (
            not math.isfinite(self.min_acceptance_rate)
            or not 0.0 <= self.min_acceptance_rate <= 1.0
        ):
            raise ValueError(
                "recovery.min_acceptance_rate must be finite and within [0, 1]"
            )
        if (
            not math.isfinite(self.training_source_fraction)
            or not 0.0 < self.training_source_fraction <= 1.0
        ):
            raise ValueError(
                "recovery.training_source_fraction must be finite and within (0, 1]"
            )


@dataclass(frozen=True)
class SequenceConfig:
    enabled: bool = False
    horizon: int = 1
    future_loss_decay: float = 0.9
    temporal_decay: float = 0.25
    gripper_close_threshold: float = 0.35
    gripper_open_threshold: float = 0.65
    recovery_loss_fraction: float = 0.0
    rollout_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("sequence.horizon must be positive")
        if (
            not math.isfinite(self.future_loss_decay)
            or not 0.0 < self.future_loss_decay <= 1.0
        ):
            raise ValueError(
                "sequence.future_loss_decay must be finite and within (0, 1]"
            )
        if not math.isfinite(self.temporal_decay) or self.temporal_decay < 0.0:
            raise ValueError(
                "sequence.temporal_decay must be finite and non-negative"
            )
        if not (
            0.0
            <= self.gripper_close_threshold
            < self.gripper_open_threshold
            <= 1.0
        ):
            raise ValueError(
                "sequence gripper thresholds must satisfy 0 <= close < open <= 1"
            )
        if (
            not math.isfinite(self.recovery_loss_fraction)
            or not 0.0 <= self.recovery_loss_fraction < 1.0
        ):
            raise ValueError(
                "sequence.recovery_loss_fraction must be finite and within [0, 1)"
            )
        if self.rollout_device not in {"cpu", "mps"}:
            raise ValueError("sequence.rollout_device must be cpu or mps")
        if self.enabled and self.rollout_device != "cpu":
            raise ValueError(
                "enabled sequence experiments require sequence.rollout_device=cpu"
            )


@dataclass(frozen=True)
class RandomizationConfig:
    enabled: bool = False
    object_mass_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    joint_damping_scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        _validate_positive_range("physics.randomization.object_mass_scale", self.object_mass_scale)
        _validate_positive_range("physics.randomization.friction_scale", self.friction_scale)
        _validate_positive_range(
            "physics.randomization.joint_damping_scale", self.joint_damping_scale
        )


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.002
    policy_hz: int = 20
    substeps: int = 25
    translation_delta: float = 0.02
    rotation_delta: float = math.radians(3.0)
    settle_steps: int = 250
    stable_grasp_frames: int = 10
    stable_lift_height: float = 0.01
    ik_damping: float = 0.05
    ik_iterations: int = 20
    ik_position_tolerance: float = 0.002
    ik_orientation_tolerance: float = math.radians(2.0)
    expert_gate_cases_per_condition: int = 20
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    def __post_init__(self) -> None:
        positive_floats = (
            self.timestep,
            self.translation_delta,
            self.rotation_delta,
            self.stable_lift_height,
            self.ik_damping,
            self.ik_position_tolerance,
            self.ik_orientation_tolerance,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("physics scalar limits must be finite and positive")
        positive_ints = (
            self.policy_hz,
            self.substeps,
            self.settle_steps,
            self.stable_grasp_frames,
            self.ik_iterations,
            self.expert_gate_cases_per_condition,
        )
        if min(positive_ints) < 1:
            raise ValueError("physics counts and frequencies must be positive")
        if not math.isclose(
            self.timestep * self.policy_hz * self.substeps,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "physics frequency must satisfy timestep * policy_hz * substeps == 1"
            )


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False
    width: int = 256
    height: int = 256
    cameras: tuple[str, ...] = (
        "agentview",
        "wristview",
        "sideview",
        "topview",
    )

    def __post_init__(self) -> None:
        expected = ("agentview", "wristview", "sideview", "topview")
        if self.width < 1 or self.height < 1:
            raise ValueError("recording width and height must be positive")
        if self.cameras != expected:
            raise ValueError(f"recording cameras must be exactly {expected}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "interaction_graph_pilot"
    seed: int = 42
    device: str = "auto"
    max_objects: int = 5
    data_dir: str = "outputs/interaction_vla/data"
    output_dir: str = "outputs/interaction_vla"
    backend: str = "kinematic"
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def __post_init__(self) -> None:
        requested_counts = self.train.object_counts + self.eval.object_counts
        if self.max_objects < 2 or self.max_objects < max(requested_counts):
            raise ValueError("max_objects must cover every configured object count and be >= 2")
        if set(self.train.object_counts) & set(self.eval.ood_object_counts):
            raise ValueError("OOD object counts must not overlap training object counts")
        if set(self.train.object_counts) & set(self.eval.crowded_object_counts):
            raise ValueError("crowded object counts must not overlap training object counts")
        if self.device not in DEVICE_REQUESTS:
            raise ValueError("device must be one of: auto, cpu, mps, cuda")
        if self.backend not in {"kinematic", "franka_contact"}:
            raise ValueError("backend must be one of: kinematic, franka_contact")
        expected_action_dim = 4 if self.backend == "kinematic" else 7
        if self.model.action_dim != expected_action_dim:
            raise ValueError(
                f"backend {self.backend} requires model.action_dim={expected_action_dim}"
            )
        if self.backend == "franka_contact" and self.max_objects > 5:
            raise ValueError("franka_contact backend supports at most 5 objects")
        if self.sequence.enabled and self.model.action_dim != 7:
            raise ValueError("sequence control currently requires physical 7D actions")
        if self.sequence.enabled and not self.recovery.enabled:
            raise ValueError("sequence v3 requires configured recovery augmentation")
        if self.sequence.enabled and self.sequence.recovery_loss_fraction == 0.0:
            raise ValueError("sequence v3 requires a positive recovery loss fraction")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    train_raw = dict(raw.pop("train", {}))
    eval_raw = dict(raw.pop("eval", {}))
    model_raw = dict(raw.pop("model", {}))
    environment_raw = dict(raw.pop("environment", {}))
    recovery_raw = dict(raw.pop("recovery", {}))
    sequence_raw = dict(raw.pop("sequence", {}))
    physics_raw = dict(raw.pop("physics", {}))
    randomization_raw = dict(physics_raw.pop("randomization", {}))
    recording_raw = dict(raw.pop("recording", {}))

    for key in ("object_counts", "model_seeds"):
        if key in train_raw:
            train_raw[key] = _as_tuple(train_raw[key])
    if "crowded_object_counts" not in eval_raw and "ood_object_counts" in eval_raw:
        eval_raw["crowded_object_counts"] = eval_raw["ood_object_counts"]
    for key in ("object_counts", "ood_object_counts", "crowded_object_counts"):
        if key in eval_raw:
            eval_raw[key] = _as_tuple(eval_raw[key])
    for key in ("workspace_low", "workspace_high"):
        if key in environment_raw:
            environment_raw[key] = _float_tuple(environment_raw[key])
    for key in ("object_mass_scale", "friction_scale", "joint_damping_scale"):
        if key in randomization_raw:
            randomization_raw[key] = _float_tuple(randomization_raw[key])
    if "cameras" in recording_raw:
        recording_raw["cameras"] = tuple(str(value) for value in recording_raw["cameras"])

    return ExperimentConfig(
        **raw,
        train=TrainConfig(**train_raw),
        eval=EvalConfig(**eval_raw),
        model=ModelConfig(**model_raw),
        environment=EnvironmentConfig(**environment_raw),
        recovery=RecoveryConfig(**recovery_raw),
        sequence=SequenceConfig(**sequence_raw),
        physics=PhysicsConfig(
            **physics_raw,
            randomization=RandomizationConfig(**randomization_raw),
        ),
        recording=RecordingConfig(**recording_raw),
    )
