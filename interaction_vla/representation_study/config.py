from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


CONFIG_SCHEMA_VERSION = "interaction_representation_runtime_v1"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _only(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ", ".join(sorted(unknown)))


@dataclass(frozen=True)
class StudyDatasetConfig:
    repo_id: str
    root: Path
    split_manifest: Path
    bridge_config: Path

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("dataset.repo_id must be non-empty")


@dataclass(frozen=True)
class StudyTraceConfig:
    root: Path
    condition: str

    def __post_init__(self) -> None:
        if not self.condition.strip():
            raise ValueError("trace.condition must be non-empty")


@dataclass(frozen=True)
class StateBankBuildConfig:
    output_dir: Path
    selection_seed: int
    split_ratios: tuple[float, float, float]
    expert_per_phase: int
    policy_per_stratum: int
    replay_position_tolerance: float

    def __post_init__(self) -> None:
        ratios = np.asarray(self.split_ratios, dtype=np.float64)
        if (
            ratios.shape != (3,)
            or not np.isfinite(ratios).all()
            or np.any(ratios <= 0.0)
            or not np.isclose(ratios.sum(), 1.0)
        ):
            raise ValueError("state_bank.split_ratios must be positive and sum to one")
        if self.selection_seed < 0:
            raise ValueError("state_bank.selection_seed must be non-negative")
        if self.expert_per_phase < 1 or self.policy_per_stratum < 1:
            raise ValueError("State Bank selection counts must be positive")
        if (
            not np.isfinite(self.replay_position_tolerance)
            or self.replay_position_tolerance <= 0.0
        ):
            raise ValueError("replay_position_tolerance must be finite and positive")


@dataclass(frozen=True)
class StudyStageConfig:
    checkpoint: str
    trainable_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.checkpoint).strip():
            raise ValueError("stage checkpoint must be non-empty")
        if len(set(self.trainable_groups)) != len(self.trainable_groups):
            raise ValueError("stage trainable_groups must be unique")


@dataclass(frozen=True)
class ExtractionConfig:
    output_dir: Path
    device: str
    batch_size: int

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("extraction.device must be auto, cpu, mps, or cuda")
        if self.batch_size < 1:
            raise ValueError("extraction.batch_size must be positive")


@dataclass(frozen=True)
class ProbeConfig:
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decays: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.seed < 0:
            raise ValueError("probe epochs/batch_size must be positive and seed non-negative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("probe learning_rate must be finite and positive")
        if not self.weight_decays or any(
            not np.isfinite(value) or value < 0.0 for value in self.weight_decays
        ):
            raise ValueError("probe weight_decays must be finite and non-negative")


@dataclass(frozen=True)
class InterventionConfig:
    output_dir: Path
    partition: str
    batch_size: int
    max_states: int
    modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.partition not in {"train", "validation", "test"}:
            raise ValueError("intervention.partition must be train, validation, or test")
        if self.batch_size < 2 or self.max_states < 2:
            raise ValueError("intervention batch_size/max_states must be at least two")
        if not self.modes or set(self.modes) - {"zero", "mean", "matched_random"}:
            raise ValueError("intervention modes are incompatible")


@dataclass(frozen=True)
class SFTConfig:
    output_dir: Path
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    save_every: int
    num_workers: int
    seed: int

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.save_every) < 1:
            raise ValueError("sft steps/batch_size/save_every must be positive")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("sft num_workers/seed must be non-negative")
        for name in ("learning_rate", "grad_clip_norm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"sft.{name} must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("sft.weight_decay must be finite and non-negative")


@dataclass(frozen=True)
class ResidualRLConfig:
    output_dir: Path
    device: str
    total_steps: int
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    learning_rate: float
    representation_learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    residual_scale: tuple[float, ...]
    reward_mode: str
    progress_reward_scale: float
    max_episode_steps: int
    object_counts: tuple[int, ...]
    layouts: tuple[str, ...]
    eval_interval: int
    eval_episodes: int
    success_threshold: float
    seed: int

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("rl.device must be auto, cpu, mps, or cuda")
        positive = (
            self.total_steps,
            self.rollout_steps,
            self.update_epochs,
            self.minibatch_size,
            self.max_episode_steps,
            self.eval_interval,
            self.eval_episodes,
        )
        if min(positive) < 1 or self.seed < 0:
            raise ValueError("RL counts must be positive and seed non-negative")
        if self.total_steps % self.rollout_steps:
            raise ValueError("rl.total_steps must be divisible by rollout_steps")
        if self.rollout_steps % self.minibatch_size:
            raise ValueError("rl.rollout_steps must be divisible by minibatch_size")
        if self.eval_interval % self.rollout_steps:
            raise ValueError("rl.eval_interval must be divisible by rollout_steps")
        if len(self.residual_scale) != 7 or any(
            not np.isfinite(value) or value < 0.0 for value in self.residual_scale
        ):
            raise ValueError("rl.residual_scale must contain seven non-negative finite values")
        if self.reward_mode not in {"sparse", "progress"}:
            raise ValueError("rl.reward_mode must be sparse or progress")
        if self.reward_mode == "sparse" and self.progress_reward_scale != 0.0:
            raise ValueError("sparse RL must set progress_reward_scale to zero")
        if not self.object_counts or min(self.object_counts) < 2:
            raise ValueError("rl.object_counts must contain values >= 2")
        if not self.layouts or set(self.layouts) - {"normal", "crowded"}:
            raise ValueError("rl.layouts must contain normal/crowded values")
        unit_interval = ("gamma", "gae_lambda", "clip_coef", "success_threshold")
        for name in unit_interval:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"rl.{name} must lie within [0, 1]")
        for name in (
            "learning_rate",
            "representation_learning_rate",
            "value_coef",
            "max_grad_norm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"rl.{name} must be finite and positive")
        if not np.isfinite(self.entropy_coef) or self.entropy_coef < 0.0:
            raise ValueError("rl.entropy_coef must be finite and non-negative")


@dataclass(frozen=True)
class AnalysisConfig:
    output_dir: Path
    bootstrap_samples: int
    confidence_level: float
    seed: int

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 100 or self.seed < 0:
            raise ValueError("analysis needs >=100 bootstrap samples and a non-negative seed")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("analysis.confidence_level must lie within (0, 1)")


@dataclass(frozen=True)
class RepresentationStudyConfig:
    config_path: Path
    study_id: str
    dataset: StudyDatasetConfig
    trace: StudyTraceConfig
    state_bank: StateBankBuildConfig
    stages: Mapping[str, Mapping[str, StudyStageConfig]]
    extraction: ExtractionConfig
    probes: ProbeConfig
    interventions: InterventionConfig
    sft: SFTConfig
    rl: ResidualRLConfig
    analysis: AnalysisConfig
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError("representation study config schema_version is incompatible")
        if not self.study_id.strip():
            raise ValueError("study_id must be non-empty")
        for backend, stages in self.stages.items():
            if backend not in {"act", "smolvla", "pi0"}:
                raise ValueError(f"unsupported study backend: {backend}")
            for stage, stage_config in stages.items():
                if stage not in {"pretrained", "sft", "continued_sft", "rl_head", "rl_representation"}:
                    raise ValueError(f"unsupported study stage: {stage}")
                if not isinstance(stage_config, StudyStageConfig):
                    raise ValueError("study stages must contain StudyStageConfig values")
                if stage == "pretrained" and stage_config.trainable_groups:
                    raise ValueError("pretrained stage cannot declare trainable groups")
                if stage != "pretrained" and not stage_config.trainable_groups:
                    raise ValueError(f"{stage} must declare trainable groups")

    def stage_config(self, backend: str, stage: str) -> StudyStageConfig:
        try:
            return self.stages[backend][stage]
        except KeyError as error:
            raise ValueError(f"stage is not configured: {backend}/{stage}") from error

    def state_bank_selection_payload(self) -> dict[str, object]:
        return {
            "schema_version": "interaction_state_bank_selection_v1",
            "study_id": self.study_id,
            "dataset": {
                "repo_id": self.dataset.repo_id,
                "root": self.dataset.root.as_posix(),
                "split_manifest": self.dataset.split_manifest.as_posix(),
                "bridge_config": self.dataset.bridge_config.as_posix(),
            },
            "trace": {
                "root": self.trace.root.as_posix(),
                "condition": self.trace.condition,
            },
            "state_bank": {
                "selection_seed": self.state_bank.selection_seed,
                "split_ratios": list(self.state_bank.split_ratios),
                "expert_per_phase": self.state_bank.expert_per_phase,
                "policy_per_stratum": self.state_bank.policy_per_stratum,
                "replay_position_tolerance": self.state_bank.replay_position_tolerance,
            },
        }

    def state_bank_selection_sha256(self) -> str:
        encoded = json.dumps(
            self.state_bank_selection_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_study_config(path: str | Path) -> RepresentationStudyConfig:
    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _mapping(loaded, "representation study config")
    _only(
        raw,
        {
            "schema_version",
            "study_id",
            "dataset",
            "trace",
            "state_bank",
            "stages",
            "extraction",
            "probes",
            "interventions",
            "sft",
            "rl",
            "analysis",
        },
        "config",
    )
    dataset = _mapping(raw.get("dataset"), "dataset")
    trace = _mapping(raw.get("trace"), "trace")
    state_bank = _mapping(raw.get("state_bank"), "state_bank")
    stages = _mapping(raw.get("stages", {}), "stages")
    extraction = _mapping(raw.get("extraction"), "extraction")
    probes = _mapping(raw.get("probes"), "probes")
    interventions = _mapping(raw.get("interventions"), "interventions")
    sft = _mapping(raw.get("sft"), "sft")
    rl = _mapping(raw.get("rl"), "rl")
    analysis = _mapping(raw.get("analysis"), "analysis")
    _only(dataset, {"repo_id", "root", "split_manifest", "bridge_config"}, "dataset")
    _only(trace, {"root", "condition"}, "trace")
    _only(
        state_bank,
        {
            "output_dir",
            "selection_seed",
            "split_ratios",
            "expert_per_phase",
            "policy_per_stratum",
            "replay_position_tolerance",
        },
        "state_bank",
    )
    _only(extraction, {"output_dir", "device", "batch_size"}, "extraction")
    _only(
        probes,
        {"output_dir", "epochs", "batch_size", "learning_rate", "weight_decays", "seed"},
        "probes",
    )
    _only(
        interventions,
        {"output_dir", "partition", "batch_size", "max_states", "modes"},
        "interventions",
    )
    _only(
        sft,
        {
            "output_dir", "steps", "batch_size", "learning_rate", "weight_decay",
            "grad_clip_norm", "save_every", "num_workers", "seed",
        },
        "sft",
    )
    _only(
        rl,
        {
            "output_dir", "device", "total_steps", "rollout_steps", "update_epochs",
            "minibatch_size", "learning_rate", "representation_learning_rate", "gamma",
            "gae_lambda", "clip_coef", "value_coef", "entropy_coef", "max_grad_norm",
            "residual_scale", "reward_mode", "progress_reward_scale", "max_episode_steps",
            "object_counts", "layouts", "eval_interval", "eval_episodes",
            "success_threshold", "seed",
        },
        "rl",
    )
    _only(analysis, {"output_dir", "bootstrap_samples", "confidence_level", "seed"}, "analysis")
    parsed_stages: dict[str, dict[str, StudyStageConfig]] = {}
    for backend, backend_value in stages.items():
        backend_stages = _mapping(backend_value, f"stages.{backend}")
        parsed_stages[str(backend)] = {}
        for stage, stage_value in backend_stages.items():
            stage_raw = _mapping(stage_value, f"stages.{backend}.{stage}")
            _only(stage_raw, {"checkpoint", "trainable_groups"}, f"stages.{backend}.{stage}")
            parsed_stages[str(backend)][str(stage)] = StudyStageConfig(
                checkpoint=str(stage_raw["checkpoint"]),
                trainable_groups=tuple(str(value) for value in stage_raw.get("trainable_groups", [])),
            )
    return RepresentationStudyConfig(
        config_path=config_path,
        schema_version=str(raw.get("schema_version", "")),
        study_id=str(raw.get("study_id", "")),
        dataset=StudyDatasetConfig(
            repo_id=str(dataset["repo_id"]),
            root=Path(dataset["root"]),
            split_manifest=Path(dataset["split_manifest"]),
            bridge_config=Path(dataset["bridge_config"]),
        ),
        trace=StudyTraceConfig(
            root=Path(trace["root"]), condition=str(trace["condition"])
        ),
        state_bank=StateBankBuildConfig(
            output_dir=Path(state_bank["output_dir"]),
            selection_seed=int(state_bank["selection_seed"]),
            split_ratios=tuple(float(value) for value in state_bank["split_ratios"]),
            expert_per_phase=int(state_bank["expert_per_phase"]),
            policy_per_stratum=int(state_bank["policy_per_stratum"]),
            replay_position_tolerance=float(state_bank["replay_position_tolerance"]),
        ),
        stages=parsed_stages,
        extraction=ExtractionConfig(
            output_dir=Path(extraction["output_dir"]),
            device=str(extraction["device"]),
            batch_size=int(extraction["batch_size"]),
        ),
        probes=ProbeConfig(
            output_dir=Path(probes["output_dir"]),
            epochs=int(probes["epochs"]),
            batch_size=int(probes["batch_size"]),
            learning_rate=float(probes["learning_rate"]),
            weight_decays=tuple(float(value) for value in probes["weight_decays"]),
            seed=int(probes["seed"]),
        ),
        interventions=InterventionConfig(
            output_dir=Path(interventions["output_dir"]),
            partition=str(interventions["partition"]),
            batch_size=int(interventions["batch_size"]),
            max_states=int(interventions["max_states"]),
            modes=tuple(str(value) for value in interventions["modes"]),
        ),
        sft=SFTConfig(
            output_dir=Path(sft["output_dir"]),
            steps=int(sft["steps"]),
            batch_size=int(sft["batch_size"]),
            learning_rate=float(sft["learning_rate"]),
            weight_decay=float(sft["weight_decay"]),
            grad_clip_norm=float(sft["grad_clip_norm"]),
            save_every=int(sft["save_every"]),
            num_workers=int(sft["num_workers"]),
            seed=int(sft["seed"]),
        ),
        rl=ResidualRLConfig(
            output_dir=Path(rl["output_dir"]),
            device=str(rl["device"]),
            total_steps=int(rl["total_steps"]),
            rollout_steps=int(rl["rollout_steps"]),
            update_epochs=int(rl["update_epochs"]),
            minibatch_size=int(rl["minibatch_size"]),
            learning_rate=float(rl["learning_rate"]),
            representation_learning_rate=float(rl["representation_learning_rate"]),
            gamma=float(rl["gamma"]),
            gae_lambda=float(rl["gae_lambda"]),
            clip_coef=float(rl["clip_coef"]),
            value_coef=float(rl["value_coef"]),
            entropy_coef=float(rl["entropy_coef"]),
            max_grad_norm=float(rl["max_grad_norm"]),
            residual_scale=tuple(float(value) for value in rl["residual_scale"]),
            reward_mode=str(rl["reward_mode"]),
            progress_reward_scale=float(rl["progress_reward_scale"]),
            max_episode_steps=int(rl["max_episode_steps"]),
            object_counts=tuple(int(value) for value in rl["object_counts"]),
            layouts=tuple(str(value) for value in rl["layouts"]),
            eval_interval=int(rl["eval_interval"]),
            eval_episodes=int(rl["eval_episodes"]),
            success_threshold=float(rl["success_threshold"]),
            seed=int(rl["seed"]),
        ),
        analysis=AnalysisConfig(
            output_dir=Path(analysis["output_dir"]),
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            confidence_level=float(analysis["confidence_level"]),
            seed=int(analysis["seed"]),
        ),
    )
