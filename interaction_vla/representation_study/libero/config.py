from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


LIBERO_CONFIG_SCHEMA = "libero_interaction_representation_v1"
SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FORMAL_SUITES = ("libero_spatial", "libero_object")
TAP_NAMES = (
    "vision_output",
    "multimodal_fusion",
    "action_expert_input",
    "pre_action",
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _relative_path(value: object, name: str) -> Path:
    raw = "" if value is None else str(value).strip()
    path = Path(raw)
    if not raw or path == Path("."):
        raise ValueError(f"{name} must be a non-empty repository-relative path")
    if path.is_absolute():
        raise ValueError(f"{name} must be repository-relative")
    if ".." in path.parts:
        raise ValueError(f"{name} must not escape the repository")
    return path


def _positive(value: object, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True)
class SourceConfig:
    lerobot_repo_id: str
    lerobot_revision: str
    raw_hdf5_root: Path
    lerobot_root: Path | None = None


@dataclass(frozen=True)
class CoverageConfig:
    suites: tuple[str, ...]
    fail_on_unsupported_task: bool
    tasks_per_suite: int | None


@dataclass(frozen=True)
class ReplayConfig:
    control_freq: int
    state_l2_p95_tolerance: float
    state_max_abs_tolerance: float
    minimum_acceptance_rate: float
    action_atol: float


@dataclass(frozen=True)
class AnnotationConfig:
    stable_window_frames: int
    relative_translation_drift_m: float
    relative_rotation_drift_deg: float
    minimum_comotion_m: float
    lift_clearance_m: float
    approach_surface_distance_m: float
    hysteresis_m: float
    minimum_finger_groups: int


@dataclass(frozen=True)
class StateBankConfig:
    states_per_episode: int | None
    holdout_episodes_per_task: int
    timeline_count: int


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    task_ratios: tuple[float, float, float]
    episode_ratios: tuple[float, float, float]


@dataclass(frozen=True)
class StageConfig:
    base_model: str
    base_revision: str
    fractions: tuple[float, ...]
    epochs: int
    seed: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class TapConfig:
    names: tuple[str, ...]
    pooling: str
    denoising_call: str


@dataclass(frozen=True)
class ProbeConfig:
    bootstrap_samples: int
    confidence_level: float
    linear_l2: tuple[float, ...]
    mlp_hidden_dim: int
    linear_epochs: int
    mlp_epochs: int


@dataclass(frozen=True)
class LiberoStudyConfig:
    schema_version: str
    seed: int
    output_dir: Path
    sources: SourceConfig
    coverage: CoverageConfig
    replay: ReplayConfig
    annotations: AnnotationConfig
    state_bank: StateBankConfig
    splits: SplitConfig
    stages: StageConfig
    taps: TapConfig
    probes: ProbeConfig
    source_path: Path


def _ratios(value: object, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive values")
    if abs(sum(result) - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to one")
    return result  # type: ignore[return-value]


def load_libero_study_config(path: str | Path) -> LiberoStudyConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    schema = str(root.get("schema_version", ""))
    if schema != LIBERO_CONFIG_SCHEMA:
        raise ValueError(f"schema_version must be {LIBERO_CONFIG_SCHEMA}")

    sources = _mapping(root.get("sources"), "sources")
    coverage = _mapping(root.get("coverage"), "coverage")
    replay = _mapping(root.get("replay"), "replay")
    annotations = _mapping(root.get("annotations"), "annotations")
    bank = _mapping(root.get("state_bank"), "state_bank")
    splits = _mapping(root.get("splits"), "splits")
    stages = _mapping(root.get("stages"), "stages")
    taps = _mapping(root.get("taps"), "taps")
    probes = _mapping(root.get("probes"), "probes")

    suites = tuple(str(item) for item in coverage.get("suites", ()))
    unknown_suites = sorted(set(suites).difference(SUPPORTED_SUITES))
    if not suites or unknown_suites:
        raise ValueError(f"coverage.suites contains unsupported values: {unknown_suites}")
    unreviewed_suites = sorted(set(suites).difference(FORMAL_SUITES))
    if unreviewed_suites:
        raise ValueError(
            f"coverage.suites task semantics are not reviewed: {unreviewed_suites}"
        )
    if coverage.get("fail_on_unsupported_task") is not True:
        raise ValueError("coverage.fail_on_unsupported_task must be true")

    fractions = tuple(float(item) for item in stages.get("fractions", (0.25, 0.5, 1.0)))
    if (
        not fractions
        or any(item <= 0 or item > 1 for item in fractions)
        or any(left >= right for left, right in zip(fractions, fractions[1:], strict=False))
        or fractions[-1] != 1.0
    ):
        raise ValueError("stages.fractions must be strictly increasing in (0, 1] and end at 1")
    if fractions != (0.25, 0.5, 1.0):
        raise ValueError("stages.fractions must be the preregistered nested values 0.25/0.50/1.00")

    tap_names = tuple(str(item) for item in taps.get("names", TAP_NAMES))
    if tap_names != TAP_NAMES:
        raise ValueError(f"taps.names must be the preregistered order {TAP_NAMES}")
    pooling = str(taps.get("pooling", "valid_token_mean"))
    if pooling != "valid_token_mean":
        raise ValueError("taps.pooling primary must be valid_token_mean")

    seed = int(root.get("seed", 2057736129))
    if seed < 0:
        raise ValueError("seed must be non-negative")
    stage_seed = int(stages.get("seed", seed))
    if stage_seed < 0:
        raise ValueError("stages.seed must be non-negative")

    states_per_episode = bank.get("states_per_episode")
    tasks_per_suite = coverage.get("tasks_per_suite")
    lerobot_root_value = sources.get("lerobot_root")
    result = LiberoStudyConfig(
        schema_version=schema,
        seed=seed,
        output_dir=_relative_path(root.get("output_dir", ""), "output_dir"),
        sources=SourceConfig(
            lerobot_repo_id=str(sources.get("lerobot_repo_id", "")),
            lerobot_revision=str(sources.get("lerobot_revision", "main")),
            raw_hdf5_root=_relative_path(
                sources.get("raw_hdf5_root", ""), "sources.raw_hdf5_root"
            ),
            lerobot_root=(
                _relative_path(lerobot_root_value, "sources.lerobot_root")
                if lerobot_root_value is not None
                else None
            ),
        ),
        coverage=CoverageConfig(
            suites=suites,
            fail_on_unsupported_task=True,
            tasks_per_suite=(
                None if tasks_per_suite is None else int(tasks_per_suite)
            ),
        ),
        replay=ReplayConfig(
            control_freq=int(replay.get("control_freq", 20)),
            state_l2_p95_tolerance=_positive(
                replay.get("state_l2_p95_tolerance", 0.01),
                "replay.state_l2_p95_tolerance",
            ),
            state_max_abs_tolerance=_positive(
                replay.get("state_max_abs_tolerance", 0.05),
                "replay.state_max_abs_tolerance",
            ),
            minimum_acceptance_rate=float(replay.get("minimum_acceptance_rate", 0.95)),
            action_atol=_positive(replay.get("action_atol", 1e-5), "replay.action_atol"),
        ),
        annotations=AnnotationConfig(
            stable_window_frames=int(annotations.get("stable_window_frames", 5)),
            relative_translation_drift_m=_positive(
                annotations.get("relative_translation_drift_m", 0.015),
                "annotations.relative_translation_drift_m",
            ),
            relative_rotation_drift_deg=_positive(
                annotations.get("relative_rotation_drift_deg", 15.0),
                "annotations.relative_rotation_drift_deg",
            ),
            minimum_comotion_m=_positive(
                annotations.get("minimum_comotion_m", 0.005),
                "annotations.minimum_comotion_m",
            ),
            lift_clearance_m=_positive(
                annotations.get("lift_clearance_m", 0.01),
                "annotations.lift_clearance_m",
            ),
            approach_surface_distance_m=_positive(
                annotations.get("approach_surface_distance_m", 0.05),
                "annotations.approach_surface_distance_m",
            ),
            hysteresis_m=_positive(
                annotations.get("hysteresis_m", 0.01), "annotations.hysteresis_m"
            ),
            minimum_finger_groups=int(annotations.get("minimum_finger_groups", 2)),
        ),
        state_bank=StateBankConfig(
            states_per_episode=(
                int(states_per_episode) if states_per_episode is not None else None
            ),
            holdout_episodes_per_task=int(bank.get("holdout_episodes_per_task", 5)),
            timeline_count=int(bank.get("timeline_count", 8)),
        ),
        splits=SplitConfig(
            seed=int(splits.get("seed", seed)),
            task_ratios=_ratios(
                splits.get("task_ratios", (0.7, 0.15, 0.15)), "splits.task_ratios"
            ),
            episode_ratios=_ratios(
                splits.get("episode_ratios", (0.7, 0.15, 0.15)),
                "splits.episode_ratios",
            ),
        ),
        stages=StageConfig(
            base_model=str(stages.get("base_model", "")),
            base_revision=str(stages.get("base_revision", "main")),
            fractions=fractions,
            epochs=int(stages.get("epochs", 20)),
            seed=stage_seed,
            batch_size=int(stages.get("batch_size", 8)),
            num_workers=int(stages.get("num_workers", 4)),
        ),
        taps=TapConfig(
            names=tap_names,
            pooling=pooling,
            denoising_call=str(taps.get("denoising_call", "final")),
        ),
        probes=ProbeConfig(
            bootstrap_samples=int(probes.get("bootstrap_samples", 2000)),
            confidence_level=float(probes.get("confidence_level", 0.95)),
            linear_l2=tuple(float(item) for item in probes.get("linear_l2", (0.0, 1e-4, 1e-3))),
            mlp_hidden_dim=int(probes.get("mlp_hidden_dim", 256)),
            linear_epochs=int(probes.get("linear_epochs", 150)),
            mlp_epochs=int(probes.get("mlp_epochs", 75)),
        ),
        source_path=source,
    )
    if not result.sources.lerobot_repo_id or not result.stages.base_model:
        raise ValueError("sources.lerobot_repo_id and stages.base_model are required")
    if not 0 < result.replay.minimum_acceptance_rate <= 1:
        raise ValueError("replay.minimum_acceptance_rate must be in (0, 1]")
    if result.replay.control_freq <= 0:
        raise ValueError("replay.control_freq must be positive")
    if result.annotations.stable_window_frames < 2:
        raise ValueError("annotations.stable_window_frames must be at least two")
    if result.annotations.minimum_finger_groups < 1:
        raise ValueError("annotations.minimum_finger_groups must be positive")
    if result.coverage.tasks_per_suite is not None and result.coverage.tasks_per_suite < 3:
        raise ValueError("coverage.tasks_per_suite must be at least three when set")
    if (
        result.state_bank.states_per_episode is not None
        and result.state_bank.states_per_episode <= 0
    ):
        raise ValueError("state_bank.states_per_episode must be positive when set")
    if result.state_bank.holdout_episodes_per_task < 3:
        raise ValueError(
            "state_bank.holdout_episodes_per_task must be at least three for episode-group splitting"
        )
    if result.state_bank.timeline_count <= 0:
        raise ValueError("state_bank.timeline_count must be positive")
    if result.splits.seed < 0:
        raise ValueError("splits.seed must be non-negative")
    if result.stages.epochs <= 0:
        raise ValueError("stages.epochs must be positive")
    if result.stages.batch_size <= 0 or result.stages.num_workers < 0:
        raise ValueError("stages batch_size must be positive and num_workers non-negative")
    if result.probes.linear_epochs <= 0 or result.probes.mlp_epochs <= 0:
        raise ValueError("probe epoch budgets must be positive")
    if result.probes.bootstrap_samples <= 0:
        raise ValueError("probes.bootstrap_samples must be positive")
    if not 0.0 < result.probes.confidence_level < 1.0:
        raise ValueError("probes.confidence_level must be in (0, 1)")
    if not result.probes.linear_l2 or any(value < 0 for value in result.probes.linear_l2):
        raise ValueError("probes.linear_l2 must contain non-negative values")
    if result.probes.mlp_hidden_dim <= 0:
        raise ValueError("probes.mlp_hidden_dim must be positive")
    if result.taps.denoising_call != "final":
        raise ValueError("taps.denoising_call must be the preregistered value final")
    return result
