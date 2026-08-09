from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from interaction_vla.config import ExperimentConfig, load_config


@dataclass(frozen=True)
class DatasetBridgeConfig:
    repo_id: str
    root: Path
    episodes: int
    object_counts: tuple[int, ...]
    task: str
    fps: int = 20
    image_size: tuple[int, int] = (256, 256)
    state_dim: int = 10
    action_dim: int = 7
    max_attempt_multiplier: int = 10

    def __post_init__(self) -> None:
        if not self.repo_id or not self.task.strip():
            raise ValueError("dataset repo_id and task must be non-empty")
        if self.episodes < 1 or self.max_attempt_multiplier < 1:
            raise ValueError("dataset episode and attempt counts must be positive")
        if not self.object_counts or min(self.object_counts) < 2:
            raise ValueError("dataset object_counts must contain values >= 2")
        if self.fps != 20 or self.image_size != (256, 256):
            raise ValueError("the first bridge requires 20 Hz and 256x256 images")
        if self.state_dim != 10 or self.action_dim != 7:
            raise ValueError("the first bridge requires state_dim=10 and action_dim=7")


@dataclass(frozen=True)
class TeacherConfig:
    distractor_count: int = 2
    replacement_margin: float = 0.10
    replacement_frames: int = 3
    dropout_frames: int = 3
    safety_margin_m: float = 0.01
    goal_horizon: int = 8
    goal_improvement_margin: float = 0.05


@dataclass(frozen=True)
class ACTBridgeConfig:
    output_dir: Path
    device: str = "auto"
    chunk_size: int = 8
    n_action_steps: int = 8
    batch_size: int = 2
    num_workers: int = 0
    steps: int | None = 500
    epochs: int | None = None
    maximum_epochs: int = 10
    learning_rate: float = 1e-5
    seed: int = 0
    dim_model: int = 256
    dim_feedforward: int = 1024
    encoder_layers: int = 2
    vae_encoder_layers: int = 2

    def __post_init__(self) -> None:
        if (self.steps is None) == (self.epochs is None):
            raise ValueError("ACT requires exactly one of steps or epochs")
        if self.chunk_size != 8 or self.n_action_steps != 8:
            raise ValueError("the first ACT bridge requires an 8-step chunk")
        if self.batch_size not in {1, 2} or self.num_workers != 0:
            raise ValueError("macOS ACT requires batch size 1/2 and num_workers=0")
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("ACT device must be auto, cpu, or mps")


@dataclass(frozen=True)
class BridgeConfig:
    config_path: Path
    source_config_path: Path
    expert_gate: Path
    dataset: DatasetBridgeConfig
    teacher: TeacherConfig
    act: ACTBridgeConfig
    seed: int
    required_smoke_report: Path | None
    source: ExperimentConfig


def _path_or_none(value: Any) -> Path | None:
    return None if value is None else Path(value)


def load_bridge_config(path: str | Path) -> BridgeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    raw = dict(loaded)
    dataset_raw = dict(raw.pop("dataset", {}))
    teacher_raw = dict(raw.pop("teacher", {}))
    act_raw = dict(raw.pop("act", {}))

    source_config_path = Path(raw.pop("source_config"))
    source = load_config(source_config_path)

    dataset_raw["root"] = Path(dataset_raw["root"])
    dataset_raw["object_counts"] = tuple(int(value) for value in dataset_raw["object_counts"])
    dataset_raw["image_size"] = tuple(int(value) for value in dataset_raw["image_size"])
    act_raw["output_dir"] = Path(act_raw["output_dir"])

    dataset = DatasetBridgeConfig(**dataset_raw)
    teacher = TeacherConfig(**teacher_raw)
    act = ACTBridgeConfig(**act_raw)

    if source.backend != "franka_contact":
        raise ValueError("LeRobot bridge source backend must be franka_contact")
    if source.physics.policy_hz != dataset.fps:
        raise ValueError("source policy_hz must equal dataset fps")
    if source.max_objects < max(dataset.object_counts):
        raise ValueError("source max_objects must cover dataset object_counts")

    return BridgeConfig(
        config_path=config_path,
        source_config_path=source_config_path,
        expert_gate=Path(raw.pop("expert_gate")),
        dataset=dataset,
        teacher=teacher,
        act=act,
        seed=int(raw.pop("seed")),
        required_smoke_report=_path_or_none(raw.pop("required_smoke_report", None)),
        source=source,
    )
