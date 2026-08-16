from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from interaction_vla.device import DEVICE_REQUESTS


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    root: Path
    reflect_checkpoint: Path
    split_seed: int
    split_ratios: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("dataset repo_id must be non-empty")
        values = tuple(float(value) for value in self.split_ratios)
        if (
            len(values) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in values)
            or not math.isclose(sum(values), 1.0)
        ):
            raise ValueError("split_ratios must be three positive values summing to one")
        object.__setattr__(self, "split_ratios", values)


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 128
    max_language_tokens: int = 32
    image_embedding_dim: int = 128
    text_embedding_dim: int = 64
    graph_embedding_dim: int = 128

    def __post_init__(self) -> None:
        if self.image_size < 8 or self.max_language_tokens < 1:
            raise ValueError("image size and language token count must be positive")
        if min(
            self.image_embedding_dim,
            self.text_embedding_dim,
            self.graph_embedding_dim,
        ) < 1:
            raise ValueError("model embedding dimensions must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    required_oracle_report: Path | None = None
    device: str = "auto"
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    fractions: tuple[float, ...] = (1.0,)
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.device not in DEVICE_REQUESTS:
            raise ValueError("training device must be auto, cpu, mps, or cuda")
        if self.batch_size < 1 or self.num_workers != 0 or self.epochs < 1:
            raise ValueError("training requires positive batch/epochs and num_workers=0")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        fractions = tuple(float(value) for value in self.fractions)
        if (
            not fractions
            or len(set(fractions)) != len(fractions)
            or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions)
        ):
            raise ValueError("fractions must be unique values within (0, 1]")
        seeds = tuple(int(value) for value in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
            raise ValueError("seeds must be unique non-negative integers")
        object.__setattr__(self, "fractions", fractions)
        object.__setattr__(self, "seeds", seeds)


@dataclass(frozen=True)
class GraphFinetuneConfig:
    config_path: Path
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


def load_graph_finetune_config(path: str | Path) -> GraphFinetuneConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("graph fine-tune config must be a mapping")
    unknown = set(raw) - {"dataset", "model", "training"}
    if unknown:
        raise ValueError("unknown graph fine-tune sections: " + ", ".join(sorted(unknown)))
    dataset_raw: dict[str, Any] = dict(raw.get("dataset", {}))
    model_raw: dict[str, Any] = dict(raw.get("model", {}))
    training_raw: dict[str, Any] = dict(raw.get("training", {}))
    for name in ("root", "reflect_checkpoint"):
        if name not in dataset_raw:
            raise ValueError(f"dataset.{name} is required")
        dataset_raw[name] = Path(dataset_raw[name])
    if "split_ratios" in dataset_raw:
        dataset_raw["split_ratios"] = tuple(dataset_raw["split_ratios"])
    if "output_dir" not in training_raw:
        raise ValueError("training.output_dir is required")
    training_raw["output_dir"] = Path(training_raw["output_dir"])
    if training_raw.get("required_oracle_report") is not None:
        training_raw["required_oracle_report"] = Path(
            training_raw["required_oracle_report"]
        )
    if "fractions" in training_raw:
        training_raw["fractions"] = tuple(training_raw["fractions"])
    if "seeds" in training_raw:
        training_raw["seeds"] = tuple(training_raw["seeds"])
    return GraphFinetuneConfig(
        config_path=config_path,
        dataset=DatasetConfig(**dataset_raw),
        model=ModelConfig(**model_raw),
        training=TrainingConfig(**training_raw),
    )
