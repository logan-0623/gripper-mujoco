from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReflectDatasetConfig:
    repo_id: str
    source_split: str = "train"
    split_seed: int = 0
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    max_rows: int | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.repo_id.strip() or not self.source_split.strip():
            raise ValueError("dataset repo_id and source_split must be non-empty")
        if len(self.split_ratios) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.split_ratios
        ):
            raise ValueError("split_ratios must contain three positive finite values")
        if not math.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("split_ratios must sum to one")
        if self.max_rows is not None and self.max_rows < 3:
            raise ValueError("max_rows must be null or at least three")


@dataclass(frozen=True)
class ReflectModelConfig:
    image_size: int = 128
    max_history_tokens: int = 64
    image_embedding_dim: int = 128
    text_embedding_dim: int = 64
    graph_embedding_dim: int = 128

    def __post_init__(self) -> None:
        if self.image_size < 8 or self.max_history_tokens < 1:
            raise ValueError("image_size and max_history_tokens must be positive")
        if min(
            self.image_embedding_dim,
            self.text_embedding_dim,
            self.graph_embedding_dim,
        ) < 1:
            raise ValueError("model embedding dimensions must be positive")


@dataclass(frozen=True)
class ReflectTrainingConfig:
    output_dir: Path
    device: str = "auto"
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("training device must be auto, cpu, or mps")
        if self.batch_size < 1 or self.num_workers != 0 or self.epochs < 1:
            raise ValueError("training requires positive batch/epochs and num_workers=0")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")


@dataclass(frozen=True)
class GraphPretrainConfig:
    config_path: Path
    dataset: ReflectDatasetConfig
    model: ReflectModelConfig
    training: ReflectTrainingConfig


def load_graph_pretrain_config(path: str | Path) -> GraphPretrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("graph pretrain config must be a mapping")
    unknown = set(raw) - {"dataset", "model", "training"}
    if unknown:
        raise ValueError("unknown graph pretrain config sections: " + ", ".join(sorted(unknown)))
    dataset_raw: dict[str, Any] = dict(raw.get("dataset", {}))
    model_raw: dict[str, Any] = dict(raw.get("model", {}))
    training_raw: dict[str, Any] = dict(raw.get("training", {}))
    if "split_ratios" in dataset_raw:
        dataset_raw["split_ratios"] = tuple(float(x) for x in dataset_raw["split_ratios"])
    if dataset_raw.get("cache_dir") is not None:
        dataset_raw["cache_dir"] = Path(dataset_raw["cache_dir"])
    if "output_dir" not in training_raw:
        raise ValueError("training.output_dir is required")
    training_raw["output_dir"] = Path(training_raw["output_dir"])
    return GraphPretrainConfig(
        config_path=config_path,
        dataset=ReflectDatasetConfig(**dataset_raw),
        model=ReflectModelConfig(**model_raw),
        training=ReflectTrainingConfig(**training_raw),
    )
