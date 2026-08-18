from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import ABLATION_CONDITIONS


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ValueError(f"missing {name} fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ", ".join(sorted(unknown)))


@dataclass(frozen=True)
class AblationConfig:
    config_path: Path
    base_graph_control_config: Path
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    shuffle_seed: int
    cache_dir: Path
    training_output_dir: Path
    smoke_steps: int
    formal_epochs: int

    def __post_init__(self) -> None:
        if self.conditions != ABLATION_CONDITIONS:
            raise ValueError("conditions must be the exact progressive ablation matrix")
        if len(self.seeds) < 3:
            raise ValueError("progressive ablation requires at least three policy seeds")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("ablation seeds must be unique and non-negative")
        if self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must be non-negative")
        if self.smoke_steps < 1:
            raise ValueError("training.smoke_steps must be positive")
        if self.formal_epochs != 10:
            raise ValueError("training.formal_epochs must be exactly 10")
        if self.cache_dir == self.training_output_dir:
            raise ValueError("ablation cache and training outputs must differ")

    @property
    def smoke_output_dir(self) -> Path:
        return self.training_output_dir.parent / "smoke"


def load_ablation_config(path: str | Path) -> AblationConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    raw = _mapping(loaded, "ablation config")
    _require_exact_keys(
        raw,
        {
            "base_graph_control_config",
            "conditions",
            "seeds",
            "shuffle_seed",
            "cache",
            "training",
        },
        "ablation config",
    )
    cache = _mapping(raw["cache"], "ablation cache")
    training = _mapping(raw["training"], "ablation training")
    _require_exact_keys(cache, {"directory"}, "ablation cache")
    _require_exact_keys(
        training,
        {"output_dir", "smoke_steps", "formal_epochs"},
        "ablation training",
    )
    return AblationConfig(
        config_path=config_path,
        base_graph_control_config=Path(raw["base_graph_control_config"]),
        conditions=tuple(str(value) for value in raw["conditions"]),
        seeds=tuple(int(value) for value in raw["seeds"]),
        shuffle_seed=int(raw["shuffle_seed"]),
        cache_dir=Path(cache["directory"]),
        training_output_dir=Path(training["output_dir"]),
        smoke_steps=int(training["smoke_steps"]),
        formal_epochs=int(training["formal_epochs"]),
    )

