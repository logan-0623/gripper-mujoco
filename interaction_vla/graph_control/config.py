from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import ALL_CONDITIONS, ORACLE_CONDITIONS


def _require_keys(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        label = f"{section} " if section else ""
        raise ValueError(f"unknown {label}fields: " + ", ".join(sorted(unknown)))


@dataclass(frozen=True)
class CacheConfig:
    directory: Path
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("cache.batch_size must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    smoke_steps: int = 1
    formal_epochs: int = 10

    def __post_init__(self) -> None:
        if self.smoke_steps < 1:
            raise ValueError("training.smoke_steps must be positive")
        if self.formal_epochs != 10:
            raise ValueError("training.formal_epochs must be exactly 10")


@dataclass(frozen=True)
class EvaluationConfig:
    layouts: tuple[str, ...]
    object_counts: tuple[int, ...]
    cases_per_cell: int
    master_seed: int
    max_steps: int

    def __post_init__(self) -> None:
        valid_matrix = (
            self.layouts == ("normal",) and self.object_counts == (2,)
        ) or (
            self.layouts == ("normal", "crowded")
            and self.object_counts == (2, 3)
        )
        if not valid_matrix:
            raise ValueError(
                "evaluation.layouts/object_counts must be normal/2 or the full "
                "normal,crowded x 2,3 matrix"
            )
        if self.cases_per_cell < 1:
            raise ValueError("evaluation.cases_per_cell must be positive")
        if self.master_seed < 0:
            raise ValueError("evaluation.master_seed must be non-negative")
        if self.max_steps < 1:
            raise ValueError("evaluation.max_steps must be positive")

    @property
    def cells(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (layout, object_count)
            for layout in self.layouts
            for object_count in self.object_counts
        )


@dataclass(frozen=True)
class GraphControlConfig:
    config_path: Path
    bridge_config: Path
    required_recovery_report: Path
    split_manifest: Path
    graph_runs_root: Path | None
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    cache: CacheConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        if self.conditions not in {ORACLE_CONDITIONS, ALL_CONDITIONS}:
            raise ValueError(
                "conditions must be exactly the oracle pair or full Graph v2 "
                "condition matrix"
            )
        if self.conditions == ORACLE_CONDITIONS and self.graph_runs_root is not None:
            raise ValueError("oracle-only Graph control must set graph_runs_root to null")
        if self.conditions == ALL_CONDITIONS and self.graph_runs_root is None:
            raise ValueError("the full Graph v2 matrix requires graph_runs_root")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        destinations = (self.cache.directory, self.training.output_dir)
        if len(set(destinations)) != len(destinations):
            raise ValueError("cache and training output directories must differ")

    def graph_checkpoint(self, condition: str, seed: int) -> Path | None:
        if condition not in self.conditions:
            raise ValueError(f"unknown graph control condition: {condition}")
        if seed not in self.seeds:
            raise ValueError(f"seed {seed} is outside the configured seed matrix")
        if condition in ORACLE_CONDITIONS:
            return None
        initialization = (
            "random_init"
            if condition == "predicted_random_v2"
            else "reflectvlm_init"
        )
        assert self.graph_runs_root is not None
        return (
            self.graph_runs_root
            / initialization
            / "fraction_1"
            / f"seed_{seed}"
            / "checkpoint.pt"
        )


def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return dict(value)


def load_graph_control_config(path: str | Path) -> GraphControlConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    raw = _mapping(loaded, "graph control config")
    _require_keys(
        raw,
        {
            "bridge_config",
            "required_recovery_report",
            "split_manifest",
            "graph_runs_root",
            "conditions",
            "seeds",
            "cache",
            "training",
            "evaluation",
        },
        "graph control config",
    )
    required = {
        "bridge_config",
        "required_recovery_report",
        "split_manifest",
        "graph_runs_root",
        "conditions",
        "seeds",
        "cache",
        "training",
        "evaluation",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError("missing graph control fields: " + ", ".join(sorted(missing)))

    cache_raw = _mapping(raw["cache"], "cache")
    training_raw = _mapping(raw["training"], "training")
    evaluation_raw = _mapping(raw["evaluation"], "evaluation")
    _require_keys(cache_raw, {"directory", "batch_size"}, "cache")
    _require_keys(
        training_raw, {"output_dir", "smoke_steps", "formal_epochs"}, "training"
    )
    _require_keys(
        evaluation_raw,
        {"layouts", "object_counts", "cases_per_cell", "master_seed", "max_steps"},
        "evaluation",
    )

    return GraphControlConfig(
        config_path=config_path,
        bridge_config=Path(raw["bridge_config"]),
        required_recovery_report=Path(raw["required_recovery_report"]),
        split_manifest=Path(raw["split_manifest"]),
        graph_runs_root=(
            None if raw["graph_runs_root"] is None else Path(raw["graph_runs_root"])
        ),
        conditions=tuple(str(value) for value in raw["conditions"]),
        seeds=tuple(int(value) for value in raw["seeds"]),
        cache=CacheConfig(
            directory=Path(cache_raw["directory"]),
            batch_size=int(cache_raw.get("batch_size", 16)),
        ),
        training=TrainingConfig(
            output_dir=Path(training_raw["output_dir"]),
            smoke_steps=int(training_raw.get("smoke_steps", 1)),
            formal_epochs=int(training_raw.get("formal_epochs", 10)),
        ),
        evaluation=EvaluationConfig(
            layouts=tuple(str(value) for value in evaluation_raw["layouts"]),
            object_counts=tuple(int(value) for value in evaluation_raw["object_counts"]),
            cases_per_cell=int(evaluation_raw["cases_per_cell"]),
            master_seed=int(evaluation_raw["master_seed"]),
            max_steps=int(evaluation_raw["max_steps"]),
        ),
    )
