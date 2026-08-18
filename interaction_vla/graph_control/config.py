from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
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
class DiagnosticsConfig:
    output_dir: Path
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 2057736129
    max_lag: int = 3
    active_epsilon: float = 1.0e-6
    sensitivity_rows_per_episode: int = 4
    sensitivity_batch_size: int = 4
    sensitivity_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 1:
            raise ValueError("diagnostics.bootstrap_samples must be positive")
        if self.bootstrap_seed < 0:
            raise ValueError("diagnostics.bootstrap_seed must be non-negative")
        if self.max_lag < 0:
            raise ValueError("diagnostics.max_lag must be non-negative")
        if not np.isfinite(self.active_epsilon) or self.active_epsilon <= 0.0:
            raise ValueError(
                "diagnostics.active_epsilon must be finite and positive"
            )
        if self.sensitivity_rows_per_episode < 1:
            raise ValueError(
                "diagnostics.sensitivity_rows_per_episode must be positive"
            )
        if self.sensitivity_batch_size < 1:
            raise ValueError("diagnostics.sensitivity_batch_size must be positive")
        if (
            not np.isfinite(self.sensitivity_scale)
            or not 0.0 < self.sensitivity_scale <= 1.0
        ):
            raise ValueError(
                "diagnostics.sensitivity_scale must lie within (0, 1]"
            )


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool
    output_dir: Path
    resume: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("trace.enabled must be boolean")
        if not isinstance(self.resume, bool):
            raise ValueError("trace.resume must be boolean")


@dataclass(frozen=True)
class GraphControlConfig:
    config_path: Path
    bridge_config: Path
    required_recovery_report: Path
    required_oracle_report: Path | None
    split_manifest: Path
    graph_runs_root: Path | None
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    cache: CacheConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    diagnostics: DiagnosticsConfig | None = None
    trace: TraceConfig | None = None

    def __post_init__(self) -> None:
        if self.conditions not in {ORACLE_CONDITIONS, ALL_CONDITIONS}:
            raise ValueError(
                "conditions must be exactly the oracle pair or full Graph v2 "
                "condition matrix"
            )
        if self.conditions == ORACLE_CONDITIONS and self.graph_runs_root is not None:
            raise ValueError("oracle-only Graph control must set graph_runs_root to null")
        if (
            self.conditions == ORACLE_CONDITIONS
            and self.required_oracle_report is not None
        ):
            raise ValueError(
                "oracle-only Graph control required_oracle_report must be null"
            )
        if self.conditions == ALL_CONDITIONS and self.graph_runs_root is None:
            raise ValueError("the full Graph v2 matrix requires graph_runs_root")
        if self.conditions == ALL_CONDITIONS and self.required_oracle_report is None:
            raise ValueError("the full Graph v2 matrix requires required_oracle_report")
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
            "required_oracle_report",
            "split_manifest",
            "graph_runs_root",
            "conditions",
            "seeds",
            "cache",
            "training",
            "evaluation",
            "diagnostics",
            "trace",
        },
        "graph control config",
    )
    required = {
        "bridge_config",
        "required_recovery_report",
        "required_oracle_report",
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
    diagnostics_raw = (
        None
        if raw.get("diagnostics") is None
        else _mapping(raw["diagnostics"], "diagnostics")
    )
    trace_raw = (
        None if raw.get("trace") is None else _mapping(raw["trace"], "trace")
    )
    _require_keys(cache_raw, {"directory", "batch_size"}, "cache")
    _require_keys(
        training_raw, {"output_dir", "smoke_steps", "formal_epochs"}, "training"
    )
    _require_keys(
        evaluation_raw,
        {"layouts", "object_counts", "cases_per_cell", "master_seed", "max_steps"},
        "evaluation",
    )
    if diagnostics_raw is not None:
        _require_keys(
            diagnostics_raw,
            {
                "output_dir",
                "bootstrap_samples",
                "bootstrap_seed",
                "max_lag",
                "active_epsilon",
                "sensitivity_rows_per_episode",
                "sensitivity_batch_size",
                "sensitivity_scale",
            },
            "diagnostics",
        )
        if "output_dir" not in diagnostics_raw:
            raise ValueError("missing diagnostics fields: output_dir")
    if trace_raw is not None:
        _require_keys(trace_raw, {"enabled", "output_dir", "resume"}, "trace")
        missing_trace = {"enabled", "output_dir"} - set(trace_raw)
        if missing_trace:
            raise ValueError(
                "missing trace fields: " + ", ".join(sorted(missing_trace))
            )

    return GraphControlConfig(
        config_path=config_path,
        bridge_config=Path(raw["bridge_config"]),
        required_recovery_report=Path(raw["required_recovery_report"]),
        required_oracle_report=(
            None
            if raw["required_oracle_report"] is None
            else Path(raw["required_oracle_report"])
        ),
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
        diagnostics=(
            None
            if diagnostics_raw is None
            else DiagnosticsConfig(
                output_dir=Path(diagnostics_raw["output_dir"]),
                bootstrap_samples=int(
                    diagnostics_raw.get("bootstrap_samples", 2000)
                ),
                bootstrap_seed=int(
                    diagnostics_raw.get("bootstrap_seed", 2057736129)
                ),
                max_lag=int(diagnostics_raw.get("max_lag", 3)),
                active_epsilon=float(
                    diagnostics_raw.get("active_epsilon", 1.0e-6)
                ),
                sensitivity_rows_per_episode=int(
                    diagnostics_raw.get("sensitivity_rows_per_episode", 4)
                ),
                sensitivity_batch_size=int(
                    diagnostics_raw.get("sensitivity_batch_size", 4)
                ),
                sensitivity_scale=float(
                    diagnostics_raw.get("sensitivity_scale", 0.25)
                ),
            )
        ),
        trace=(
            None
            if trace_raw is None
            else TraceConfig(
                enabled=trace_raw["enabled"],
                output_dir=Path(trace_raw["output_dir"]),
                resume=trace_raw.get("resume", True),
            )
        ),
    )
