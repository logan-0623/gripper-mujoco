from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..state_bank.io import write_json_atomic
from ..statistics import clustered_bootstrap_mean
from .config import LiberoStudyConfig
from .latents import _file_sha256
from .recruitment import PRIMARY_CONDITIONS, _profile_root, phase_stratum


PRIMARY_METRIC = "first_action_l2"
ACTION_COMPONENTS = (
    "translation_l2",
    "rotation_l2",
    "gripper_abs",
    "gripper_flip",
)
CHUNK_METRIC = "chunk_l2"
PAIRED_CONTRASTS = {
    "emergence": ("pretrained", "d25_u16070"),
    "data_coverage": ("d25_u16070", "d100_u16617"),
    "optimization_time": ("d100_u16617", "d100_u66470"),
}
SCHEMA_VERSION = "libero_stablegrasp_cached_analysis_v1"


def _effect(row: Mapping[str, object], metric: str) -> float:
    effects = row.get("effects")
    if not isinstance(effects, Mapping) or not isinstance(effects.get(metric), Mapping):
        raise ValueError(f"cached action row is missing metric: {metric}")
    value = float(effects[metric]["target_minus_random"])
    if not np.isfinite(value):
        raise ValueError(f"cached action effect is non-finite: {metric}")
    return value


def _interval(
    values: Sequence[float],
    rows: Sequence[Mapping[str, object]],
    *,
    config: LiberoStudyConfig,
    cluster: str,
) -> dict[str, float]:
    return clustered_bootstrap_mean(
        values,
        [str(row[cluster]) for row in rows],
        samples=max(100, config.probes.bootstrap_samples),
        confidence=config.probes.confidence_level,
        seed=config.seed + (1 if cluster == "task" else 0),
    )


def _metric_summary(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    config: LiberoStudyConfig,
    values: Sequence[float] | None = None,
) -> dict[str, object]:
    effect = list(values) if values is not None else [_effect(row, metric) for row in rows]
    return {
        "states": len(rows),
        "episode_ci": _interval(effect, rows, config=config, cluster="episode"),
        "task_ci": _interval(effect, rows, config=config, cluster="task"),
    }


def _state_summaries(
    rows: Sequence[Mapping[str, object]], *, config: LiberoStudyConfig
) -> dict[str, object]:
    groups = {
        "stable_grasp": {
            "false": [row for row in rows if not bool(row["stable_grasp"])],
            "true": [row for row in rows if bool(row["stable_grasp"])],
        },
        "phase_stratum": {
            stratum: [row for row in rows if row["stratum"] == stratum]
            for stratum in sorted({str(row["stratum"]) for row in rows})
        },
    }
    result: dict[str, object] = {}
    for grouping, levels in groups.items():
        result[grouping] = {
            level: {
                metric: _metric_summary(subset, metric, config=config)
                for metric in (PRIMARY_METRIC, CHUNK_METRIC)
            }
            if subset
            else {"status": "not_estimable", "states": 0}
            for level, subset in levels.items()
        }
    return result


def _align_cached_rows(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[Mapping[str, object]]]:
    missing = set(PRIMARY_CONDITIONS) - set(rows_by_condition)
    if missing:
        raise ValueError(f"cached action conditions are missing: {sorted(missing)}")
    reference_rows = list(rows_by_condition[PRIMARY_CONDITIONS[0]])
    reference_ids = tuple(str(row["state_id"]) for row in reference_rows)
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("cached action rows contain duplicate state IDs")
    aligned = {PRIMARY_CONDITIONS[0]: reference_rows}
    for condition in PRIMARY_CONDITIONS[1:]:
        by_id = {str(row["state_id"]): row for row in rows_by_condition[condition]}
        if len(by_id) != len(rows_by_condition[condition]) or set(by_id) != set(reference_ids):
            raise ValueError("cached checkpoint rows must contain identical state IDs")
        aligned[condition] = [by_id[state_id] for state_id in reference_ids]
    for condition, rows in aligned.items():
        for reference, row in zip(reference_rows, rows, strict=True):
            for field in ("episode", "task", "phase", "stratum", "stable_grasp"):
                if row[field] != reference[field]:
                    raise ValueError(
                        f"cached state metadata differs across checkpoints: {condition}:{field}"
                    )
            if row["stratum"] != phase_stratum(str(row["phase"])):
                raise ValueError("cached phase stratum does not match the registered phase")
    return aligned


def _summarize_cached_recruitment_rows(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    config: LiberoStudyConfig,
) -> dict[str, object]:
    aligned = _align_cached_rows(rows_by_condition)
    metrics = (PRIMARY_METRIC, *ACTION_COMPONENTS, CHUNK_METRIC)
    conditions: dict[str, object] = {}
    for condition in PRIMARY_CONDITIONS:
        rows = aligned[condition]
        conditions[condition] = {
            "primary_u": _metric_summary(rows, PRIMARY_METRIC, config=config),
            "state_conditioned_u": _state_summaries(rows, config=config),
            "action_component_conditioned_u": {
                metric: _metric_summary(rows, metric, config=config)
                for metric in ACTION_COMPONENTS
            },
            "full_chunk_secondary": _metric_summary(rows, CHUNK_METRIC, config=config),
        }
    contrasts: dict[str, object] = {}
    for name, (before, after) in PAIRED_CONTRASTS.items():
        before_rows = aligned[before]
        after_rows = aligned[after]
        contrasts[name] = {
            "before": before,
            "after": after,
            "paired_states": len(before_rows),
            "metrics": {
                metric: _metric_summary(
                    before_rows,
                    metric,
                    config=config,
                    values=[
                        _effect(after_row, metric) - _effect(before_row, metric)
                        for before_row, after_row in zip(before_rows, after_rows, strict=True)
                    ],
                )
                for metric in metrics
            },
        }
    return {
        "conditions": conditions,
        "paired_checkpoint_deltas": contrasts,
    }


def analyze_cached_longitudinal_recruitment(
    config: LiberoStudyConfig, *, max_states: int
) -> dict[str, object]:
    profile_root = _profile_root(Path(config.output_dir) / "protocol_v3", max_states)
    source_path = profile_root / "action_sensitivity.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"action sensitivity report not found: {source_path}")
    source = json.loads(source_path.read_text())
    if source.get("status") != "complete" or not source.get("integrity_passed"):
        raise ValueError("cached analysis requires a complete integrity-passed action report")
    rows_by_condition: dict[str, list[Mapping[str, object]]] = {}
    row_hashes: dict[str, str] = {}
    for condition in PRIMARY_CONDITIONS:
        condition_report = source["conditions"][condition]
        path = Path(str(condition_report["rows"]))
        digest = _file_sha256(path)
        if digest != condition_report["rows_sha256"]:
            raise ValueError(f"cached action rows changed: {condition}")
        rows_by_condition[condition] = json.loads(gzip.decompress(path.read_bytes()))
        row_hashes[condition] = digest
    binding = hashlib.sha256(
        json.dumps(
            {
                "implementation_sha256": _file_sha256(Path(__file__)),
                "source_action_sensitivity_sha256": _file_sha256(source_path),
                "source_rows_sha256": row_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report_path = profile_root / "cached_analysis.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text())
        if existing.get("binding_sha256") != binding:
            raise FileExistsError(f"cached analysis has a different binding: {report_path}")
        return existing
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "integrity_passed": True,
        "no_model_loading": True,
        "no_new_inference": True,
        "factor": "stable_grasp",
        "tap": "action_expert_input",
        "primary_metric": PRIMARY_METRIC,
        "primary_cluster": "episode",
        "robustness_cluster": "task",
        "full_chunk_role": "secondary",
        "paired_delta_sign": "after_minus_before",
        "matched_random_basis": "checkpoint_specific_fold_held_out",
        "binding_sha256": binding,
        "source_action_sensitivity": str(source_path),
        "source_action_sensitivity_sha256": _file_sha256(source_path),
        "source_rows_sha256": row_hashes,
        "functional_recruitment_gate_passed": bool(source.get("passed")),
        "closed_loop_authorized": bool(source.get("passed")),
        **_summarize_cached_recruitment_rows(rows_by_condition, config=config),
    }
    write_json_atomic(report_path, report)
    return report
