from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..state_bank.io import write_json_atomic
from ..statistics import (
    clustered_bootstrap_mean,
    paired_sign_flip_pvalue,
    spearman_correlation,
)
from .core import normalized_curve_auc
from .formal import (
    CONSTANT_CONTROL_CONDITIONS,
    FORMAL_CONDITIONS,
    FORMAL_TRAINING_CONDITIONS,
)
from .v2_config import RecoveryRLV2Config


FORMAL_REPORT_SCHEMA = "recovery_representation_report_v2"


@dataclass(frozen=True)
class ExpansionDecision:
    passed: bool
    positive_seeds: int
    seed_count: int
    directions: tuple[float, ...]
    rule: str = "strictly positive in at least two of three paired seeds"


def expansion_gate(directions: Sequence[float]) -> ExpansionDecision:
    values = tuple(float(value) for value in directions)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("formal expansion gate requires three finite seed directions")
    positive = sum(value > 0.0 for value in values)
    return ExpansionDecision(
        passed=positive >= 2,
        positive_seeds=positive,
        seed_count=3,
        directions=values,
    )


def assemble_formal_evidence(
    *,
    accessible: Sequence[Mapping[str, object]],
    useful: Sequence[Mapping[str, object]],
    plasticity: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups = {
        "accessible": accessible,
        "useful": useful,
        "plasticity": plasticity,
    }
    rows: list[dict[str, object]] = []
    for expected_axis, values in groups.items():
        for raw in values:
            row = dict(raw)
            if row.get("axis") != expected_axis:
                raise ValueError(f"formal evidence row crosses {expected_axis} axis")
            if expected_axis == "useful" and row.get("unit") != "episode":
                raise ValueError("formal useful evidence must use episode units")
            rows.append(row)
    return rows


def _load(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"formal report artifact must be a JSON object: {path}")
    return value


def _evaluation_path(
    config: RecoveryRLV2Config,
    condition: str,
    seed_index: int,
    name: str,
) -> Path:
    return (
        config.output_dir
        / "formal"
        / "evaluations"
        / condition
        / f"seed_{seed_index}"
        / name
    )


def _measurement_path(
    config: RecoveryRLV2Config,
    condition: str,
    seed_index: int,
) -> Path:
    return (
        config.output_dir
        / "formal"
        / "measurements"
        / condition
        / f"seed_{seed_index}"
        / "ledger.json"
    )


def _primary_metric(target: str) -> str:
    return "r2" if target == "geometry" else "balanced_accuracy"


def _probe_rows(
    config: RecoveryRLV2Config,
    *,
    missing: list[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for condition in FORMAL_CONDITIONS:
        seed_indices = (0,) if condition in CONSTANT_CONTROL_CONDITIONS else range(3)
        for seed_index in seed_indices:
            ledger_path = _measurement_path(config, condition, seed_index)
            ledger = _load(ledger_path)
            if ledger is None:
                missing.append(ledger_path.as_posix())
                continue
            for point in ledger.get("points", []):
                if not isinstance(point, Mapping):
                    raise ValueError("formal measurement point must be a mapping")
                step = int(point["environment_steps"])
                reports = point.get("probe_reports")
                if not isinstance(reports, Mapping):
                    raise ValueError("formal measurement point has no probe reports")
                for model_kind, raw_path in reports.items():
                    report_path = Path(str(raw_path))
                    probe = _load(report_path)
                    if probe is None:
                        missing.append(report_path.as_posix())
                        continue
                    for raw in probe.get("rows", []):
                        row = dict(raw)
                        target = str(row["target"])
                        metrics = row.get("metrics")
                        if not isinstance(metrics, Mapping):
                            raise ValueError("formal probe row has no metrics")
                        test = metrics.get("test")
                        if not isinstance(test, Mapping):
                            raise ValueError("formal probe row has no test metrics")
                        for metric, value in test.items():
                            result.append(
                                {
                                    "axis": "accessible",
                                    "unit": "state",
                                    "backend": "act",
                                    "condition": condition,
                                    "representation_seed_index": seed_index,
                                    "environment_steps": step,
                                    "tap": row["tap"],
                                    "target": target,
                                    "target_role": row["target_role"],
                                    "model_kind": model_kind,
                                    "partition": "test",
                                    "metric": metric,
                                    "primary_metric": metric == _primary_metric(target),
                                    "value": float(value),
                                    "source": report_path.as_posix(),
                                }
                            )
    return result


def _curve_and_utility_rows(
    config: RecoveryRLV2Config,
    *,
    missing: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, int], dict[str, object]]]:
    useful: list[dict[str, object]] = []
    plasticity: list[dict[str, object]] = []
    final_reports: dict[tuple[str, int], dict[str, object]] = {}
    metrics = (
        "success_rate",
        "timeout_rate",
        "drop_rate",
        "wrong_object_rate",
        "mean_steps",
        "mean_return",
        "mean_residual_norm",
        "action_clipping_rate",
        "action_smoothness",
        "mean_ik_projection_scale",
    )
    for condition in FORMAL_CONDITIONS:
        for seed_index in range(3):
            curve_path = _evaluation_path(
                config, condition, seed_index, "curve_report.json"
            )
            final_path = _evaluation_path(
                config,
                condition,
                seed_index,
                f"final/step_{config.formal_steps:06d}/report.json",
            )
            curve = _load(curve_path)
            final = _load(final_path)
            if curve is None:
                missing.append(curve_path.as_posix())
            if final is None:
                missing.append(final_path.as_posix())
            if curve is None or final is None:
                continue
            final_reports[(condition, seed_index)] = final
            points = curve.get("points")
            if not isinstance(points, list) or len(points) != len(config.snapshot_steps):
                raise ValueError("formal curve report does not contain six points")
            for point in points:
                if not isinstance(point, Mapping):
                    raise ValueError("formal curve point must be a mapping")
                for family in ("nominal", "recovery"):
                    outcome = point.get(family)
                    if not isinstance(outcome, Mapping):
                        raise ValueError(f"formal curve point has no {family} outcome")
                    for metric in metrics:
                        useful.append(
                            {
                                "axis": "useful",
                                "unit": "episode",
                                "scope": "curve",
                                "backend": "act",
                                "condition": condition,
                                "seed_index": seed_index,
                                "environment_steps": int(point["environment_steps"]),
                                "distribution": family,
                                "metric": metric,
                                "value": float(outcome[metric]),
                                "episodes": int(outcome["episodes"]),
                                "source": curve_path.as_posix(),
                            }
                        )
            for family in ("nominal", "recovery"):
                outcome = final.get(family)
                if not isinstance(outcome, Mapping):
                    raise ValueError(f"formal final report has no {family} outcome")
                for metric in metrics:
                    useful.append(
                        {
                            "axis": "useful",
                            "unit": "episode",
                            "scope": "held_out_final",
                            "backend": "act",
                            "condition": condition,
                            "seed_index": seed_index,
                            "environment_steps": config.formal_steps,
                            "distribution": family,
                            "metric": metric,
                            "value": float(outcome[metric]),
                            "episodes": int(outcome["episodes"]),
                            "source": final_path.as_posix(),
                        }
                    )
            steps = [int(point["environment_steps"]) for point in points]
            for family in ("nominal", "recovery"):
                success = [float(point[family]["success_rate"]) for point in points]
                auc = normalized_curve_auc(
                    steps,
                    success,
                    budget=config.formal_steps,
                )
                plasticity.extend(
                    (
                        {
                            "axis": "plasticity",
                            "unit": "training_run",
                            "backend": "act",
                            "condition": condition,
                            "seed_index": seed_index,
                            "distribution": family,
                            "metric": "normalized_success_auc",
                            "value": auc,
                            "source": curve_path.as_posix(),
                        },
                        {
                            "axis": "plasticity",
                            "unit": "training_run",
                            "backend": "act",
                            "condition": condition,
                            "seed_index": seed_index,
                            "distribution": family,
                            "metric": "final_minus_initial_success",
                            "value": success[-1] - success[0],
                            "source": curve_path.as_posix(),
                        },
                    )
                )
    return useful, plasticity, final_reports


def _paired_effects(
    final_reports: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for condition in FORMAL_CONDITIONS:
        if condition == "sft":
            continue
        for family in ("nominal", "recovery"):
            differences: list[float] = []
            clusters: list[str] = []
            seed_directions: list[float] = []
            for seed_index in range(3):
                baseline = final_reports.get(("sft", seed_index))
                candidate = final_reports.get((condition, seed_index))
                if baseline is None or candidate is None:
                    continue
                baseline_rows = {
                    str(row["case_id"]): row
                    for row in baseline["rows"]
                    if row["family"] == family
                }
                candidate_rows = {
                    str(row["case_id"]): row
                    for row in candidate["rows"]
                    if row["family"] == family
                }
                if set(baseline_rows) != set(candidate_rows):
                    raise ValueError("formal final paired case ids differ")
                local = [
                    float(candidate_rows[case_id]["success"])
                    - float(baseline_rows[case_id]["success"])
                    for case_id in sorted(baseline_rows)
                ]
                differences.extend(local)
                clusters.extend([f"seed_{seed_index}"] * len(local))
                seed_directions.append(float(np.mean(local)))
            if not differences:
                continue
            interval = clustered_bootstrap_mean(
                differences,
                clusters,
                samples=5000,
                confidence=0.95,
                seed=seed + len(result),
            )
            result.append(
                {
                    "contrast": f"{condition}-minus-sft",
                    "condition": condition,
                    "baseline": "sft",
                    "distribution": family,
                    "metric": "paired_success_delta",
                    **interval,
                    "paired_sign_flip_p": paired_sign_flip_pvalue(
                        seed_directions,
                        samples=5000,
                        seed=seed + 1000 + len(result),
                    ),
                    "episode_pairs": len(differences),
                    "seed_directions": seed_directions,
                    "cluster_unit": "evaluation_seed",
                }
            )
    return result


def _required_protocol_artifacts(
    config: RecoveryRLV2Config,
) -> list[Path]:
    result = [
        config.output_dir / "gates" / f"{name}.json"
        for name in ("distribution", "backend", "oracle", "anchoring")
    ]
    result.extend(
        (
            config.output_dir / "manifests" / "cases.json",
            config.output_dir / "manifests" / "oracle_normalization.json",
            config.output_dir / "state_bank_v2" / "manifest.json",
        )
    )
    for condition in FORMAL_TRAINING_CONDITIONS:
        for seed_index in range(3):
            run = (
                config.output_dir
                / "formal"
                / "runs"
                / condition
                / f"seed_{seed_index}"
            )
            result.append(run / "training_report.json")
            result.extend(
                run / "snapshots" / f"step_{step:06d}" / "COMPLETED"
                for step in config.snapshot_steps
            )
    return result


def _trajectory_rows(
    accessible: Sequence[Mapping[str, object]],
    useful: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    success = {
        (
            str(row["condition"]),
            int(row["seed_index"]),
            int(row["environment_steps"]),
        ): float(row["value"])
        for row in useful
        if row.get("scope") == "curve"
        and row.get("distribution") == "recovery"
        and row.get("metric") == "success_rate"
    }
    primary = [
        row
        for row in accessible
        if row.get("model_kind") == "linear"
        and row.get("primary_metric") is True
        and row.get("target_role") == "primary"
        and row.get("condition") not in CONSTANT_CONTROL_CONDITIONS
    ]
    by_series: dict[tuple[str, int, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in primary:
        key = (
            str(row["condition"]),
            int(row["representation_seed_index"]),
            str(row["tap"]),
            str(row["target"]),
        )
        by_series[key].append(row)
    trajectories: list[dict[str, object]] = []
    associations_by_factor: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (condition, seed_index, tap, target), values in sorted(by_series.items()):
        ordered = sorted(values, key=lambda row: int(row["environment_steps"]))
        if not ordered:
            continue
        baseline_probe = float(ordered[0]["value"])
        baseline_success = success.get((condition, seed_index, 0))
        for row in ordered:
            step = int(row["environment_steps"])
            behavior = success.get((condition, seed_index, step))
            if baseline_success is None or behavior is None:
                continue
            delta_probe = float(row["value"]) - baseline_probe
            delta_success = behavior - baseline_success
            trajectories.append(
                {
                    "condition": condition,
                    "seed_index": seed_index,
                    "environment_steps": step,
                    "tap": tap,
                    "target": target,
                    "probe_metric": row["metric"],
                    "probe_value": float(row["value"]),
                    "recovery_success": behavior,
                    "delta_probe": delta_probe,
                    "delta_recovery_success": delta_success,
                    "interpretation": "descriptive trajectory; not causal",
                }
            )
            associations_by_factor[(tap, target)].append(
                (delta_probe, delta_success)
            )
    associations: list[dict[str, object]] = []
    for (tap, target), pairs in sorted(associations_by_factor.items()):
        if len(pairs) < 3:
            continue
        coefficient = spearman_correlation(
            [value[0] for value in pairs],
            [value[1] for value in pairs],
        )
        if not math.isfinite(coefficient):
            continue
        associations.append(
            {
                "tap": tap,
                "target": target,
                "pairs": len(pairs),
                "spearman_delta_probe_vs_delta_recovery_success": coefficient,
                "interpretation": "descriptive multi-seed/time association; not causal",
            }
        )
    return trajectories, associations


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Formal ACT recovery representation study",
        "",
        f"Complete: **{str(bool(report['complete'])).lower()}**",
        "",
        "This report keeps representation accessibility, closed-loop utility, and online plasticity as separate evidence axes.",
        "",
        "## Protocol",
        "",
        "| Condition | Role | Training seeds |",
        "|---|---|---:|",
        "| SFT | frozen control | 0 (three paired evaluation seeds) |",
        "| Continued SFT | extra-imitation control | 0 (three paired evaluation seeds) |",
        "| Oracle State | privileged interface validation | 3 |",
        "| RL Head | frozen ACT representation | 3 |",
        "| RL Representation | ACT fusion adaptation | 3 |",
        "",
        "## Expansion gate",
        "",
    ]
    gate = report.get("expansion_gate")
    if gate is None:
        lines.append("Pending complete RL-head and RL-representation curves.")
    else:
        lines.append(
            f"Passed: **{str(bool(gate['passed'])).lower()}**; positive paired seed directions: {gate['positive_seeds']}/3."
        )
    if report.get("missing"):
        lines.extend(("", "## Missing artifacts", ""))
        lines.extend(f"- `{value}`" for value in report["missing"])
    lines.extend(
        (
            "",
            "Associations between probe change and recovery-success change are descriptive. Probe accessibility alone is not treated as functional policy use.",
        )
    )
    return "\n".join(lines) + "\n"


def build_formal_report(config: RecoveryRLV2Config) -> dict[str, object]:
    destination = config.output_dir / "formal"
    destination.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    missing.extend(
        path.as_posix()
        for path in _required_protocol_artifacts(config)
        if not path.is_file()
    )
    accessible = _probe_rows(config, missing=missing)
    useful, plasticity, final_reports = _curve_and_utility_rows(
        config, missing=missing
    )
    result_rows = assemble_formal_evidence(
        accessible=accessible,
        useful=useful,
        plasticity=plasticity,
    )
    effects = _paired_effects(final_reports, seed=config.seed + 800_000)
    trajectories, associations = _trajectory_rows(accessible, useful)
    auc = {
        (str(row["condition"]), int(row["seed_index"])): float(row["value"])
        for row in plasticity
        if row.get("distribution") == "recovery"
        and row.get("metric") == "normalized_success_auc"
    }
    directions = [
        auc[("rl_representation", seed_index)] - auc[("rl_head", seed_index)]
        for seed_index in range(3)
        if ("rl_representation", seed_index) in auc
        and ("rl_head", seed_index) in auc
    ]
    gate = expansion_gate(directions) if len(directions) == 3 else None
    nominal_changes = {
        int(row["seed_index"]): float(row["value"])
        for row in plasticity
        if row.get("condition") == "rl_representation"
        and row.get("distribution") == "nominal"
        and row.get("metric") == "final_minus_initial_success"
    }
    retention_gate = (
        {
            "passed": all(
                nominal_changes[index] >= -0.10 - 1.0e-12
                for index in range(3)
            ),
            "threshold": -0.10,
            "seed_directions": [nominal_changes[index] for index in range(3)],
            "rule": "RL-representation nominal forgetting is at most 10 percentage points in every seed",
        }
        if len(nominal_changes) == 3
        else None
    )
    missing = sorted(set(missing))
    complete = not missing
    report = {
        "schema_version": FORMAL_REPORT_SCHEMA,
        "complete": complete,
        "passed": complete,
        "result_rows": len(result_rows),
        "curve_rows": len(useful),
        "probe_trajectory_rows": len(trajectories),
        "pairwise_effects": len(effects),
        "trajectory_associations": associations,
        "expansion_gate": None if gate is None else asdict(gate),
        "nominal_retention_gate": retention_gate,
        "modern_vla_ready": bool(
            complete
            and gate is not None
            and gate.passed
            and retention_gate is not None
            and retention_gate["passed"]
        ),
        "missing": missing,
        "claims": {
            "correctness_vs_utility": "requires comparison of probe accessibility and paired behavior; no implication is assumed",
            "utility_vs_plasticity": "reported as separate axes",
            "rl_representation": "supported for expansion only if at least two of three paired AUC directions are positive",
        },
    }
    write_json_atomic(destination / "result_rows.json", {"rows": result_rows})
    write_json_atomic(destination / "curve_rows.json", {"rows": useful})
    write_json_atomic(
        destination / "probe_trajectory_rows.json",
        {"rows": trajectories, "associations": associations},
    )
    write_json_atomic(destination / "pairwise_effects.json", {"rows": effects})
    write_json_atomic(destination / "study_report.json", report)
    _write_text_atomic(destination / "study_report.md", _markdown(report))
    return report
