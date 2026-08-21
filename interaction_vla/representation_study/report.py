from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from interaction_vla.lerobot_bridge.provenance import sha256_file

from .config import RepresentationStudyConfig
from .statistics import spearman_correlation
from .state_bank.io import write_json_atomic


STUDY_REPORT_SCHEMA_VERSION = "interaction_representation_study_report_v1"


def representation_relationships(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join stage-level probe accessibility to closed-loop utility descriptively."""
    utility = {
        (str(row["backend"]), str(row["stage"])): float(row["value"])
        for row in rows
        if row.get("axis") == "useful" and row.get("metric") == "success_rate"
    }
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        target = str(row.get("target", ""))
        primary_metric = "r2" if target == "geometry" else "balanced_accuracy"
        if (
            row.get("axis") != "accessible"
            or row.get("partition") != "test"
            or row.get("metric") != primary_metric
        ):
            continue
        key = (
            str(row["backend"]),
            str(row["tap"]),
            target,
            primary_metric,
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for (backend, tap, target, metric), values in sorted(grouped.items()):
        aligned = sorted(
            (
                str(row["stage"]),
                float(row["value"]),
                utility[(backend, str(row["stage"]))],
            )
            for row in values
            if (backend, str(row["stage"])) in utility
        )
        if len(aligned) < 3:
            continue
        coefficient = spearman_correlation(
            [row[1] for row in aligned], [row[2] for row in aligned]
        )
        if not math.isfinite(coefficient):
            continue
        result.append(
            {
                "relationship": "accessible_vs_utility",
                "backend": backend,
                "tap": tap,
                "target": target,
                "probe_metric": metric,
                "utility_metric": "success_rate",
                "stages": [row[0] for row in aligned],
                "pairs": len(aligned),
                "spearman": coefficient,
                "interpretation": "descriptive cross-stage association; not causal",
            }
        )
    return result


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"study artifact must contain a JSON object: {path}")
    return value


def probe_result_rows(
    report: Mapping[str, object], *, source: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in report.get("rows", []):  # type: ignore[assignment]
        row = dict(raw)
        for field, axis in (
            ("metrics", "accessible"),
            ("baseline_metrics", "accessible_control"),
        ):
            metrics = dict(row.get(field, {}))
            for partition, raw_values in metrics.items():
                values = dict(raw_values)
                for metric, value in values.items():
                    result.append(
                        {
                            "axis": axis,
                            "backend": row["backend"],
                            "stage": row["stage"],
                            "tap": row["tap"],
                            "target": row["target"],
                            "model_kind": row["model_kind"],
                            "partition": partition,
                            "metric": metric,
                            "value": float(value),
                            "source": source,
                        }
                    )
    return result


def intervention_result_rows(
    report: Mapping[str, object], *, source: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for key, raw_values in dict(report.get("aggregate", {})).items():
        tap, mode = str(key).split("/", 1)
        for metric, value in dict(raw_values).items():
            result.append(
                {
                    "axis": "used",
                    "backend": report["backend"],
                    "stage": report["stage"],
                    "tap": tap,
                    "intervention": mode,
                    "metric": metric,
                    "value": float(value),
                    "source": source,
                    "interpretation": "offline functional use; not closed-loop success",
                }
            )
    return result


def rl_result_rows(report: Mapping[str, object], *, source: str) -> list[dict[str, object]]:
    result = []
    for metric in (
        "normalized_learning_curve_auc",
        "initial_success_rate",
        "final_success_rate",
        "final_return_gain",
        "steps_to_fixed_threshold",
    ):
        value = report.get(metric)
        if value is not None:
            result.append(
                {
                    "axis": "plasticity" if metric in {
                        "normalized_learning_curve_auc", "final_return_gain", "steps_to_fixed_threshold"
                    } else "useful",
                    "backend": report["backend"],
                    "stage": report["stage"],
                    "metric": metric,
                    "value": float(value),
                    "source": source,
                }
            )
    return result


def utility_result_rows(
    report: Mapping[str, object], *, source: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for metric in (
        "success_rate", "timeout_rate", "drop_rate", "wrong_object_rate", "mean_steps",
        "action_clipping_rate", "mean_ik_projection_scale",
    ):
        if metric in report:
            result.append(
                {
                    "axis": "useful",
                    "backend": report["backend"],
                    "stage": report["stage"],
                    "metric": metric,
                    "value": float(report[metric]),
                    "episodes": int(report["episodes"]),
                    "source": source,
                }
            )
    return result


def closed_loop_intervention_rows(
    report: Mapping[str, object], *, source: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in report.get("summaries", []):  # type: ignore[assignment]
        row = dict(raw)
        for metric in (
            "success_rate", "paired_success_delta", "ci_low", "ci_high",
            "paired_sign_flip_p", "bh_adjusted_p",
        ):
            result.append(
                {
                    "axis": "used_to_useful",
                    "backend": report["backend"],
                    "stage": report["stage"],
                    "tap": row["tap"],
                    "intervention": row["mode"],
                    "metric": metric,
                    "value": float(row[metric]),
                    "episodes": int(row["episodes"]),
                    "source": source,
                }
            )
    return result


def _legacy_rows(path: Path) -> list[dict[str, object]]:
    report = _load(path)
    if report is None:
        return []
    rows: list[dict[str, object]] = []
    for condition, values in dict(report.get("by_condition", {})).items():
        for metric, value in dict(values).items():
            rows.append(
                {
                    "axis": "controlled_outcome",
                    "backend": "act",
                    "stage": "legacy_graph_control",
                    "condition": condition,
                    "metric": metric,
                    "value": float(value),
                    "source": path.as_posix(),
                }
            )
    return rows


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown(
    config: RepresentationStudyConfig,
    stages: Sequence[Mapping[str, object]],
    *,
    missing: Sequence[str],
    rows: int,
) -> str:
    lines = [
        "# ICRA interaction-representation study",
        "",
        "The graph ontology is used as a measurement language. ACT is the controlled mechanism study; SmolVLA is the required modern-VLA validation; pi0 remains optional.",
        "",
        "## Evidence status",
        "",
        "| Backend | Stage | Latents | Linear probes | Offline use | Causal rollout | Closed loop | Training report |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stages:
        lines.append(
            "| {backend} | {stage} | {latents} | {linear_probe} | {intervention} | {causal_intervention} | {utility} | {training} |".format(
                **{
                    key: ("yes" if row[key] else "no") if isinstance(row[key], bool) else row[key]
                    for key in ("backend", "stage", "latents", "linear_probe", "intervention", "causal_intervention", "utility", "training")
                }
            )
        )
    lines.extend(
        [
            "",
            f"Machine-readable evidence rows: {rows}.",
            "",
            "The report separates accessible (frozen probes), used (latent interventions), useful (closed-loop outcomes), and plasticity (fixed-budget RL AUC). Correlations are descriptive and are not labeled causal.",
        ]
    )
    if missing:
        lines.extend(["", "## Remaining artifacts", ""])
        lines.extend(f"- `{value}`" for value in missing)
    return "\n".join(lines) + "\n"


def build_study_report(config: RepresentationStudyConfig) -> dict[str, object]:
    report_id = "+".join(sorted(config.stages))
    destination = config.analysis.output_dir / "reports" / report_id
    result_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for backend, stages in config.stages.items():
        for stage in stages:
            latent = config.extraction.output_dir / backend / stage / "all" / "latents.npz"
            linear = config.probes.output_dir / backend / stage / "linear" / "report.json"
            mlp = config.probes.output_dir / backend / stage / "shallow_mlp" / "report.json"
            intervention = config.interventions.output_dir / backend / stage / config.interventions.partition / "report.json"
            causal_intervention = config.interventions.output_dir / backend / stage / "closed_loop" / "report.json"
            utility = config.analysis.output_dir / "policy_evaluation" / backend / stage / "report.json"
            if stage in {"sft", "continued_sft"}:
                training = config.sft.output_dir / backend / stage / "report.json"
            elif stage in {"rl_head", "rl_representation"}:
                training = config.rl.output_dir / backend / stage / "report.json"
            else:
                training = None
            checkpoint_ready = Path(stages[stage].checkpoint).is_dir() or stage == "pretrained"
            training_ready = (
                True
                if training is None
                else training.is_file() or (stage == "sft" and checkpoint_ready)
            )
            row = {
                "backend": backend,
                "stage": stage,
                "checkpoint": checkpoint_ready,
                "latents": latent.is_file(),
                "linear_probe": linear.is_file(),
                "shallow_mlp_probe": mlp.is_file(),
                "intervention": intervention.is_file(),
                "causal_intervention": causal_intervention.is_file(),
                "utility": utility.is_file(),
                "training": training_ready,
            }
            stage_rows.append(row)
            for required in (latent, linear, intervention, causal_intervention, utility):
                if not required.is_file():
                    missing.append(required.as_posix())
            if training is not None and not training_ready:
                missing.append(training.as_posix())
            for probe_path in (linear, mlp):
                probe = _load(probe_path)
                if probe is not None:
                    result_rows.extend(probe_result_rows(probe, source=probe_path.as_posix()))
            use = _load(intervention)
            if use is not None:
                result_rows.extend(intervention_result_rows(use, source=intervention.as_posix()))
            causal_use = _load(causal_intervention)
            if causal_use is not None:
                result_rows.extend(
                    closed_loop_intervention_rows(
                        causal_use, source=causal_intervention.as_posix()
                    )
                )
            closed_loop = _load(utility)
            if closed_loop is not None:
                result_rows.extend(
                    utility_result_rows(closed_loop, source=utility.as_posix())
                )
            if stage in {"rl_head", "rl_representation"} and training is not None:
                rl = _load(training)
                if rl is not None:
                    result_rows.extend(rl_result_rows(rl, source=training.as_posix()))
    legacy = Path("outputs/graph_control/graph_v2_pilot/runs/evaluation/report.json")
    result_rows.extend(_legacy_rows(legacy))
    relationships = representation_relationships(result_rows)
    evidence = {
        "schema_version": STUDY_REPORT_SCHEMA_VERSION,
        "study_id": config.study_id,
        "report_id": report_id,
        "report_dir": destination.as_posix(),
        "passed": True,
        "complete": not missing,
        "graph_role": "measurement_ontology",
        "roles": {
            "act": "controlled_mechanism_study",
            "smolvla": "modern_vla_validation",
            "pi0": "optional_external_validity",
        },
        "stage_status": stage_rows,
        "missing_artifacts": sorted(set(missing)),
        "result_rows": len(result_rows),
        "relationship_rows": len(relationships),
        "interpretation_contract": {
            "accessible": "information recoverable by a frozen lightweight probe",
            "accessible_control": "train-label constant baseline for the same held-out target",
            "used": "action sensitivity under a paired latent intervention",
            "useful": "closed-loop task outcome",
            "plasticity": "fixed-budget normalized online-learning AUC",
            "causal_scope": "interventions support functional use; cross-stage correlations remain descriptive",
        },
        "legacy_controlled_evidence": (
            {"uri": legacy.as_posix(), "sha256": sha256_file(legacy)} if legacy.is_file() else None
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination / "result_rows.json", {
        "schema_version": STUDY_REPORT_SCHEMA_VERSION,
        "study_id": config.study_id,
        "rows": result_rows,
    })
    write_json_atomic(
        destination / "relationship_rows.json",
        {
            "schema_version": STUDY_REPORT_SCHEMA_VERSION,
            "study_id": config.study_id,
            "rows": relationships,
            "interpretation": "descriptive cross-stage association; not causal",
        },
    )
    write_json_atomic(destination / "study_report.json", evidence)
    _write_text_atomic(
        destination / "study_report.md",
        _markdown(config, stage_rows, missing=sorted(set(missing)), rows=len(result_rows)),
    )
    return evidence
