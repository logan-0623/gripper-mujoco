from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .crossfit_probes import (
    _read_gzip_json,
    _write_immutable_gzip_json,
    _write_immutable_json,
    build_crossfit_manifest,
    run_crossfit_cell,
)
from .latents import (
    _file_sha256,
    _tree_sha256,
    extract_smolvla_latents_from_checkpoint,
    load_latent_cache,
)
from .state_bank import load_state_bank


POSITIVE_CONTROL_SCHEMA = "libero_smolvla_official_positive_control_v1"
PRIMARY_TAP = "action_expert_input"
SUPPORTED_FACTORS = ("stable_grasp", "contact")
PROBE_FACTORS = ("stable_grasp", "contact", "phase", "geometry")
MINIMUM_SUCCESS_RATE = 0.20


def positive_control_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "protocol_v4" / "positive_control"


def official_success_rate(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("LeRobot evaluation has no overall metrics")
    percent = float(overall.get("pc_success", float("nan")))
    episodes = int(overall.get("n_episodes", 0))
    if not np.isfinite(percent) or not 0.0 <= percent <= 100.0 or episodes <= 0:
        raise ValueError("LeRobot evaluation metrics must be finite and non-empty")
    return percent / 100.0, episodes


def positive_control_decision(
    *,
    success_rate: float,
    accessible: bool,
    specificity_passed: bool,
    usage_ci: tuple[float, float],
    factor: str,
) -> dict[str, object]:
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"unsupported positive-control factor: {factor}")
    if not np.isfinite((success_rate, *usage_ci)).all():
        raise ValueError("positive-control decision values must be finite")
    if success_rate <= MINIMUM_SUCCESS_RATE:
        decision = "failed_policy_floor"
    elif not accessible:
        decision = (
            "replicate_contact_once"
            if factor == "stable_grasp"
            else "pivot_interaction_supervised_sft"
        )
    elif not specificity_passed:
        decision = "failed_specificity"
    elif usage_ci[0] > 0.0:
        decision = "continue_official_longitudinal"
    elif factor == "stable_grasp":
        decision = "replicate_contact_once"
    else:
        decision = "pivot_interaction_supervised_sft"
    return {
        "decision": decision,
        "authorize_longitudinal_training": (
            decision == "continue_official_longitudinal"
        ),
    }


def plan_positive_control(
    config: LiberoStudyConfig,
    *,
    checkpoint: str | Path,
    eval_dir: str | Path,
) -> dict[str, object]:
    checkpoint = Path(checkpoint)
    eval_dir = Path(eval_dir)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"official checkpoint is incomplete: {checkpoint}")
    eval_report = eval_dir / "eval_info.json"
    if not eval_report.is_file():
        raise FileNotFoundError(f"LeRobot evaluation report is missing: {eval_report}")
    state_bank = Path(config.output_dir) / "state_bank" / "manifest.json"
    if not state_bank.is_file():
        raise FileNotFoundError(f"State Bank manifest is missing: {state_bank}")
    success_rate, episodes = official_success_rate(eval_report)
    if success_rate <= MINIMUM_SUCCESS_RATE:
        raise ValueError(
            "official checkpoint remains at the preregistered closed-loop floor"
        )
    report: dict[str, object] = {
        "schema_version": POSITIVE_CONTROL_SCHEMA,
        "passed": True,
        "status": "ready_for_latents",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _tree_sha256(checkpoint),
        "evaluation_dir": str(eval_dir),
        "evaluation_report_sha256": _file_sha256(eval_report),
        "baseline_success_rate": success_rate,
        "baseline_episodes": episodes,
        "state_bank_sha256": _file_sha256(state_bank),
        "config_sha256": _file_sha256(config.source_path),
        "runtime_contract": {
            "source": "lerobot.scripts.lerobot_eval",
            "n_action_steps": 10,
            "num_denoising_steps": 10,
            "empty_cameras": 1,
            "camera_name_mapping": {
                "agentview_image": "camera1",
                "robot0_eye_in_hand_image": "camera2",
            },
            "max_parallel_tasks": 1,
            "recording": False,
        },
        "interpretation": "successful-policy floor screen, not a benchmark claim",
    }
    path = positive_control_root(config.output_dir) / "plan.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError(f"positive-control binding changed: {path}")
        return existing
    write_json_atomic(path, report)
    return report


def load_positive_control_plan(config: LiberoStudyConfig) -> dict[str, object]:
    path = positive_control_root(config.output_dir) / "plan.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != POSITIVE_CONTROL_SCHEMA or not report.get(
        "passed"
    ):
        raise ValueError("positive-control plan is incomplete or incompatible")
    checkpoint = Path(str(report["checkpoint"]))
    if _tree_sha256(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("positive-control checkpoint hash changed")
    eval_report = Path(str(report["evaluation_dir"])) / "eval_info.json"
    if _file_sha256(eval_report) != report.get("evaluation_report_sha256"):
        raise ValueError("positive-control evaluation report hash changed")
    state_bank = Path(config.output_dir) / "state_bank" / "manifest.json"
    if _file_sha256(state_bank) != report.get("state_bank_sha256"):
        raise ValueError("positive-control State Bank hash changed")
    if _file_sha256(config.source_path) != report.get("config_sha256"):
        raise ValueError("positive-control config hash changed")
    return report


def extract_positive_control(
    config: LiberoStudyConfig, *, batch_size: int = 32
) -> dict[str, object]:
    plan = load_positive_control_plan(config)
    root = positive_control_root(config.output_dir)
    return extract_smolvla_latents_from_checkpoint(
        config,
        checkpoint=str(plan["checkpoint"]),
        checkpoint_id=f"official:{str(plan['checkpoint_sha256'])[:16]}",
        checkpoint_hash=str(plan["checkpoint_sha256"]),
        output_dir=root / "latents",
        label="official_smolvla_libero",
        batch_size=batch_size,
        runtime_binding=True,
        report_schema="libero_smolvla_positive_control_latents_v1",
        report_fields={"plan_sha256": _file_sha256(root / "plan.json")},
        taps=(PRIMARY_TAP,),
    )


def summarize_positive_control_probe_results(
    cells: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(cells) != set(PROBE_FACTORS):
        raise ValueError("positive-control probe factors are incomplete")
    status_counts = {
        status: sum(row.get("status") == status for row in cells.values())
        for status in ("complete", "not_estimable", "failed_gate")
    }
    stable = cells["stable_grasp"]
    passed = status_counts["complete"] == len(PROBE_FACTORS)
    return {
        "passed": passed,
        "status_counts": status_counts,
        "stable_grasp_accessible": bool(
            passed and stable.get("accessible") is True
        ),
    }


def _positive_control_probe_binding(
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    latent_manifest: Mapping[str, object],
    folds_sha256: str,
    factor: str,
) -> dict[str, object]:
    return {
        "schema_version": "libero_smolvla_positive_control_crossfit_cell_v1",
        "condition": "official_smolvla_libero",
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "tap": PRIMARY_TAP,
        "factor": factor,
        "split": "episode_group",
        "plan_sha256": plan_sha256,
        "folds_sha256": folds_sha256,
        "latent_values_sha256": latent_manifest["values_sha256"],
        "state_bank_sha256": plan["state_bank_sha256"],
    }


def run_positive_control_probe(config: LiberoStudyConfig) -> dict[str, object]:
    plan = load_positive_control_plan(config)
    root = positive_control_root(config.output_dir)
    plan_sha256 = _file_sha256(root / "plan.json")
    latent_report_path = root / "latents" / "report.json"
    latent_report = json.loads(latent_report_path.read_text(encoding="utf-8"))
    if (
        latent_report.get("schema_version")
        != "libero_smolvla_positive_control_latents_v1"
        or not latent_report.get("passed")
        or latent_report.get("checkpoint_sha256") != plan["checkpoint_sha256"]
        or latent_report.get("plan_sha256") != plan_sha256
    ):
        raise ValueError("positive-control latent report is stale or incompatible")
    records, _, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    state_ids, features, latent_manifest = load_latent_cache(
        root / "latents" / PRIMARY_TAP
    )
    if state_ids != tuple(record.state_id for record in records):
        raise ValueError("positive-control latent rows differ from the State Bank")
    if latent_manifest.get("state_bank_sha256") != plan["state_bank_sha256"]:
        raise ValueError("positive-control latent State Bank binding is stale")

    manifest = build_crossfit_manifest(
        records,
        split_name="episode_group",
        folds=config.probes.crossfit_folds,
        seed=config.seed,
    )
    probe_root = root / "probe"
    folds_path = probe_root / "folds.json"
    _write_immutable_json(
        folds_path,
        {
            "schema_version": "libero_smolvla_positive_control_folds_v1",
            "passed": True,
            "manifest": asdict(manifest),
            "plan_sha256": plan_sha256,
        },
    )
    folds_sha256 = _file_sha256(folds_path)
    cells: dict[str, Mapping[str, object]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for factor in PROBE_FACTORS:
        path = probe_root / "cells" / f"{factor}.json.gz"
        binding = _positive_control_probe_binding(
            plan=plan,
            plan_sha256=plan_sha256,
            latent_manifest=latent_manifest,
            folds_sha256=folds_sha256,
            factor=factor,
        )
        if path.is_file():
            cell = _read_gzip_json(path)
            if cell.get("binding") != binding:
                raise FileExistsError(f"positive-control probe binding changed: {path}")
        else:
            cell = {
                "binding": binding,
                "result": run_crossfit_cell(
                    config=config,
                    records=records,
                    features=features,
                    manifest=manifest,
                    tap=PRIMARY_TAP,
                    factor=factor,
                ),
            }
            _write_immutable_gzip_json(path, cell)
            cell = _read_gzip_json(path)
        result = cell.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"positive-control probe cell is malformed: {path}")
        cells[factor] = result
        artifacts[factor] = {"path": str(path), "sha256": _file_sha256(path)}

    summary = summarize_positive_control_probe_results(cells)
    report = {
        "schema_version": "libero_smolvla_positive_control_probe_v1",
        **summary,
        "condition": "official_smolvla_libero",
        "tap": PRIMARY_TAP,
        "split": "episode_group",
        "plan_sha256": plan_sha256,
        "folds_sha256": folds_sha256,
        "artifacts": artifacts,
        "cells": {
            factor: {
                key: cells[factor].get(key)
                for key in (
                    "status",
                    "reason",
                    "accessible",
                    "primary_metric_name",
                    "primary_metric",
                    "accessibility_threshold",
                    "accessibility_utility",
                    "accessibility_utility_ci",
                )
            }
            for factor in PROBE_FACTORS
        },
        "interpretation_boundary": {
            "accessible": "cross-fitted utility exceeds the strongest registered shortcut",
            "functionally_used": "not measured by this report",
        },
    }
    report_path = probe_root / "report.json"
    _write_immutable_json(report_path, report)
    return json.loads(report_path.read_text(encoding="utf-8"))
