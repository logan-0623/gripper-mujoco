from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .latents import _file_sha256, _tree_sha256


POSITIVE_CONTROL_SCHEMA = "libero_smolvla_official_positive_control_v1"
PRIMARY_TAP = "action_expert_input"
SUPPORTED_FACTORS = ("stable_grasp", "contact")
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
