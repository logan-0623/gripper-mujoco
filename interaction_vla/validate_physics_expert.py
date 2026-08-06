from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config
from .env import TerminationReason
from .franka import FINGER_JOINT_NAMES, FRANKA_SCENE_PATH, OBJECT_NAMES
from .physics_env import FrankaContactEnv
from .physics_expert import PhysicsScriptedExpert
from .franka_controller import FrankaCartesianController
from .contact_physics import StableGraspTracker
from .physics_provenance import (
    config_file_hash,
    controller_source_hash,
    learned_rollout_source_hash,
    scene_asset_hash,
)
from .chunked_controller import ChunkedPolicyController, TemporalActionEnsembler
from .physics_data import prepare_physics_recovery_start


@dataclass(frozen=True)
class ExpertValidationCase:
    condition: str
    seed: int
    object_count: int


@dataclass(frozen=True)
class ExpertValidationResult:
    condition: str
    seed: int
    object_count: int
    success: bool
    reason: str
    steps: int
    stable_lift: bool
    physics_failure: bool


def make_validation_cases(
    *, config_seed: int, cases_per_condition: int
) -> tuple[ExpertValidationCase, ...]:
    if cases_per_condition < 1:
        raise ValueError("cases_per_condition must be positive")
    cases: list[ExpertValidationCase] = []
    for condition_id, condition in enumerate(("normal", "crowded")):
        for index in range(cases_per_condition):
            sequence = np.random.SeedSequence(
                (int(config_seed), 0x56414C47, condition_id, index)
            )
            seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
            cases.append(
                ExpertValidationCase(
                    condition=condition,
                    seed=seed,
                    object_count=2 + index % 2,
                )
            )
    if len({case.seed for case in cases}) != len(cases):
        raise RuntimeError("validation seed namespace produced a collision")
    return tuple(cases)


def build_gate_report(
    results: Iterable[ExpertValidationResult],
    *,
    threshold: float,
    controller_hash: str,
    scene_hash: str,
    config_hash: str,
    attachment_audit_passed: bool,
    rollout_integrity_hash: str = "",
) -> dict[str, object]:
    values = tuple(results)
    if not values:
        raise ValueError("expert validation requires at least one result")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    conditions: dict[str, dict[str, float | int]] = {}
    condition_passes: list[bool] = []
    for condition in ("normal", "crowded"):
        selected = [result for result in values if result.condition == condition]
        if not selected:
            raise ValueError(f"expert validation is missing {condition} cases")
        success = sum(result.success for result in selected)
        rate = success / len(selected)
        conditions[condition] = {
            "success": success,
            "total": len(selected),
            "rate": rate,
        }
        condition_passes.append(rate >= threshold)
    success_count = sum(result.success for result in values)
    success_rate = success_count / len(values)
    successful_lifts_are_physical = all(
        not result.success or result.stable_lift for result in values
    )
    no_physics_failure = not any(result.physics_failure for result in values)
    passed = bool(
        success_rate >= threshold
        and all(condition_passes)
        and successful_lifts_are_physical
        and no_physics_failure
        and attachment_audit_passed
    )
    return {
        "passed": passed,
        "threshold": threshold,
        "success_rate": success_rate,
        "controller_hash": controller_hash,
        "rollout_integrity_hash": rollout_integrity_hash,
        "scene_hash": scene_hash,
        "config_hash": config_hash,
        "attachment_audit_passed": attachment_audit_passed,
        "successful_lifts_are_physical": successful_lifts_are_physical,
        "no_physics_failure": no_physics_failure,
        "conditions": conditions,
        "episodes": [asdict(result) for result in values],
    }


def no_attachment_audit() -> bool:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    object_joint_ids = {
        model.joint(f"{name}_joint").id for name in OBJECT_NAMES
    }
    if model.neq != 1 or model.eq_type[0] != mujoco.mjtEq.mjEQ_JOINT:
        return False
    finger_joint_ids = {model.joint(name).id for name in FINGER_JOINT_NAMES}
    if {int(model.eq_obj1id[0]), int(model.eq_obj2id[0])} != finger_joint_ids:
        return False
    for name in OBJECT_NAMES:
        if model.body_mocapid[model.body(name).id] != -1:
            return False
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT:
            if int(model.actuator_trnid[actuator_id, 0]) in object_joint_ids:
                return False
    rollout_methods = (
        FrankaContactEnv.step,
        FrankaContactEnv.advance_intervention,
        FrankaCartesianController.apply_action,
        PhysicsScriptedExpert.act,
        StableGraspTracker.update,
        ChunkedPolicyController.act,
        TemporalActionEnsembler.action_for_step,
        prepare_physics_recovery_start,
    )
    if any(
        token in inspect.getsource(method)
        for method in rollout_methods
        for token in (
            ".qpos[",
            ".qvel[",
            "mocap_pos",
            "mocap_quat",
            "eq_active",
        )
    ):
        return False
    return True


def run_validation_case(
    config: ExperimentConfig, case: ExpertValidationCase
) -> ExpertValidationResult:
    env = FrankaContactEnv(
        max_objects=config.max_objects,
        max_steps=config.environment.max_steps,
        min_object_distance=config.environment.min_object_distance,
        workspace_low=config.environment.workspace_low,
        workspace_high=config.environment.workspace_high,
        crowded_anchor_min_distance=config.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=config.environment.crowded_anchor_max_distance,
        physics=config.physics,
    )
    expert = PhysicsScriptedExpert(config.physics)
    try:
        snapshot = env.reset(
            seed=case.seed,
            object_count=case.object_count,
            layout_mode=case.condition,
        )
        expert.reset(seed=case.seed)
        transition = None
        for _ in range(config.environment.max_steps):
            action = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
            transition = env.step(action)
            snapshot = transition.snapshot
            if transition.done:
                break
        if transition is None:
            raise RuntimeError("validation episode executed no policy step")
        return ExpertValidationResult(
            condition=case.condition,
            seed=case.seed,
            object_count=case.object_count,
            success=transition.reason is TerminationReason.SUCCESS,
            reason=transition.reason.value,
            steps=env.step_count,
            stable_lift=env.grasp_state.ever_stable_target,
            physics_failure=transition.reason is TerminationReason.PHYSICS_FAILURE,
        )
    except Exception as error:
        return ExpertValidationResult(
            condition=case.condition,
            seed=case.seed,
            object_count=case.object_count,
            success=False,
            reason=f"exception:{type(error).__name__}:{error}",
            steps=getattr(env, "step_count", 0),
            stable_lift=False,
            physics_failure=True,
        )


def validate_expert_from_config(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    show_progress: bool = False,
) -> Path:
    path = Path(config_path)
    config = load_config(path)
    if config.backend != "franka_contact":
        raise ValueError("expert physics validation requires backend=franka_contact")
    cases = make_validation_cases(
        config_seed=config.seed,
        cases_per_condition=config.physics.expert_gate_cases_per_condition,
    )
    progress = (
        tqdm(
            total=len(cases),
            desc="expert gate",
            unit="case",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    results: list[ExpertValidationResult] = []
    passed = 0
    try:
        for case in cases:
            result = run_validation_case(config, case)
            results.append(result)
            passed += int(result.success)
            if progress is not None:
                progress.set_postfix(
                    condition=case.condition,
                    objects=case.object_count,
                    seed=case.seed,
                    success=int(result.success),
                    passed=passed,
                    failed=len(results) - passed,
                )
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    report = build_gate_report(
        results,
        threshold=0.9,
        controller_hash=controller_source_hash(),
        scene_hash=scene_asset_hash(),
        config_hash=config_file_hash(path),
        attachment_audit_passed=no_attachment_audit(),
        rollout_integrity_hash=learned_rollout_source_hash(),
    )
    destination = (
        Path(output) if output is not None else Path(config.output_dir) / "expert_gate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the physical Franka scripted expert before representation training"
    )
    parser.add_argument("--config", default="configs/physics_smoke_macos.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = validate_expert_from_config(
        args.config,
        output=args.output,
        show_progress=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    print(output)
    print(
        f"success_rate={report['success_rate']:.3f} "
        f"normal={report['conditions']['normal']['rate']:.3f} "
        f"crowded={report['conditions']['crowded']['rate']:.3f}"
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
