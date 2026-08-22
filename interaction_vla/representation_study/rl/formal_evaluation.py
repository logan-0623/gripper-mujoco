from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.provenance import sha256_file

from ..state_bank.io import write_json_atomic
from .actors import LatentResidualActor, OracleResidualActor
from .distributions import RecoveryCase, load_case_manifest
from .evaluation_v2 import EvaluationReport, evaluate_case_manifest
from .formal import (
    CONSTANT_CONTROL_CONDITIONS,
    FORMAL_SCHEMA,
    FormalRun,
    _anchor_coefficients,
    _example_latent_dim,
    prepare_formal_run,
)
from .foundation import (
    _ActorEvaluationPolicy,
    _ZeroResidualPolicy,
    _runtime,
    paired_evaluation_cases,
)
from .snapshots import SnapshotStore
from .v2_config import RecoveryRLV2Config


FORMAL_EVALUATION_SCHEMA = "recovery_formal_evaluation_v2"


def validate_curve_case_alignment(
    reports: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if not reports:
        raise ValueError("formal curve evaluation requires reports")
    case_ids = tuple(str(value) for value in reports[0].get("case_ids", []))
    if not case_ids:
        raise ValueError("formal curve evaluation has no paired cases")
    if any(
        tuple(str(value) for value in report.get("case_ids", [])) != case_ids
        for report in reports[1:]
    ):
        raise ValueError("formal curve must use the same paired cases at every checkpoint")
    return case_ids


def validate_final_distribution_counts(
    report: Mapping[str, object],
    *,
    expected: int,
) -> None:
    nominal = report.get("nominal")
    recovery = report.get("recovery")
    if (
        not isinstance(nominal, Mapping)
        or not isinstance(recovery, Mapping)
        or int(nominal.get("episodes", -1)) != expected
        or int(recovery.get("episodes", -1)) != expected
    ):
        raise ValueError(
            f"formal final evaluation requires {expected} nominal and {expected} recovery episodes"
        )


def formal_policy_seed(base_seed: int, seed_index: int) -> int:
    if base_seed < 0 or not 0 <= seed_index < 3:
        raise ValueError("formal evaluation seed index must lie within [0, 2]")
    return int(
        np.random.SeedSequence(
            (base_seed, seed_index, 0x45564C32)
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _evaluation_run(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> FormalRun:
    return prepare_formal_run(
        config,
        condition=condition,
        seed_index=(0 if condition in CONSTANT_CONTROL_CONDITIONS else seed_index),
    )


def _load_snapshot_policy(
    config: RecoveryRLV2Config,
    run: FormalRun,
    runtime: Any,
    *,
    environment_steps: int,
    manifest: Any,
) -> Any:
    if run.condition in CONSTANT_CONTROL_CONDITIONS:
        return _ZeroResidualPolicy()
    snapshot_root = run.output_dir / "snapshots"
    payload = SnapshotStore(snapshot_root).load(
        step=environment_steps,
        expected_binding=run.binding,
        map_location=runtime.backend.device,
    )
    if payload.get("schema_version") != FORMAL_SCHEMA:
        raise ValueError("formal evaluation snapshot payload is incompatible")
    policy_state = payload.get("policy_state")
    if policy_state is not None:
        if not isinstance(policy_state, Mapping):
            raise ValueError("formal evaluation ACT policy state is incompatible")
        runtime.backend.policy.load_state_dict(policy_state, strict=False)
    algorithm = payload.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError("formal evaluation snapshot has no algorithm state")
    actor_state = algorithm.get("actor")
    if not isinstance(actor_state, Mapping):
        raise ValueError("formal evaluation snapshot has no residual actor")
    if run.condition == "oracle_state":
        actor = OracleResidualActor(config.oracle_state_dim)
        observation = "oracle"
    else:
        actor = LatentResidualActor(_example_latent_dim(runtime, manifest))
        observation = "latent"
    actor.load_state_dict(actor_state)
    actor.to(runtime.backend.device).eval()
    return _ActorEvaluationPolicy(
        actor,
        observation=observation,
        device=runtime.backend.device,
    )


def _decorate_report(
    report: EvaluationReport,
    *,
    run: FormalRun,
    evaluation_seed_index: int,
    environment_steps: int,
    partition: str,
) -> dict[str, object]:
    value = report.to_dict()
    rows = []
    for row in value["rows"]:
        rows.append(
            {
                "condition": run.condition,
                "training_seed_index": (
                    None if run.constant_control else run.seed_index
                ),
                "training_seed": None if run.constant_control else run.seed,
                "evaluation_seed_index": evaluation_seed_index,
                "environment_steps": environment_steps,
                "partition": partition,
                **row,
            }
        )
    result = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": run.condition,
        "training_seed_index": None if run.constant_control else run.seed_index,
        "training_seed": None if run.constant_control else run.seed,
        "evaluation_seed_index": evaluation_seed_index,
        "environment_steps": environment_steps,
        "partition": partition,
        "formal_binding": run.binding,
        "case_ids": value["case_ids"],
        "policy_seed": value["policy_seed"],
        "all": value["all"],
        "nominal": value["nominal"],
        "perturbation": value["perturbation"],
        "recovery": value["recovery"],
        "rows": rows,
    }
    if run.constant_control:
        result["constant_control"] = True
        result["referenced_evaluation_step"] = 0
    return result


def _write_report(path: Path, value: Mapping[str, object]) -> Path:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"formal evaluation report is immutable: {path}")
        return path
    write_json_atomic(path, dict(value))
    return path


def _evaluate_point(
    config: RecoveryRLV2Config,
    run: FormalRun,
    *,
    evaluation_seed_index: int,
    environment_steps: int,
    partition: str,
    cases: Sequence[RecoveryCase],
) -> dict[str, object]:
    destination = (
        config.output_dir
        / "formal"
        / "evaluations"
        / run.condition
        / f"seed_{evaluation_seed_index}"
        / partition
        / f"step_{environment_steps:06d}"
        / "report.json"
    )
    if destination.is_file():
        report = json.loads(destination.read_text(encoding="utf-8"))
        if report.get("formal_binding") != run.binding:
            raise ValueError("formal evaluation point binding differs")
        if tuple(report.get("case_ids", ())) != tuple(
            case.case_id for case in cases
        ):
            raise ValueError("formal evaluation point case set differs")
        return report
    residual_coefficient, _, _ = _anchor_coefficients(config, run.anchoring)
    checkpoint = (
        run.parent_checkpoint
        if run.condition in CONSTANT_CONTROL_CONDITIONS
        else config.sft_checkpoint
    )
    manifest = load_case_manifest(
        config.output_dir / "manifests" / "cases.json"
    )
    policy_seed = formal_policy_seed(config.seed, evaluation_seed_index)
    with _runtime(
        config,
        seed=policy_seed,
        residual_coefficient=residual_coefficient,
        checkpoint=checkpoint,
    ) as runtime:
        policy = _load_snapshot_policy(
            config,
            run,
            runtime,
            environment_steps=environment_steps,
            manifest=manifest,
        )
        evaluation = evaluate_case_manifest(
            policy,
            runtime,
            cases,
            policy_seed=policy_seed,
        )
    report = _decorate_report(
        evaluation,
        run=run,
        evaluation_seed_index=evaluation_seed_index,
        environment_steps=environment_steps,
        partition=partition,
    )
    _write_report(destination, report)
    return report


def evaluate_formal_timeline(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> dict[str, object]:
    run = _evaluation_run(
        config, condition=condition, seed_index=seed_index
    )
    if not run.constant_control:
        training = run.output_dir / "training_report.json"
        if not training.is_file():
            raise FileNotFoundError(f"formal training report not found: {training}")
        training_report = json.loads(training.read_text(encoding="utf-8"))
        if (
            training_report.get("passed") is not True
            or training_report.get("binding") != run.binding
        ):
            raise ValueError("formal training report is incompatible")
    manifest = load_case_manifest(
        config.output_dir / "manifests" / "cases.json"
    )
    cases = paired_evaluation_cases(
        manifest,
        partition="curve",
        count=config.distribution.curve_case_count,
    )
    points: list[dict[str, object]] = []
    if run.constant_control:
        measured = _evaluate_point(
            config,
            run,
            evaluation_seed_index=seed_index,
            environment_steps=0,
            partition="curve",
            cases=cases,
        )
        for step in config.snapshot_steps:
            if step == 0:
                points.append(measured)
                continue
            point = json.loads(json.dumps(measured))
            point["environment_steps"] = step
            point["constant_control"] = True
            point["referenced_evaluation_step"] = 0
            for row in point["rows"]:
                row["environment_steps"] = step
            destination = (
                config.output_dir
                / "formal"
                / "evaluations"
                / run.condition
                / f"seed_{seed_index}"
                / "curve"
                / f"step_{step:06d}"
                / "report.json"
            )
            _write_report(destination, point)
            points.append(point)
    else:
        progress = tqdm(
            config.snapshot_steps,
            desc=f"formal/{condition}/seed={seed_index} curve",
            unit="checkpoint",
            dynamic_ncols=True,
        )
        for step in progress:
            points.append(
                _evaluate_point(
                    config,
                    run,
                    evaluation_seed_index=seed_index,
                    environment_steps=step,
                    partition="curve",
                    cases=cases,
                )
            )
    case_ids = validate_curve_case_alignment(points)
    destination = (
        config.output_dir
        / "formal"
        / "evaluations"
        / condition
        / f"seed_{seed_index}"
        / "curve_report.json"
    )
    report = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": condition,
        "training_seed_index": None if run.constant_control else run.seed_index,
        "evaluation_seed_index": seed_index,
        "formal_binding": run.binding,
        "case_ids": list(case_ids),
        "case_count_per_distribution": config.distribution.curve_case_count,
        "points": points,
        "point_report_hashes": [
            sha256_file(
                config.output_dir
                / "formal"
                / "evaluations"
                / condition
                / f"seed_{seed_index}"
                / "curve"
                / f"step_{step:06d}"
                / "report.json"
            )
            for step in config.snapshot_steps
        ],
    }
    _write_report(destination, report)
    return report


def evaluate_formal_final(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> dict[str, object]:
    run = _evaluation_run(
        config, condition=condition, seed_index=seed_index
    )
    manifest = load_case_manifest(
        config.output_dir / "manifests" / "cases.json"
    )
    cases = paired_evaluation_cases(
        manifest,
        partition="final",
        count=config.distribution.final_case_count,
    )
    report = _evaluate_point(
        config,
        run,
        evaluation_seed_index=seed_index,
        environment_steps=config.formal_steps,
        partition="final",
        cases=cases,
    )
    validate_final_distribution_counts(
        report,
        expected=config.distribution.final_case_count,
    )
    return report


def evaluate_formal_run(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> dict[str, object]:
    curve = evaluate_formal_timeline(
        config, condition=condition, seed_index=seed_index
    )
    final = evaluate_formal_final(
        config, condition=condition, seed_index=seed_index
    )
    destination = (
        config.output_dir
        / "formal"
        / "evaluations"
        / condition
        / f"seed_{seed_index}"
        / "report.json"
    )
    report = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": condition,
        "seed_index": seed_index,
        "formal_binding": curve["formal_binding"],
        "curve_report": (
            destination.parent / "curve_report.json"
        ).as_posix(),
        "curve_report_sha256": sha256_file(
            destination.parent / "curve_report.json"
        ),
        "final_report": (
            destination.parent
            / "final"
            / f"step_{config.formal_steps:06d}"
            / "report.json"
        ).as_posix(),
        "final_report_sha256": sha256_file(
            destination.parent
            / "final"
            / f"step_{config.formal_steps:06d}"
            / "report.json"
        ),
        "curve_points": len(curve["points"]),
        "final_nominal": final["nominal"],
        "final_recovery": final["recovery"],
    }
    _write_report(destination, report)
    return report
