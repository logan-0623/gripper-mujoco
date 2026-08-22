from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.env import TerminationReason
from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file

from ..state_bank.io import write_json_atomic
from .actors import LatentResidualActor, OracleResidualActor
from .distributions import RecoveryCase, load_case_manifest
from .evaluation_v2 import (
    EpisodeOutcome,
    EvaluationReport,
    _aggregate,
    _episode_seed,
    evaluate_case_manifest,
)
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


def _policy_artifact_metadata(
    run: FormalRun, *, environment_steps: int
) -> dict[str, object]:
    if run.constant_control:
        checkpoint = Path(run.parent_checkpoint)
        digest = fingerprint_tree(checkpoint) if checkpoint.is_dir() else sha256_file(checkpoint)
        return {"checkpoint_sha256": digest, "snapshot_sha256": None}
    snapshot = run.output_dir / "snapshots" / f"step_{environment_steps:06d}"
    inspected = SnapshotStore(snapshot.parent).inspect(
        step=environment_steps, expected_binding=run.binding
    )
    return {
        "checkpoint_sha256": None,
        "snapshot_sha256": inspected["snapshot_sha256"],
    }


def validate_evaluation_point(
    report: Mapping[str, object],
    *,
    run: FormalRun,
    evaluation_seed_index: int,
    environment_steps: int,
    partition: str,
    cases: Sequence[RecoveryCase],
    policy_seed: int,
    policy_artifact: Mapping[str, object],
) -> None:
    expected = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": run.condition,
        "training_seed_index": None if run.constant_control else run.seed_index,
        "training_seed": None if run.constant_control else run.seed,
        "evaluation_seed_index": evaluation_seed_index,
        "environment_steps": environment_steps,
        "partition": partition,
        "formal_binding": run.binding,
        "policy_seed": policy_seed,
        "checkpoint_sha256": policy_artifact.get("checkpoint_sha256"),
        "snapshot_sha256": policy_artifact.get("snapshot_sha256"),
    }
    differing = [name for name, value in expected.items() if report.get(name) != value]
    if differing:
        raise ValueError(
            "formal evaluation point metadata differs: " + ", ".join(differing)
        )
    if tuple(report.get("case_ids", ())) != tuple(case.case_id for case in cases):
        raise ValueError("formal evaluation point case set differs")
    _validate_episode_rows_and_aggregates(
        report,
        run=run,
        evaluation_seed_index=evaluation_seed_index,
        environment_steps=environment_steps,
        partition=partition,
        cases=cases,
        policy_seed=policy_seed,
    )


def _aggregate_matches(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if not isinstance(actual, Mapping):
        return False
    expected_values = asdict(expected)
    if set(actual) != set(expected_values):
        return False
    for name, value in expected_values.items():
        observed = actual[name]
        if isinstance(value, int):
            if type(observed) is not int or observed != value:
                return False
        elif not np.isfinite(float(observed)) or not np.isclose(
            float(observed), float(value), rtol=0.0, atol=1.0e-12
        ):
            return False
    return True


def _validate_episode_rows_and_aggregates(
    report: Mapping[str, object],
    *,
    run: FormalRun,
    evaluation_seed_index: int,
    environment_steps: int,
    partition: str,
    cases: Sequence[RecoveryCase],
    policy_seed: int,
) -> None:
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(cases):
        raise ValueError("formal evaluation must contain one episode row per case")
    outcome_fields = {field.name for field in fields(EpisodeOutcome)}
    decoration = {
        "condition",
        "training_seed_index",
        "training_seed",
        "evaluation_seed_index",
        "environment_steps",
        "partition",
    }
    outcomes: list[EpisodeOutcome] = []
    for raw, case in zip(raw_rows, cases, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != outcome_fields | decoration:
            raise ValueError("formal evaluation episode row schema differs")
        row_expected = {
            "condition": run.condition,
            "training_seed_index": None if run.constant_control else run.seed_index,
            "training_seed": None if run.constant_control else run.seed,
            "evaluation_seed_index": evaluation_seed_index,
            "environment_steps": environment_steps,
            "partition": partition,
            "case_id": case.case_id,
            "source_seed": case.source_seed,
            "variant_id": case.variant_id,
            "family": case.family,
            "intervention_kind": case.intervention_kind,
            "policy_seed": _episode_seed(policy_seed, case),
        }
        if any(raw.get(name) != value for name, value in row_expected.items()):
            raise ValueError("formal evaluation episode identity differs")
        success = raw.get("success")
        if type(success) is not bool:
            raise ValueError("formal evaluation success must be boolean")
        valid_reasons = {reason.value for reason in TerminationReason}
        termination_reason = str(raw.get("termination_reason", ""))
        if (
            termination_reason not in valid_reasons
            or success != (termination_reason == TerminationReason.SUCCESS.value)
        ):
            raise ValueError("formal evaluation outcome and termination reason differ")
        steps = raw.get("steps")
        if type(steps) is not int or steps < 1:
            raise ValueError("formal evaluation episode steps must be positive")
        numeric = (
            "episode_return",
            "reward_terminal",
            "reward_progress",
            "reward_residual",
            "mean_residual_norm",
            "action_clipping_rate",
            "action_smoothness",
            "mean_ik_projection_scale",
        )
        if any(not np.isfinite(float(raw[name])) for name in numeric):
            raise ValueError("formal evaluation episode metrics must be finite")
        if (
            float(raw["mean_residual_norm"]) < 0.0
            or float(raw["action_smoothness"]) < 0.0
            or not np.isclose(
                float(raw["episode_return"]),
                float(raw["reward_terminal"])
                + float(raw["reward_progress"])
                + float(raw["reward_residual"]),
                rtol=0.0,
                atol=1.0e-5,
            )
        ):
            raise ValueError("formal evaluation reward/action diagnostics differ")
        if not 0.0 <= float(raw["action_clipping_rate"]) <= 1.0:
            raise ValueError("formal evaluation clipping rate is outside [0, 1]")
        if not 0.0 <= float(raw["mean_ik_projection_scale"]) <= 1.0:
            raise ValueError("formal evaluation IK scale is outside [0, 1]")
        outcomes.append(
            EpisodeOutcome(**{name: raw[name] for name in outcome_fields})
        )
    if tuple(row.case_id for row in outcomes) != tuple(report["case_ids"]):
        raise ValueError("formal evaluation row order differs from case ids")
    if len({row.case_id for row in outcomes}) != len(outcomes):
        raise ValueError("formal evaluation contains duplicate episode rows")
    expected_aggregates = {
        "all": _aggregate(outcomes),
        **{
            family: (
                None
                if not (selected := [row for row in outcomes if row.family == family])
                else _aggregate(selected)
            )
            for family in ("nominal", "perturbation", "recovery")
        },
    }
    if any(
        not _aggregate_matches(report.get(name), expected)
        for name, expected in expected_aggregates.items()
    ):
        raise ValueError("formal evaluation aggregate does not match episode rows")


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
    policy_artifact: Mapping[str, object],
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
        "checkpoint_sha256": policy_artifact.get("checkpoint_sha256"),
        "snapshot_sha256": policy_artifact.get("snapshot_sha256"),
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
        validate_evaluation_point(
            report,
            run=run,
            evaluation_seed_index=evaluation_seed_index,
            environment_steps=environment_steps,
            partition=partition,
            cases=cases,
            policy_seed=formal_policy_seed(config.seed, evaluation_seed_index),
            policy_artifact=_policy_artifact_metadata(
                run, environment_steps=environment_steps
            ),
        )
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
    policy_artifact = _policy_artifact_metadata(
        run, environment_steps=environment_steps
    )
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
        policy_artifact=policy_artifact,
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


def validate_formal_evaluation_artifacts(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> dict[str, object]:
    """Validate cached curve/final reports against cases, seeds, and policy bytes."""
    run = _evaluation_run(config, condition=condition, seed_index=seed_index)
    manifest = load_case_manifest(config.output_dir / "manifests" / "cases.json")
    curve_cases = paired_evaluation_cases(
        manifest,
        partition="curve",
        count=config.distribution.curve_case_count,
    )
    final_cases = paired_evaluation_cases(
        manifest,
        partition="final",
        count=config.distribution.final_case_count,
    )
    root = (
        config.output_dir
        / "formal"
        / "evaluations"
        / condition
        / f"seed_{seed_index}"
    )
    curve_path = root / "curve_report.json"
    if not curve_path.is_file():
        raise FileNotFoundError(f"formal curve report not found: {curve_path}")
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    curve_expected = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": condition,
        "training_seed_index": None if run.constant_control else run.seed_index,
        "evaluation_seed_index": seed_index,
        "formal_binding": run.binding,
        "case_count_per_distribution": config.distribution.curve_case_count,
    }
    if any(curve.get(name) != value for name, value in curve_expected.items()):
        raise ValueError("formal curve report metadata differs")
    point_reports: list[dict[str, object]] = []
    point_hashes: list[str] = []
    policy_seed = formal_policy_seed(config.seed, seed_index)
    for step in config.snapshot_steps:
        point_path = root / "curve" / f"step_{step:06d}" / "report.json"
        if not point_path.is_file():
            raise FileNotFoundError(f"formal curve point not found: {point_path}")
        point = json.loads(point_path.read_text(encoding="utf-8"))
        validate_evaluation_point(
            point,
            run=run,
            evaluation_seed_index=seed_index,
            environment_steps=step,
            partition="curve",
            cases=curve_cases,
            policy_seed=policy_seed,
            policy_artifact=_policy_artifact_metadata(
                run, environment_steps=step
            ),
        )
        point_reports.append(point)
        point_hashes.append(sha256_file(point_path))
    validate_curve_case_alignment(point_reports)
    if curve.get("points") != point_reports or curve.get("point_report_hashes") != point_hashes:
        raise ValueError("formal curve aggregate differs from point artifacts")

    final_path = root / "final" / f"step_{config.formal_steps:06d}" / "report.json"
    if not final_path.is_file():
        raise FileNotFoundError(f"formal final report not found: {final_path}")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    validate_evaluation_point(
        final,
        run=run,
        evaluation_seed_index=seed_index,
        environment_steps=config.formal_steps,
        partition="final",
        cases=final_cases,
        policy_seed=policy_seed,
        policy_artifact=_policy_artifact_metadata(
            run, environment_steps=config.formal_steps
        ),
    )
    validate_final_distribution_counts(
        final, expected=config.distribution.final_case_count
    )
    report_path = root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"formal evaluation summary not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_summary = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": condition,
        "seed_index": seed_index,
        "formal_binding": run.binding,
        "curve_report": curve_path.as_posix(),
        "curve_report_sha256": sha256_file(curve_path),
        "final_report": final_path.as_posix(),
        "final_report_sha256": sha256_file(final_path),
        "curve_points": len(config.snapshot_steps),
        "final_nominal": final["nominal"],
        "final_recovery": final["recovery"],
    }
    if any(report.get(name) != value for name, value in expected_summary.items()):
        raise ValueError("formal evaluation summary binding or hashes differ")
    return report
