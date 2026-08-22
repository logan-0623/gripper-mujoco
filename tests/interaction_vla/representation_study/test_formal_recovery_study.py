from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from interaction_vla.representation_study.rl.formal import (
    FORMAL_CONDITIONS,
    FORMAL_TRAINING_CONDITIONS,
    build_constant_control_timeline,
    formal_matrix,
    prepare_formal_run,
    FormalRun,
)
from interaction_vla.representation_study.rl.formal_evaluation import (
    FORMAL_EVALUATION_SCHEMA,
    _decorate_report,
    validate_evaluation_point,
    validate_curve_case_alignment,
    validate_final_distribution_counts,
)
from interaction_vla.representation_study.rl.evaluation_v2 import (
    EVALUATION_V2_SCHEMA,
    EpisodeOutcome,
    EvaluationReport,
    _aggregate,
    _episode_seed,
)
from interaction_vla.representation_study.rl.protocol import GATE_SCHEMA
from interaction_vla.representation_study.rl.foundation import (
    _packed_replay_observations,
    replay_policy_seeds,
)
from interaction_vla.representation_study.rl.replay import ReplayBatch
from interaction_vla.representation_study.rl.distributions import (
    build_case_manifest,
    save_case_manifest,
)
from interaction_vla.representation_study.rl.oracle_state import OracleNormalization

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_config(tmp_path: Path):
    from interaction_vla.representation_study.rl.v2_config import RecoveryRLV2Config

    config_path = tmp_path / "recovery.yaml"
    bridge = tmp_path / "bridge.yaml"
    sft = tmp_path / "sft"
    continued = tmp_path / "continued"
    bridge.write_text("dataset: test\n", encoding="utf-8")
    sft.mkdir()
    continued.mkdir()
    (sft / "weights.bin").write_bytes(b"sft")
    (continued / "weights.bin").write_bytes(b"continued")
    raw = {
        "schema_version": "recovery_rl_v2",
        "output_dir": str(tmp_path / "study"),
        "bridge_config": str(bridge),
        "sft_checkpoint": str(sft),
        "continued_sft_checkpoint": str(continued),
        "device": "cpu",
        "distribution": {
            "recovery_probability": 0.50,
            "perturbation_probability": 0.30,
            "nominal_probability": 0.20,
            "calibration_seed_count": 50,
            "training_seed_count": 200,
            "curve_case_count": 30,
            "final_case_count": 50,
            "severity_candidates": [0.50, 0.75, 1.00],
            "minimum_accepted_per_kind": 10,
        },
        "screen_steps": 8192,
        "formal_steps": 20480,
        "snapshot_steps": [0, 4096, 8192, 12288, 16384, 20480],
        "oracle_state_dim": 36,
        "residual_scale": [0.10] * 6 + [0.20],
        "gamma": 0.99,
        "progress_coefficient": 0.10,
        "residual_coefficient": 0.01,
        "nominal_anchor_coefficient": 1.0,
        "latent_anchor_coefficient": 0.10,
        "representation_learning_rate": 1.0e-5,
        "ppo": {
            "rollout_steps": 256,
            "update_epochs": 4,
            "minibatch_size": 64,
            "actor_learning_rate": 3.0e-4,
            "value_learning_rate": 3.0e-4,
            "gae_lambda": 0.95,
            "clip_coefficient": 0.20,
            "entropy_coefficient": 0.01,
            "max_grad_norm": 1.0,
        },
        "sac": {
            "replay_capacity": 100000,
            "warmup_steps": 1024,
            "batch_size": 256,
            "actor_learning_rate": 3.0e-4,
            "critic_learning_rate": 3.0e-4,
            "temperature_learning_rate": 3.0e-4,
            "polyak": 0.005,
            "updates_per_environment_step": 1,
            "target_entropy": -7.0,
        },
        "seed": 2057736129,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return RecoveryRLV2Config.from_mapping(raw, config_path=config_path)


def _write_gate(
    root: Path,
    name: str,
    *,
    binding: str,
    case_hash: str = "a" * 64,
    normalization_hash: str | None = None,
) -> None:
    inputs: dict[str, object] = {
        "binding": binding,
        "case_manifest_sha256": case_hash,
    }
    gate: dict[str, object] = {
        "schema_version": GATE_SCHEMA,
        "gate": name,
        "passed": True,
        "reasons": [],
        "inputs": inputs,
    }
    if name == "backend":
        gate["selected_backend"] = "ppo"
        inputs["distribution_gate_sha256"] = _sha256(
            root / "gates" / "distribution.json"
        )
    if name == "distribution":
        inputs["oracle_normalization_sha256"] = normalization_hash
    if name == "oracle":
        inputs["selected_backend"] = "ppo"
        inputs["backend_gate_sha256"] = _sha256(
            root / "gates" / "backend.json"
        )
    if name == "anchoring":
        inputs.update(
            {
                "selected_backend": "ppo",
                "selected_variant": "full_anchoring",
                "oracle_gate_sha256": _sha256(root / "gates" / "oracle.json"),
            }
        )
    path = root / "gates" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate), encoding="utf-8")


def _write_required_manifests(config, *, binding: str) -> None:
    root = config.output_dir
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = build_case_manifest(
        seed=17, calibration=1, training=1, curve=1, final=1, severity=0.75
    )
    save_case_manifest(manifests / "cases.json", manifest)
    normalization_path = manifests / "oracle_normalization.json"
    normalization_path.write_text(
        json.dumps(OracleNormalization().to_json()), encoding="utf-8"
    )
    bank = root / "state_bank_v2"
    bank.mkdir(parents=True, exist_ok=True)
    artifact = bank / "records.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    (bank / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "recovery_state_bank_v2",
                "source_case_manifest_sha256": manifest.sha256,
                "artifact_hashes": {"records.jsonl": _sha256(artifact)},
            }
        ),
        encoding="utf-8",
    )
    for name in ("distribution", "backend", "oracle", "anchoring"):
        _write_gate(
            root,
            name,
            binding=binding,
            case_hash=manifest.sha256,
            normalization_hash=_sha256(normalization_path),
        )


def test_formal_matrix_has_registered_conditions_and_three_seeds() -> None:
    matrix = formal_matrix(base_seed=2057736129)
    assert matrix.conditions == FORMAL_CONDITIONS
    assert matrix.training_conditions == FORMAL_TRAINING_CONDITIONS
    assert len(matrix.training_seeds) == 3
    assert len(set(matrix.training_seeds)) == 3


def test_formal_training_requires_all_foundation_gates(tmp_path: Path) -> None:
    from interaction_vla.representation_study.rl.foundation import foundation_binding

    config = _formal_config(tmp_path)
    binding = foundation_binding(config)
    for name in ("distribution", "backend", "oracle"):
        _write_gate(config.output_dir, name, binding=binding)
    with pytest.raises(FileNotFoundError, match="anchoring gate"):
        prepare_formal_run(config, condition="rl_head", seed_index=0)


def test_formal_run_is_bound_to_selected_protocol(tmp_path: Path) -> None:
    from interaction_vla.representation_study.rl.foundation import foundation_binding

    config = _formal_config(tmp_path)
    binding = foundation_binding(config)
    _write_required_manifests(config, binding=binding)
    run = prepare_formal_run(config, condition="rl_representation", seed_index=2)
    assert run.backend == "ppo"
    assert run.anchoring == "full_anchoring"
    assert run.trainable_groups == ("fusion",)
    assert run.output_dir == config.output_dir / "formal/runs/rl_representation/seed_2"
    assert len(run.binding) == 64


def test_formal_run_rejects_semantically_replaced_case_manifest(tmp_path: Path) -> None:
    from interaction_vla.representation_study.rl.foundation import foundation_binding

    config = _formal_config(tmp_path)
    binding = foundation_binding(config)
    _write_required_manifests(config, binding=binding)
    path = config.output_dir / "manifests" / "cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_hash"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic hash"):
        prepare_formal_run(config, condition="rl_head", seed_index=0)


def test_formal_run_rejects_replaced_oracle_normalization(tmp_path: Path) -> None:
    from interaction_vla.representation_study.rl.foundation import foundation_binding

    config = _formal_config(tmp_path)
    binding = foundation_binding(config)
    _write_required_manifests(config, binding=binding)
    path = config.output_dir / "manifests" / "oracle_normalization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["distance_scale"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization hash"):
        prepare_formal_run(config, condition="rl_head", seed_index=0)


def test_constant_control_timeline_reuses_parent_without_copy(tmp_path: Path) -> None:
    from interaction_vla.representation_study.rl.foundation import foundation_binding

    config = _formal_config(tmp_path)
    binding = foundation_binding(config)
    _write_required_manifests(config, binding=binding)
    run = prepare_formal_run(config, condition="sft", seed_index=0)
    timeline = build_constant_control_timeline(config, run)
    assert timeline["constant_control"] is True
    assert [point["environment_steps"] for point in timeline["points"]] == list(
        config.snapshot_steps
    )
    assert {point["checkpoint"] for point in timeline["points"]} == {
        config.sft_checkpoint
    }
    assert not (run.output_dir / "snapshots").exists()


def test_formal_condition_rejects_extra_seed_for_constant_control(tmp_path: Path) -> None:
    from interaction_vla.representation_study.rl.foundation import foundation_binding

    config = _formal_config(tmp_path)
    binding = foundation_binding(config)
    _write_required_manifests(config, binding=binding)
    with pytest.raises(ValueError, match="seed_index=0"):
        prepare_formal_run(config, condition="continued_sft", seed_index=1)


def test_curve_evaluation_uses_same_cases_at_every_checkpoint() -> None:
    case_ids = ["nominal:1", "recovery:1"]
    reports = [
        {"environment_steps": step, "case_ids": case_ids}
        for step in (0, 4096, 8192, 12288, 16384, 20480)
    ]
    assert validate_curve_case_alignment(reports) == tuple(case_ids)
    reports[-1]["case_ids"] = ["nominal:2", "recovery:2"]
    with pytest.raises(ValueError, match="same paired cases"):
        validate_curve_case_alignment(reports)


def test_replay_reconstructs_distinct_current_and_next_policy_inputs() -> None:
    current_image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    next_image = np.full((1, 256, 256, 3), 255, dtype=np.uint8)
    batch = ReplayBatch(
        transition_ids=("transition-0",),
        case_ids=("case-0",),
        families=("recovery",),
        tasks=("pick the target",),
        agent_rgb=current_image,
        wrist_rgb=current_image,
        state=np.zeros((1, 10), dtype=np.float32),
        next_agent_rgb=next_image,
        next_wrist_rgb=next_image,
        next_state=np.ones((1, 10), dtype=np.float32),
        oracle_state=np.zeros((1, 36), dtype=np.float32),
        next_oracle_state=np.ones((1, 36), dtype=np.float32),
        actor_observation=np.zeros((1, 16), dtype=np.float32),
        next_actor_observation=np.ones((1, 16), dtype=np.float32),
        residual=np.zeros((1, 7), dtype=np.float32),
        reward=np.zeros(1, dtype=np.float32),
        done=np.zeros(1, dtype=np.float32),
    )

    current = _packed_replay_observations(batch)
    following = _packed_replay_observations(batch, next_observation=True)
    assert float(current[0]["observation.state"].sum()) == 0.0
    assert float(following[0]["observation.state"].sum()) == 10.0
    assert replay_policy_seeds(batch.transition_ids, next_observation=False) != (
        replay_policy_seeds(batch.transition_ids, next_observation=True)
    )


def test_cached_evaluation_rejects_stale_policy_seed() -> None:
    case = build_case_manifest(
        seed=29, calibration=1, training=1, curve=1, final=1, severity=0.75
    ).partition("curve")[0]
    run = FormalRun(
        condition="rl_head",
        seed_index=1,
        seed=101,
        backend="ppo",
        anchoring="full_anchoring",
        output_dir=Path("formal/runs/rl_head/seed_1"),
        binding="b" * 64,
        parent_checkpoint="sft",
        trainable_groups=(),
        constant_control=False,
    )
    report = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": "rl_head",
        "training_seed_index": 1,
        "training_seed": 101,
        "evaluation_seed_index": 2,
        "environment_steps": 4096,
        "partition": "curve",
        "formal_binding": "b" * 64,
        "policy_seed": 999,
        "checkpoint_sha256": None,
        "snapshot_sha256": "c" * 64,
        "case_ids": [case.case_id],
    }
    with pytest.raises(ValueError, match="policy_seed"):
        validate_evaluation_point(
            report,
            run=run,
            evaluation_seed_index=2,
            environment_steps=4096,
            partition="curve",
            cases=(case,),
            policy_seed=1000,
            policy_artifact={
                "checkpoint_sha256": None,
                "snapshot_sha256": "c" * 64,
            },
        )


def test_cached_evaluation_rejects_missing_episode_rows() -> None:
    case = build_case_manifest(
        seed=31, calibration=1, training=1, curve=1, final=1, severity=0.75
    ).partition("curve")[0]
    run = FormalRun(
        condition="rl_head",
        seed_index=0,
        seed=101,
        backend="ppo",
        anchoring="full_anchoring",
        output_dir=Path("formal/runs/rl_head/seed_0"),
        binding="b" * 64,
        parent_checkpoint="sft",
        trainable_groups=(),
        constant_control=False,
    )
    report = {
        "schema_version": FORMAL_EVALUATION_SCHEMA,
        "passed": True,
        "condition": "rl_head",
        "training_seed_index": 0,
        "training_seed": 101,
        "evaluation_seed_index": 0,
        "environment_steps": 4096,
        "partition": "curve",
        "formal_binding": "b" * 64,
        "policy_seed": 1000,
        "checkpoint_sha256": None,
        "snapshot_sha256": "c" * 64,
        "case_ids": [case.case_id],
        "rows": [],
    }
    with pytest.raises(ValueError, match="one episode row per case"):
        validate_evaluation_point(
            report,
            run=run,
            evaluation_seed_index=0,
            environment_steps=4096,
            partition="curve",
            cases=(case,),
            policy_seed=1000,
            policy_artifact={
                "checkpoint_sha256": None,
                "snapshot_sha256": "c" * 64,
            },
        )


def test_generated_evaluation_rows_and_aggregates_validate() -> None:
    case = build_case_manifest(
        seed=37, calibration=1, training=1, curve=1, final=1, severity=0.75
    ).partition("curve")[0]
    run = FormalRun(
        condition="rl_head",
        seed_index=0,
        seed=101,
        backend="ppo",
        anchoring="full_anchoring",
        output_dir=Path("formal/runs/rl_head/seed_0"),
        binding="b" * 64,
        parent_checkpoint="sft",
        trainable_groups=(),
        constant_control=False,
    )
    outcome = EpisodeOutcome(
        case_id=case.case_id,
        source_seed=case.source_seed,
        variant_id=case.variant_id,
        family=case.family,
        intervention_kind=case.intervention_kind,
        policy_seed=_episode_seed(1000, case),
        success=True,
        termination_reason="success",
        steps=12,
        episode_return=1.0,
        reward_terminal=1.0,
        reward_progress=0.1,
        reward_residual=-0.1,
        mean_residual_norm=0.2,
        action_clipping_rate=0.1,
        action_smoothness=0.3,
        mean_ik_projection_scale=0.9,
    )
    aggregate = _aggregate((outcome,))
    families = {
        name: aggregate if case.family == name else None
        for name in ("nominal", "perturbation", "recovery")
    }
    raw = EvaluationReport(
        schema_version=EVALUATION_V2_SCHEMA,
        policy_seed=1000,
        case_ids=(case.case_id,),
        rows=(outcome,),
        all=aggregate,
        nominal=families["nominal"],
        perturbation=families["perturbation"],
        recovery=families["recovery"],
    )
    artifact = {"checkpoint_sha256": None, "snapshot_sha256": "c" * 64}
    report = _decorate_report(
        raw,
        run=run,
        evaluation_seed_index=0,
        environment_steps=4096,
        partition="curve",
        policy_artifact=artifact,
    )

    validate_evaluation_point(
        report,
        run=run,
        evaluation_seed_index=0,
        environment_steps=4096,
        partition="curve",
        cases=(case,),
        policy_seed=1000,
        policy_artifact=artifact,
    )


def test_final_evaluation_has_fifty_cases_per_distribution() -> None:
    report = {
        "nominal": {"episodes": 50},
        "recovery": {"episodes": 50},
    }
    validate_final_distribution_counts(report, expected=50)
    report["recovery"]["episodes"] = 49
    with pytest.raises(ValueError, match="50 nominal and 50 recovery"):
        validate_final_distribution_counts(report, expected=50)
