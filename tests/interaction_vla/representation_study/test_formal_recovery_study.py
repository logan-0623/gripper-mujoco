from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from interaction_vla.representation_study.rl.formal import (
    FORMAL_CONDITIONS,
    FORMAL_TRAINING_CONDITIONS,
    build_constant_control_timeline,
    formal_matrix,
    prepare_formal_run,
)
from interaction_vla.representation_study.rl.formal_evaluation import (
    validate_curve_case_alignment,
    validate_final_distribution_counts,
)
from interaction_vla.representation_study.rl.protocol import GATE_SCHEMA

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


def _write_gate(root: Path, name: str, *, binding: str) -> None:
    inputs: dict[str, object] = {
        "binding": binding,
        "case_manifest_sha256": "a" * 64,
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
    if name == "oracle":
        inputs["selected_backend"] = "ppo"
    if name == "anchoring":
        inputs.update(
            {"selected_backend": "ppo", "selected_variant": "full_anchoring"}
        )
    path = root / "gates" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate), encoding="utf-8")


def _write_required_manifests(config, *, binding: str) -> None:
    root = config.output_dir
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    cases = {"schema_version": "recovery_case_manifest_v2", "sha256": "a" * 64}
    (manifests / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    normalization = {"schema_version": "compact_oracle_normalization_v1"}
    (manifests / "oracle_normalization.json").write_text(
        json.dumps(normalization), encoding="utf-8"
    )
    bank = root / "state_bank_v2"
    bank.mkdir(parents=True, exist_ok=True)
    artifact = bank / "records.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    (bank / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "recovery_state_bank_v2",
                "source_case_manifest_sha256": "a" * 64,
                "artifact_hashes": {"records.jsonl": _sha256(artifact)},
            }
        ),
        encoding="utf-8",
    )
    for name in ("distribution", "backend", "oracle", "anchoring"):
        _write_gate(root, name, binding=binding)


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


def test_final_evaluation_has_fifty_cases_per_distribution() -> None:
    report = {
        "nominal": {"episodes": 50},
        "recovery": {"episodes": 50},
    }
    validate_final_distribution_counts(report, expected=50)
    report["recovery"]["episodes"] = 49
    with pytest.raises(ValueError, match="50 nominal and 50 recovery"):
        validate_final_distribution_counts(report, expected=50)
