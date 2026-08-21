from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from interaction_vla.representation_study.rl.v2_config import (
    RecoveryRLV2Config,
    load_recovery_rl_v2_config,
)


def minimal_v2_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "recovery_rl_v2",
        "output_dir": str(tmp_path / "outputs"),
        "bridge_config": "bridge.yaml",
        "sft_checkpoint": "sft/checkpoint",
        "continued_sft_checkpoint": "continued/checkpoint",
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
        "residual_scale": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
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


def test_v2_config_rejects_v1_output_root(tmp_path: Path) -> None:
    raw = minimal_v2_mapping(tmp_path)
    raw["output_dir"] = "outputs/representation_study/icra"
    with pytest.raises(ValueError, match="immutable v1 root"):
        RecoveryRLV2Config.from_mapping(raw, config_path=tmp_path / "v2.yaml")


def test_v2_config_fixes_distribution_and_snapshot_contract(tmp_path: Path) -> None:
    config = RecoveryRLV2Config.from_mapping(
        minimal_v2_mapping(tmp_path), config_path=tmp_path / "v2.yaml"
    )
    assert config.distribution.probabilities == pytest.approx((0.50, 0.30, 0.20))
    assert config.snapshot_steps == (0, 4096, 8192, 12288, 16384, 20480)
    assert config.oracle_state_dim == 36
    assert config.ppo.rollout_steps == 256
    assert config.sac.target_entropy == -7.0
    assert config.representation_learning_rate == 1.0e-5


def test_v2_config_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    raw = minimal_v2_mapping(tmp_path)
    assert isinstance(raw["sac"], dict)
    raw["sac"]["mystery"] = True
    with pytest.raises(ValueError, match="unknown sac fields: mystery"):
        RecoveryRLV2Config.from_mapping(raw, config_path=tmp_path / "v2.yaml")


def test_v2_config_rejects_invalid_probability_mass(tmp_path: Path) -> None:
    raw = minimal_v2_mapping(tmp_path)
    assert isinstance(raw["distribution"], dict)
    raw["distribution"]["nominal_probability"] = 0.40
    with pytest.raises(ValueError, match="sum to one"):
        RecoveryRLV2Config.from_mapping(raw, config_path=tmp_path / "v2.yaml")


def test_load_v2_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "recovery.yaml"
    path.write_text(yaml.safe_dump(minimal_v2_mapping(tmp_path)), encoding="utf-8")
    loaded = load_recovery_rl_v2_config(path)
    assert loaded.config_path == path
    assert loaded.continued_sft_checkpoint == "continued/checkpoint"


def test_checked_in_profiles_are_output_isolated() -> None:
    root = Path(__file__).resolve().parents[3]
    mac = load_recovery_rl_v2_config(
        root / "configs/representation_study/recovery_rl_v2_act_macos.yaml"
    )
    cuda = load_recovery_rl_v2_config(
        root / "configs/representation_study/recovery_rl_v2_act_linux_cuda.yaml"
    )
    assert mac.output_dir == Path("outputs/representation_study/icra_rl_v2")
    assert cuda.output_dir == Path("outputs/representation_study/icra_rl_v2_cuda")
    assert mac.device == "auto"
    assert cuda.device == "cuda"
    assert mac.output_dir != cuda.output_dir
