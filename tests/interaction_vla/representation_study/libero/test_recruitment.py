import numpy as np
import pytest
import torch
from torch import nn
from dataclasses import replace
import json

from interaction_vla.representation_study.libero.recruitment import (
    PRIMARY_CONDITIONS,
    FinalDenoisingDeltaHook,
    binary_raw_probe,
    consensus_direction,
    phase_stratum,
    same_norm_random_delta,
    specificity_gate,
    validate_probe_reconstruction,
    _action_sensitivity,
    _matched_training_audit,
)
from interaction_vla.representation_study.libero.config import load_libero_study_config


def test_primary_conditions_are_the_four_frozen_contrasts() -> None:
    assert PRIMARY_CONDITIONS == (
        "pretrained",
        "d25_u16070",
        "d100_u16617",
        "d100_u66470",
    )


def test_binary_probe_is_converted_from_standardized_to_raw_coordinates() -> None:
    direction, bias = binary_raw_probe(
        weight=np.asarray([[1.0, 2.0], [5.0, 8.0]]),
        bias=np.asarray([0.5, 1.5]),
        feature_mean=np.asarray([10.0, 20.0]),
        feature_scale=np.asarray([2.0, 4.0]),
    )
    assert np.allclose(direction, [2.0, 1.5])
    assert bias == pytest.approx(1.0 - np.dot([10.0, 20.0], [2.0, 1.5]))


def test_consensus_and_random_control_are_rank_one_orthogonal_and_same_norm() -> None:
    target = consensus_direction(
        np.asarray([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [3.0, 0.1, 0.0]])
    )
    delta = np.asarray([[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    random = same_norm_random_delta(delta, target_direction=target, seed=9)
    assert np.allclose(np.linalg.norm(random, axis=1), np.linalg.norm(delta, axis=1))
    assert np.allclose(random @ target, 0.0, atol=1e-7)
    assert np.allclose(random, same_norm_random_delta(delta, target_direction=target, seed=9))


def test_final_denoising_hook_changes_only_the_final_call_and_all_tokens() -> None:
    module = nn.Identity()
    delta = torch.tensor([[1.0, -2.0]])
    hook = FinalDenoisingDeltaHook(module, expected_calls=3, delta=delta)
    with hook:
        first = module(torch.zeros(1, 4, 2))
        second = module(torch.zeros(1, 4, 2))
        final = module(torch.zeros(1, 4, 2))
    assert torch.equal(first, torch.zeros_like(first))
    assert torch.equal(second, torch.zeros_like(second))
    assert torch.equal(final, delta[:, None, :].expand_as(final))
    assert hook.calls == 3


def test_probe_reconstruction_rejects_changed_held_out_scores() -> None:
    archived = {
        "paired_payload": {
            "state_ids": ["a", "b"],
            "replicates": [
                {"seed_offset": 0, "prediction": [0, 1], "score": [0.1, 0.9]}
            ],
        }
    }
    validate_probe_reconstruction(
        result={"test_prediction": [0, 1], "test_score": [0.1, 0.9]},
        test_state_ids=("a", "b"),
        archived_result=archived,
        seed_offset=0,
    )
    with pytest.raises(ValueError, match="score"):
        validate_probe_reconstruction(
            result={"test_prediction": [0, 1], "test_score": [0.2, 0.9]},
            test_state_ids=("a", "b"),
            archived_result=archived,
            seed_offset=0,
        )


def test_probe_reconstruction_allows_roundoff_for_continuous_predictions() -> None:
    archived = {
        "paired_payload": {
            "state_ids": ["a", "b"],
            "replicates": [
                {
                    "seed_offset": 0,
                    "prediction": [[0.1, 0.2], [0.3, 0.4]],
                    "score": None,
                }
            ],
        }
    }
    validate_probe_reconstruction(
        result={
            "test_prediction": [[0.1 + 2e-8, 0.2], [0.3, 0.4 - 2e-8]],
            "test_score": None,
        },
        test_state_ids=("a", "b"),
        archived_result=archived,
        seed_offset=0,
    )
    with pytest.raises(ValueError, match="prediction"):
        validate_probe_reconstruction(
            result={
                "test_prediction": [[0.1 + 2e-5, 0.2], [0.3, 0.4]],
                "test_score": None,
            },
            test_state_ids=("a", "b"),
            archived_result=archived,
            seed_offset=0,
        )


def test_specificity_gate_fails_when_phase_changes_as_much_as_stable_grasp() -> None:
    failed = specificity_gate(
        target_minus_random={"estimate": 0.4, "ci_low": 0.2, "ci_high": 0.6},
        target_effect=1.0,
        non_target_effects={"phase": 1.0, "contact": 0.2},
        activation_norm_ratio=1.01,
        place_target_minus_random={"estimate": 0.1, "ci_low": -0.1, "ci_high": 0.3},
    )
    assert not failed["passed"]
    assert "phase" in " ".join(failed["failures"])


def test_phase_strata_are_preregistered() -> None:
    assert phase_stratum("approach") == "pre_contact"
    assert phase_stratum("contact") == "contact_grasp"
    assert phase_stratum("transport") == "post_grasp"
    assert phase_stratum("release_retreat") == "place_release"


def test_failed_specificity_blocks_policy_and_dataset_loading(tmp_path) -> None:
    config = replace(
        load_libero_study_config(
            "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
        ),
        output_dir=tmp_path,
    )
    report = (
        tmp_path
        / "protocol_v3"
        / "recruitment"
        / "stable_grasp"
        / "n_0064"
        / "specificity.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"passed": False}), encoding="utf-8")

    result = _action_sensitivity(config, max_states=64, batch_size=8)

    assert result["status"] == "blocked_by_specificity"


def test_matched_training_audit_allows_only_nested_coverage_and_step_differences() -> None:
    shared = {
        "seed": 7,
        "batch_size": 8,
        "num_workers": 4,
        "cudnn_deterministic": True,
        "optimizer": {"type": "adamw"},
        "scheduler": {"type": "cosine"},
        "policy": {"type": "smolvla", "freeze_vision_encoder": True},
        "rename_map": {"image": "camera1"},
    }
    d25 = {**shared, "steps": 16070, "dataset": {"repo_id": "x", "revision": "r", "episodes": [1, 2]}}
    d100 = {**shared, "steps": 16617, "dataset": {"repo_id": "x", "revision": "r", "episodes": [1, 2, 3]}}
    assert _matched_training_audit(
        d25, d100, d25_updates=16070, d100_updates=16617
    )["passed"]
    changed = {**d100, "optimizer": {"type": "sgd"}}
    assert not _matched_training_audit(
        d25, changed, d25_updates=16070, d100_updates=16617
    )["passed"]
