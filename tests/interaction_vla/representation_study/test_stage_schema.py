from __future__ import annotations

import json

import pytest

from interaction_vla.representation_study.schemas.stages import (
    ArtifactBinding,
    StageManifest,
    SUPPORTED_BACKENDS,
    SUPPORTED_STAGES,
)


def _binding(name: str) -> ArtifactBinding:
    return ArtifactBinding(uri=name, sha256="a" * 64)


def _manifest(**updates) -> StageManifest:
    values = {
        "study_id": "act_seed_0",
        "backend": "act",
        "stage": "sft",
        "checkpoint": _binding("checkpoint"),
        "config": _binding("config.yaml"),
        "dataset": _binding("dataset"),
        "source": _binding("source-tree"),
        "state_bank": _binding("state-bank"),
        "trainable_groups": ("policy",),
        "latent_taps": (
            "vision_backbone",
            "temporal_fused",
            "decoder_input",
            "pre_action",
        ),
    }
    values.update(updates)
    return StageManifest(**values)


def test_stage_schema_supports_all_policy_families_and_training_stages() -> None:
    assert SUPPORTED_BACKENDS == ("act", "smolvla", "pi0")
    assert SUPPORTED_STAGES == (
        "pretrained",
        "sft",
        "continued_sft",
        "rl_head",
        "rl_representation",
    )
    assert _manifest().backend == "act"
    assert _manifest(backend="smolvla").backend == "smolvla"
    assert _manifest(backend="pi0").backend == "pi0"


def test_artifact_binding_requires_content_hash_and_nonempty_uri() -> None:
    with pytest.raises(ValueError, match="uri"):
        ArtifactBinding(uri="", sha256="a" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ArtifactBinding(uri="checkpoint", sha256="not-a-hash")


def test_stage_manifest_rejects_unknown_stage_duplicate_taps_and_groups() -> None:
    with pytest.raises(ValueError, match="stage"):
        _manifest(stage="rl_everything")
    with pytest.raises(ValueError, match="latent taps"):
        _manifest(latent_taps=("vision_backbone", "vision_backbone"))
    with pytest.raises(ValueError, match="trainable groups"):
        _manifest(trainable_groups=("policy", "policy"))


def test_stage_manifest_round_trip_is_canonical_and_immutable() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()
    restored = StageManifest.from_dict(payload)

    assert restored == manifest
    assert json.loads(manifest.to_json()) == payload
    assert manifest.to_json() == restored.to_json()
    with pytest.raises(Exception):
        manifest.stage = "rl_head"


def test_pretrained_stage_forbids_trainable_groups() -> None:
    assert _manifest(stage="pretrained", trainable_groups=()).stage == "pretrained"
    with pytest.raises(ValueError, match="pretrained"):
        _manifest(stage="pretrained", trainable_groups=("policy",))

