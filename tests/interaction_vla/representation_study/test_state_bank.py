from __future__ import annotations

import json

import pytest

from interaction_vla.representation_study.schemas.stages import ArtifactBinding
from interaction_vla.representation_study.state_bank.schema import (
    CanonicalJson,
    ObservationReference,
    StateBankManifest,
    StateBankRecord,
    StateBankSplit,
)
from interaction_vla.representation_study.state_bank.validation import (
    validate_state_bank,
)


def _binding(name: str) -> ArtifactBinding:
    return ArtifactBinding(uri=name, sha256="c" * 64)


def _record(
    state_id: str,
    episode: str,
    row: int,
    *,
    stratum: str,
    domain: str,
) -> StateBankRecord:
    return StateBankRecord(
        state_id=state_id,
        source_episode_id=episode,
        split_group_id=episode,
        frame_index=row,
        task_id="pick_place",
        stratum=stratum,
        domain=domain,
        phase="approach",
        seed=17,
        instruction="pick up the red cube",
        observation=ObservationReference(
            source_uri="local/franka_lerobot_act_pilot",
            source_index=row,
            agent_rgb_key="observation.images.agent",
            wrist_rgb_key="observation.images.wrist",
        ),
        robot_state=(0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0),
        privileged_state=CanonicalJson.from_value(
            {"target_position": [0.4, 0.1, 0.03], "contact": False}
        ),
        ontology_labels=CanonicalJson.from_value(
            {"target": "red_cube", "recovery_needed": domain == "policy_shift"}
        ),
        provenance=CanonicalJson.from_value({"collector": "mujoco", "seed": 17}),
    )


def _records() -> tuple[StateBankRecord, ...]:
    return (
        _record("train-nominal", "episode-0", 0, stratum="nominal", domain="expert_support"),
        _record(
            "validation-perturbation",
            "episode-1",
            1,
            stratum="perturbation",
            domain="policy_shift",
        ),
        _record(
            "test-recovery",
            "episode-2",
            2,
            stratum="recovery",
            domain="policy_shift",
        ),
    )


def _split() -> StateBankSplit:
    return StateBankSplit(
        train=("train-nominal",),
        validation=("validation-perturbation",),
        test=("test-recovery",),
    )


def test_canonical_json_is_order_invariant_and_rejects_nonfinite_values() -> None:
    first = CanonicalJson.from_value({"b": [2, 1], "a": True})
    second = CanonicalJson.from_value({"a": True, "b": [2, 1]})

    assert first == second
    assert first.to_value() == {"a": True, "b": [2, 1]}
    with pytest.raises(ValueError, match="finite"):
        CanonicalJson.from_value({"bad": float("nan")})


def test_state_bank_record_validates_scientific_axes_and_robot_state() -> None:
    record = _records()[0]
    assert record.stratum == "nominal"
    assert record.domain == "expert_support"
    assert record.to_dict()["ontology_labels"]["target"] == "red_cube"

    with pytest.raises(ValueError, match="stratum"):
        _record("bad", "episode-3", 3, stratum="unknown", domain="policy_shift")
    with pytest.raises(ValueError, match="robot_state"):
        StateBankRecord.from_dict(
            {**record.to_dict(), "state_id": "bad-state", "robot_state": [float("inf")]}
        )


def test_state_bank_validation_enforces_episode_level_splits() -> None:
    report = validate_state_bank(_records(), _split())

    assert report["passed"] is True
    assert report["episode_overlap"] is False
    assert report["record_count"] == 3
    assert report["stratum_counts"] == {
        "nominal": 1,
        "perturbation": 1,
        "recovery": 1,
        "terminal": 0,
    }
    assert report["partition_episodes"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert report["partition_groups"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_state_bank_validation_rejects_duplicate_rows_and_episode_leakage() -> None:
    records = list(_records())
    records[1] = StateBankRecord.from_dict(
        {
            **records[1].to_dict(),
            "observation": records[0].observation.to_dict(),
        }
    )
    with pytest.raises(ValueError, match="observation reference"):
        validate_state_bank(records, _split())

    records = list(_records())
    records[2] = StateBankRecord.from_dict(
        {**records[2].to_dict(), "source_episode_id": "episode-0"}
    )
    with pytest.raises(ValueError, match="source episode leakage"):
        validate_state_bank(records, _split())

    records = list(_records())
    records[2] = StateBankRecord.from_dict(
        {**records[2].to_dict(), "split_group_id": "episode-0"}
    )
    with pytest.raises(ValueError, match="split group leakage"):
        validate_state_bank(records, _split())


def test_state_bank_manifest_round_trip_binds_all_source_artifacts() -> None:
    manifest = StateBankManifest(
        bank_id="icra_state_bank_v1",
        dataset=_binding("dataset"),
        ontology=_binding("ontology"),
        source=_binding("source-tree"),
        selection_config=_binding("configs/representation_study/base.yaml"),
        records_sha256="d" * 64,
        record_count=3,
        state_dim=7,
        split=_split(),
    )

    restored = StateBankManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert json.loads(manifest.to_json()) == manifest.to_dict()
    assert len(manifest.sha256()) == 64
