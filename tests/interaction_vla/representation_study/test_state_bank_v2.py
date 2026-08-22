from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from interaction_vla.representation_study.state_bank.v2_builder import (
    PRIMARY_STRATA,
    StateBankV2Candidate,
    build_state_bank_v2,
    load_v2_split,
    _next_relation_from_oracle,
    _inspect_existing,
    _write_collection_report,
)
from interaction_vla.lerobot_bridge.teacher_schema import (
    OPERATOR_IDS,
    PREDICATE_IDS,
    RELATION_TYPE_IDS,
)


def _candidates() -> tuple[StateBankV2Candidate, ...]:
    counts = {"train": 300, "validation": 80, "test": 80}
    result: list[StateBankV2Candidate] = []
    cursor = 0
    for split_index, (partition, count) in enumerate(counts.items()):
        for family_index, family in enumerate(PRIMARY_STRATA):
            for index in range(count):
                source = split_index * 10_000 + family_index * 1000 + index // 10
                image = np.full((8, 8, 3), cursor % 255, dtype=np.uint8)
                result.append(
                    StateBankV2Candidate(
                        candidate_id=f"candidate-{cursor:05d}",
                        case_id=f"case-{source}-{family}",
                        source_seed=source,
                        partition=partition,
                        family=family,
                        intervention_kind=("nominal" if family == "nominal" else f"{family}_kind_{index % 3}"),
                        step=index,
                        phase=index % 6,
                        robot_state=np.full(10, index, dtype=np.float32),
                        oracle_state=np.full(36, index / 100.0, dtype=np.float32),
                        agent_rgb=image,
                        wrist_rgb=image,
                        labels={
                            "geometry": [float(index % 7)] * 16,
                            "phase": index % 6,
                            "recovery_state": family_index,
                            "recovery_type": index % 7,
                            "next_relation": {
                                "relation_id": index % 8,
                                "operator_id": index % 5,
                                "predicate_id": index % 7,
                            },
                            "contact": index % 2,
                            "stable_grasp": (index // 2) % 2,
                        },
                    )
                )
                cursor += 1
    return tuple(result)


def test_state_bank_v2_has_exact_primary_balance(tmp_path: Path) -> None:
    report = build_state_bank_v2(
        _candidates(),
        output_dir=tmp_path / "bank",
        manifest_hash="a" * 64,
        seed=11,
    )
    assert report.record_count == 1200
    assert report.stratum_counts == {
        "nominal": 400,
        "perturbation": 400,
        "recovery": 400,
    }


def test_state_bank_v2_has_no_source_seed_overlap(tmp_path: Path) -> None:
    build_state_bank_v2(
        _candidates(),
        output_dir=tmp_path / "bank",
        manifest_hash="a" * 64,
        seed=11,
    )
    split = load_v2_split(tmp_path / "bank" / "split.json")
    assert split.source_seeds("train").isdisjoint(split.source_seeds("validation"))
    assert split.source_seeds("train").isdisjoint(split.source_seeds("test"))
    assert split.source_seeds("validation").isdisjoint(split.source_seeds("test"))


def test_existing_state_bank_must_match_bound_case_manifest(tmp_path: Path) -> None:
    build_state_bank_v2(
        _candidates(),
        output_dir=tmp_path / "bank",
        manifest_hash="a" * 64,
        seed=11,
    )
    with pytest.raises(ValueError, match="bound case manifest"):
        _inspect_existing(
            tmp_path / "bank", expected_case_manifest_sha256="b" * 64
        )


def test_fresh_collection_report_uses_validated_foundation_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bank"
    root.mkdir()
    path = _write_collection_report(
        root,
        foundation_binding="b" * 64,
        case_manifest_sha256="c" * 64,
        rejected_cases=(),
        registered_interventions=("nominal",),
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["binding"] == "b" * 64
    assert report["source_case_manifest_sha256"] == "c" * 64


def test_state_bank_v2_preserves_multiple_test_recovery_groups(tmp_path: Path) -> None:
    build_state_bank_v2(
        _candidates(),
        output_dir=tmp_path / "bank",
        manifest_hash="a" * 64,
        seed=11,
    )
    split = load_v2_split(tmp_path / "bank" / "split.json")
    assert len(split.source_seeds("test", family="recovery")) >= 3


def test_next_relation_is_a_task_relation_triple_not_a_phase_alias() -> None:
    approach = np.zeros(36, dtype=np.float32)
    carrying = np.zeros(36, dtype=np.float32)
    carrying[13] = 0.8
    carrying[17:19] = 1.0

    assert _next_relation_from_oracle(approach) == {
        "relation_id": RELATION_TYPE_IDS["gripper_to_target"],
        "operator_id": OPERATOR_IDS["establish"],
        "predicate_id": PREDICATE_IDS["proximity"],
    }
    assert _next_relation_from_oracle(carrying) == {
        "relation_id": RELATION_TYPE_IDS["target_to_receptacle"],
        "operator_id": OPERATOR_IDS["establish"],
        "predicate_id": PREDICATE_IDS["containment"],
    }

    released = np.zeros(36, dtype=np.float32)
    released[13] = 0.10
    released[16] = 1.0
    released[19] = 1.0
    assert _next_relation_from_oracle(released) == {
        "relation_id": RELATION_TYPE_IDS["gripper_to_receptacle"],
        "operator_id": OPERATOR_IDS["increase"],
        "predicate_id": PREDICATE_IDS["clearance"],
    }
