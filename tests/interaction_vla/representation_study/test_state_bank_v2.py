from __future__ import annotations

from pathlib import Path

import numpy as np

from interaction_vla.representation_study.state_bank.v2_builder import (
    PRIMARY_STRATA,
    StateBankV2Candidate,
    build_state_bank_v2,
    load_v2_split,
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
                            "next_relation": index % 6,
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


def test_state_bank_v2_preserves_multiple_test_recovery_groups(tmp_path: Path) -> None:
    build_state_bank_v2(
        _candidates(),
        output_dir=tmp_path / "bank",
        manifest_hash="a" * 64,
        seed=11,
    )
    split = load_v2_split(tmp_path / "bank" / "split.json")
    assert len(split.source_seeds("test", family="recovery")) >= 3
