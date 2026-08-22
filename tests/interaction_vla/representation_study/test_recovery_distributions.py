from __future__ import annotations

from pathlib import Path

import pytest

from interaction_vla.representation_study.rl.distributions import (
    RecoveryCaseSampler,
    build_case_manifest,
    load_case_manifest,
    save_case_manifest,
)


def _manifest():
    return build_case_manifest(
        seed=7,
        calibration=8,
        training=16,
        curve=6,
        final=6,
        severity=0.75,
    )


def test_source_seed_never_crosses_distribution_partitions() -> None:
    manifest = _manifest()
    groups = [set(manifest.source_seeds(name)) for name in manifest.partition_names]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    )
    assert {len(group) for group in groups} == {6, 8, 16}


def test_each_source_has_one_nominal_and_six_intervention_cases() -> None:
    manifest = _manifest()
    source_seed = manifest.source_seeds("training")[0]
    cases = tuple(case for case in manifest.cases if case.source_seed == source_seed)
    assert len(cases) == 7
    assert sum(case.family == "nominal" for case in cases) == 1
    assert sum(case.family == "perturbation" for case in cases) == 3
    assert sum(case.family == "recovery" for case in cases) == 3
    assert {case.severity for case in cases if case.family == "perturbation"} == {0.75}
    assert {case.severity for case in cases if case.family == "recovery"} == {1.0}
    assert next(case for case in cases if case.family == "nominal").severity == 0.0


def test_mixture_sampler_is_resume_exact() -> None:
    sampler = RecoveryCaseSampler(
        _manifest(), probabilities=(0.5, 0.3, 0.2), seed=9
    )
    state = sampler.state_dict()
    expected = [sampler.next_case().case_id for _ in range(20)]
    sampler.load_state_dict(state)
    assert [sampler.next_case().case_id for _ in range(20)] == expected


def test_replacement_sampling_preserves_rejected_family() -> None:
    sampler = RecoveryCaseSampler(
        _manifest(), probabilities=(0.5, 0.3, 0.2), seed=9
    )
    for family in ("recovery", "perturbation", "nominal"):
        assert all(sampler.next_case(family=family).family == family for _ in range(10))


def test_mixture_sampler_uses_training_partition_only() -> None:
    sampler = RecoveryCaseSampler(
        _manifest(), probabilities=(0.5, 0.3, 0.2), seed=9
    )
    assert {sampler.next_case().partition for _ in range(100)} == {"training"}


def test_sampler_rejects_state_from_another_manifest() -> None:
    first = RecoveryCaseSampler(_manifest(), probabilities=(0.5, 0.3, 0.2), seed=9)
    second_manifest = build_case_manifest(
        seed=8,
        calibration=8,
        training=16,
        curve=6,
        final=6,
        severity=0.75,
    )
    second = RecoveryCaseSampler(
        second_manifest, probabilities=(0.5, 0.3, 0.2), seed=9
    )
    with pytest.raises(ValueError, match="manifest hash"):
        second.load_state_dict(first.state_dict())


def test_manifest_round_trip_preserves_canonical_hash(tmp_path: Path) -> None:
    manifest = _manifest()
    destination = save_case_manifest(tmp_path / "cases.json", manifest)
    loaded = load_case_manifest(destination)
    assert loaded == manifest
    assert loaded.sha256 == manifest.sha256
