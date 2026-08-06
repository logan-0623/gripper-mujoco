from __future__ import annotations

import json

import pytest

from interaction_vla.source_split import (
    SourceSplit,
    deterministic_source_split,
    load_source_split,
    save_source_split,
    select_training_recovery_sources,
    source_split_hash,
    validate_derived_sources,
)


def test_two_hundred_sources_split_exactly_160_20_20() -> None:
    seeds = tuple(range(10_000, 10_200))
    first = deterministic_source_split(seeds, seed=42)
    second = deterministic_source_split(reversed(seeds), seed=42)

    assert first == second
    assert len(first.train) == 160
    assert len(first.validation) == 20
    assert len(first.test) == 20
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)


def test_ten_sources_split_exactly_8_1_1() -> None:
    split = deterministic_source_split(range(10), seed=42)

    assert (len(split.train), len(split.validation), len(split.test)) == (8, 1, 1)


def test_training_recovery_uses_exactly_twenty_five_percent_of_train_sources() -> None:
    split = deterministic_source_split(range(200), seed=42)
    selected = select_training_recovery_sources(
        split.train,
        fraction=0.25,
        seed=42,
    )

    assert len(selected) == 40
    assert set(selected).issubset(split.train)


def test_derived_source_validation_rejects_cross_split_leakage() -> None:
    split = SourceSplit(train=(1, 2), validation=(3,), test=(4,))

    with pytest.raises(ValueError, match="training recovery"):
        validate_derived_sources(
            split,
            training_recovery_sources=(2, 3),
            benchmark_sources=(3, 4),
        )
    with pytest.raises(ValueError, match="benchmark"):
        validate_derived_sources(
            split,
            training_recovery_sources=(2,),
            benchmark_sources=(1, 3, 4),
        )


def test_saved_source_split_round_trips_with_a_canonical_hash(tmp_path) -> None:
    split = deterministic_source_split(range(10), seed=42)
    recovery = select_training_recovery_sources(split.train, fraction=0.25, seed=42)
    benchmark = tuple(sorted(split.validation + split.test))
    path = tmp_path / "source_split.json"

    save_source_split(
        path,
        split,
        training_recovery_sources=recovery,
        benchmark_sources=benchmark,
    )
    loaded = load_source_split(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert loaded.split == split
    assert loaded.training_recovery_sources == recovery
    assert loaded.benchmark_sources == benchmark
    assert loaded.content_hash == source_split_hash(payload)
