from __future__ import annotations

from types import SimpleNamespace

import pytest

from interaction_vla.graph_control.ablation_pipeline import (
    ablation_inspect_from_config,
    build_test_reservoir_schedule,
    partition_episode_rows,
)
from interaction_vla.graph_control.training import ControlSplit
from interaction_vla.graph_control.schema import ABLATION_CONDITIONS


def test_partition_episode_rows_binds_split_to_dataset_metadata() -> None:
    source = SimpleNamespace(
        meta=SimpleNamespace(
            episodes=[
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 2},
                {"episode_index": 1, "dataset_from_index": 2, "dataset_to_index": 5},
                {"episode_index": 2, "dataset_from_index": 5, "dataset_to_index": 7},
            ]
        )
    )
    split = ControlSplit(
        path=SimpleNamespace(),
        episodes={"train": (0,), "validation": (1,), "test": (2,)},
        rows={"train": (0, 1), "validation": (2, 3, 4), "test": (5, 6)},
        split_seed=0,
        sha256="a" * 64,
    )

    rows = partition_episode_rows(source, split)

    assert rows == {
        "train": {0: (0, 1)},
        "validation": {1: (2, 3, 4)},
        "test": {2: (5, 6)},
    }
    split.rows["test"] = (4, 5)
    with pytest.raises(ValueError, match="test row alignment"):
        partition_episode_rows(source, split)


def test_test_reservoir_schedule_is_deterministic_and_balanced() -> None:
    case_ids = tuple(f"case-{index}" for index in range(13))
    first = build_test_reservoir_schedule(case_ids, (40, 41, 42, 43, 44), seed=17)
    second = build_test_reservoir_schedule(case_ids, (40, 41, 42, 43, 44), seed=17)

    assert first == second
    assert set(first) == set(case_ids)
    counts = [list(first.values()).count(episode) for episode in (40, 41, 42, 43, 44)]
    assert max(counts) - min(counts) <= 1


def test_ablation_inspect_consumes_the_seven_item_context(monkeypatch) -> None:
    source = SimpleNamespace(
        meta=SimpleNamespace(
            episodes=[
                {"episode_index": index, "dataset_from_index": index, "dataset_to_index": index + 1}
                for index in range(6)
            ]
        )
    )
    split = ControlSplit(
        path=SimpleNamespace(),
        episodes={"train": (0, 1), "validation": (2, 3), "test": (4, 5)},
        rows={"train": (0, 1), "validation": (2, 3), "test": (4, 5)},
        split_seed=0,
        sha256="a" * 64,
    )
    config = SimpleNamespace(seeds=(0, 1, 2), conditions=ABLATION_CONDITIONS)
    base = SimpleNamespace(
        graph_checkpoint=lambda condition, seed: SimpleNamespace(
            as_posix=lambda: f"seed-{seed}.pt"
        )
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.ablation_pipeline._ablation_context",
        lambda path: (config, base, object(), split, source, "b" * 64, "c" * 64),
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.ablation_pipeline._source_cache",
        lambda *args, seed, **kwargs: SimpleNamespace(sha256=str(seed) * 64),
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.ablation_pipeline.sha256_file",
        lambda path: "d" * 64,
    )

    report = ablation_inspect_from_config("ablation.yaml")

    assert report["passed"] is True
    assert report["source_caches"] == {"0": "0" * 64, "1": "1" * 64, "2": "2" * 64}
