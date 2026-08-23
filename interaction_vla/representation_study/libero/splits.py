from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .schema import StateRecord


SPLIT_SCHEMA = "libero_interaction_state_bank_split_v1"
PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    name: str
    group_unit: str
    seed: int
    ratios: tuple[float, float, float]
    assignments: Mapping[str, str]
    schema_version: str = SPLIT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "group_unit": self.group_unit,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "assignments": dict(sorted(self.assignments.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SplitManifest":
        if value.get("schema_version") != SPLIT_SCHEMA:
            raise ValueError(f"split schema must be {SPLIT_SCHEMA}")
        return cls(
            name=str(value["name"]),
            group_unit=str(value["group_unit"]),
            seed=int(value["seed"]),
            ratios=tuple(float(item) for item in value["ratios"]),  # type: ignore[arg-type]
            assignments={str(key): str(item) for key, item in value["assignments"].items()},  # type: ignore[union-attr]
        )

    def with_assignments(self, assignments: Mapping[str, str]) -> "SplitManifest":
        return SplitManifest(
            name=self.name,
            group_unit=self.group_unit,
            seed=self.seed,
            ratios=self.ratios,
            assignments=dict(assignments),
        )


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")


def _partition_names(count: int, ratios: tuple[float, float, float]) -> tuple[str, ...]:
    if count <= 0:
        return ()
    raw = np.asarray(ratios) * count
    counts = np.floor(raw).astype(int)
    if count >= 3:
        counts = np.maximum(counts, 1)
    while int(counts.sum()) > count:
        candidates = [index for index, value in enumerate(counts) if value > 1]
        if not candidates:
            break
        index = min(candidates, key=lambda item: raw[item] - counts[item])
        counts[index] -= 1
    while int(counts.sum()) < count:
        index = max(range(3), key=lambda item: raw[item] - counts[item])
        counts[index] += 1
    return tuple(
        partition
        for partition, partition_count in zip(PARTITIONS, counts, strict=True)
        for _ in range(int(partition_count))
    )


def build_task_group_split(
    records: Sequence[StateRecord],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitManifest:
    _validate_ratios(ratios)
    tasks_by_suite: dict[str, set[int]] = {}
    for record in records:
        tasks_by_suite.setdefault(record.suite, set()).add(record.task_id)
    task_partition: dict[tuple[str, int], str] = {}
    for suite_index, (suite, task_ids) in enumerate(sorted(tasks_by_suite.items())):
        tasks = sorted(task_ids)
        suite_seed = np.random.SeedSequence([seed, suite_index]).generate_state(1)[0]
        order = np.random.default_rng(suite_seed).permutation(len(tasks))
        partitions = _partition_names(len(tasks), ratios)
        for position, index in enumerate(order):
            task_partition[(suite, tasks[int(index)])] = partitions[position]
    split = SplitManifest(
        name="task_group",
        group_unit="task",
        seed=seed,
        ratios=ratios,
        assignments={
            record.state_id: task_partition[(record.suite, record.task_id)]
            for record in records
        },
    )
    validate_split(records, split)
    return split


def build_episode_group_split(
    records: Sequence[StateRecord],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitManifest:
    _validate_ratios(ratios)
    by_task: dict[tuple[str, int], set[str]] = {}
    for record in records:
        by_task.setdefault((record.suite, record.task_id), set()).add(
            record.source_episode_id
        )
    episode_partition: dict[tuple[str, int, str], str] = {}
    for task_index, (task, episodes_set) in enumerate(sorted(by_task.items())):
        episodes = sorted(episodes_set)
        task_seed = np.random.SeedSequence([seed, task_index]).generate_state(1)[0]
        order = np.random.default_rng(task_seed).permutation(len(episodes))
        partitions = _partition_names(len(episodes), ratios)
        for position, index in enumerate(order):
            episode_partition[(task[0], task[1], episodes[int(index)])] = partitions[
                position
            ]
    split = SplitManifest(
        name="episode_group",
        group_unit="episode",
        seed=seed,
        ratios=ratios,
        assignments={
            record.state_id: episode_partition[
                (record.suite, record.task_id, record.source_episode_id)
            ]
            for record in records
        },
    )
    validate_split(records, split)
    return split


def validate_split(
    records: Sequence[StateRecord], split: SplitManifest
) -> dict[str, object]:
    state_ids = {record.state_id for record in records}
    if len(state_ids) != len(records):
        raise ValueError("State Bank contains duplicate state IDs")
    if set(split.assignments) != state_ids:
        raise ValueError("split assignments must exactly cover State Bank state IDs")
    unknown = sorted(set(split.assignments.values()).difference(PARTITIONS))
    if unknown:
        raise ValueError(f"split contains unknown partitions: {unknown}")

    episode_partitions: dict[tuple[str, int, str], set[str]] = {}
    task_partitions: dict[tuple[str, int], set[str]] = {}
    counts = {partition: 0 for partition in PARTITIONS}
    for record in records:
        partition = split.assignments[record.state_id]
        counts[partition] += 1
        episode_partitions.setdefault(
            (record.suite, record.task_id, record.source_episode_id), set()
        ).add(partition)
        task_partitions.setdefault((record.suite, record.task_id), set()).add(partition)
    episode_overlap = any(len(values) != 1 for values in episode_partitions.values())
    task_overlap = any(len(values) != 1 for values in task_partitions.values())
    if episode_overlap:
        raise ValueError("frames from one episode span multiple partitions")
    if split.group_unit == "task" and task_overlap:
        raise ValueError("task-group split contains task leakage")
    return {
        "passed": True,
        "group_unit": split.group_unit,
        "states_by_partition": counts,
        "episode_overlap": episode_overlap,
        "task_overlap": task_overlap,
        "episodes": len(episode_partitions),
        "tasks": len(task_partitions),
    }
