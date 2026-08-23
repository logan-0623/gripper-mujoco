import pytest

from interaction_vla.representation_study.libero.splits import (
    build_episode_group_split,
    build_task_group_split,
    validate_split,
)

from .helpers import make_record


def _records():
    return tuple(
        make_record(task_id=task, episode=task * 10 + episode, frame=frame)
        for task in range(6)
        for episode in range(3)
        for frame in range(2)
    )


def test_task_group_split_has_no_task_or_episode_leakage() -> None:
    records = _records()
    split = build_task_group_split(records, ratios=(0.5, 0.25, 0.25), seed=7)
    report = validate_split(records, split)
    assert report["passed"]
    assert report["task_overlap"] is False
    assert report["episode_overlap"] is False
    assert set(split.assignments) == {record.state_id for record in records}


def test_task_group_split_is_stratified_by_suite_when_each_suite_is_large_enough() -> None:
    records = tuple(
        make_record(
            suite=suite,
            task_id=task,
            episode=task,
            frame=0,
        )
        for suite in ("libero_spatial", "libero_object")
        for task in range(3)
    )
    split = build_task_group_split(records, ratios=(0.5, 0.25, 0.25), seed=7)

    for suite in ("libero_spatial", "libero_object"):
        suite_partitions = {
            split.assignments[record.state_id]
            for record in records
            if record.suite == suite
        }
        assert suite_partitions == {"train", "validation", "test"}


def test_episode_group_split_keeps_frames_together_but_tasks_span_partitions() -> None:
    records = _records()
    split = build_episode_group_split(records, ratios=(0.5, 0.25, 0.25), seed=7)
    report = validate_split(records, split)
    assert report["passed"]
    assert report["episode_overlap"] is False
    assert split.group_unit == "episode"
    task_partitions = {
        split.assignments[record.state_id]
        for record in records
        if record.task_id == 0
    }
    assert len(task_partitions) == 3


def test_split_validation_rejects_missing_and_unknown_state_ids() -> None:
    records = _records()
    split = build_task_group_split(records, ratios=(0.5, 0.25, 0.25), seed=7)
    assignments = dict(split.assignments)
    assignments.pop(records[0].state_id)
    assignments["unknown"] = "train"
    with pytest.raises(ValueError, match="exactly cover"):
        validate_split(records, split.with_assignments(assignments))
