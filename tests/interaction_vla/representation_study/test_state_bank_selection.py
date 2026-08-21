from __future__ import annotations

from interaction_vla.representation_study.state_bank.selection import (
    assign_groups_to_partitions,
    classify_trace_strata,
    select_stratified_indices,
)


def _trace(step: int, **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "step": step,
        "done": False,
        "action_was_clipped": False,
        "ik_projection_scale": 1.0,
        "wrong_object_contact": False,
        "target_drop": False,
        "minimum_distractor_clearance": 0.1,
    }
    value.update(updates)
    return value


def test_trace_strata_mark_perturbation_then_recovery_and_terminal() -> None:
    rows = [
        _trace(0),
        _trace(1, action_was_clipped=True),
        _trace(2),
        _trace(3, done=True),
    ]
    assert classify_trace_strata(rows) == (
        "nominal",
        "perturbation",
        "recovery",
        "terminal",
    )


def test_stratified_selection_is_deterministic_and_keeps_boundaries() -> None:
    strata = ("nominal",) * 5 + ("perturbation",) * 5 + ("recovery",) * 5
    first = select_stratified_indices(strata, per_stratum=2)
    second = select_stratified_indices(strata, per_stratum=2)
    assert first == second
    assert first == (0, 4, 5, 9, 10, 14)


def test_group_partition_assignment_keeps_case_groups_together() -> None:
    groups = tuple(f"case-{index}" for index in range(10))
    mapping = assign_groups_to_partitions(groups, seed=7, ratios=(0.8, 0.1, 0.1))
    assert set(mapping) == set(groups)
    assert list(mapping.values()).count("train") == 8
    assert list(mapping.values()).count("validation") == 1
    assert list(mapping.values()).count("test") == 1

