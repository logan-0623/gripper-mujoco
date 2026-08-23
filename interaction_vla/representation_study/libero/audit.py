from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .schema import FACTORS, StateRecord


def _range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def build_state_bank_audit(
    records: Sequence[StateRecord],
    *,
    replay_statistics: Mapping[str, object],
    minimum_acceptance_rate: float,
    l2_p95_tolerance: float,
    max_abs_tolerance: float,
) -> dict[str, object]:
    if not records:
        raise ValueError("State Bank audit requires at least one record")
    state_ids = [record.state_id for record in records]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("State Bank audit found duplicate state IDs")

    phase = Counter(
        record.labels.phase
        for record in records
        if record.labels.applicability.phase and record.labels.phase is not None
    )
    applicable = {
        factor: sum(bool(getattr(record.labels.applicability, factor)) for record in records)
        for factor in FACTORS
    }
    contact_rows = [
        record.labels.contact
        for record in records
        if record.labels.applicability.contact and record.labels.contact is not None
    ]
    contacts = [row.gripper_target for row in contact_rows]
    grasps = [
        record.labels.stable_grasp
        for record in records
        if record.labels.applicability.stable_grasp
        and record.labels.stable_grasp is not None
    ]
    gripper_target = [
        record.labels.geometry.gripper_target_distance
        for record in records
        if record.labels.applicability.geometry and record.labels.geometry is not None
    ]
    target_goal = [
        record.labels.geometry.target_goal_distance
        for record in records
        if record.labels.applicability.geometry and record.labels.geometry is not None
    ]
    gripper_target_pose = [
        record.labels.geometry.gripper_to_target
        for record in records
        if record.labels.applicability.geometry and record.labels.geometry is not None
    ]
    target_goal_pose = [
        record.labels.geometry.target_to_goal
        for record in records
        if record.labels.applicability.geometry and record.labels.geometry is not None
    ]

    def pose_ranges(rows: list[tuple[float, ...]]) -> dict[str, dict[str, float | None]]:
        names = ("tx", "ty", "tz", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5")
        if not rows:
            return {name: _range([]) for name in names}
        matrix = np.asarray(rows, dtype=np.float64)
        return {name: _range(matrix[:, index].tolist()) for index, name in enumerate(names)}
    entities = Counter(
        record.labels.entity.target
        for record in records
        if record.labels.applicability.entity and record.labels.entity is not None
    )
    goals = Counter(
        str(record.labels.entity.goal)
        for record in records
        if record.labels.applicability.entity
        and record.labels.entity is not None
        and record.labels.entity.goal is not None
    )
    sources = Counter(
        str(record.labels.entity.source)
        for record in records
        if record.labels.applicability.entity
        and record.labels.entity is not None
        and record.labels.entity.source is not None
    )
    distractors = Counter(
        distractor
        for record in records
        if record.labels.applicability.entity and record.labels.entity is not None
        for distractor in record.labels.entity.distractors
    )
    next_relations = Counter(
        "|".join(
            (
                record.labels.next_relation.subject_role,
                record.labels.next_relation.predicate,
                record.labels.next_relation.object_role,
                record.labels.next_relation.operator,
            )
        )
        for record in records
        if record.labels.applicability.next_relation
        and record.labels.next_relation is not None
    )

    def temporal_statistics(factor: str) -> dict[str, int]:
        by_episode: dict[tuple[str, int, str], list[StateRecord]] = {}
        for record in records:
            by_episode.setdefault(
                (record.suite, record.task_id, record.source_episode_id), []
            ).append(record)
        transitions = 0
        isolated = 0
        for episode_records in by_episode.values():
            ordered = sorted(episode_records, key=lambda item: item.frame_index)
            values: list[bool] = []
            for item in ordered:
                if factor == "contact":
                    value = bool(item.labels.contact and item.labels.contact.gripper_target)
                else:
                    value = bool(item.labels.stable_grasp)
                values.append(value)
            transitions += sum(left != right for left, right in zip(values, values[1:], strict=False))
            isolated += sum(
                value
                and (index == 0 or not values[index - 1])
                and (index == len(values) - 1 or not values[index + 1])
                for index, value in enumerate(values)
            )
        return {"transitions": transitions, "isolated_positive_frames": isolated}

    terminal_by_episode: dict[tuple[str, int, str], StateRecord] = {}
    for record in records:
        key = (record.suite, record.task_id, record.source_episode_id)
        current = terminal_by_episode.get(key)
        if current is None or record.frame_index > current.frame_index:
            terminal_by_episode[key] = record
    goal_relation_observed = sum(
        item.labels.phase in {"place", "release_retreat"}
        for item in terminal_by_episode.values()
    )
    acceptance = float(
        replay_statistics.get(
            "acceptance_rate",
            float(replay_statistics.get("accepted", 0))
            / max(int(replay_statistics.get("episodes", 0)), 1),
        )
    )
    l2_p95 = float(replay_statistics.get("l2_p95", float("inf")))
    max_abs = float(replay_statistics.get("max_abs", float("inf")))
    reasons: list[str] = []
    if acceptance < minimum_acceptance_rate:
        reasons.append("replay acceptance rate is below threshold")
    if l2_p95 > l2_p95_tolerance:
        reasons.append("replay l2 p95 exceeds tolerance")
    if max_abs > max_abs_tolerance:
        reasons.append("replay maximum absolute error exceeds tolerance")
    expected_tasks = int(replay_statistics.get("expected_tasks", 0))
    accepted_tasks = int(replay_statistics.get("accepted_tasks", expected_tasks))
    if expected_tasks and accepted_tasks != expected_tasks:
        reasons.append("replay did not preserve complete configured task coverage")
    if len(set(contacts)) < 2:
        reasons.append("contact labels do not contain both classes")
    if len(set(grasps)) < 2:
        reasons.append("stable-grasp labels do not contain both classes")
    missing_or_invalid = 0
    for record in records:
        try:
            StateRecord.from_dict(record.to_dict())
        except (KeyError, TypeError, ValueError):
            missing_or_invalid += 1
    if missing_or_invalid:
        reasons.append("records contain missing or invalid values")
    return {
        "schema_version": "libero_state_bank_audit_v1",
        "passed": not reasons,
        "gate_reasons": reasons,
        "tasks": len({(record.suite, record.task_id) for record in records}),
        "episodes": len(
            {
                (record.suite, record.task_id, record.source_episode_id)
                for record in records
            }
        ),
        "states": len(records),
        "factor_applicability": applicable,
        "entity_distribution": dict(sorted(entities.items())),
        "entity_subfield_distribution": {
            "target": dict(sorted(entities.items())),
            "goal": dict(sorted(goals.items())),
            "source": dict(sorted(sources.items())),
        },
        "distractor_distribution": dict(sorted(distractors.items())),
        "phase_distribution": dict(sorted(phase.items())),
        "next_relation_distribution": dict(sorted(next_relations.items())),
        "contact": {
            "applicable": len(contacts),
            "positive": sum(contacts),
            "positive_rate": sum(contacts) / len(contacts) if contacts else None,
            **temporal_statistics("contact"),
            "subfields": {
                name: {
                    "applicable": len(contact_rows),
                    "positive": sum(bool(getattr(row, name)) for row in contact_rows),
                    "positive_rate": (
                        sum(bool(getattr(row, name)) for row in contact_rows)
                        / len(contact_rows)
                        if contact_rows
                        else None
                    ),
                }
                for name in ("gripper_target", "target_goal", "target_source")
            },
        },
        "stable_grasp": {
            "applicable": len(grasps),
            "positive": sum(bool(item) for item in grasps),
            "positive_rate": (
                sum(bool(item) for item in grasps) / len(grasps) if grasps else None
            ),
            **temporal_statistics("stable_grasp"),
        },
        "geometry": {
            "gripper_to_target": pose_ranges(gripper_target_pose),
            "target_to_goal": pose_ranges(target_goal_pose),
            "gripper_target_distance": _range(gripper_target),
            "target_goal_distance": _range(target_goal),
        },
        "missing_or_invalid": missing_or_invalid,
        "trajectory_outcomes": {
            "source_outcome_available": False,
            "terminal_goal_relation_observed": goal_relation_observed,
            "terminal_goal_relation_not_observed": (
                len(terminal_by_episode) - goal_relation_observed
            ),
            "interpretation": (
                "diagnostic only: the last observation can precede the final action, "
                "so absence is not labeled as demonstration failure"
            ),
        },
        "replay": dict(replay_statistics),
    }
