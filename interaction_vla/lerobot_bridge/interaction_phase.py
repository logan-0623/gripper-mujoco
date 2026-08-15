from __future__ import annotations

import numpy as np

from interaction_vla.lerobot_bridge.teacher_schema import (
    PROBABILITY_0,
    PROBABILITY_1,
    RELATIVE_POSITION,
    SIGNED_MARGIN_0,
)


PHASE_NAMES = ("approach", "grasp", "lift", "transport", "place", "release")
PHASE_IDS = {name: index for index, name in enumerate(PHASE_NAMES)}


def causal_phase_step(values: object, previous: int) -> int:
    relation = np.asarray(values, dtype=np.float32)
    if relation.shape != (8, 24) or not np.isfinite(relation).all():
        raise ValueError("current relations must be finite with shape [8, 24]")
    if int(previous) not in range(len(PHASE_NAMES)):
        raise ValueError("previous phase is invalid")

    contact = relation[0, PROBABILITY_0] >= 0.45
    co_motion = relation[0, PROBABILITY_1] >= 0.70
    grasped = bool(contact and co_motion)
    on_support = relation[2, PROBABILITY_0] >= 0.50
    target_distance = float(np.linalg.norm(relation[0, RELATIVE_POSITION]))
    goal_distance = float(np.linalg.norm(relation[1, RELATIVE_POSITION]))
    support_gap = float(relation[2, SIGNED_MARGIN_0])
    post_grasp = previous in {
        PHASE_IDS["lift"],
        PHASE_IDS["transport"],
        PHASE_IDS["place"],
        PHASE_IDS["release"],
    }
    if previous == PHASE_IDS["release"]:
        current = PHASE_IDS["release"]
    elif post_grasp and goal_distance <= 0.04 and not grasped:
        current = PHASE_IDS["release"]
    elif post_grasp and goal_distance <= 0.14:
        current = PHASE_IDS["place"]
    elif grasped and (on_support or support_gap < 0.06):
        current = PHASE_IDS["lift"]
    elif grasped and goal_distance > 0.08:
        current = PHASE_IDS["transport"]
    elif grasped:
        current = PHASE_IDS["place"]
    elif target_distance <= 0.07:
        current = PHASE_IDS["grasp"]
    else:
        current = PHASE_IDS["approach"]
    return current


def causal_phase_ids(values: object) -> np.ndarray:
    relations = np.asarray(values, dtype=np.float32)
    if (
        relations.ndim != 3
        or relations.shape[1:] != (8, 24)
        or not np.isfinite(relations).all()
    ):
        raise ValueError("relation sequence must be finite with shape [T, 8, 24]")
    phases = np.empty(len(relations), dtype=np.int64)
    previous = PHASE_IDS["approach"]
    for index, relation in enumerate(relations):
        previous = causal_phase_step(relation, previous)
        phases[index] = previous
    return phases
