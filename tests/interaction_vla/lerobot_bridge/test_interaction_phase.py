import numpy as np

from interaction_vla.lerobot_bridge.interaction_phase import (
    PHASE_IDS,
    causal_phase_ids,
    causal_phase_step,
)
from interaction_vla.lerobot_bridge.teacher_schema import (
    PROBABILITY_0,
    PROBABILITY_1,
    RELATIVE_POSITION,
    SIGNED_MARGIN_0,
)


def phase_relation_fixture() -> np.ndarray:
    relation = np.zeros((6, 8, 24), dtype=np.float32)
    relation[:, 0, RELATIVE_POSITION] = (0.20, 0.0, 0.0)
    relation[:, 1, RELATIVE_POSITION] = (0.20, 0.0, 0.0)
    relation[:, 2, SIGNED_MARGIN_0] = 0.10
    relation[1, 0, RELATIVE_POSITION] = (0.05, 0.0, 0.0)
    relation[2:5, 0, PROBABILITY_0] = 0.80
    relation[2:5, 0, PROBABILITY_1] = 0.90
    relation[2, 2, PROBABILITY_0] = 0.80
    relation[3, 1, RELATIVE_POSITION] = (0.20, 0.0, 0.0)
    relation[4, 1, RELATIVE_POSITION] = (0.05, 0.0, 0.0)
    relation[5, 0, RELATIVE_POSITION] = (0.20, 0.0, 0.0)
    relation[5, 1, RELATIVE_POSITION] = (0.02, 0.0, 0.0)
    relation[5, 1, PROBABILITY_0] = 0.80
    return relation


def test_causal_phase_labels_follow_current_interaction_relations() -> None:
    phases = causal_phase_ids(phase_relation_fixture())

    assert phases.tolist() == [
        PHASE_IDS["approach"],
        PHASE_IDS["grasp"],
        PHASE_IDS["lift"],
        PHASE_IDS["transport"],
        PHASE_IDS["place"],
        PHASE_IDS["release"],
    ]


def test_causal_phase_labels_use_only_current_and_previous_relations() -> None:
    relation = phase_relation_fixture()
    first = causal_phase_ids(relation)
    changed_future = relation.copy()
    changed_future[-1, :, :] = 100.0
    second = causal_phase_ids(changed_future)

    np.testing.assert_array_equal(first[:-1], second[:-1])
    assert set(first.tolist()) <= set(range(6))


def test_release_uses_causal_goal_geometry_when_probability_is_miscalibrated() -> None:
    relation = np.zeros((8, 24), dtype=np.float32)
    relation[0, RELATIVE_POSITION] = (0.20, 0.0, 0.0)
    relation[1, RELATIVE_POSITION] = (0.03, 0.0, 0.0)
    relation[1, PROBABILITY_0] = 0.05

    phase = causal_phase_step(relation, PHASE_IDS["place"])

    assert phase == PHASE_IDS["release"]
