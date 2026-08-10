import numpy as np

from interaction_vla.lerobot_bridge.teacher import (
    BREAK,
    CLEARANCE,
    CO_MOTION,
    ESTABLISH,
    INCREASE,
    PROXIMITY,
    label_relation_goals,
    validate_typed_relation_goals,
)
from interaction_vla.lerobot_bridge.teacher_schema import (
    CONFIDENCE,
    ERROR_0,
    ERROR_1,
    PROBABILITY_0,
    PROBABILITY_1,
)


def test_goal_label_selects_largest_confident_future_improvement() -> None:
    errors = np.full((9, 8, 2), 0.8, dtype=np.float32)
    confidence = np.ones((9, 8), dtype=np.float32)
    errors[:, 0, 0] = np.linspace(0.8, 0.1, 9)
    errors[:, 1, 0] = np.linspace(0.8, 0.6, 9)

    labels = label_relation_goals(
        errors, confidence, horizon=8, minimum_improvement=0.05
    )

    assert labels[0, 0] == 0
    assert labels[0, 1] == 4
    assert labels[0, 2] == 0
    assert labels[0, 3] < 0.0
    assert 0.0 <= labels[0, 4] <= 1.0


def test_no_improvement_preserves_previous_relation() -> None:
    errors = np.ones((3, 8, 2), dtype=np.float32)
    confidence = np.ones((3, 8), dtype=np.float32)

    labels = label_relation_goals(
        errors, confidence, horizon=2, minimum_improvement=0.05
    )

    assert labels[:, 1].tolist() == [3.0, 3.0, 3.0]


def _typed_values(frames: int = 9) -> np.ndarray:
    values = np.zeros((frames, 8, 24), dtype=np.float32)
    values[:, :, CONFIDENCE] = 1.0
    values[:, :, ERROR_0 : ERROR_1 + 1] = 0.8
    values[:, 0, PROBABILITY_0] = 0.2
    values[:, 0, PROBABILITY_1] = 0.2
    return values


def _typed_labels(values: np.ndarray) -> np.ndarray:
    return label_relation_goals(
        values[:, :, (ERROR_0, ERROR_1)],
        values[:, :, CONFIDENCE],
        relation_values=values,
        horizon=8,
        minimum_improvement=0.05,
    )


def test_typed_goal_uses_establish_proximity_not_generic_decrease() -> None:
    values = _typed_values()
    values[:, 0, ERROR_0] = np.linspace(0.8, 0.1, len(values))

    label = _typed_labels(values)[0]

    assert label[:3].tolist() == [0.0, float(ESTABLISH), float(PROXIMITY)]
    assert label[3] < 0.0


def test_typed_goal_can_break_co_motion_after_place_and_support() -> None:
    values = _typed_values()
    values[:, 1, ERROR_0] = 0.1
    values[:, 2, ERROR_0] = 0.1
    values[:, 0, PROBABILITY_0] = 0.9
    values[:, 0, PROBABILITY_1] = np.linspace(0.9, 0.1, len(values))

    label = _typed_labels(values)[0]

    assert label[:3].tolist() == [0.0, float(BREAK), float(CO_MOTION)]


def test_typed_goal_can_increase_clearance_after_release() -> None:
    values = _typed_values()
    values[:, 1, ERROR_0] = 0.1
    values[:, 2, ERROR_0] = 0.1
    values[:, 0, PROBABILITY_0] = 0.9
    values[:, 0, PROBABILITY_1] = 0.1
    values[:, 7, ERROR_0] = np.linspace(0.9, 0.1, len(values))

    label = _typed_labels(values)[0]

    assert label[:3].tolist() == [7.0, float(INCREASE), float(CLEARANCE)]


def test_typed_goal_validation_rejects_operator_corruption() -> None:
    values = _typed_values()
    values[:, 0, ERROR_0] = np.linspace(0.8, 0.1, len(values))
    labels = _typed_labels(values)
    labels[0, 1] = 4.0

    with np.testing.assert_raises_regex(ValueError, "semantic"):
        validate_typed_relation_goals(
            values,
            labels,
            horizon=8,
            minimum_improvement=0.05,
        )
