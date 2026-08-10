import numpy as np

from interaction_vla.lerobot_bridge.teacher import label_relation_goals


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
