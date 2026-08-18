from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES
from interaction_vla.graph_control.sensitivity import (
    action_change_metrics,
    finite_difference_interventions,
    mask_token_group,
    standardized_perturbation_magnitude,
    training_feature_statistics,
)


def _tokens(rows: int = 4) -> np.ndarray:
    values = np.zeros((rows, TOKEN_DIM), dtype=np.float64)
    values[:, TOKEN_SLICES["goal_residual"]] = np.linspace(-1.0, 1.0, rows)[:, None]
    values[:, TOKEN_SLICES["phase"].start] = 1.0
    return values


def test_group_masking_changes_only_the_named_slice() -> None:
    values = _tokens()
    values[:, TOKEN_SLICES["gripper_target_geometry"]] = 0.5

    masked = mask_token_group(values, "gripper_target_geometry")

    assert np.all(masked[:, TOKEN_SLICES["gripper_target_geometry"]] == 0.0)
    untouched = np.ones(TOKEN_DIM, dtype=bool)
    untouched[TOKEN_SLICES["gripper_target_geometry"]] = False
    np.testing.assert_array_equal(masked[:, untouched], values[:, untouched])
    assert not np.shares_memory(masked, values)


def test_group_masking_rejects_unknown_or_nonfinite_inputs() -> None:
    values = _tokens()
    with pytest.raises(ValueError, match="unknown Graph token group"):
        mask_token_group(values, "unknown")
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mask_token_group(values, "phase")


def test_training_statistics_are_featurewise_and_finite() -> None:
    values = _tokens()
    statistics = training_feature_statistics(values)

    assert set(statistics) == {"std", "p01", "p99"}
    assert all(array.shape == (TOKEN_DIM,) for array in statistics.values())
    assert statistics["std"][TOKEN_SLICES["goal_residual"].start] > 0.0
    assert statistics["std"][TOKEN_SLICES["phase"].start] == 0.0


def test_continuous_finite_difference_uses_train_std_and_quantile_clipping() -> None:
    values = _tokens()
    bounds = TOKEN_SLICES["goal_residual"]
    statistics = {
        "std": np.ones(TOKEN_DIM, dtype=np.float64) * 2.0,
        "p01": np.ones(TOKEN_DIM, dtype=np.float64) * -0.4,
        "p99": np.ones(TOKEN_DIM, dtype=np.float64) * 0.6,
    }

    minus, plus = finite_difference_interventions(
        values,
        "goal_residual",
        statistics=statistics,
        scale=0.25,
    )

    np.testing.assert_allclose(minus[:, bounds].ravel(), [-0.4, -0.4, -1 / 6, 0.5])
    np.testing.assert_allclose(plus[:, bounds].ravel(), [-0.4, 1 / 6, 0.6, 0.6])
    untouched = np.ones(TOKEN_DIM, dtype=bool)
    untouched[bounds] = False
    np.testing.assert_array_equal(minus[:, untouched], values[:, untouched])
    np.testing.assert_array_equal(plus[:, untouched], values[:, untouched])


def test_categorical_finite_difference_preserves_simplex_and_missing_zero() -> None:
    values = _tokens(rows=3)
    bounds = TOKEN_SLICES["phase"]
    values[:, bounds] = 0.0
    values[0, bounds.start] = 1.0
    values[1, bounds.start : bounds.start + 2] = (0.75, 0.25)
    statistics = training_feature_statistics(values)

    toward_uniform, away_from_uniform = finite_difference_interventions(
        values,
        "phase",
        statistics=statistics,
        scale=0.25,
    )

    np.testing.assert_allclose(toward_uniform[:2, bounds].sum(axis=1), 1.0)
    np.testing.assert_allclose(away_from_uniform[:2, bounds].sum(axis=1), 1.0)
    assert toward_uniform[0, bounds.start] < values[0, bounds.start]
    assert away_from_uniform[1, bounds.start] > values[1, bounds.start]
    np.testing.assert_array_equal(toward_uniform[2, bounds], 0.0)
    np.testing.assert_array_equal(away_from_uniform[2, bounds], 0.0)
    assert np.all(toward_uniform[:, bounds] >= 0.0)
    assert np.all(away_from_uniform[:, bounds] >= 0.0)


def test_standardized_magnitude_handles_zero_variance_and_categorical_groups() -> None:
    values = _tokens(rows=2)
    changed = values.copy()
    bounds = TOKEN_SLICES["goal_residual"]
    changed[:, bounds] += 0.5
    std = np.zeros(TOKEN_DIM, dtype=np.float64)
    std[bounds] = 2.0
    np.testing.assert_allclose(
        standardized_perturbation_magnitude(
            values, changed, "goal_residual", training_std=std
        ),
        0.25,
    )

    phase_changed = values.copy()
    phase_changed[:, TOKEN_SLICES["phase"]] = 0.0
    phase_changed[:, TOKEN_SLICES["phase"].start + 1] = 1.0
    expected = np.sqrt(2.0)
    np.testing.assert_allclose(
        standardized_perturbation_magnitude(
            values, phase_changed, "phase", training_std=std
        ),
        expected,
    )


def test_action_change_metrics_separate_motion_and_gripper_effects() -> None:
    baseline = np.asarray(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25], [0.0] * 7],
        dtype=np.float64,
    )
    changed = baseline.copy()
    changed[0, :3] = (-1.0, 0.0, 0.0)
    changed[0, 3:6] = (0.0, 1.0, 1.0)
    changed[0, 6] = 0.75
    changed[1, 6] = 0.5
    magnitude = np.asarray([2.0, 0.0])

    metrics = action_change_metrics(
        baseline, changed, perturbation_magnitude=magnitude
    )

    assert set(metrics) == {
        "action_l1",
        "action_l2",
        "translation_l2",
        "rotation_l2",
        "gripper_absolute_change",
        "action_direction_cosine_change",
        "translation_sign_changed",
        "standardized_perturbation_magnitude",
        "normalized_action_l2",
    }
    np.testing.assert_allclose(metrics["translation_l2"], [2.0, 0.0])
    np.testing.assert_allclose(metrics["rotation_l2"], [1.0, 0.0])
    np.testing.assert_allclose(metrics["gripper_absolute_change"], [0.5, 0.5])
    np.testing.assert_array_equal(metrics["translation_sign_changed"], [True, False])
    assert metrics["normalized_action_l2"][0] == pytest.approx(np.sqrt(5.0) / 2.0)
    assert np.isnan(metrics["normalized_action_l2"][1])
