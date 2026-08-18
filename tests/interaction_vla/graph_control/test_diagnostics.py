from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph_control.diagnostics import (
    build_representation_diagnostics,
    categorical_sequence_metrics,
    cluster_bootstrap_mean,
    covariance_effective_rank,
    feature_distribution,
    lagged_feature_correlation,
    temporal_feature_metrics,
    teacher_distance_metrics,
    validate_episode_layout,
    validate_tokens,
)
from interaction_vla.graph_control.schema import (
    ALL_CONDITIONS,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SLICES,
)


def _layout():
    return validate_episode_layout(
        row_indices=np.array([4, 5, 9, 10, 11]),
        episode_indices=np.array([2, 2, 7, 7, 7]),
        frame_indices=np.array([0, 1, 0, 1, 2]),
    )


def test_episode_layout_preserves_rows_and_never_crosses_episode_boundaries() -> None:
    layout = _layout()

    assert layout.episode_slices == (slice(0, 2), slice(2, 5))
    assert layout.episode_ids == (2, 7)
    np.testing.assert_array_equal(layout.row_indices, [4, 5, 9, 10, 11])
    assert not layout.row_indices.flags.writeable
    assert not layout.episode_indices.flags.writeable
    assert not layout.frame_indices.flags.writeable


@pytest.mark.parametrize(
    ("rows", "episodes", "frames", "message"),
    [
        ([4, 4], [2, 2], [0, 1], "unique"),
        ([4, 5, 6], [2, 7, 2], [0, 0, 1], "contiguous"),
        ([4, 5], [2, 2], [1, 2], "frame 0"),
        ([4, 5], [2, 2], [0, 2], "frame indices"),
    ],
)
def test_episode_layout_rejects_ambiguous_alignment(
    rows, episodes, frames, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_episode_layout(
            row_indices=np.asarray(rows),
            episode_indices=np.asarray(episodes),
            frame_indices=np.asarray(frames),
        )


def test_token_validation_requires_finite_2d_graph_tokens() -> None:
    tokens = np.zeros((5, TOKEN_DIM), dtype=np.float64)

    validated = validate_tokens(tokens, rows=5)

    assert validated.shape == (5, TOKEN_DIM)
    assert validated.dtype == np.float64
    assert not validated.flags.writeable

    with pytest.raises(ValueError, match="shape"):
        validate_tokens(tokens[:, :-1], rows=5)
    tokens[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_tokens(tokens, rows=5)


def test_feature_distribution_reports_robust_range_activity_and_saturation() -> None:
    metrics = feature_distribution(
        np.asarray([-1.0, 0.0, 1.0, 2.0]), active_epsilon=1.0e-6
    )
    quantiles = np.quantile(
        np.asarray([-1.0, 0.0, 1.0, 2.0]),
        [0.05, 0.25, 0.5, 0.75, 0.95],
        method="linear",
    )

    assert metrics["count"] == 4
    assert metrics["mean"] == 0.5
    assert metrics["std"] == pytest.approx(np.std([-1.0, 0.0, 1.0, 2.0]))
    assert metrics["min"] == -1.0
    assert metrics["max"] == 2.0
    assert metrics["p05"] == pytest.approx(quantiles[0])
    assert metrics["p25"] == pytest.approx(quantiles[1])
    assert metrics["median"] == pytest.approx(quantiles[2])
    assert metrics["p75"] == pytest.approx(quantiles[3])
    assert metrics["p95"] == pytest.approx(quantiles[4])
    assert metrics["robust_range"] == pytest.approx(quantiles[4] - quantiles[0])
    assert metrics["active_fraction"] == 0.75
    assert metrics["negative_saturation_fraction"] == 0.25
    assert metrics["positive_saturation_fraction"] == 0.5


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.inf])
def test_feature_distribution_rejects_invalid_inputs(epsilon: float) -> None:
    with pytest.raises(ValueError):
        feature_distribution(np.ones(3), active_epsilon=epsilon)

    with pytest.raises(ValueError, match="non-empty"):
        feature_distribution(np.array([]), active_epsilon=1.0e-6)
    with pytest.raises(ValueError, match="finite"):
        feature_distribution(np.array([np.nan]), active_epsilon=1.0e-6)


def test_temporal_metrics_do_not_cross_episode_boundaries() -> None:
    values = np.array([0.0, 1.0, 100.0, 102.0, 105.0])

    metrics = temporal_feature_metrics(values, _layout())

    assert metrics["first_difference_count"] == 3
    assert metrics["first_difference_mae"] == 2.0
    assert metrics["first_difference_rms"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert metrics["second_difference_count"] == 1
    assert metrics["second_difference_mae"] == 1.0


def test_temporal_metrics_report_missing_second_difference() -> None:
    layout = validate_episode_layout(
        row_indices=np.array([0, 1]),
        episode_indices=np.array([0, 0]),
        frame_indices=np.array([0, 1]),
    )

    metrics = temporal_feature_metrics(np.array([1.0, 2.0]), layout)

    assert metrics["second_difference_count"] == 0
    assert metrics["second_difference_mae"] is None


def test_categorical_metrics_treat_zero_vectors_as_missing_and_count_false_flips() -> None:
    predicted = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    teacher = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    metrics = categorical_sequence_metrics(
        predicted, _layout(), teacher_probabilities=teacher
    )

    assert metrics["valid_frames"] == 4
    assert metrics["missing_frames"] == 1
    assert metrics["mean_entropy"] == 0.0
    assert metrics["flip_numerator"] == 1
    assert metrics["flip_denominator"] == 2
    assert metrics["flip_rate"] == 0.5
    assert metrics["mean_dwell_length"] == pytest.approx(4.0 / 3.0)
    assert metrics["false_flip_numerator"] == 1
    assert metrics["false_flip_denominator"] == 2
    assert metrics["false_flip_rate"] == 0.5


def test_categorical_metrics_validate_shapes_and_probability_mass() -> None:
    with pytest.raises(ValueError, match="shape"):
        categorical_sequence_metrics(np.ones((4, 2)), _layout())
    invalid = np.ones((5, 2))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        categorical_sequence_metrics(invalid, _layout())
    invalid = np.ones((5, 2))
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        categorical_sequence_metrics(invalid, _layout())


def test_covariance_effective_rank_handles_degenerate_and_balanced_groups() -> None:
    assert covariance_effective_rank(np.ones((4, 2))) == 0.0
    assert covariance_effective_rank(np.array([[0.0], [1.0], [2.0]])) == 1.0
    balanced = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert covariance_effective_rank(balanced) == pytest.approx(2.0)


def test_covariance_effective_rank_rejects_nonfinite_or_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        covariance_effective_rank(np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        covariance_effective_rank(np.array([[np.nan]]))


def test_teacher_distance_skips_cosine_for_zero_vectors() -> None:
    predicted = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    teacher = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])

    metrics = teacher_distance_metrics(predicted, teacher)

    assert metrics["frames"] == 3
    assert metrics["mean_l1"] == 1.0
    assert metrics["mean_l2"] == pytest.approx((np.sqrt(2.0) + 1.0) / 3.0)
    assert metrics["cosine_frames"] == 2
    assert metrics["mean_cosine_distance"] == 0.5


def test_teacher_distance_rejects_shape_and_nonfinite_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        teacher_distance_metrics(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        teacher_distance_metrics(np.array([[np.nan]]), np.ones((1, 1)))


def test_lagged_correlation_finds_predicted_one_step_delay_without_boundary_pairs() -> None:
    layout = validate_episode_layout(
        row_indices=np.arange(8),
        episode_indices=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        frame_indices=np.array([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    teacher = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0])
    predicted = np.array([999.0, 0.0, 1.0, 2.0, -999.0, 100.0, 101.0, 102.0])

    metrics = lagged_feature_correlation(
        predicted, teacher, layout, max_lag=2
    )

    assert metrics["lag_zero_pairs"] == 8
    assert metrics["best_lag"] == 1
    assert metrics["best_pairs"] == 6
    assert metrics["best_correlation"] == pytest.approx(1.0)


def test_lagged_correlation_reports_constant_features_as_undefined() -> None:
    metrics = lagged_feature_correlation(
        np.ones(5), np.ones(5), _layout(), max_lag=3
    )

    assert metrics["lag_zero_correlation"] is None
    assert metrics["best_lag"] is None
    assert metrics["best_pairs"] == 0
    assert metrics["best_correlation"] is None


@pytest.mark.parametrize("max_lag", [-1, 1.5])
def test_lagged_correlation_rejects_invalid_lag(max_lag) -> None:
    with pytest.raises(ValueError, match="max_lag"):
        lagged_feature_correlation(
            np.arange(5), np.arange(5), _layout(), max_lag=max_lag
        )


def test_cluster_bootstrap_is_deterministic_over_episode_values() -> None:
    values = {2: 0.0, 7: 1.0, 9: 2.0}

    first = cluster_bootstrap_mean(values, samples=500, seed=17)
    second = cluster_bootstrap_mean(values, samples=500, seed=17)

    assert first == second
    assert first["estimate"] == 1.0
    assert first["episodes"] == 3
    assert first["bootstrap_samples"] == 500
    assert first["confidence"] == 0.95
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]


@pytest.mark.parametrize(
    ("values", "samples", "seed", "message"),
    [
        ({}, 100, 0, "non-empty"),
        ({0: np.nan}, 100, 0, "finite"),
        ({0: 1.0}, 0, 0, "samples"),
        ({0: 1.0}, 100, -1, "seed"),
    ],
)
def test_cluster_bootstrap_rejects_invalid_inputs(
    values, samples: int, seed: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cluster_bootstrap_mean(values, samples=samples, seed=seed)


def _diagnostic_token_matrix() -> np.ndarray:
    values = np.linspace(0.1, 0.9, TOKEN_DIM, dtype=np.float64)
    tokens = np.stack([values + frame * 0.01 for frame in range(5)])
    for name in ("phase", "next_relation", "relation_operator", "predicate"):
        bounds = TOKEN_SLICES[name]
        tokens[:, bounds] = 0.0
        width = bounds.stop - bounds.start
        for frame in range(len(tokens)):
            tokens[frame, bounds.start + frame % width] = 1.0
    return tokens


def _condition_matrix() -> tuple[dict[tuple[int, str], np.ndarray], np.ndarray]:
    teacher = _diagnostic_token_matrix()
    return (
        {
            (0, "flat"): np.zeros_like(teacher),
            (0, "oracle_graph_v2"): teacher.copy(),
            (0, "predicted_random_v2"): teacher * 0.9,
            (0, "predicted_reflect_v2"): teacher * 1.1,
        },
        teacher,
    )


def test_complete_diagnostics_report_covers_features_groups_and_episodes() -> None:
    condition_tokens, teacher = _condition_matrix()

    report, episode_records = build_representation_diagnostics(
        condition_tokens=condition_tokens,
        teacher_tokens=teacher,
        layout=_layout(),
        partition="test",
        bootstrap_samples=100,
        bootstrap_seed=17,
        max_lag=2,
        active_epsilon=1.0e-6,
    )

    assert report["schema_version"] == "graph_representation_diagnostics_v1"
    assert report["partition"] == "test"
    assert report["rows"] == 5
    assert report["episodes"] == 2
    assert report["conditions"] == list(ALL_CONDITIONS)
    assert report["estimator_seeds"] == [0]
    assert report["teacher_deduplicated"] is True
    assert set(report["by_seed_condition"]) == {
        "shared/flat",
        "shared/oracle_graph_v2",
        "seed_0/predicted_random_v2",
        "seed_0/predicted_reflect_v2",
    }

    predicted = report["by_seed_condition"]["seed_0/predicted_random_v2"]
    teacher_report = report["by_seed_condition"]["shared/oracle_graph_v2"]
    flat_report = report["by_seed_condition"]["shared/flat"]
    assert list(predicted["features"]) == list(TOKEN_FEATURE_NAMES)
    assert list(predicted["groups"]) == list(TOKEN_SLICES)
    assert "categorical" in predicted["groups"]["phase"]
    assert "hard_state" in predicted["groups"]["entity_presence"]
    assert "teacher_distance" in predicted["groups"]["phase"]
    assert "teacher_distance" not in teacher_report["groups"]["phase"]
    assert "teacher_distance" not in flat_report["groups"]["phase"]
    assert "teacher_lag" in predicted["features"][TOKEN_FEATURE_NAMES[0]]
    assert "teacher_lag" not in teacher_report["features"][TOKEN_FEATURE_NAMES[0]]
    assert "teacher_lag" not in flat_report["features"][TOKEN_FEATURE_NAMES[0]]
    assert "episode_clustered" in predicted["features"][TOKEN_FEATURE_NAMES[0]]
    assert "episode_clustered" in predicted["groups"]["phase"]

    assert len(episode_records) == 8
    assert {
        (record["condition"], record["estimator_seed"], record["episode_id"])
        for record in episode_records
    } == {
        (condition, seed, episode)
        for condition, seed in (
            ("flat", None),
            ("oracle_graph_v2", None),
            ("predicted_random_v2", 0),
            ("predicted_reflect_v2", 0),
        )
        for episode in (2, 7)
    }
    for record in episode_records:
        assert record["frames"] in {2, 3}
        assert list(record["features"]) == list(TOKEN_FEATURE_NAMES)
        assert list(record["groups"]) == list(TOKEN_SLICES)


def test_complete_diagnostics_requires_full_matrix_and_exact_teacher_cache() -> None:
    condition_tokens, teacher = _condition_matrix()
    del condition_tokens[(0, "predicted_reflect_v2")]
    with pytest.raises(ValueError, match="condition matrix"):
        build_representation_diagnostics(
            condition_tokens=condition_tokens,
            teacher_tokens=teacher,
            layout=_layout(),
            partition="test",
            bootstrap_samples=10,
            bootstrap_seed=0,
            max_lag=1,
            active_epsilon=1.0e-6,
        )

    condition_tokens, teacher = _condition_matrix()
    condition_tokens[(0, "oracle_graph_v2")][0, 0] += 1.0
    with pytest.raises(ValueError, match="Teacher"):
        build_representation_diagnostics(
            condition_tokens=condition_tokens,
            teacher_tokens=teacher,
            layout=_layout(),
            partition="test",
            bootstrap_samples=10,
            bootstrap_seed=0,
            max_lag=1,
            active_epsilon=1.0e-6,
        )
