from __future__ import annotations

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES
from interaction_vla.graph_control.sensitivity import (
    ALL_TOKENS_GROUP,
    LEGACY_SENSITIVITY_METRICS,
    LEGACY_SENSITIVITY_SCHEMA_VERSION,
    PREVIOUS_SENSITIVITY_SCHEMA_VERSION,
    SENSITIVITY_METRICS,
    SENSITIVITY_SCHEMA_VERSION,
    action_change_metrics,
    build_sensitivity_report,
    finite_difference_interventions,
    make_sensitivity_records,
    mask_token_group,
    predict_first_actions,
    select_episode_balanced_positions,
    standardized_perturbation_magnitude,
    temporally_matched_random_tokens,
    training_action_statistics,
    training_feature_statistics,
)
from interaction_vla.graph_control.diagnostics import validate_episode_layout


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


def test_all_token_mask_is_an_explicit_zero_control() -> None:
    values = _tokens()
    values[:, TOKEN_SLICES["gripper_target_geometry"]] = 0.5

    masked = mask_token_group(values, ALL_TOKENS_GROUP)

    np.testing.assert_array_equal(masked, 0.0)
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


def test_temporally_matched_random_deranges_episodes_at_normalized_time() -> None:
    layout = validate_episode_layout(
        row_indices=np.arange(9),
        episode_indices=np.asarray([10, 10, 10, 20, 20, 30, 30, 30, 30]),
        frame_indices=np.asarray([0, 1, 2, 0, 1, 0, 1, 2, 3]),
    )
    values = np.zeros((9, TOKEN_DIM), dtype=np.float64)
    values[:, 0] = np.asarray([100, 101, 102, 200, 201, 300, 301, 302, 303])

    changed, provenance = temporally_matched_random_tokens(
        values, layout, seed=17
    )
    repeated, repeated_provenance = temporally_matched_random_tokens(
        values, layout, seed=17
    )

    np.testing.assert_array_equal(changed, repeated)
    assert provenance == repeated_provenance
    assert provenance["alignment"] == "normalized_episode_progress_nearest"
    assert set(provenance["episode_mapping"]) == {"10", "20", "30"}
    assert all(
        destination != source
        for destination, source in provenance["episode_mapping"].items()
    )
    # Each destination sequence comes from one other episode and remains ordered.
    for bounds in layout.episode_slices:
        assert len(set((changed[bounds, 0] // 100).astype(int).tolist())) == 1
        assert np.all(np.diff(changed[bounds, 0]) >= 0.0)
    assert not np.shares_memory(changed, values)


def test_temporally_matched_random_requires_multiple_complete_episodes() -> None:
    layout = validate_episode_layout(
        row_indices=np.arange(3),
        episode_indices=np.asarray([10, 10, 10]),
        frame_indices=np.asarray([0, 1, 2]),
    )
    with pytest.raises(ValueError, match="at least two episodes"):
        temporally_matched_random_tokens(_tokens(rows=3), layout, seed=0)


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
        baseline,
        changed,
        perturbation_magnitude=magnitude,
        action_scale=np.ones(7, dtype=np.float64),
    )

    assert set(metrics) == set(SENSITIVITY_METRICS)
    np.testing.assert_allclose(metrics["action_l2"], [np.sqrt(5.25), 0.5])
    np.testing.assert_allclose(
        metrics["action_rms"], [np.sqrt(5.25 / 7.0), 0.5 / np.sqrt(7.0)]
    )
    np.testing.assert_allclose(metrics["translation_l2"], [2.0, 0.0])
    np.testing.assert_allclose(metrics["translation_rms"], [2.0 / np.sqrt(3.0), 0.0])
    np.testing.assert_allclose(metrics["rotation_l2"], [1.0, 0.0])
    np.testing.assert_allclose(metrics["rotation_rms"], [1.0 / np.sqrt(3.0), 0.0])
    np.testing.assert_allclose(metrics["gripper_absolute_change"], [0.5, 0.5])
    np.testing.assert_array_equal(metrics["translation_sign_changed"], [True, False])
    np.testing.assert_allclose(
        metrics["action_scale_normalized_rms"], metrics["action_rms"]
    )
    np.testing.assert_allclose(
        metrics["standardized_token_perturbation_l2"], magnitude
    )


def test_sensitivity_v3_preserves_v1_and_v2_schema_history() -> None:
    assert LEGACY_SENSITIVITY_SCHEMA_VERSION == "graph_policy_sensitivity_v1"
    assert "normalized_action_l2" in LEGACY_SENSITIVITY_METRICS
    assert PREVIOUS_SENSITIVITY_SCHEMA_VERSION == "graph_policy_sensitivity_v2"
    assert SENSITIVITY_SCHEMA_VERSION == "graph_policy_sensitivity_v3"
    assert "normalized_action_l2" not in SENSITIVITY_METRICS
    assert "action_scale_normalized_rms" in SENSITIVITY_METRICS


def test_training_action_statistics_use_iqr_with_declared_floor() -> None:
    actions = np.asarray(
        [
            [0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 1.0, 0.1, 0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.2, 0.0, 0.0, 1.0],
            [3.0, 1.0, 1.0, 0.3, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    statistics = training_action_statistics(actions, minimum_scale=0.05)

    assert set(statistics) == {
        "q25",
        "q75",
        "iqr",
        "effective_scale",
        "minimum_scale",
    }
    np.testing.assert_allclose(statistics["iqr"][:4], [1.5, 1.5, 0.0, 0.15])
    np.testing.assert_allclose(
        statistics["effective_scale"], [1.5, 1.5, 0.05, 0.15, 0.05, 0.05, 0.05]
    )
    assert statistics["minimum_scale"] == pytest.approx(0.05)


def test_action_scale_normalization_is_finite_for_tiny_token_perturbations() -> None:
    baseline = np.zeros((2, 7), dtype=np.float64)
    changed = np.ones((2, 7), dtype=np.float64) * 0.25

    metrics = action_change_metrics(
        baseline,
        changed,
        perturbation_magnitude=np.asarray([0.0, 1.0e-15]),
        action_scale=np.asarray([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.5]),
    )

    assert np.isfinite(metrics["action_scale_normalized_rms"]).all()
    np.testing.assert_allclose(
        metrics["action_scale_normalized_rms"],
        np.sqrt(
            np.mean(
                np.asarray([0.25, 0.25, 0.25, 2.5, 2.5, 2.5, 0.5]) ** 2
            )
        ),
    )


def test_episode_balanced_selection_is_deterministic_and_includes_boundaries() -> None:
    layout = validate_episode_layout(
        row_indices=np.arange(8),
        episode_indices=np.asarray([2, 2, 2, 2, 2, 7, 7, 7]),
        frame_indices=np.asarray([0, 1, 2, 3, 4, 0, 1, 2]),
    )

    np.testing.assert_array_equal(
        select_episode_balanced_positions(layout, rows_per_episode=2),
        [0, 4, 5, 7],
    )
    np.testing.assert_array_equal(
        select_episode_balanced_positions(layout, rows_per_episode=1),
        [2, 6],
    )


def test_sensitivity_report_clusters_frames_by_episode_and_policy_seed() -> None:
    metrics = {
        name: np.asarray(value)
        for name, value in {
            "action_l1": [0.0, 2.0, 10.0],
            "action_l2": [0.0, 2.0, 10.0],
            "translation_l2": [0.0, 2.0, 10.0],
            "action_rms": [0.0, 2.0 / np.sqrt(7.0), 10.0 / np.sqrt(7.0)],
            "translation_rms": [0.0, 2.0 / np.sqrt(3.0), 10.0 / np.sqrt(3.0)],
            "rotation_l2": [0.0, 0.0, 0.0],
            "rotation_rms": [0.0, 0.0, 0.0],
            "gripper_absolute_change": [0.0, 0.0, 0.0],
            "action_direction_cosine_change": [0.0, 0.0, 0.0],
            "translation_sign_changed": [False, True, True],
            "standardized_token_perturbation_l2": [1.0, 1.0, 1.0],
            "action_scale_normalized_rms": [0.0, 2.0, np.nan],
        }.items()
    }
    records = []
    for seed, offset in ((0, 0.0), (1, 1.0)):
        shifted = {name: value.copy() for name, value in metrics.items()}
        shifted["action_l2"] = shifted["action_l2"] + offset
        records.extend(
            make_sensitivity_records(
                policy_seed=seed,
                condition="predicted_random_v2",
                group="phase",
                intervention="mask",
                row_indices=np.asarray([10, 11, 20]),
                episode_indices=np.asarray([2, 2, 7]),
                frame_indices=np.asarray([0, 4, 1]),
                metrics=shifted,
            )
        )

    report = build_sensitivity_report(
        records,
        partition="test",
        bootstrap_samples=100,
        bootstrap_seed=17,
    )

    assert report["passed"] is True
    assert report["rows"] == 6
    assert report["policy_seeds"] == [0, 1]
    seed_zero = report["by_seed_condition"]["seed_0/predicted_random_v2"]
    action_l2 = seed_zero["phase"]["mask"]["action_l2"]
    assert action_l2["estimate"] == pytest.approx(5.5)
    assert action_l2["episodes"] == 2
    normalized = seed_zero["phase"]["mask"]["action_scale_normalized_rms"]
    assert normalized["estimate"] == pytest.approx(1.0)
    across = report["across_policy_seeds"]["predicted_random_v2"]["phase"][
        "mask"
    ]["action_l2"]
    assert across == {
        "estimate": pytest.approx(6.0),
        "policy_seed_std": pytest.approx(np.sqrt(0.5)),
        "policy_seeds": 2,
    }


class _FakePolicy:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def eval(self) -> None:
        return None

    def predict_action_chunk(self, batch):
        token = batch["observation.environment_state"]
        result = torch.zeros((len(token), 8, 7), dtype=torch.float32)
        result[:, 0, :] = token[:, :7]
        return result


def test_first_action_inference_resets_policy_for_each_independent_batch() -> None:
    policy = _FakePolicy()
    raw_batches = [
        {"observation.state": torch.zeros((2, 10))},
        {"observation.state": torch.zeros((1, 10))},
    ]
    tokens = np.zeros((3, TOKEN_DIM), dtype=np.float32)
    tokens[:, :7] = np.arange(21, dtype=np.float32).reshape(3, 7)

    actions = predict_first_actions(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        raw_batches=raw_batches,
        tokens=tokens,
    )

    assert policy.reset_count == 2
    assert actions.shape == (3, 7)
    np.testing.assert_array_equal(actions, tokens[:, :7])


def test_first_action_inference_rejects_batch_or_action_shape_mismatch() -> None:
    policy = _FakePolicy()
    with pytest.raises(ValueError, match="batch rows"):
        predict_first_actions(
            policy=policy,
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            raw_batches=[{"observation.state": torch.zeros((2, 10))}],
            tokens=np.zeros((1, TOKEN_DIM), dtype=np.float32),
        )

    class WrongPolicy(_FakePolicy):
        def predict_action_chunk(self, batch):
            return torch.zeros((len(batch["observation.environment_state"]), 7))

    with pytest.raises(ValueError, match="action chunk"):
        predict_first_actions(
            policy=WrongPolicy(),
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            raw_batches=[{"observation.state": torch.zeros((1, 10))}],
            tokens=np.zeros((1, TOKEN_DIM), dtype=np.float32),
        )
