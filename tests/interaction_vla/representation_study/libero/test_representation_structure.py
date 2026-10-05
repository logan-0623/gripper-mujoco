from types import SimpleNamespace

import numpy as np

from interaction_vla.representation_study.libero.representation_structure import (
    build_targets, grouped_probe, select_tokens, transition_consistency, transition_triplets)


def _record(episode, frame, distance=0.5, grasp=False):
    return SimpleNamespace(
        suite="libero_spatial", task_id=0, source_episode_id=episode, frame_index=frame,
        observation=SimpleNamespace(robot_state=[float(frame), 0.0]),
        labels=SimpleNamespace(geometry=SimpleNamespace(gripper_target_distance=distance,
                                                        target_goal_distance=1.0 - distance),
                               stable_grasp=grasp))


def test_future_targets_use_the_dense_bank_and_stop_at_episode_end():
    bank = [_record(0, f, distance=0.1 * f, grasp=f >= 2) for f in range(4)]
    by_frame = {(r.suite, r.task_id, r.source_episode_id, r.frame_index): r for r in bank}
    targets = build_targets(bank[:3], by_frame, horizon=1)
    np.testing.assert_allclose(targets["delta_gripper_target_distance_k1"][1], [0.1] * 3)
    np.testing.assert_allclose(targets["stable_grasp_k1"][1], [0.0, 1.0, 1.0])
    assert np.isnan(build_targets(bank[3:], by_frame, horizon=1)["stable_grasp_k1"][1][0])


def test_token_selection_never_averages_beyond_the_requested_tokens():
    values = np.zeros((2, 3, 50, 4))
    values[:, :, 0] = 1.0
    np.testing.assert_allclose(select_tokens(values, "token0"), 1.0)
    np.testing.assert_allclose(select_tokens(values, "executed_mean"), 0.1)
    np.testing.assert_allclose(select_tokens(values, "all_mean"), 0.02)


def test_hidden_information_beyond_controls_gives_gain_and_finite_fragility():
    rng = np.random.default_rng(0)
    n, groups = 400, np.repeat(np.arange(20), 20)
    controls = rng.normal(size=(n, 3))
    latent = rng.normal(size=n)
    features = np.column_stack((latent, rng.normal(scale=0.1, size=(n, 39))))
    result = grouped_probe(features, controls, latent, "regression", groups, folds=5, pca_dim=8, seed=0)
    assert result["status"] == "complete"
    assert result["conditional_gain"] > 0.5
    assert 0 < result["fragility_sigma"] < np.inf
    redundant = grouped_probe(features, controls, controls[:, 0], "regression", groups,
                              folds=5, pca_dim=8, seed=0)
    assert abs(redundant["conditional_gain"]) < 0.05


def test_probe_reports_too_few_episodes_instead_of_scoring():
    result = grouped_probe(np.ones((10, 4)), np.ones((10, 2)), np.arange(10.0), "regression",
                           [0] * 10, folds=5, pca_dim=2, seed=0)
    assert result["status"] == "not_estimable"


def test_linear_dynamics_are_composition_consistent_and_sparse_traces_are_flagged():
    rng = np.random.default_rng(1)
    rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0] * 0.95
    rows, features = [], []
    for episode in range(12):
        state = rng.normal(size=6)
        for frame in range(0, 12, 2):
            rows.append(_record(episode, frame))
            features.append(state.copy())
            state = rotation @ state
    features = np.asarray(features)
    assert len(transition_triplets(rows, 2)) == 12 * 4
    result = transition_consistency(features, rows, 2, minimum=10, pca_dim=6, seed=0)
    assert result["status"] == "complete"
    assert result["composition_over_direct"] < 1.5
    sparse = transition_consistency(features, rows, 3, minimum=10, pca_dim=6, seed=0)
    assert sparse["status"] == "not_estimable"
