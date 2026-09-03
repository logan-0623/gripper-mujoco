import numpy as np
import torch

from interaction_vla.representation_study.libero.sparse_autoencoder import (
    TopKSparseAutoencoder,
    load_sparse_autoencoder,
    match_decoder_features,
    save_sparse_autoencoder,
    train_sparse_autoencoder,
)
from interaction_vla.representation_study.libero.feature_discovery import (
    feature_removal_delta,
    matched_random_feature_delta,
    select_stable_features,
    sparse_feature_decision,
    summarize_action_usage,
    summarize_activations,
)


def test_topk_sparse_autoencoder_round_trip_and_matching(tmp_path) -> None:
    torch.manual_seed(7)
    model = TopKSparseAutoencoder(input_dim=6, feature_dim=12, top_k=3)
    values = torch.randn(9, 6)
    _, activations = model(values)
    assert torch.all((activations > 0).sum(dim=1) <= 3)

    path = tmp_path / "sae.npz"
    save_sparse_autoencoder(path, model, mean=np.arange(6), scale=np.ones(6))
    restored, mean, scale = load_sparse_autoencoder(path)
    assert np.array_equal(mean, np.arange(6))
    assert np.array_equal(scale, np.ones(6))
    assert torch.equal(model.encoder.weight, restored.encoder.weight)
    assert torch.equal(model.decoder.weight, restored.decoder.weight)

    reference = np.eye(4, dtype=np.float64)
    candidate = reference[[2, 0, 3, 1]]
    matches = match_decoder_features(reference, candidate)
    assert matches == ((0, 1, 1.0), (1, 3, 1.0), (2, 0, 1.0), (3, 2, 1.0))

    data = np.random.default_rng(4).normal(size=(24, 6)).astype(np.float32)
    first = train_sparse_autoencoder(
        data, feature_dim=12, top_k=3, steps=5, batch_size=8, seed=9, device="cpu", progress=False
    )[0]
    second = train_sparse_autoencoder(
        data, feature_dim=12, top_k=3, steps=5, batch_size=8, seed=9, device="cpu", progress=False
    )[0]
    assert torch.equal(first.encoder.weight, second.encoder.weight)


def test_label_blind_feature_profiles_and_stable_selection() -> None:
    rng = np.random.default_rng(11)
    activations = np.abs(rng.normal(size=(40, 4)))
    tasks = np.repeat(["a", "b"], 20)
    episodes = np.repeat([f"e{i}" for i in range(8)], 5)
    frames = np.tile(np.arange(5), 8)
    profiles = summarize_activations(activations, tasks, episodes, frames, seed=5)
    assert profiles[0]["task_count"] == 2
    assert profiles[0]["episode_count"] == 8
    assert 0 <= profiles[0]["temporal_difference_ratio"]

    permutation = np.asarray([2, 0, 3, 1])
    decoders = np.eye(4)
    candidates = select_stable_features(
        [activations, activations[:, permutation], activations[:, permutation]],
        [decoders, decoders[permutation], decoders[permutation]],
        profiles,
        limit=3,
        min_tasks=2,
        min_episodes=4,
        min_decoder_cosine=0.99,
        min_activation_correlation=0.99,
    )
    assert len(candidates) == 3
    assert all(row["minimum_decoder_cosine"] == 1.0 for row in candidates)


def test_feature_removal_and_causal_decision() -> None:
    delta = feature_removal_delta(
        np.asarray([2.0, 0.5]),
        np.asarray([1.0, -2.0]),
        np.asarray([3.0, 4.0]),
    )
    assert np.array_equal(delta, [[-6.0, 16.0], [-1.5, 4.0]])
    assert sparse_feature_decision(candidate_count=0, causal_rows=[]) == "stop_no_stable_features"
    assert sparse_feature_decision(
        candidate_count=4,
        causal_rows=[{"ci_low": -0.1, "q_value": 0.01}],
    ) == "stop_no_causal_features"
    assert sparse_feature_decision(
        candidate_count=4,
        causal_rows=[{"ci_low": 0.01, "q_value": 0.04}],
    ) == "authorize_separate_longitudinal_design"


def test_matched_random_and_episode_clustered_action_usage() -> None:
    target = np.asarray([[3.0, 4.0], [0.6, 0.8], [1.5, 2.0], [0.3, 0.4]])
    random = matched_random_feature_delta(target, direction=np.asarray([3.0, 4.0]), seed=3)
    assert np.allclose(np.linalg.norm(target, axis=1), np.linalg.norm(random, axis=1))
    assert np.allclose(random @ np.asarray([3.0, 4.0]), 0.0)

    summary = summarize_action_usage(
        np.asarray([0.4, 0.5, 0.2, 0.3]),
        np.asarray([0.1, 0.1, 0.1, 0.1]),
        ["e0", "e0", "e1", "e1"],
        samples=200,
        confidence=0.95,
        seed=5,
    )
    assert summary["target_minus_random"]["ci_low"] > 0
    assert 0 <= summary["p_value"] <= 1
