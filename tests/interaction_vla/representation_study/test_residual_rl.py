from __future__ import annotations

import numpy as np
import pytest
import torch

from interaction_vla.representation_study.rl.core import (
    ResidualActorCritic,
    clipped_ppo_loss,
    combine_residual_action,
    generalized_advantage_estimate,
    normalized_curve_auc,
)
from interaction_vla.representation_study.rl.checkpoint import (
    capture_rng_state,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_residual_actor_is_initialized_as_zero_mean_and_optional_adapter_identity() -> None:
    latent = torch.randn(3, 16)
    head = ResidualActorCritic(16, adapt_representation=False)
    representation = ResidualActorCritic(16, adapt_representation=True)
    assert torch.allclose(head.sample(latent, deterministic=True).residual, torch.zeros(3, 7))
    assert torch.allclose(representation.representation(latent), latent)


def test_residual_action_respects_policy_bounds() -> None:
    result = combine_residual_action(
        np.asarray([0.95] * 6 + [0.9]), np.ones(7), np.asarray([0.2] * 7)
    )
    assert np.all(result[:6] <= 1.0)
    assert result[6] == 1.0


def test_gae_stops_bootstrap_at_episode_boundaries() -> None:
    advantages, returns = generalized_advantage_estimate(
        [0.0, 1.0, 0.0], [0.2, 0.3, 0.4], [False, True, False],
        last_value=0.5, gamma=1.0, gae_lambda=1.0
    )
    assert np.allclose(returns[:2], [1.0, 1.0])
    assert np.isclose(returns[2], 0.5)


def test_clipped_ppo_loss_is_finite_and_differentiable() -> None:
    policy = ResidualActorCritic(8, adapt_representation=False)
    latent = torch.randn(6, 8)
    sampled = policy.sample(latent, deterministic=False)
    evaluation = policy.evaluate(latent, sampled.residual)
    loss, metrics = clipped_ppo_loss(
        evaluation,
        old_log_prob=sampled.log_prob.detach(),
        advantages=torch.randn(6),
        returns=torch.randn(6),
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert np.isfinite(list(metrics.values())).all()


def test_squashed_gaussian_actions_are_bounded_and_re_evaluate_exactly() -> None:
    torch.manual_seed(7)
    policy = ResidualActorCritic(8, adapt_representation=False)
    latent = torch.randn(32, 8)

    sampled = policy.sample(latent, deterministic=False)
    evaluated = policy.evaluate(latent, sampled.residual)

    assert torch.all(sampled.residual > -1.0)
    assert torch.all(sampled.residual < 1.0)
    assert torch.allclose(sampled.log_prob, evaluated.log_prob, atol=1e-5, rtol=1e-5)


def test_normalized_auc_uses_fixed_budget() -> None:
    assert normalized_curve_auc([0, 5, 10], [0.0, 0.5, 1.0], budget=10) == pytest.approx(0.5)
    assert normalized_curve_auc([0, 5], [0.0, 1.0], budget=10) == pytest.approx(0.75)


def test_residual_checkpoint_round_trip_restores_progress(tmp_path) -> None:
    policy = ResidualActorCritic(8, adapt_representation=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    path = tmp_path / "training_state.pt"
    save_training_checkpoint(
        path,
        policy=policy,
        optimizer=optimizer,
        policy_state=None,
        environment_steps=32,
        update=2,
        curve=[{"steps": 0, "success_rate": 0.0}],
        rng_state=capture_rng_state(),
        metadata={"backend": "act", "stage": "rl_head"},
    )
    loaded = load_training_checkpoint(path, map_location="cpu")
    assert loaded["environment_steps"] == 32
    assert loaded["update"] == 2
    assert loaded["metadata"]["stage"] == "rl_head"
