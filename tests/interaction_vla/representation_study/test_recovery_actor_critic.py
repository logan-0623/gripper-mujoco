from __future__ import annotations

import torch

from interaction_vla.representation_study.rl.actors import (
    LatentResidualActor,
    OracleResidualActor,
)
from interaction_vla.representation_study.rl.critics import (
    OracleTwinQ,
    OracleValueCritic,
)


def test_critic_loss_never_reaches_actor() -> None:
    actor = LatentResidualActor(latent_dim=32)
    critic = OracleValueCritic(state_dim=36)
    critic(torch.randn(8, 36)).square().mean().backward()
    assert all(parameter.grad is None for parameter in actor.parameters())


def test_sac_q_loss_does_not_update_visual_actor() -> None:
    actor = LatentResidualActor(latent_dim=32)
    critics = OracleTwinQ(state_dim=36, action_dim=7)
    residual = actor.sample(
        torch.randn(8, 32), deterministic=False
    ).residual.detach()
    q1, q2 = critics(torch.randn(8, 36), residual)
    (q1.square().mean() + q2.square().mean()).backward()
    assert all(parameter.grad is None for parameter in actor.parameters())


def test_latent_actor_sample_is_bounded_and_reparameterized() -> None:
    actor = LatentResidualActor(latent_dim=16)
    latent = torch.randn(5, 16, requires_grad=True)
    sample = actor.sample(latent, deterministic=False)
    assert sample.residual.shape == (5, 7)
    assert sample.log_prob.shape == (5,)
    assert torch.isfinite(sample.log_prob).all()
    assert torch.all(sample.residual.abs() <= 1.0)
    sample.residual.square().mean().backward()
    assert latent.grad is not None


def test_oracle_actor_and_critics_have_registered_shapes() -> None:
    state = torch.randn(4, 36)
    actor = OracleResidualActor()
    value = OracleValueCritic()
    q = OracleTwinQ()
    residual = actor.sample(state, deterministic=True).residual
    q1, q2 = q(state, residual)
    assert residual.shape == (4, 7)
    assert value(state).shape == (4,)
    assert q1.shape == q2.shape == (4,)
