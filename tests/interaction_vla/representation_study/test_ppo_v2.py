from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from interaction_vla.representation_study.rl.actors import LatentResidualActor
from interaction_vla.representation_study.rl.critics import OracleValueCritic
from interaction_vla.representation_study.rl.ppo_v2 import OnPolicyBatch, PPOV2
from interaction_vla.representation_study.rl.v2_config import PPOV2Config


def _config() -> PPOV2Config:
    return PPOV2Config(
        rollout_steps=64,
        update_epochs=2,
        minibatch_size=16,
        actor_learning_rate=3.0e-4,
        value_learning_rate=3.0e-4,
        gae_lambda=0.95,
        clip_coefficient=0.20,
        entropy_coefficient=0.01,
        max_grad_norm=1.0,
    )


def _batch(actor: LatentResidualActor, rows: int = 32) -> OnPolicyBatch:
    observation = torch.linspace(-1.0, 1.0, rows * 16).reshape(rows, 16)
    with torch.no_grad():
        sampled = actor.sample(observation, deterministic=True)
    return OnPolicyBatch(
        actor_observation=observation,
        oracle_state=torch.linspace(-0.5, 0.5, rows * 36).reshape(rows, 36),
        residual=sampled.residual,
        old_log_prob=sampled.log_prob,
        advantages=torch.linspace(-1.0, 1.0, rows),
        returns=torch.linspace(0.0, 1.0, rows),
    )


def _backend() -> PPOV2:
    return PPOV2(
        actor=LatentResidualActor(latent_dim=16),
        critic=OracleValueCritic(state_dim=36),
        config=_config(),
    )


def test_ppo_v2_update_is_finite_and_critic_isolated() -> None:
    torch.manual_seed(3)
    backend = _backend()
    report = backend.update(_batch(backend.actor))
    assert np.isfinite(report.policy_loss)
    assert np.isfinite(report.value_loss)
    assert report.critic_gradient_on_actor == 0.0


def test_ppo_v2_actor_pass_can_update_declared_representation_only() -> None:
    torch.manual_seed(3)
    adapter = torch.nn.Linear(16, 16, bias=False)
    actor = LatentResidualActor(latent_dim=16)
    with torch.no_grad():
        actor.mean.weight.normal_(mean=0.0, std=1.0e-2)
    backend = PPOV2(
        actor=actor,
        critic=OracleValueCritic(state_dim=36),
        config=_config(),
        representation_parameters=tuple(adapter.parameters()),
    )
    batch = _batch(actor)
    batch = OnPolicyBatch(
        actor_observation=adapter(batch.actor_observation),
        oracle_state=batch.oracle_state,
        residual=batch.residual,
        old_log_prob=batch.old_log_prob,
        advantages=batch.advantages,
        returns=batch.returns,
    )
    before = adapter.weight.detach().clone()
    report = backend.update(batch)
    assert report.critic_gradient_on_representation == 0.0
    assert not torch.equal(adapter.weight.detach(), before)


def test_ppo_v2_state_resume_matches_uninterrupted_updates() -> None:
    torch.manual_seed(11)
    uninterrupted = _backend()
    initial = deepcopy(uninterrupted.state_dict())
    resumed = _backend()
    resumed.load_state_dict(initial)
    first_batch = _batch(uninterrupted.actor)
    uninterrupted.update(first_batch)
    second_batch = _batch(uninterrupted.actor)
    uninterrupted.update(second_batch)

    split = _backend()
    split.load_state_dict(initial)
    split.update(_batch(split.actor))
    checkpoint = deepcopy(split.state_dict())
    resumed.load_state_dict(checkpoint)
    resumed.update(_batch(resumed.actor))

    expected = uninterrupted.state_dict()
    actual = resumed.state_dict()
    for group in ("actor", "critic"):
        for name, tensor in expected[group].items():
            torch.testing.assert_close(tensor, actual[group][name], rtol=0.0, atol=0.0)
