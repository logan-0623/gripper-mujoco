from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from interaction_vla.representation_study.rl.actors import LatentResidualActor
from interaction_vla.representation_study.rl.critics import OracleTwinQ
from interaction_vla.representation_study.rl.sac import SAC, SACBatch, sac_target
from interaction_vla.representation_study.rl.v2_config import SACConfig


def _config() -> SACConfig:
    return SACConfig(
        replay_capacity=512,
        warmup_steps=32,
        batch_size=16,
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        temperature_learning_rate=3.0e-4,
        polyak=0.005,
        updates_per_environment_step=1,
        target_entropy=-7.0,
    )


def _batch(rows: int = 16) -> SACBatch:
    return SACBatch(
        actor_observation=torch.linspace(-1.0, 1.0, rows * 16).reshape(rows, 16),
        next_actor_observation=torch.linspace(1.0, -1.0, rows * 16).reshape(rows, 16),
        oracle_state=torch.linspace(-0.5, 0.5, rows * 36).reshape(rows, 36),
        next_oracle_state=torch.linspace(0.5, -0.5, rows * 36).reshape(rows, 36),
        residual=torch.linspace(-0.2, 0.2, rows * 7).reshape(rows, 7),
        reward=torch.linspace(-1.0, 1.0, rows),
        done=(torch.arange(rows) % 5 == 0).float(),
    )


def _backend() -> SAC:
    return SAC(
        actor=LatentResidualActor(latent_dim=16),
        critics=OracleTwinQ(state_dim=36, action_dim=7),
        config=_config(),
        gamma=0.99,
    )


def test_sac_target_uses_minimum_target_q() -> None:
    target = sac_target(
        reward=torch.tensor([1.0]),
        done=torch.tensor([0.0]),
        next_log_prob=torch.tensor([-0.2]),
        q1=torch.tensor([3.0]),
        q2=torch.tensor([2.0]),
        alpha=torch.tensor(0.1),
        gamma=0.99,
    )
    assert target.item() == pytest.approx(1.0 + 0.99 * (2.0 + 0.02))


def test_one_sac_update_is_finite_and_q_path_is_isolated() -> None:
    torch.manual_seed(5)
    backend = _backend()
    report = backend.update(_batch())
    assert all(np.isfinite(value) for value in asdict(report).values())
    assert report.critic_gradient_on_actor == 0.0
    assert report.critic_gradient_on_representation == 0.0


def test_sac_temperature_is_created_on_the_policy_device() -> None:
    actor = LatentResidualActor(latent_dim=16)
    critics = OracleTwinQ(state_dim=36, action_dim=7)
    backend = SAC(actor=actor, critics=critics, config=_config(), gamma=0.99)

    assert backend.log_alpha.device == next(actor.parameters()).device
    assert backend.log_alpha.device == next(critics.parameters()).device


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    reason="accelerator is unavailable",
)
def test_sac_temperature_is_created_on_a_non_cpu_policy_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    actor = LatentResidualActor(latent_dim=16).to(device)
    critics = OracleTwinQ(state_dim=36, action_dim=7).to(device)
    backend = SAC(actor=actor, critics=critics, config=_config(), gamma=0.99)

    assert backend.log_alpha.device == device


def test_sac_actor_pass_can_update_declared_representation() -> None:
    torch.manual_seed(7)
    adapter = torch.nn.Linear(16, 16, bias=False)
    actor = LatentResidualActor(latent_dim=16)
    with torch.no_grad():
        actor.mean.weight.normal_(mean=0.0, std=1.0e-2)
    backend = SAC(
        actor=actor,
        critics=OracleTwinQ(state_dim=36, action_dim=7),
        config=_config(),
        gamma=0.99,
        representation_parameters=tuple(adapter.parameters()),
    )
    raw = _batch()
    batch = SACBatch(
        actor_observation=adapter(raw.actor_observation),
        next_actor_observation=adapter(raw.next_actor_observation).detach(),
        oracle_state=raw.oracle_state,
        next_oracle_state=raw.next_oracle_state,
        residual=raw.residual,
        reward=raw.reward,
        done=raw.done,
    )
    before = adapter.weight.detach().clone()
    backend.update(batch)
    assert not torch.equal(adapter.weight.detach(), before)


def test_sac_state_resume_matches_uninterrupted_updates() -> None:
    torch.manual_seed(13)
    uninterrupted = _backend()
    initial = deepcopy(uninterrupted.state_dict())
    uninterrupted.update(_batch())
    uninterrupted.update(_batch())

    split = _backend()
    split.load_state_dict(initial)
    split.update(_batch())
    checkpoint = deepcopy(split.state_dict())
    resumed = _backend()
    resumed.load_state_dict(checkpoint)
    resumed.update(_batch())

    expected = uninterrupted.state_dict()
    actual = resumed.state_dict()
    for group in ("actor", "critics", "target_critics"):
        for name, tensor in expected[group].items():
            torch.testing.assert_close(tensor, actual[group][name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(expected["log_alpha"], actual["log_alpha"], rtol=0.0, atol=0.0)
