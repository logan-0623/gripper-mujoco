from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .actors import ResidualActor
from .anchoring import latent_drift_loss, nominal_residual_loss
from .critics import OracleValueCritic
from .v2_config import PPOV2Config


PPO_V2_STATE_SCHEMA = "isolated_ppo_v2"


@dataclass(frozen=True)
class OnPolicyBatch:
    actor_observation: torch.Tensor
    oracle_state: torch.Tensor
    residual: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    nominal_actor_observation: torch.Tensor | None = None
    representation_latent: torch.Tensor | None = None
    sft_target_latent: torch.Tensor | None = None

    def validate(self, *, actor_dim: int, state_dim: int, action_dim: int) -> int:
        if self.actor_observation.ndim != 2:
            raise ValueError("PPO actor observation must be a matrix")
        rows = int(self.actor_observation.shape[0])
        expected = {
            "actor_observation": (rows, actor_dim),
            "oracle_state": (rows, state_dim),
            "residual": (rows, action_dim),
            "old_log_prob": (rows,),
            "advantages": (rows,),
            "returns": (rows,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"PPO {name} must be finite with shape {shape}")
        if self.nominal_actor_observation is not None and (
            self.nominal_actor_observation.ndim != 2
            or self.nominal_actor_observation.shape[1] != actor_dim
            or not torch.isfinite(self.nominal_actor_observation).all()
        ):
            raise ValueError("PPO nominal actor observation is incompatible")
        if (self.representation_latent is None) != (self.sft_target_latent is None):
            raise ValueError("PPO latent anchoring requires current and SFT latents")
        if self.representation_latent is not None and (
            self.representation_latent.shape != self.sft_target_latent.shape
            or self.representation_latent.ndim != 2
        ):
            raise ValueError("PPO latent anchor tensors are incompatible")
        return rows


@dataclass(frozen=True)
class PPOUpdateReport:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    nominal_anchor_loss: float
    latent_anchor_loss: float
    actor_grad_norm: float
    critic_grad_norm: float
    critic_gradient_on_actor: float
    critic_gradient_on_representation: float


def _maximum_gradient(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [
        float(parameter.grad.detach().abs().max().item())
        for parameter in parameters
        if parameter.grad is not None
    ]
    return max(values, default=0.0)


class PPOV2:
    def __init__(
        self,
        *,
        actor: ResidualActor,
        critic: OracleValueCritic,
        config: PPOV2Config,
        representation_parameters: Sequence[torch.nn.Parameter] = (),
        representation_learning_rate: float | None = None,
        nominal_anchor_coefficient: float = 1.0,
        latent_anchor_coefficient: float = 0.10,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.representation_parameters = tuple(representation_parameters)
        actor_parameters = tuple(actor.parameters())
        if {id(value) for value in actor_parameters} & {
            id(value) for value in self.representation_parameters
        }:
            raise ValueError("PPO actor and representation parameters must be disjoint")
        if nominal_anchor_coefficient < 0.0 or latent_anchor_coefficient < 0.0:
            raise ValueError("PPO anchor coefficients must be non-negative")
        self.nominal_anchor_coefficient = float(nominal_anchor_coefficient)
        self.latent_anchor_coefficient = float(latent_anchor_coefficient)
        groups: list[dict[str, object]] = [
            {
                "params": actor_parameters,
                "lr": config.actor_learning_rate,
                "name": "residual_actor",
            }
        ]
        if self.representation_parameters:
            learning_rate = (
                config.actor_learning_rate
                if representation_learning_rate is None
                else float(representation_learning_rate)
            )
            if learning_rate <= 0.0:
                raise ValueError("PPO representation learning rate must be positive")
            groups.append(
                {
                    "params": self.representation_parameters,
                    "lr": learning_rate,
                    "name": "policy_representation",
                }
            )
        self.actor_optimizer = torch.optim.Adam(groups)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config.value_learning_rate
        )

    def update(self, batch: OnPolicyBatch) -> PPOUpdateReport:
        batch.validate(
            actor_dim=self.actor.input_dim,
            state_dim=self.critic.state_dim,
            action_dim=self.actor.action_dim,
        )
        actor_parameters = tuple(self.actor.parameters())
        critic_parameters = tuple(self.critic.parameters())
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        values = self.critic(batch.oracle_state)
        value_loss = 0.5 * (values - batch.returns.detach()).square().mean()
        value_loss.backward()
        critic_on_actor = _maximum_gradient(actor_parameters)
        critic_on_representation = _maximum_gradient(
            self.representation_parameters
        )
        critic_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                critic_parameters,
                self.config.max_grad_norm,
            ).item()
        )
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        evaluation = self.actor.evaluate(
            batch.actor_observation,
            batch.residual.detach(),
        )
        ratio = (evaluation.log_prob - batch.old_log_prob.detach()).exp()
        advantages = batch.advantages.detach()
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1.0e-8)
        unclipped = ratio * advantages
        clipped = ratio.clamp(
            1.0 - self.config.clip_coefficient,
            1.0 + self.config.clip_coefficient,
        ) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = evaluation.entropy_estimate.mean()
        nominal_anchor = torch.zeros((), device=policy_loss.device)
        if batch.nominal_actor_observation is not None:
            nominal = self.actor.sample(
                batch.nominal_actor_observation,
                deterministic=True,
            ).residual
            nominal_anchor = nominal_residual_loss(nominal)
        latent_anchor = torch.zeros((), device=policy_loss.device)
        if batch.representation_latent is not None:
            assert batch.sft_target_latent is not None
            latent_anchor = latent_drift_loss(
                batch.representation_latent,
                batch.sft_target_latent,
            )
        actor_loss = (
            policy_loss
            - self.config.entropy_coefficient * entropy
            + self.nominal_anchor_coefficient * nominal_anchor
            + self.latent_anchor_coefficient * latent_anchor
        )
        actor_loss.backward()
        actor_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (*actor_parameters, *self.representation_parameters),
                self.config.max_grad_norm,
            ).item()
        )
        self.actor_optimizer.step()
        report = PPOUpdateReport(
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            entropy=float(entropy.detach().item()),
            approx_kl=float(
                (batch.old_log_prob.detach() - evaluation.log_prob)
                .mean()
                .detach()
                .item()
            ),
            clip_fraction=float(
                ((ratio - 1.0).abs() > self.config.clip_coefficient)
                .float()
                .mean()
                .detach()
                .item()
            ),
            nominal_anchor_loss=float(nominal_anchor.detach().item()),
            latent_anchor_loss=float(latent_anchor.detach().item()),
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            critic_gradient_on_actor=critic_on_actor,
            critic_gradient_on_representation=critic_on_representation,
        )
        if not all(
            torch.isfinite(torch.tensor(value))
            for value in report.__dict__.values()
        ):
            raise FloatingPointError("PPO v2 update produced a non-finite diagnostic")
        return report

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": PPO_V2_STATE_SCHEMA,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != PPO_V2_STATE_SCHEMA:
            raise ValueError("PPO v2 state schema is incompatible")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
