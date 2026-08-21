from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Iterator, Mapping, Sequence

import torch

from .actors import ResidualActor
from .anchoring import latent_drift_loss, nominal_residual_loss
from .critics import OracleTwinQ
from .v2_config import SACConfig


SAC_STATE_SCHEMA = "isolated_sac_v1"


@dataclass(frozen=True)
class SACBatch:
    actor_observation: torch.Tensor
    next_actor_observation: torch.Tensor
    oracle_state: torch.Tensor
    next_oracle_state: torch.Tensor
    residual: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    nominal_actor_observation: torch.Tensor | None = None
    representation_latent: torch.Tensor | None = None
    sft_target_latent: torch.Tensor | None = None

    def validate(self, *, actor_dim: int, state_dim: int, action_dim: int) -> int:
        if self.actor_observation.ndim != 2:
            raise ValueError("SAC actor observation must be a matrix")
        rows = int(self.actor_observation.shape[0])
        expected = {
            "actor_observation": (rows, actor_dim),
            "next_actor_observation": (rows, actor_dim),
            "oracle_state": (rows, state_dim),
            "next_oracle_state": (rows, state_dim),
            "residual": (rows, action_dim),
            "reward": (rows,),
            "done": (rows,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"SAC {name} must be finite with shape {shape}")
        if torch.any((self.done < 0.0) | (self.done > 1.0)):
            raise ValueError("SAC done mask must remain within [0, 1]")
        if self.nominal_actor_observation is not None and (
            self.nominal_actor_observation.ndim != 2
            or self.nominal_actor_observation.shape[1] != actor_dim
            or not torch.isfinite(self.nominal_actor_observation).all()
        ):
            raise ValueError("SAC nominal actor observation is incompatible")
        if (self.representation_latent is None) != (self.sft_target_latent is None):
            raise ValueError("SAC latent anchoring requires current and SFT latents")
        if self.representation_latent is not None and (
            self.representation_latent.ndim != 2
            or self.representation_latent.shape != self.sft_target_latent.shape
            or not torch.isfinite(self.representation_latent).all()
            or not torch.isfinite(self.sft_target_latent).all()
        ):
            raise ValueError("SAC latent anchor tensors are incompatible")
        return rows


@dataclass(frozen=True)
class SACUpdateReport:
    critic_loss: float
    actor_loss: float
    temperature_loss: float
    alpha: float
    mean_target_q: float
    mean_policy_q: float
    mean_log_prob: float
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


@contextmanager
def _frozen(parameters: Sequence[torch.nn.Parameter]) -> Iterator[None]:
    previous = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in zip(parameters, previous, strict=True):
            parameter.requires_grad_(required)


def sac_target(
    *,
    reward: torch.Tensor,
    done: torch.Tensor,
    next_log_prob: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    tensors = (reward, done, next_log_prob, q1, q2)
    if not tensors or any(value.shape != reward.shape for value in tensors):
        raise ValueError("SAC target inputs must share shape")
    if not all(torch.isfinite(value).all() for value in (*tensors, alpha)):
        raise ValueError("SAC target inputs must be finite")
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("SAC gamma must lie within (0, 1]")
    with torch.no_grad():
        soft_value = torch.minimum(q1, q2) - alpha.detach() * next_log_prob
        return reward + gamma * (1.0 - done) * soft_value


class SAC:
    def __init__(
        self,
        *,
        actor: ResidualActor,
        critics: OracleTwinQ,
        config: SACConfig,
        gamma: float,
        representation_parameters: Sequence[torch.nn.Parameter] = (),
        representation_learning_rate: float | None = None,
        nominal_anchor_coefficient: float = 1.0,
        latent_anchor_coefficient: float = 0.10,
        initial_temperature: float = 0.10,
    ) -> None:
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("SAC gamma must lie within (0, 1]")
        if initial_temperature <= 0.0 or not math.isfinite(initial_temperature):
            raise ValueError("SAC initial temperature must be finite and positive")
        if nominal_anchor_coefficient < 0.0 or latent_anchor_coefficient < 0.0:
            raise ValueError("SAC anchor coefficients must be non-negative")
        self.actor = actor
        self.critics = critics
        self.target_critics = deepcopy(critics)
        self.target_critics.requires_grad_(False)
        self.config = config
        self.gamma = float(gamma)
        self.nominal_anchor_coefficient = float(nominal_anchor_coefficient)
        self.latent_anchor_coefficient = float(latent_anchor_coefficient)
        self.representation_parameters = tuple(representation_parameters)
        actor_parameters = tuple(actor.parameters())
        critic_parameters = tuple(critics.parameters())
        actor_ids = {id(value) for value in actor_parameters}
        representation_ids = {id(value) for value in self.representation_parameters}
        critic_ids = {id(value) for value in critic_parameters}
        if actor_ids & representation_ids:
            raise ValueError("SAC actor and representation parameters must be disjoint")
        if critic_ids & (actor_ids | representation_ids):
            raise ValueError("SAC critics must not share policy parameters")
        actor_groups: list[dict[str, object]] = [
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
            if learning_rate <= 0.0 or not math.isfinite(learning_rate):
                raise ValueError("SAC representation learning rate must be positive")
            actor_groups.append(
                {
                    "params": self.representation_parameters,
                    "lr": learning_rate,
                    "name": "policy_representation",
                }
            )
        self.actor_optimizer = torch.optim.Adam(actor_groups)
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=config.critic_learning_rate,
        )
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(initial_temperature), dtype=torch.float32)
        )
        self.temperature_optimizer = torch.optim.Adam(
            (self.log_alpha,),
            lr=config.temperature_learning_rate,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: SACBatch) -> SACUpdateReport:
        batch.validate(
            actor_dim=self.actor.input_dim,
            state_dim=self.critics.state_dim,
            action_dim=self.actor.action_dim,
        )
        actor_parameters = tuple(self.actor.parameters())
        critic_parameters = tuple(self.critics.parameters())
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_sample = self.actor.sample(
                batch.next_actor_observation.detach(),
                deterministic=False,
            )
            target_q1, target_q2 = self.target_critics(
                batch.next_oracle_state,
                next_sample.residual,
            )
            target = sac_target(
                reward=batch.reward,
                done=batch.done,
                next_log_prob=next_sample.log_prob,
                q1=target_q1,
                q2=target_q2,
                alpha=self.alpha,
                gamma=self.gamma,
            )
        q1, q2 = self.critics(batch.oracle_state, batch.residual.detach())
        critic_loss = 0.5 * (
            (q1 - target).square().mean() + (q2 - target).square().mean()
        )
        critic_loss.backward()
        critic_on_actor = _maximum_gradient(actor_parameters)
        critic_on_representation = _maximum_gradient(
            self.representation_parameters
        )
        critic_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(critic_parameters, 10.0).item()
        )
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        sample = self.actor.sample(batch.actor_observation, deterministic=False)
        with _frozen(critic_parameters):
            policy_q1, policy_q2 = self.critics(
                batch.oracle_state,
                sample.residual,
            )
            policy_q = torch.minimum(policy_q1, policy_q2)
            policy_loss = (self.alpha.detach() * sample.log_prob - policy_q).mean()
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
            + self.nominal_anchor_coefficient * nominal_anchor
            + self.latent_anchor_coefficient * latent_anchor
        )
        actor_loss.backward()
        actor_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (*actor_parameters, *self.representation_parameters),
                10.0,
            ).item()
        )
        self.actor_optimizer.step()

        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss = -(
            self.log_alpha * (sample.log_prob.detach() + self.config.target_entropy)
        ).mean()
        temperature_loss.backward()
        self.temperature_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critics.parameters(),
                self.critics.parameters(),
                strict=True,
            ):
                target_parameter.mul_(1.0 - self.config.polyak)
                target_parameter.add_(parameter, alpha=self.config.polyak)

        report = SACUpdateReport(
            critic_loss=float(critic_loss.detach().item()),
            actor_loss=float(actor_loss.detach().item()),
            temperature_loss=float(temperature_loss.detach().item()),
            alpha=float(self.alpha.detach().item()),
            mean_target_q=float(target.mean().detach().item()),
            mean_policy_q=float(policy_q.mean().detach().item()),
            mean_log_prob=float(sample.log_prob.mean().detach().item()),
            nominal_anchor_loss=float(nominal_anchor.detach().item()),
            latent_anchor_loss=float(latent_anchor.detach().item()),
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            critic_gradient_on_actor=critic_on_actor,
            critic_gradient_on_representation=critic_on_representation,
        )
        if not all(math.isfinite(value) for value in report.__dict__.values()):
            raise FloatingPointError("SAC update produced a non-finite diagnostic")
        return report

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": SAC_STATE_SCHEMA,
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != SAC_STATE_SCHEMA:
            raise ValueError("SAC state schema is incompatible")
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["target_critics"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
