from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class ActorSample:
    residual: torch.Tensor
    log_prob: torch.Tensor

    @property
    def entropy_estimate(self) -> torch.Tensor:
        return -self.log_prob


class ResidualActor(nn.Module):
    input_dim: int
    action_dim: int

    def _distribution(self, observation: torch.Tensor) -> Normal:
        if observation.ndim != 2 or observation.shape[1] != self.input_dim:
            raise ValueError(
                f"actor observation must have shape [batch, {self.input_dim}]"
            )
        hidden = self.encoder(observation)
        mean = self.mean(hidden)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _log_prob(
        distribution: Normal,
        raw: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log1p(-residual.square() + 1.0e-6)
        return (distribution.log_prob(raw) - correction).sum(dim=-1)

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool,
    ) -> ActorSample:
        distribution = self._distribution(observation)
        raw = distribution.mean if deterministic else distribution.rsample()
        residual = torch.tanh(raw)
        return ActorSample(
            residual=residual,
            log_prob=self._log_prob(distribution, raw, residual),
        )

    def evaluate(
        self,
        observation: torch.Tensor,
        residual: torch.Tensor,
    ) -> ActorSample:
        distribution = self._distribution(observation)
        if residual.shape != (observation.shape[0], self.action_dim):
            raise ValueError(
                f"residual must have shape [batch, {self.action_dim}]"
            )
        bounded = residual.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        raw = torch.atanh(bounded)
        return ActorSample(
            residual=bounded,
            log_prob=self._log_prob(distribution, raw, bounded),
        )


class OracleResidualActor(ResidualActor):
    def __init__(self, state_dim: int = 36, action_dim: int = 7) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 1:
            raise ValueError("Oracle actor dimensions must be positive")
        self.input_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)


class LatentResidualActor(ResidualActor):
    def __init__(self, latent_dim: int, action_dim: int = 7) -> None:
        super().__init__()
        if latent_dim < 1 or action_dim < 1:
            raise ValueError("latent actor dimensions must be positive")
        self.input_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
