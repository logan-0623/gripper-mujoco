from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class PolicyEvaluation(NamedTuple):
    residual: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class ResidualActorCritic(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int = 7, *, adapt_representation: bool) -> None:
        super().__init__()
        if latent_dim < 1 or action_dim < 1:
            raise ValueError("residual policy dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.adapt_representation = bool(adapt_representation)
        if adapt_representation:
            bottleneck = min(128, max(16, latent_dim // 4))
            self.adapter = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, bottleneck),
                nn.Tanh(),
                nn.Linear(bottleneck, latent_dim),
            )
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        else:
            self.adapter = None
        hidden = min(256, max(64, latent_dim // 2))
        self.trunk = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))
        self.critic = nn.Linear(hidden, 1)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def representation(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("residual latent must have shape [batch, latent_dim]")
        return latent if self.adapter is None else latent + self.adapter(latent)

    def distribution_and_value(self, latent: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        hidden = self.trunk(self.representation(latent))
        mean = self.actor_mean(hidden)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), self.critic(hidden).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal, raw: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        correction = torch.log1p(-residual.square() + 1.0e-6)
        return (distribution.log_prob(raw) - correction).sum(dim=-1)

    def sample(self, latent: torch.Tensor, *, deterministic: bool) -> PolicyEvaluation:
        distribution, value = self.distribution_and_value(latent)
        raw = distribution.mean if deterministic else distribution.sample()
        residual = torch.tanh(raw)
        log_prob = self._squashed_log_prob(distribution, raw, residual)
        return PolicyEvaluation(
            residual=residual,
            log_prob=log_prob,
            # A squashed Gaussian has no simple analytic entropy. This unbiased
            # single-sample estimate is consistent with the sampled action.
            entropy=-log_prob,
            value=value,
        )

    def evaluate(self, latent: torch.Tensor, residual: torch.Tensor) -> PolicyEvaluation:
        distribution, value = self.distribution_and_value(latent)
        bounded = residual.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        raw = torch.atanh(bounded)
        log_prob = self._squashed_log_prob(distribution, raw, bounded)
        return PolicyEvaluation(
            residual=bounded,
            log_prob=log_prob,
            entropy=-log_prob,
            value=value,
        )


def combine_residual_action(
    base_action: object, residual: object, scale: object
) -> np.ndarray:
    base = np.asarray(base_action, dtype=np.float64)
    delta = np.asarray(residual, dtype=np.float64)
    alpha = np.asarray(scale, dtype=np.float64)
    if base.shape != (7,) or delta.shape != (7,) or alpha.shape != (7,):
        raise ValueError("base action, residual, and scale must have shape [7]")
    if not np.isfinite(base).all() or not np.isfinite(delta).all() or not np.isfinite(alpha).all():
        raise ValueError("residual action inputs must be finite")
    if np.any(alpha < 0.0):
        raise ValueError("residual scales must be non-negative")
    result = base + alpha * delta
    result[:6] = np.clip(result[:6], -1.0, 1.0)
    result[6] = np.clip(result[6], 0.0, 1.0)
    return result.astype(np.float32)


def generalized_advantage_estimate(
    rewards: object,
    values: object,
    dones: object,
    *,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    reward = np.asarray(rewards, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    done = np.asarray(dones, dtype=np.bool_)
    if reward.ndim != 1 or value.shape != reward.shape or done.shape != reward.shape or not len(reward):
        raise ValueError("GAE inputs must be aligned non-empty vectors")
    if not np.isfinite(reward).all() or not np.isfinite(value).all() or not np.isfinite(last_value):
        raise ValueError("GAE inputs must be finite")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("GAE gamma/lambda must lie within [0, 1]")
    advantages = np.zeros_like(reward)
    running = 0.0
    next_value = float(last_value)
    for index in range(len(reward) - 1, -1, -1):
        nonterminal = 0.0 if done[index] else 1.0
        delta = reward[index] + gamma * next_value * nonterminal - value[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
        next_value = value[index]
    returns = advantages + value
    return advantages.astype(np.float32), returns.astype(np.float32)


def clipped_ppo_loss(
    evaluation: PolicyEvaluation,
    *,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = (evaluation.log_prob - old_log_prob).exp()
    normalized_advantage = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    unclipped = ratio * normalized_advantage
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * normalized_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (evaluation.value - returns).square().mean()
    entropy = evaluation.entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "approx_kl": float((old_log_prob - evaluation.log_prob).mean().detach().item()),
        "clip_fraction": float(((ratio - 1.0).abs() > clip_coef).float().mean().detach().item()),
    }


def normalized_curve_auc(steps: object, values: object, *, budget: int) -> float:
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("learning curve AUC requires aligned vectors with at least two points")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or budget <= 0:
        raise ValueError("learning curve AUC inputs must be finite and budget positive")
    if x[0] != 0 or np.any(np.diff(x) <= 0) or x[-1] > budget:
        raise ValueError("learning curve steps must start at zero, increase, and remain within budget")
    if x[-1] < budget:
        x = np.append(x, float(budget))
        y = np.append(y, y[-1])
    return float(np.trapezoid(y, x) / budget)
