from __future__ import annotations

import torch
from torch import nn


class OracleValueCritic(nn.Module):
    def __init__(self, state_dim: int = 36) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("value critic state_dim must be positive")
        self.state_dim = int(state_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"value state must have shape [batch, {self.state_dim}]"
            )
        return self.value(self.encoder(state)).squeeze(-1)


def _q_network(width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


class OracleTwinQ(nn.Module):
    def __init__(self, state_dim: int = 36, action_dim: int = 7) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 1:
            raise ValueError("Q critic dimensions must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        width = state_dim + action_dim
        self.q1 = _q_network(width)
        self.q2 = _q_network(width)

    def forward(
        self,
        state: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"Q state must have shape [batch, {self.state_dim}]"
            )
        if residual.shape != (state.shape[0], self.action_dim):
            raise ValueError(
                f"Q residual must have shape [batch, {self.action_dim}]"
            )
        features = torch.cat((state, residual), dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)
