from __future__ import annotations

import torch
from torch import Tensor, nn

from interaction_vla.graph.schema import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

from .encoders import SceneBatch, build_matched_encoders


class ActionPolicy(nn.Module):
    """Shared behavior-cloning head over scene context and gripper state."""

    def __init__(
        self,
        scene_encoder: nn.Module | None,
        *,
        embedding_dim: int,
        proprio_dim: int = 7,
        hidden_dim: int = 64,
        xyz_action_scale: float = 0.04,
        action_dim: int = 4,
        action_mode: str = "legacy_cartesian_4d",
        action_horizon: int = 1,
    ) -> None:
        super().__init__()
        if xyz_action_scale <= 0:
            raise ValueError("xyz_action_scale must be positive")
        expected_action_dim = {
            "legacy_cartesian_4d": 4,
            "cartesian_7d": 7,
        }
        if action_mode not in expected_action_dim:
            raise ValueError(f"unknown action_mode: {action_mode}")
        if action_dim != expected_action_dim[action_mode]:
            raise ValueError(
                f"action_mode {action_mode} requires action_dim={expected_action_dim[action_mode]}"
            )
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.scene_encoder = scene_encoder
        self.embedding_dim = embedding_dim
        self.xyz_action_scale = xyz_action_scale
        self.action_dim = action_dim
        self.action_mode = action_mode
        self.action_horizon = int(action_horizon)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, embedding_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_horizon * action_dim),
        )

    def _scene_context(
        self,
        scene: SceneBatch | None,
        proprio_context: Tensor,
    ) -> Tensor:
        if self.scene_encoder is None:
            return torch.zeros_like(proprio_context)
        if scene is None:
            raise ValueError("scene input is required when a scene encoder is configured")
        return self.scene_encoder(scene)

    def predict_action_chunk(
        self,
        scene: SceneBatch | None,
        proprioception: Tensor,
    ) -> Tensor:
        proprio_context = self.proprio_encoder(proprioception)
        scene_context = self._scene_context(scene, proprio_context)
        raw_actions = self.action_head(torch.cat((scene_context, proprio_context), dim=-1))
        raw_actions = raw_actions.reshape(
            -1,
            self.action_horizon,
            self.action_dim,
        )
        if self.action_mode == "legacy_cartesian_4d":
            xyz = torch.tanh(raw_actions[:, :, :3]) * self.xyz_action_scale
            gripper = torch.sigmoid(raw_actions[:, :, 3:4])
            return torch.cat((xyz, gripper), dim=-1)
        pose_command = torch.tanh(raw_actions[:, :, :6])
        gripper = torch.sigmoid(raw_actions[:, :, 6:7])
        return torch.cat((pose_command, gripper), dim=-1)

    def forward(self, scene: SceneBatch | None, proprioception: Tensor) -> Tensor:
        chunk = self.predict_action_chunk(scene, proprioception)
        return chunk[:, 0] if self.action_horizon == 1 else chunk


def build_action_policy(
    *,
    representation: str,
    max_nodes: int = 8,
    max_edges: int = 56,
    node_feature_dim: int = NODE_FEATURE_DIM,
    edge_feature_dim: int = EDGE_FEATURE_DIM,
    graph_hidden_dim: int = 64,
    embedding_dim: int = 64,
    policy_hidden_dim: int = 64,
    message_rounds: int = 2,
    proprio_dim: int = 7,
    action_dim: int = 4,
    action_mode: str = "legacy_cartesian_4d",
    action_horizon: int = 1,
) -> ActionPolicy:
    if representation not in {"flat", "graph", "proprio"}:
        raise ValueError("representation must be one of: flat, graph, proprio")

    scene_encoder: nn.Module | None
    if representation == "proprio":
        scene_encoder = None
    else:
        flat, graph = build_matched_encoders(
            max_nodes=max_nodes,
            max_edges=max_edges,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            graph_hidden_dim=graph_hidden_dim,
            embedding_dim=embedding_dim,
            message_rounds=message_rounds,
        )
        scene_encoder = flat if representation == "flat" else graph
    return ActionPolicy(
        scene_encoder,
        embedding_dim=embedding_dim,
        proprio_dim=proprio_dim,
        hidden_dim=policy_hidden_dim,
        action_dim=action_dim,
        action_mode=action_mode,
        action_horizon=action_horizon,
    )
