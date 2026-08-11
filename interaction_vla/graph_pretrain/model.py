from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from interaction_vla.lerobot_bridge.teacher_schema import OPERATOR_IDS, PREDICATE_IDS

from .schema import OBJECT_COUNT, PHASE_NAMES, STATE_NAMES, UPRIGHT_NAMES


class ReflectGraphEstimator(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        graph_embedding_dim: int,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include pad and unknown tokens")
        if min(image_embedding_dim, text_embedding_dim, graph_embedding_dim) < 1:
            raise ValueError("embedding dimensions must be positive")
        self.image_embedding_dim = int(image_embedding_dim)
        self.text_embedding_dim = int(text_embedding_dim)
        self.graph_embedding_dim = int(graph_embedding_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, image_embedding_dim),
            nn.ReLU(),
        )
        self.token_embedding = nn.Embedding(
            vocab_size, text_embedding_dim, padding_idx=0
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_embedding_dim + text_embedding_dim, graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.ReLU(),
        )
        self.target_head = nn.Linear(graph_embedding_dim, OBJECT_COUNT)
        self.in_hand_head = nn.Linear(graph_embedding_dim, OBJECT_COUNT + 1)
        self.state_head = nn.Linear(
            graph_embedding_dim, OBJECT_COUNT * len(STATE_NAMES)
        )
        self.upright_head = nn.Linear(
            graph_embedding_dim, OBJECT_COUNT * len(UPRIGHT_NAMES)
        )
        self.dependency_head = nn.Linear(
            graph_embedding_dim, OBJECT_COUNT * OBJECT_COUNT
        )
        self.phase_head = nn.Linear(graph_embedding_dim, len(PHASE_NAMES))
        self.operator_head = nn.Linear(graph_embedding_dim, len(OPERATOR_IDS))
        self.predicate_head = nn.Linear(graph_embedding_dim, len(PREDICATE_IDS))

    def forward(
        self, image: Tensor, history_tokens: Tensor, history_mask: Tensor
    ) -> dict[str, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")
        if history_tokens.ndim != 2 or history_mask.shape != history_tokens.shape:
            raise ValueError("history tokens and mask must share shape (batch, tokens)")
        if history_tokens.shape[0] != image.shape[0]:
            raise ValueError("image and history batch sizes differ")
        image_context = self.image_encoder(image)
        mask = history_mask.to(dtype=image_context.dtype).unsqueeze(-1)
        token_context = self.token_embedding(history_tokens) * mask
        history_context = token_context.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        graph_embedding = self.fusion(
            torch.cat((image_context, history_context), dim=-1)
        )
        batch_size = image.shape[0]
        return {
            "target_logits": self.target_head(graph_embedding),
            "in_hand_logits": self.in_hand_head(graph_embedding),
            "state_logits": self.state_head(graph_embedding).reshape(
                batch_size, OBJECT_COUNT, len(STATE_NAMES)
            ),
            "upright_logits": self.upright_head(graph_embedding).reshape(
                batch_size, OBJECT_COUNT, len(UPRIGHT_NAMES)
            ),
            "dependency_logits": self.dependency_head(graph_embedding).reshape(
                batch_size, OBJECT_COUNT, OBJECT_COUNT
            ),
            "phase_logits": self.phase_head(graph_embedding),
            "operator_logits": self.operator_head(graph_embedding),
            "predicate_logits": self.predicate_head(graph_embedding),
            "graph_embedding": graph_embedding,
        }


def graph_prediction_loss(
    outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    object_mask = batch["object_mask"].bool()
    if object_mask.shape != batch["state_ids"].shape or not object_mask.any():
        raise ValueError("object_mask must select at least one object slot")
    dependency_mask = batch["dependency_mask"].bool()
    if dependency_mask.shape != batch["dependency"].shape or not dependency_mask.any():
        raise ValueError("dependency_mask must select at least one dependency edge")
    losses = {
        "target": F.cross_entropy(outputs["target_logits"], batch["target_index"]),
        "in_hand": F.cross_entropy(
            outputs["in_hand_logits"], batch["in_hand_index"]
        ),
        "state": F.cross_entropy(
            outputs["state_logits"][object_mask], batch["state_ids"][object_mask]
        ),
        "upright": F.cross_entropy(
            outputs["upright_logits"][object_mask],
            batch["upright_ids"][object_mask],
        ),
        "dependency": F.binary_cross_entropy_with_logits(
            outputs["dependency_logits"][dependency_mask],
            batch["dependency"][dependency_mask],
        ),
        "phase": F.cross_entropy(outputs["phase_logits"], batch["phase_id"]),
        "operator": F.cross_entropy(
            outputs["operator_logits"], batch["goal_operator_id"]
        ),
        "predicate": F.cross_entropy(
            outputs["predicate_logits"], batch["goal_predicate_id"]
        ),
    }
    losses["total"] = torch.stack(tuple(losses.values())).sum()
    return losses
