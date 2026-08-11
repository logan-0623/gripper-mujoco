from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from interaction_vla.graph_pretrain.reflectvlm import Vocabulary
from interaction_vla.graph_pretrain.schema import SCHEMA_VERSION as REFLECT_SCHEMA
from interaction_vla.lerobot_bridge.teacher_schema import OPERATOR_IDS, PREDICATE_IDS

from .schema import ENTITY_COUNT, RELATION_COUNT, SEMANTIC_DIM


@dataclass(frozen=True)
class TransferReport:
    copied_modules: tuple[str, ...]
    copied_tensors: tuple[str, ...]
    skipped_tensors: tuple[str, ...]
    copied_token_count: int
    target_token_count: int


class MuJoCoGraphEstimator(nn.Module):
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
        self.state_encoder = nn.Sequential(
            nn.Linear(10, image_embedding_dim),
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
        self.entity_mask_head = nn.Linear(graph_embedding_dim, ENTITY_COUNT)
        self.entity_visibility_head = nn.Linear(
            graph_embedding_dim, ENTITY_COUNT * 2
        )
        self.relation_mask_head = nn.Linear(graph_embedding_dim, RELATION_COUNT)
        self.relation_head = nn.Linear(
            graph_embedding_dim, RELATION_COUNT * SEMANTIC_DIM
        )
        self.goal_relation_head = nn.Linear(graph_embedding_dim, RELATION_COUNT)
        self.operator_head = nn.Linear(graph_embedding_dim, len(OPERATOR_IDS))
        self.predicate_head = nn.Linear(graph_embedding_dim, len(PREDICATE_IDS))
        self.residual_head = nn.Linear(graph_embedding_dim, 1)

    def forward(
        self,
        agent_rgb: Tensor,
        wrist_rgb: Tensor,
        state: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> dict[str, Tensor]:
        if (
            agent_rgb.ndim != 4
            or agent_rgb.shape[1] != 3
            or wrist_rgb.shape != agent_rgb.shape
        ):
            raise ValueError("agent and wrist RGB must share shape [batch, 3, H, W]")
        if state.shape != (agent_rgb.shape[0], 10):
            raise ValueError("state must have shape [batch, 10]")
        if language_tokens.ndim != 2 or language_mask.shape != language_tokens.shape:
            raise ValueError("language tokens and mask must share shape [batch, tokens]")
        if language_tokens.shape[0] != agent_rgb.shape[0]:
            raise ValueError("multimodal inputs must share one batch size")
        agent = self.image_encoder(agent_rgb)
        wrist = self.image_encoder(wrist_rgb)
        state_context = self.state_encoder(state)
        mask = language_mask.to(dtype=agent.dtype).unsqueeze(-1)
        words = self.token_embedding(language_tokens) * mask
        language = words.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        graph_embedding = self.fusion(
            torch.cat(((agent + wrist) * 0.5 + state_context, language), dim=-1)
        )
        batch_size = agent_rgb.shape[0]
        return {
            "entity_mask_logits": self.entity_mask_head(graph_embedding),
            "entity_visibility": torch.sigmoid(
                self.entity_visibility_head(graph_embedding)
            ).reshape(batch_size, ENTITY_COUNT, 2),
            "relation_mask_logits": self.relation_mask_head(graph_embedding),
            "relation_semantics": self.relation_head(graph_embedding).reshape(
                batch_size, RELATION_COUNT, SEMANTIC_DIM
            ),
            "goal_relation_logits": self.goal_relation_head(graph_embedding),
            "goal_operator_logits": self.operator_head(graph_embedding),
            "goal_predicate_logits": self.predicate_head(graph_embedding),
            "goal_residual": self.residual_head(graph_embedding).squeeze(-1),
            "graph_embedding": graph_embedding,
        }


def graph_finetune_loss(
    outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    entity_mask = batch["entity_mask"].bool()
    relation_mask = batch["relation_mask"].bool()
    if not entity_mask.any() or not relation_mask.any():
        raise ValueError("graph loss requires active entity and relation labels")
    losses = {
        "entity_mask": F.binary_cross_entropy_with_logits(
            outputs["entity_mask_logits"], batch["entity_mask"].float()
        ),
        "entity_visibility": F.mse_loss(
            outputs["entity_visibility"][entity_mask],
            batch["entity_visibility"][entity_mask],
        ),
        "relation_mask": F.binary_cross_entropy_with_logits(
            outputs["relation_mask_logits"], batch["relation_mask"].float()
        ),
        "relation_semantics": F.smooth_l1_loss(
            outputs["relation_semantics"][relation_mask],
            batch["relation_semantics"][relation_mask],
        ),
        "goal_relation": F.cross_entropy(
            outputs["goal_relation_logits"], batch["goal_relation"]
        ),
        "goal_operator": F.cross_entropy(
            outputs["goal_operator_logits"], batch["goal_operator"]
        ),
        "goal_predicate": F.cross_entropy(
            outputs["goal_predicate_logits"], batch["goal_predicate"]
        ),
        "goal_residual": F.smooth_l1_loss(
            outputs["goal_residual"], batch["goal_residual"]
        ),
    }
    losses["total"] = torch.stack(tuple(losses.values())).sum()
    return losses


def _module_state(
    state: Mapping[str, Tensor], prefix: str
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    marker = prefix + "."
    selected = {
        name[len(marker) :]: value
        for name, value in state.items()
        if name.startswith(marker)
    }
    if not selected:
        raise ValueError(f"ReflectVLM checkpoint is missing {prefix}")
    return selected, tuple(sorted(marker + name for name in selected))


def _load_reflect_payload(
    path: str | Path,
    *,
    image_embedding_dim: int,
    text_embedding_dim: int,
    graph_embedding_dim: int,
) -> dict[str, object]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ReflectVLM checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != REFLECT_SCHEMA:
        raise ValueError("ReflectVLM checkpoint schema is incompatible")
    required = {"model_config", "vocabulary", "model_state"}
    missing = required - set(payload)
    if missing:
        raise ValueError("ReflectVLM checkpoint is missing: " + ", ".join(sorted(missing)))
    model_config = payload["model_config"]
    if not isinstance(model_config, Mapping):
        raise ValueError("ReflectVLM model configuration is invalid")
    expected = {
        "image_embedding_dim": int(image_embedding_dim),
        "text_embedding_dim": int(text_embedding_dim),
        "graph_embedding_dim": int(graph_embedding_dim),
    }
    differing = [
        name for name, value in expected.items() if int(model_config.get(name, -1)) != value
    ]
    if differing:
        raise ValueError(
            "ReflectVLM checkpoint dimension mismatch: " + ", ".join(differing)
        )
    state = payload["model_state"]
    if not isinstance(state, Mapping) or not all(
        isinstance(value, Tensor) and torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("ReflectVLM model state must contain finite tensors")
    vocabulary = payload["vocabulary"]
    if not isinstance(vocabulary, (list, tuple)) or len(vocabulary) < 2:
        raise ValueError("ReflectVLM vocabulary is invalid")
    return payload


def _transfer_reflect_weights(
    model: MuJoCoGraphEstimator,
    *,
    vocabulary: Vocabulary,
    payload: Mapping[str, object],
) -> TransferReport:
    state = payload["model_state"]
    assert isinstance(state, Mapping)
    copied_tensors: list[str] = []
    copied_modules = ("image_encoder", "fusion", "operator_head", "predicate_head")
    for name in copied_modules:
        module_state, tensor_names = _module_state(state, name)
        getattr(model, name).load_state_dict(module_state, strict=True)
        copied_tensors.extend(tensor_names)
    source_tokens = tuple(str(token) for token in payload["vocabulary"])
    source_mapping = {token: index for index, token in enumerate(source_tokens)}
    target_mapping = vocabulary.token_to_id
    source_embedding = state.get("token_embedding.weight")
    if not isinstance(source_embedding, Tensor):
        raise ValueError("ReflectVLM checkpoint is missing token embeddings")
    if source_embedding.shape[0] != len(source_tokens):
        raise ValueError("ReflectVLM token embedding and vocabulary sizes differ")
    if source_embedding.shape[1] != model.token_embedding.weight.shape[1]:
        raise ValueError("ReflectVLM token embedding dimension mismatch")
    copied_tokens = sorted(set(source_mapping) & set(target_mapping))
    with torch.no_grad():
        for token in copied_tokens:
            model.token_embedding.weight[target_mapping[token]].copy_(
                source_embedding[source_mapping[token]]
            )
    copied_tensors.append("token_embedding.weight[matching_tokens]")
    transferred_prefixes = tuple(name + "." for name in copied_modules)
    skipped = tuple(
        sorted(
            name
            for name in state
            if not name.startswith(transferred_prefixes)
            and name != "token_embedding.weight"
        )
    )
    return TransferReport(
        copied_modules=copied_modules,
        copied_tensors=tuple(copied_tensors),
        skipped_tensors=skipped,
        copied_token_count=len(copied_tokens),
        target_token_count=len(vocabulary.tokens),
    )


def initialize_paired_models(
    *,
    vocab_size: int,
    vocabulary: Vocabulary,
    reflect_checkpoint: str | Path,
    seed: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
    graph_embedding_dim: int,
) -> tuple[MuJoCoGraphEstimator, MuJoCoGraphEstimator, TransferReport]:
    if vocab_size != len(vocabulary.tokens):
        raise ValueError("vocab_size must match the target vocabulary")
    torch.manual_seed(int(seed))
    random_model = MuJoCoGraphEstimator(
        vocab_size=vocab_size,
        image_embedding_dim=image_embedding_dim,
        text_embedding_dim=text_embedding_dim,
        graph_embedding_dim=graph_embedding_dim,
    )
    pretrained_model = copy.deepcopy(random_model)
    payload = _load_reflect_payload(
        reflect_checkpoint,
        image_embedding_dim=image_embedding_dim,
        text_embedding_dim=text_embedding_dim,
        graph_embedding_dim=graph_embedding_dim,
    )
    report = _transfer_reflect_weights(
        pretrained_model,
        vocabulary=vocabulary,
        payload=payload,
    )
    return random_model, pretrained_model, report
