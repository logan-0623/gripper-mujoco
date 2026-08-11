from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from interaction_vla.device import resolve_device
from interaction_vla.graph_finetune.config import ModelConfig
from interaction_vla.graph_finetune.data import GraphNormalization, resize_rgb
from interaction_vla.graph_finetune.pipeline import load_finetune_checkpoint

from .schema import TOKEN_DIM, TOKEN_SLICES, validate_token


_PREDICTED_KEYS = {
    "entity_mask_logits",
    "entity_visibility",
    "relation_mask_logits",
    "relation_semantics",
    "goal_relation_logits",
    "goal_operator_logits",
    "goal_predicate_logits",
    "goal_residual",
}
_DISTRACTOR_RELATIONS = (3, 4, 5, 6)
_RISK_CHANNELS = (6, 7)


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class CurrentGraphFields:
    """Causal current-frame graph fields; deliberately contains no goal label."""

    entity_mask: np.ndarray
    entity_visibility: np.ndarray
    relation_mask: np.ndarray
    relation_semantics: np.ndarray

    def __post_init__(self) -> None:
        entity_mask = np.asarray(self.entity_mask)
        relation_mask = np.asarray(self.relation_mask)
        if entity_mask.shape != (6,) or entity_mask.dtype != np.dtype(np.bool_):
            raise ValueError("entity_mask must have shape (6,) and dtype bool")
        if relation_mask.shape != (8,) or relation_mask.dtype != np.dtype(np.bool_):
            raise ValueError("relation_mask must have shape (8,) and dtype bool")
        visibility = _finite_array(
            self.entity_visibility, (6, 2), "entity_visibility"
        )
        if np.any((visibility < 0.0) | (visibility > 1.0)):
            raise ValueError("entity_visibility must lie within [0, 1]")
        semantics = _finite_array(
            self.relation_semantics, (8, 10), "relation_semantics"
        )
        object.__setattr__(self, "entity_mask", entity_mask.copy())
        object.__setattr__(self, "relation_mask", relation_mask.copy())
        object.__setattr__(self, "entity_visibility", visibility)
        object.__setattr__(self, "relation_semantics", semantics)


def normalization_from_payload(payload: Mapping[str, Any]) -> GraphNormalization:
    required = {
        "state_mean",
        "state_std",
        "relation_mean",
        "relation_std",
        "residual_mean",
        "residual_std",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError("normalization is missing: " + ", ".join(sorted(missing)))
    return GraphNormalization(
        state_mean=np.asarray(payload["state_mean"], dtype=np.float32).copy(),
        state_std=np.asarray(payload["state_std"], dtype=np.float32).copy(),
        relation_mean=np.asarray(payload["relation_mean"], dtype=np.float32).copy(),
        relation_std=np.asarray(payload["relation_std"], dtype=np.float32).copy(),
        residual_mean=float(payload["residual_mean"]),
        residual_std=float(payload["residual_std"]),
    )


def _sample_tensor(
    outputs: Mapping[str, object], name: str, shape: tuple[int, ...], sample_index: int
) -> torch.Tensor:
    value = torch.as_tensor(outputs[name]).detach().cpu()
    expected_tail = shape
    if value.ndim != len(expected_tail) + 1 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(f"{name} must have batched shape [N, {', '.join(map(str, shape))}]")
    if sample_index < 0 or sample_index >= value.shape[0]:
        raise ValueError("sample_index is outside predicted output batch")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value[sample_index].float()


def pack_predicted(
    outputs: Mapping[str, object], *, sample_index: int = 0
) -> np.ndarray:
    missing = _PREDICTED_KEYS - set(outputs)
    if missing:
        raise ValueError("predicted Graph outputs are missing: " + ", ".join(sorted(missing)))

    entity = torch.sigmoid(
        _sample_tensor(outputs, "entity_mask_logits", (6,), sample_index)
    )
    visibility = _sample_tensor(outputs, "entity_visibility", (6, 2), sample_index)
    relation = torch.sigmoid(
        _sample_tensor(outputs, "relation_mask_logits", (8,), sample_index)
    )
    semantics = _sample_tensor(outputs, "relation_semantics", (8, 10), sample_index)
    goal_relation = torch.softmax(
        _sample_tensor(outputs, "goal_relation_logits", (8,), sample_index), dim=-1
    )
    goal_operator = torch.softmax(
        _sample_tensor(outputs, "goal_operator_logits", (5,), sample_index), dim=-1
    )
    goal_predicate = torch.softmax(
        _sample_tensor(outputs, "goal_predicate_logits", (7,), sample_index), dim=-1
    )
    residual = _sample_tensor(outputs, "goal_residual", (), sample_index).reshape(1)

    token = np.empty(TOKEN_DIM, dtype=np.float32)
    token[TOKEN_SLICES["entity_presence"]] = entity.numpy()
    token[TOKEN_SLICES["entity_visibility"]] = visibility.reshape(-1).numpy()
    token[TOKEN_SLICES["relation_presence"]] = relation.numpy()
    token[TOKEN_SLICES["gripper_target_semantics"]] = semantics[0].numpy()
    token[TOKEN_SLICES["target_receptacle_semantics"]] = semantics[1].numpy()
    token[TOKEN_SLICES["distractor_risks"]] = (
        semantics[list(_DISTRACTOR_RELATIONS)][:, list(_RISK_CHANNELS)]
        .reshape(-1)
        .numpy()
    )
    token[TOKEN_SLICES["next_relation"]] = goal_relation.numpy()
    token[TOKEN_SLICES["relation_operator"]] = goal_operator.numpy()
    token[TOKEN_SLICES["predicate"]] = goal_predicate.numpy()
    token[TOKEN_SLICES["goal_residual"]] = residual.numpy()
    return validate_token(token)


def pack_oracle_current(
    current: CurrentGraphFields,
    predicted_token: object,
    normalization: GraphNormalization,
) -> np.ndarray:
    token = validate_token(predicted_token).copy()
    normalized = (
        current.relation_semantics - normalization.relation_mean
    ) / normalization.relation_std
    token[TOKEN_SLICES["entity_presence"]] = current.entity_mask.astype(np.float32)
    token[TOKEN_SLICES["entity_visibility"]] = current.entity_visibility.reshape(-1)
    token[TOKEN_SLICES["relation_presence"]] = current.relation_mask.astype(np.float32)
    token[TOKEN_SLICES["gripper_target_semantics"]] = normalized[0]
    token[TOKEN_SLICES["target_receptacle_semantics"]] = normalized[1]
    token[TOKEN_SLICES["distractor_risks"]] = normalized[
        list(_DISTRACTOR_RELATIONS)
    ][:, list(_RISK_CHANNELS)].reshape(-1)
    return validate_token(token)


class FrozenGraphRuntime:
    """Inference-only wrapper around one fine-tuned MuJoCo Graph checkpoint."""

    def __init__(self, checkpoint: str | Path, *, device: str = "auto") -> None:
        self.checkpoint = Path(checkpoint)
        self.device = resolve_device(device)
        self.model, self.vocabulary, self.payload = load_finetune_checkpoint(
            self.checkpoint, device=self.device
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_config = ModelConfig(**self.payload["model_config"])
        self.normalization = normalization_from_payload(self.payload["normalization"])

    @torch.inference_mode()
    def predict_tokens(
        self,
        *,
        agent_rgb: Sequence[object],
        wrist_rgb: Sequence[object],
        state: Sequence[object],
        task: Sequence[str],
    ) -> np.ndarray:
        count = len(agent_rgb)
        if not (len(wrist_rgb) == len(state) == len(task) == count) or count < 1:
            raise ValueError("Graph inference inputs must share a positive batch size")
        agent = torch.stack(
            [resize_rgb(value, self.model_config.image_size, "agent RGB") for value in agent_rgb]
        ).to(self.device)
        wrist = torch.stack(
            [resize_rgb(value, self.model_config.image_size, "wrist RGB") for value in wrist_rgb]
        ).to(self.device)
        states = np.stack(
            [_finite_array(value, (10,), "state") for value in state], axis=0
        )
        states = (states - self.normalization.state_mean) / self.normalization.state_std
        encoded = [
            self.vocabulary.encode((str(value),), self.model_config.max_language_tokens)
            for value in task
        ]
        language_tokens = torch.from_numpy(np.stack([value[0] for value in encoded])).to(
            self.device
        )
        language_mask = torch.from_numpy(np.stack([value[1] for value in encoded])).to(
            self.device
        )
        outputs = self.model(
            agent,
            wrist,
            torch.from_numpy(states.astype(np.float32)).to(self.device),
            language_tokens,
            language_mask,
        )
        return np.stack(
            [pack_predicted(outputs, sample_index=index) for index in range(count)], axis=0
        ).astype(np.float32, copy=False)

