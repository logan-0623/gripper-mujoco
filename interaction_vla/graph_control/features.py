from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from interaction_vla.device import resolve_device
from interaction_vla.graph_finetune.config import ModelConfig
from interaction_vla.graph_finetune.data import resize_rgb
from interaction_vla.graph_finetune.pipeline import load_finetune_checkpoint
from interaction_vla.graph_finetune.schema import (
    GraphV2Normalization,
    GraphV2Targets,
    pack_oracle_target,
)

from .schema import TOKEN_DIM, TOKEN_SLICES, validate_token


_PREDICTED_KEYS = {
    "entity_mask_logits",
    "entity_visibility",
    "relation_mask_logits",
    "gripper_target_geometry",
    "target_receptacle_geometry",
    "distractor_geometry",
    "phase_logits",
    "relation_trends",
    "goal_relation_logits",
    "goal_operator_logits",
    "goal_predicate_logits",
    "goal_residual",
}


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def normalization_from_payload(payload: Mapping[str, Any]) -> GraphV2Normalization:
    required = {
        "state_mean",
        "state_std",
        "workspace_scale",
        "velocity_scale",
    }
    if set(payload) != required:
        missing = required - set(payload)
        extra = set(payload) - required
        raise ValueError(
            f"normalization fields mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return GraphV2Normalization(
        state_mean=np.asarray(payload["state_mean"], dtype=np.float32).copy(),
        state_std=np.asarray(payload["state_std"], dtype=np.float32).copy(),
        workspace_scale=float(payload["workspace_scale"]),
        velocity_scale=float(payload["velocity_scale"]),
    )


def _sample_tensor(
    outputs: Mapping[str, object],
    name: str,
    shape: tuple[int, ...],
    sample_index: int,
) -> torch.Tensor:
    value = torch.as_tensor(outputs[name]).detach().cpu()
    if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape:
        dimensions = ", ".join(map(str, shape))
        raise ValueError(f"{name} must have batched shape [N, {dimensions}]")
    if sample_index < 0 or sample_index >= value.shape[0]:
        raise ValueError("sample_index is outside predicted output batch")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value[sample_index].float()


def _bounded(value: torch.Tensor, name: str) -> None:
    if torch.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"{name} values must lie within [0, 1]")


def pack_predicted(
    outputs: Mapping[str, object], *, sample_index: int = 0
) -> np.ndarray:
    missing = _PREDICTED_KEYS - set(outputs)
    if missing:
        raise ValueError(
            "predicted Graph outputs are missing: " + ", ".join(sorted(missing))
        )
    entity = torch.sigmoid(
        _sample_tensor(outputs, "entity_mask_logits", (6,), sample_index)
    )
    visibility = _sample_tensor(
        outputs, "entity_visibility", (6, 2), sample_index
    )
    relation = torch.sigmoid(
        _sample_tensor(outputs, "relation_mask_logits", (8,), sample_index)
    )
    gripper = _sample_tensor(
        outputs, "gripper_target_geometry", (8,), sample_index
    )
    target = _sample_tensor(
        outputs, "target_receptacle_geometry", (10,), sample_index
    )
    distractors = _sample_tensor(
        outputs, "distractor_geometry", (2, 7), sample_index
    )
    phase = torch.softmax(
        _sample_tensor(outputs, "phase_logits", (6,), sample_index), dim=-1
    )
    trends = _sample_tensor(outputs, "relation_trends", (4,), sample_index)
    goal_relation = torch.softmax(
        _sample_tensor(outputs, "goal_relation_logits", (8,), sample_index), dim=-1
    )
    goal_operator = torch.softmax(
        _sample_tensor(outputs, "goal_operator_logits", (5,), sample_index), dim=-1
    )
    goal_predicate = torch.softmax(
        _sample_tensor(outputs, "goal_predicate_logits", (7,), sample_index), dim=-1
    )
    residual = _sample_tensor(outputs, "goal_residual", (), sample_index).clamp(
        -1.0, 1.0
    )
    for value, name in (
        (visibility, "entity_visibility"),
        (gripper[5:8], "gripper_target_geometry"),
        (target[8:10], "target_receptacle_geometry"),
        (distractors[:, 4:7], "distractor_geometry"),
    ):
        _bounded(value, name)

    token = np.empty(TOKEN_DIM, dtype=np.float32)
    token[TOKEN_SLICES["entity_presence"]] = entity.numpy()
    token[TOKEN_SLICES["entity_visibility"]] = visibility.reshape(-1).numpy()
    token[TOKEN_SLICES["relation_presence"]] = relation.numpy()
    token[TOKEN_SLICES["gripper_target_geometry"]] = gripper.numpy()
    token[TOKEN_SLICES["target_receptacle_geometry"]] = target.numpy()
    token[TOKEN_SLICES["distractor_geometry"]] = distractors.reshape(-1).numpy()
    token[TOKEN_SLICES["phase"]] = phase.numpy()
    token[TOKEN_SLICES["relation_trends"]] = trends.numpy()
    token[TOKEN_SLICES["next_relation"]] = goal_relation.numpy()
    token[TOKEN_SLICES["relation_operator"]] = goal_operator.numpy()
    token[TOKEN_SLICES["predicate"]] = goal_predicate.numpy()
    token[TOKEN_SLICES["goal_residual"]] = residual.numpy()
    return validate_token(token)


def pack_oracle_current(
    targets: GraphV2Targets,
    *,
    frame_index: int,
    normalization: GraphV2Normalization,
) -> np.ndarray:
    return pack_oracle_target(targets, frame_index, normalization)


class FrozenGraphRuntime:
    """Inference-only recurrent wrapper around one Graph v2 checkpoint."""

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
        self.normalization = normalization_from_payload(
            self.payload["normalization"]
        )
        self.reset()

    def reset(self) -> None:
        self.previous_graph = np.zeros(TOKEN_DIM, dtype=np.float32)

    @torch.inference_mode()
    def predict_token(
        self,
        *,
        agent_rgb: object,
        wrist_rgb: object,
        state: object,
        task: str,
    ) -> np.ndarray:
        agent = resize_rgb(
            agent_rgb, self.model_config.image_size, "agent RGB"
        )[None].to(self.device)
        wrist = resize_rgb(
            wrist_rgb, self.model_config.image_size, "wrist RGB"
        )[None].to(self.device)
        current_state = _finite_array(state, (10,), "state")
        current_state = (
            current_state - self.normalization.state_mean
        ) / self.normalization.state_std
        tokens, mask = self.vocabulary.encode(
            (str(task),), self.model_config.max_language_tokens
        )
        outputs = self.model(
            agent,
            wrist,
            torch.from_numpy(current_state[None]).to(self.device),
            torch.from_numpy(tokens).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(self.previous_graph[None]).to(self.device),
        )
        token = pack_predicted(outputs, sample_index=0)
        self.previous_graph = token.copy()
        return token
