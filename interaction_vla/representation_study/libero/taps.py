from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


SEMANTIC_TAPS = (
    "vision_output",
    "multimodal_fusion",
    "action_expert_input",
    "pre_action",
)


def valid_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or mask.ndim != 2 or values.shape[:2] != mask.shape:
        raise ValueError("valid-token pooling requires [B,T,D] values and [B,T] mask")
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    weights = mask.unsqueeze(-1).to(values.dtype)
    denominator = weights.sum(dim=1)
    pooled = (values * weights).sum(dim=1) / denominator.clamp_min(1.0)
    pooled = torch.where(denominator > 0, pooled, torch.zeros_like(pooled))
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled semantic tap contains non-finite values")
    return pooled


def _flow_model(policy: nn.Module) -> nn.Module:
    flow = getattr(policy, "model", None)
    if flow is None:
        raise ValueError("SmolVLA policy must expose policy.model")
    required = (
        "vlm_with_expert",
        "embed_prefix",
        "action_time_mlp_out",
        "action_out_proj",
    )
    missing = [name for name in required if not hasattr(flow, name)]
    if missing:
        raise ValueError(f"SmolVLA model is missing semantic tap modules: {missing}")
    return flow


class SmolVLASemanticTapCapture:
    """Capture preregistered semantic tensors without conflating denoising calls."""

    def __init__(self, policy: nn.Module) -> None:
        self.policy = policy

    def capture(
        self, invoke: Callable[[], object]
    ) -> tuple[object, dict[str, torch.Tensor], dict[str, dict[str, object]]]:
        flow = _flow_model(self.policy)
        vlm = flow.vlm_with_expert
        original_embed_image = vlm.embed_image
        original_embed_prefix = flow.embed_prefix
        original_vlm_forward = vlm.forward
        image_outputs: list[torch.Tensor] = []
        image_masks: list[torch.Tensor] = []
        prefix_mask: torch.Tensor | None = None
        prefix_hidden: torch.Tensor | None = None
        action_inputs: list[torch.Tensor] = []
        pre_actions: list[torch.Tensor] = []

        def embed_image(image: torch.Tensor) -> torch.Tensor:
            output = original_embed_image(image)
            image_outputs.append(output)
            return output

        def embed_prefix(images, img_masks, *args, **kwargs):
            nonlocal prefix_mask
            output = original_embed_prefix(images, img_masks, *args, **kwargs)
            prefix_mask = output[1]
            image_masks[:] = [mask.to(torch.bool) for mask in img_masks]
            return output

        def vlm_forward(*args, **kwargs):
            nonlocal prefix_hidden
            output = original_vlm_forward(*args, **kwargs)
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None and args:
                inputs_embeds = args[-2] if len(args) >= 2 else None
            if (
                prefix_hidden is None
                and isinstance(inputs_embeds, (list, tuple))
                and inputs_embeds
                and inputs_embeds[0] is not None
            ):
                outputs = output[0]
                if not isinstance(outputs, (list, tuple)) or not isinstance(outputs[0], torch.Tensor):
                    raise ValueError("SmolVLA prefix forward did not return prefix hidden states")
                prefix_hidden = outputs[0]
            return output

        action_handle = flow.action_time_mlp_out.register_forward_hook(
            lambda _module, _inputs, output: action_inputs.append(output)
        )
        pre_action_handle = flow.action_out_proj.register_forward_pre_hook(
            lambda _module, inputs: pre_actions.append(inputs[0])
        )
        vlm.embed_image = embed_image
        flow.embed_prefix = embed_prefix
        vlm.forward = vlm_forward
        try:
            output = invoke()
        finally:
            vlm.embed_image = original_embed_image
            flow.embed_prefix = original_embed_prefix
            vlm.forward = original_vlm_forward
            action_handle.remove()
            pre_action_handle.remove()

        if (
            not image_outputs
            or len(image_outputs) != len(image_masks)
            or prefix_mask is None
            or prefix_hidden is None
            or not action_inputs
            or not pre_actions
        ):
            raise ValueError("SmolVLA execution did not reach every preregistered semantic tap")
        view_means: list[torch.Tensor] = []
        for values, view_mask in zip(image_outputs, image_masks, strict=True):
            token_mask = view_mask[:, None].expand(values.shape[0], values.shape[1])
            view_means.append(valid_token_mean(values, token_mask))
        vision = torch.cat(view_means, dim=-1)
        fused = valid_token_mean(prefix_hidden, prefix_mask.to(torch.bool))
        action_expert = action_inputs[-1].mean(dim=1)
        pre_action = pre_actions[-1].mean(dim=1)
        tensors = {
            "vision_output": vision,
            "multimodal_fusion": fused,
            "action_expert_input": action_expert,
            "pre_action": pre_action,
        }
        result = {
            key: value.detach().to(device="cpu", dtype=torch.float32)
            for key, value in tensors.items()
        }
        if any(value.ndim != 2 or not torch.isfinite(value).all() for value in result.values()):
            raise ValueError("semantic taps must be finite [batch, features] tensors")
        metadata = {
            "vision_output": {
                "module": "model.vlm_with_expert.embed_image",
                "pooling": "valid_token_mean_per_view_then_concatenate",
                "call_selection": "all_configured_views",
                "shape_semantics": "batch_dimension_excluded",
                "raw_shapes": [list(value.shape[1:]) for value in image_outputs],
                "pooled_shape": list(result["vision_output"].shape[1:]),
            },
            "multimodal_fusion": {
                "module": "model.vlm_with_expert.forward",
                "pooling": "valid_token_mean(prefix_pad_masks)",
                "call_selection": "first_prefix_prefill",
                "shape_semantics": "batch_dimension_excluded",
                "raw_shape": list(prefix_hidden.shape[1:]),
                "mask_shape": list(prefix_mask.shape[1:]),
                "pooled_shape": list(result["multimodal_fusion"].shape[1:]),
            },
            "action_expert_input": {
                "module": "model.action_time_mlp_out",
                "pooling": "action_chunk_mean",
                "call_selection": "final_denoising",
                "shape_semantics": "batch_dimension_excluded",
                "raw_shape": list(action_inputs[-1].shape[1:]),
                "pooled_shape": list(result["action_expert_input"].shape[1:]),
            },
            "pre_action": {
                "module": "model.action_out_proj:input",
                "pooling": "action_chunk_mean",
                "call_selection": "final_denoising",
                "shape_semantics": "batch_dimension_excluded",
                "raw_shape": list(pre_actions[-1].shape[1:]),
                "pooled_shape": list(result["pre_action"].shape[1:]),
            },
        }
        return output, result, metadata
