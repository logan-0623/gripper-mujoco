from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ..schemas.stages import StageManifest
from ..taps.capture import (
    ForwardTapCapture,
    ModuleTap,
    pool_latent_with_grad,
    resolve_module,
)
from ..taps.intervene import ForwardTapIntervention
from ..taps.registry import registered_taps
from .base import validate_backend_manifest


_TAP_MODULES: dict[str, dict[str, tuple[str, bool, str, str]]] = {
    "smolvla": {
        "vision_output": ("model.vlm_with_expert.vlm.model.vision_model", False, "first", "mean"),
        "multimodal_fusion": ("model.vlm_with_expert", False, "first", "last"),
        "action_expert_input": ("model.action_in_proj", False, "last", "last"),
        "pre_action": ("model.action_out_proj", True, "last", "last"),
    },
    "pi0": {
        "vision_output": (
            "model.paligemma_with_expert.paligemma.model.vision_tower",
            False,
            "first",
            "mean",
        ),
        "multimodal_fusion": ("model.paligemma_with_expert", False, "first", "last"),
        "action_expert_input": ("model.action_in_proj", False, "last", "last"),
        "pre_action": ("model.action_out_proj", True, "last", "last"),
    },
}

_TRAINABLE_PREFIXES: dict[str, dict[str, tuple[str, ...]]] = {
    "smolvla": {
        "vision": ("model.vlm_with_expert.vlm.model.vision_model",),
        "fusion": (
            "model.vlm_with_expert.lm_expert",
            "model.state_proj",
            "model.action_time_mlp",
        ),
        "action_head": ("model.action_in_proj", "model.action_out_proj"),
    },
    "pi0": {
        "vision": (
            "model.paligemma_with_expert.paligemma.model.vision_tower",
        ),
        "fusion": (
            "model.paligemma_with_expert.gemma_expert",
            "model.state_proj",
            "model.action_time_mlp",
        ),
        "action_head": ("model.action_in_proj", "model.action_out_proj"),
    },
}


class LeRobotPolicyBackend:
    def __init__(self, backend_name: str, *, device: str = "auto") -> None:
        if backend_name not in _TAP_MODULES:
            raise ValueError(f"unsupported LeRobot backend: {backend_name}")
        from interaction_vla.device import resolve_device

        self.backend_name = backend_name
        self.device = resolve_device(device)
        self.policy: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self.manifest: StageManifest | None = None

    def _load_checkpoint(self, checkpoint: str) -> tuple[Any, Any, Any]:
        from lerobot.policies import make_pre_post_processors

        if self.backend_name == "smolvla":
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            config = SmolVLAConfig.from_pretrained(checkpoint)
            config.device = self.device.type
            policy = SmolVLAPolicy.from_pretrained(
                checkpoint, config=config, local_files_only=Path(checkpoint).is_dir()
            )
        else:
            from lerobot.policies.pi0.configuration_pi0 import PI0Config
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            config = PI0Config.from_pretrained(checkpoint)
            config.device = self.device.type
            policy = PI0Policy.from_pretrained(
                checkpoint, config=config, local_files_only=Path(checkpoint).is_dir()
            )
        overrides = {"device_processor": {"device": self.device.type}}
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides=overrides,
            postprocessor_overrides=overrides,
        )
        policy.eval()
        return policy, preprocessor, postprocessor

    def load_stage(self, manifest: StageManifest) -> None:
        validate_backend_manifest(self, manifest)
        policy, preprocessor, postprocessor = self._load_checkpoint(
            manifest.checkpoint.uri
        )
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.manifest = manifest

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        policy, preprocessor, postprocessor = self._load_checkpoint(str(checkpoint))
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.manifest = None

    def load_checkpoint_for_dataset(
        self,
        checkpoint: str | Path,
        *,
        repo_id: str,
        dataset_root: str | Path,
        rename_map: Mapping[str, str] | None = None,
    ) -> None:
        """Load foundation weights while binding policy features to one LeRobotDataset."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies import make_policy, make_pre_post_processors

        policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
        policy_config.device = self.device.type
        policy_config.pretrained_path = Path(checkpoint)
        feature_rename_map = dict(rename_map or {})
        metadata = LeRobotDatasetMetadata(repo_id, root=Path(dataset_root))
        policy = make_policy(
            policy_config,
            ds_meta=metadata,
            rename_map=feature_rename_map,
        )
        features = {**policy.config.input_features, **policy.config.output_features}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint),
            dataset_stats=metadata.stats,
            preprocessor_overrides={
                "device_processor": {"device": self.device.type},
                "rename_observations_processor": {
                    "rename_map": feature_rename_map,
                },
                "normalizer_processor": {
                    "features": features,
                    "norm_map": policy.config.normalization_mapping,
                    "stats": metadata.stats,
                },
            },
            postprocessor_overrides={
                "device_processor": {"device": self.device.type},
                "unnormalizer_processor": {
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                    "stats": metadata.stats,
                },
            },
        )
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.manifest = None

    def _loaded(self) -> tuple[Any, Any, Any]:
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError(f"{self.backend_name} backend has no loaded stage")
        return self.policy, self.preprocessor, self.postprocessor

    def encode(self, batch: Mapping[str, object]) -> object:
        _, preprocessor, _ = self._loaded()
        return preprocessor(dict(batch))

    def act(self, batch: Mapping[str, object]) -> object:
        policy, preprocessor, postprocessor = self._loaded()
        processed = preprocessor(dict(batch))
        policy.reset()
        with torch.no_grad():
            normalized = policy.predict_action_chunk(processed)
            return postprocessor(normalized)

    def _module_taps(self, tap_ids: Sequence[str]) -> tuple[ModuleTap, ...]:
        expected = {tap.tap_id for tap in registered_taps(self.backend_name)}
        requested = tuple(str(value) for value in tap_ids)
        unknown = set(requested) - expected
        if unknown:
            raise ValueError("unknown backend taps: " + ", ".join(sorted(unknown)))
        definitions = _TAP_MODULES[self.backend_name]
        return tuple(
            ModuleTap(
                tap_id=tap_id,
                module_path=definitions[tap_id][0],
                capture_input=definitions[tap_id][1],
                tensor_selector=definitions[tap_id][2],
                call_reducer=definitions[tap_id][3],
            )
            for tap_id in requested
        )

    def get_latents(
        self, batch: Mapping[str, object], taps: Sequence[str]
    ) -> Mapping[str, object]:
        policy, preprocessor, postprocessor = self._loaded()
        processed = preprocessor(dict(batch))
        state = processed.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError("processed policy batch must contain batched observation.state")
        batch_size = int(state.shape[0])
        policy.reset()
        capture = ForwardTapCapture(policy, self._module_taps(taps))
        with torch.no_grad():
            normalized, latents = capture.capture(
                lambda: policy.predict_action_chunk(processed),
                batch_size=batch_size,
            )
            actions = postprocessor(normalized)
        return {**latents, "__action__": actions.detach().to("cpu", torch.float32)}

    def set_trainable_groups(self, groups: Sequence[str]) -> None:
        policy, _, _ = self._loaded()
        requested = {str(value) for value in groups}
        allowed = {"vision", "fusion", "action_head", "all"}
        unknown = requested - allowed
        if unknown:
            raise ValueError("unknown trainable groups: " + ", ".join(sorted(unknown)))
        for parameter in policy.parameters():
            parameter.requires_grad = False
        if "all" in requested:
            for parameter in policy.parameters():
                parameter.requires_grad = True
            return
        prefixes = _TRAINABLE_PREFIXES[self.backend_name]
        for name, parameter in policy.named_parameters():
            if any(name.startswith(prefix) for group in requested for prefix in prefixes[group]):
                parameter.requires_grad = True

    def intervene_actions(
        self,
        batch: Mapping[str, object],
        *,
        tap_id: str,
        mode: str,
    ) -> torch.Tensor:
        policy, preprocessor, postprocessor = self._loaded()
        processed = preprocessor(dict(batch))
        state = processed.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError("processed intervention batch must contain batched state")
        tap = self._module_taps((tap_id,))[0]
        policy.reset()
        intervention = ForwardTapIntervention(policy, tap)
        with torch.no_grad():
            normalized = intervention.run(
                lambda: policy.predict_action_chunk(processed),
                batch_size=int(state.shape[0]),
                mode=mode,
            )
            actions = postprocessor(normalized)
            return actions.detach().to("cpu", torch.float32)

    def _predict_with_grad(self, processed: dict[str, Any]) -> torch.Tensor:
        policy, _, _ = self._loaded()
        if self.backend_name == "smolvla":
            from lerobot.utils.constants import (
                OBS_LANGUAGE_ATTENTION_MASK,
                OBS_LANGUAGE_TOKENS,
            )

            images, image_masks = policy.prepare_images(processed)
            state = policy.prepare_state(processed)
            return policy.model.sample_actions(
                images,
                image_masks,
                processed[OBS_LANGUAGE_TOKENS],
                processed[OBS_LANGUAGE_ATTENTION_MASK],
                state,
            )[:, :, : policy.config.action_feature.shape[0]]
        raise ValueError("differentiable pi0 inference is not enabled in this study")

    def differentiable_action_and_latent(
        self, batch: Mapping[str, object], *, tap_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy, preprocessor, postprocessor = self._loaded()
        processed = preprocessor(dict(batch))
        state = processed.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError("differentiable policy batch must contain batched state")
        tap = self._module_taps((tap_id,))[0]
        module = resolve_module(policy, tap.module_path)
        captured: list[object] = []
        if tap.capture_input:
            handle = module.register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs)
            )
        else:
            handle = module.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output)
            )
        policy.reset()
        try:
            normalized = self._predict_with_grad(processed)
        finally:
            handle.remove()
        if not captured:
            raise ValueError(f"differentiable policy did not reach tap: {tap_id}")
        latent = pool_latent_with_grad(
            captured[-1], batch_size=int(state.shape[0]), selector=tap.tensor_selector
        )
        actions = postprocessor(normalized)
        return actions, latent


class SmolVLABackend(LeRobotPolicyBackend):
    def __init__(self, *, device: str = "auto") -> None:
        super().__init__("smolvla", device=device)


class PI0Backend(LeRobotPolicyBackend):
    def __init__(self, *, device: str = "auto") -> None:
        super().__init__("pi0", device=device)


def make_backend(name: str, *, device: str = "auto") -> LeRobotPolicyBackend:
    constructors = {"smolvla": SmolVLABackend, "pi0": PI0Backend}
    if name not in constructors:
        raise ValueError(f"unsupported backend: {name}")
    return constructors[name](device=device)
