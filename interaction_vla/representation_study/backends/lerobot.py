from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from interaction_vla.lerobot_bridge.rollout import load_act_runtime

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
    "act": {
        "vision_backbone": ("model.backbone", False, "last", "mean"),
        "temporal_fused": ("model.encoder", False, "last", "last"),
        "decoder_input": ("model.decoder", False, "last", "last"),
        "pre_action": ("model.action_head", True, "last", "last"),
    },
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
    "act": {
        "vision": ("model.backbone",),
        "fusion": ("model.encoder", "model.decoder", "model.encoder_"),
        "action_head": ("model.action_head",),
    },
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
        self.residual_policy: Any | None = None
        self.residual_scale: torch.Tensor | None = None
        self.residual_tap_id: str | None = None
        self.last_residual_action_was_clipped = False

    def _load_plain_checkpoint(self, checkpoint: str) -> tuple[Any, Any, Any]:
        if self.backend_name == "act":
            return load_act_runtime(checkpoint, device=self.device)
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

    def _load_checkpoint(self, checkpoint: str) -> tuple[Any, Any, Any]:
        bundle = Path(checkpoint) / "residual_study.json"
        self.residual_policy = None
        self.residual_scale = None
        self.residual_tap_id = None
        self.last_residual_action_was_clipped = False
        if not bundle.is_file():
            return self._load_plain_checkpoint(checkpoint)
        metadata = json.loads(bundle.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "interaction_residual_policy_bundle_v1":
            raise ValueError("residual policy bundle schema is incompatible")
        if metadata.get("backend") != self.backend_name:
            raise ValueError("residual policy bundle backend is incompatible")
        policy, preprocessor, postprocessor = self._load_plain_checkpoint(
            str(metadata["policy_checkpoint"])
        )
        from ..rl.core import ResidualActorCritic

        payload = torch.load(
            Path(checkpoint) / "residual.pt", map_location=self.device, weights_only=False
        )
        residual = ResidualActorCritic(
            int(payload["latent_dim"]),
            action_dim=int(payload.get("action_dim", 7)),
            adapt_representation=bool(payload.get("adapt_representation", False)),
        ).to(self.device)
        residual.load_state_dict(payload["state_dict"])
        residual.eval()
        scale = torch.as_tensor(metadata["residual_scale"], dtype=torch.float32, device=self.device)
        if scale.shape != (7,) or not torch.isfinite(scale).all() or torch.any(scale < 0.0):
            raise ValueError("residual policy scale must be a non-negative finite seven-vector")
        self.residual_policy = residual
        self.residual_scale = scale
        self.residual_tap_id = str(metadata["tap_id"])
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
        if self.backend_name == "act" or (Path(checkpoint) / "residual_study.json").is_file():
            self.load_checkpoint(checkpoint)
            return
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
        self.residual_policy = None
        self.residual_scale = None
        self.residual_tap_id = None

    def _loaded(self) -> tuple[Any, Any, Any]:
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError(f"{self.backend_name} backend has no loaded stage")
        return self.policy, self.preprocessor, self.postprocessor

    def _raw_batch(self, batch: Mapping[str, object]) -> dict[str, object]:
        policy, _, _ = self._loaded()
        result = dict(batch)
        if self.backend_name == "act":
            feature = getattr(policy.config, "env_state_feature", None)
            key = "observation.environment_state"
            if feature is not None and key not in result:
                state = result.get("observation.state")
                if not isinstance(state, torch.Tensor) or state.ndim != 2:
                    raise ValueError(
                        "ACT compatibility injection requires batched observation.state"
                    )
                width = int(feature.shape[0])
                result[key] = torch.zeros(
                    (state.shape[0], width), dtype=state.dtype, device=state.device
                )
        return result

    def encode(self, batch: Mapping[str, object]) -> object:
        _, preprocessor, _ = self._loaded()
        return preprocessor(self._raw_batch(batch))

    def act(self, batch: Mapping[str, object]) -> object:
        if self.residual_policy is not None:
            assert self.residual_tap_id is not None
            result = self.get_latents(batch, (self.residual_tap_id,))
            return result["__action__"]
        policy, preprocessor, postprocessor = self._loaded()
        processed = preprocessor(self._raw_batch(batch))
        policy.reset()
        with torch.no_grad():
            normalized = policy.predict_action_chunk(processed)
            return postprocessor(normalized)

    def _apply_residual(
        self, actions: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        self.last_residual_action_was_clipped = False
        if self.residual_policy is None:
            return actions
        if self.residual_scale is None:
            raise RuntimeError("loaded residual policy has no scale")
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError("residual composition requires action chunks with width seven")
        with torch.no_grad():
            delta = self.residual_policy.sample(
                latent.to(self.device), deterministic=True
            ).residual
        result = actions.clone()
        first = result[:, 0] + self.residual_scale * delta
        unclipped = first.clone()
        first[:, :6] = first[:, :6].clamp(-1.0, 1.0)
        first[:, 6] = first[:, 6].clamp(0.0, 1.0)
        self.last_residual_action_was_clipped = bool(
            torch.any(torch.abs(first - unclipped) > 1.0e-7).item()
        )
        result[:, 0] = first
        return result

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
        self.last_residual_action_was_clipped = False
        policy, preprocessor, postprocessor = self._loaded()
        processed = preprocessor(self._raw_batch(batch))
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
            if self.residual_policy is not None:
                if self.residual_tap_id not in latents:
                    raise ValueError(
                        "loaded residual policy requires its action-proximal tap"
                    )
                actions = self._apply_residual(actions, latents[self.residual_tap_id])
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
        processed = preprocessor(self._raw_batch(batch))
        state = processed.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError("processed intervention batch must contain batched state")
        tap = self._module_taps((tap_id,))[0]
        policy.reset()
        intervention = ForwardTapIntervention(policy, tap)
        with torch.no_grad():
            batch_size = int(state.shape[0])
            residual_latent: torch.Tensor | None = None
            if self.residual_policy is not None and tap_id != self.residual_tap_id:
                assert self.residual_tap_id is not None
                capture = ForwardTapCapture(
                    policy, self._module_taps((self.residual_tap_id,))
                )
                normalized, captured = capture.capture(
                    lambda: intervention.run(
                        lambda: policy.predict_action_chunk(processed),
                        batch_size=batch_size,
                        mode=mode,
                    ),
                    batch_size=batch_size,
                )
                residual_latent = captured[self.residual_tap_id]
            else:
                normalized = intervention.run(
                    lambda: policy.predict_action_chunk(processed),
                    batch_size=batch_size,
                    mode=mode,
                )
                if self.residual_policy is not None:
                    if intervention.last_tensor is None:
                        raise ValueError("residual tap intervention captured no replacement")
                    residual_latent = pool_latent_with_grad(
                        intervention.last_tensor,
                        batch_size=batch_size,
                        selector=tap.tensor_selector,
                    )
            actions = postprocessor(normalized)
            if residual_latent is not None:
                actions = self._apply_residual(actions, residual_latent)
            return actions.detach().to("cpu", torch.float32)

    def _predict_with_grad(self, processed: dict[str, Any]) -> torch.Tensor:
        policy, _, _ = self._loaded()
        if self.backend_name == "act":
            from lerobot.utils.constants import OBS_IMAGES

            values = dict(processed)
            if policy.config.image_features:
                values[OBS_IMAGES] = [values[key] for key in policy.config.image_features]
            return policy.model(values)[0]
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
        processed = preprocessor(self._raw_batch(batch))
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


class ACTBackend(LeRobotPolicyBackend):
    def __init__(self, *, device: str = "auto") -> None:
        super().__init__("act", device=device)


class SmolVLABackend(LeRobotPolicyBackend):
    def __init__(self, *, device: str = "auto") -> None:
        super().__init__("smolvla", device=device)


class PI0Backend(LeRobotPolicyBackend):
    def __init__(self, *, device: str = "auto") -> None:
        super().__init__("pi0", device=device)


def make_backend(name: str, *, device: str = "auto") -> LeRobotPolicyBackend:
    constructors = {"act": ACTBackend, "smolvla": SmolVLABackend, "pi0": PI0Backend}
    if name not in constructors:
        raise ValueError(f"unsupported backend: {name}")
    return constructors[name](device=device)
