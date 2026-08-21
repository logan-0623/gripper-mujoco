from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..schemas.stages import SUPPORTED_BACKENDS, StageManifest


REQUIRED_TAP_ROLES: Final[tuple[str, ...]] = (
    "visual",
    "fused",
    "policy_input",
    "action_proximal",
)


@dataclass(frozen=True)
class TapSpec:
    role: str
    tap_id: str

    def __post_init__(self) -> None:
        if self.role not in REQUIRED_TAP_ROLES:
            raise ValueError(f"unknown scientific tap role: {self.role}")
        if not str(self.tap_id).strip():
            raise ValueError("tap_id must be non-empty")


_REGISTRY: Final[dict[str, tuple[TapSpec, ...]]] = {
    "act": (
        TapSpec("visual", "vision_backbone"),
        TapSpec("fused", "temporal_fused"),
        TapSpec("policy_input", "decoder_input"),
        TapSpec("action_proximal", "pre_action"),
    ),
    "smolvla": (
        TapSpec("visual", "vision_output"),
        TapSpec("fused", "multimodal_fusion"),
        TapSpec("policy_input", "action_expert_input"),
        TapSpec("action_proximal", "pre_action"),
    ),
    "pi0": (
        TapSpec("visual", "vision_output"),
        TapSpec("fused", "multimodal_fusion"),
        TapSpec("policy_input", "action_expert_input"),
        TapSpec("action_proximal", "pre_action"),
    ),
}


def registered_taps(backend: str) -> tuple[TapSpec, ...]:
    name = str(backend).strip()
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {name}")
    taps = _REGISTRY[name]
    if tuple(tap.role for tap in taps) != REQUIRED_TAP_ROLES:
        raise RuntimeError(f"backend {name} has an invalid tap-role registry")
    if len({tap.tap_id for tap in taps}) != len(taps):
        raise RuntimeError(f"backend {name} has duplicate tap identifiers")
    return taps


def validate_manifest_taps(manifest: StageManifest) -> None:
    expected = tuple(tap.tap_id for tap in registered_taps(manifest.backend))
    if manifest.latent_taps != expected:
        raise ValueError(
            f"{manifest.backend} manifest does not match its fixed tap registry"
        )

