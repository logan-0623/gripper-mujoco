from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..schemas.stages import StageManifest
from ..taps.registry import validate_manifest_taps


@runtime_checkable
class PolicyBackend(Protocol):
    backend_name: str

    def load_stage(self, manifest: StageManifest) -> None: ...

    def encode(self, batch: Mapping[str, object]) -> object: ...

    def act(self, batch: Mapping[str, object]) -> object: ...

    def get_latents(
        self, batch: Mapping[str, object], taps: Sequence[str]
    ) -> Mapping[str, object]: ...

    def set_trainable_groups(self, groups: Sequence[str]) -> None: ...


def validate_backend_manifest(
    backend: PolicyBackend, manifest: StageManifest
) -> None:
    if not isinstance(backend, PolicyBackend):
        raise ValueError("backend does not implement the PolicyBackend protocol")
    if str(backend.backend_name) != manifest.backend:
        raise ValueError(
            f"runtime backend {backend.backend_name} does not match "
            f"manifest backend {manifest.backend}"
        )
    validate_manifest_taps(manifest)

