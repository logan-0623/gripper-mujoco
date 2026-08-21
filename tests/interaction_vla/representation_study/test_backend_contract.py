from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from interaction_vla.representation_study.backends.base import (
    PolicyBackend,
    validate_backend_manifest,
)
from interaction_vla.representation_study.schemas.stages import (
    ArtifactBinding,
    StageManifest,
)
from interaction_vla.representation_study.taps.registry import (
    REQUIRED_TAP_ROLES,
    registered_taps,
    validate_manifest_taps,
)


def _binding(name: str) -> ArtifactBinding:
    return ArtifactBinding(uri=name, sha256="b" * 64)


def _manifest(backend: str = "act") -> StageManifest:
    taps = registered_taps(backend)
    return StageManifest(
        study_id=f"{backend}_seed_0",
        backend=backend,
        stage="sft",
        checkpoint=_binding("checkpoint"),
        config=_binding("config.yaml"),
        dataset=_binding("dataset"),
        source=_binding("source"),
        state_bank=_binding("state-bank"),
        trainable_groups=("policy",),
        latent_taps=tuple(tap.tap_id for tap in taps),
    )


def test_registered_backends_expose_exact_scientific_tap_roles() -> None:
    for backend in ("act", "smolvla", "pi0"):
        taps = registered_taps(backend)
        assert tuple(tap.role for tap in taps) == REQUIRED_TAP_ROLES
        assert len({tap.tap_id for tap in taps}) == len(REQUIRED_TAP_ROLES)


def test_manifest_taps_are_fixed_per_backend() -> None:
    validate_manifest_taps(_manifest("act"))
    with pytest.raises(ValueError, match="fixed tap registry"):
        validate_manifest_taps(
            StageManifest.from_dict(
                {**_manifest("act").to_dict(), "latent_taps": ["pre_action"]}
            )
        )


class _FakeBackend:
    backend_name = "act"

    def load_stage(self, manifest: StageManifest) -> None:
        self.manifest = manifest

    def encode(self, batch: Mapping[str, object]) -> object:
        return batch

    def act(self, batch: Mapping[str, object]) -> object:
        return batch

    def get_latents(
        self, batch: Mapping[str, object], taps: Sequence[str]
    ) -> Mapping[str, object]:
        return {tap: batch for tap in taps}

    def set_trainable_groups(self, groups: Sequence[str]) -> None:
        self.groups = tuple(groups)


def test_policy_backend_protocol_and_manifest_backend_match() -> None:
    backend = _FakeBackend()
    assert isinstance(backend, PolicyBackend)
    validate_backend_manifest(backend, _manifest("act"))
    with pytest.raises(ValueError, match="backend"):
        validate_backend_manifest(backend, _manifest("smolvla"))

