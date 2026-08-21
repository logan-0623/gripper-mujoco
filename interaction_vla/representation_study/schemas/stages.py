from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Final, Mapping


STAGE_MANIFEST_SCHEMA_VERSION: Final[str] = "policy_stage_manifest_v1"
SUPPORTED_BACKENDS: Final[tuple[str, ...]] = ("act", "smolvla", "pi0")
SUPPORTED_STAGES: Final[tuple[str, ...]] = (
    "pretrained",
    "sft",
    "continued_sft",
    "rl_head",
    "rl_representation",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_tuple(value: object, name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if (not allow_empty and not result) or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class ArtifactBinding:
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        uri = str(self.uri).strip()
        digest = str(self.sha256).strip().lower()
        if not uri:
            raise ValueError("artifact uri must be non-empty")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal digits")
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, str]:
        return {"uri": self.uri, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactBinding:
        if not isinstance(value, Mapping) or set(value) != {"uri", "sha256"}:
            raise ValueError("artifact binding must contain uri and sha256")
        return cls(uri=str(value["uri"]), sha256=str(value["sha256"]))


@dataclass(frozen=True)
class StageManifest:
    study_id: str
    backend: str
    stage: str
    checkpoint: ArtifactBinding
    config: ArtifactBinding
    dataset: ArtifactBinding
    source: ArtifactBinding
    state_bank: ArtifactBinding | None
    trainable_groups: tuple[str, ...]
    latent_taps: tuple[str, ...]
    schema_version: str = STAGE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        study_id = str(self.study_id).strip()
        backend = str(self.backend).strip()
        stage = str(self.stage).strip()
        if not study_id:
            raise ValueError("study_id must be non-empty")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend: {backend}")
        if stage not in SUPPORTED_STAGES:
            raise ValueError(f"unsupported stage: {stage}")
        if self.schema_version != STAGE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("stage manifest schema_version is incompatible")
        for name in ("checkpoint", "config", "dataset", "source"):
            if not isinstance(getattr(self, name), ArtifactBinding):
                raise ValueError(f"{name} must be an ArtifactBinding")
        if self.state_bank is not None and not isinstance(
            self.state_bank, ArtifactBinding
        ):
            raise ValueError("state_bank must be an ArtifactBinding or null")
        groups = _nonempty_tuple(
            self.trainable_groups, "trainable groups", allow_empty=True
        )
        taps = _nonempty_tuple(self.latent_taps, "latent taps", allow_empty=False)
        if stage == "pretrained" and groups:
            raise ValueError("pretrained stage must not declare trainable groups")
        if stage != "pretrained" and not groups:
            raise ValueError(f"{stage} stage must declare trainable groups")
        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "trainable_groups", groups)
        object.__setattr__(self, "latent_taps", taps)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "backend": self.backend,
            "stage": self.stage,
            "checkpoint": self.checkpoint.to_dict(),
            "config": self.config.to_dict(),
            "dataset": self.dataset.to_dict(),
            "source": self.source.to_dict(),
            "state_bank": (
                None if self.state_bank is None else self.state_bank.to_dict()
            ),
            "trainable_groups": list(self.trainable_groups),
            "latent_taps": list(self.latent_taps),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: object) -> StageManifest:
        required = {
            "schema_version",
            "study_id",
            "backend",
            "stage",
            "checkpoint",
            "config",
            "dataset",
            "source",
            "state_bank",
            "trainable_groups",
            "latent_taps",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("stage manifest fields are incompatible")
        state_bank = value["state_bank"]
        return cls(
            schema_version=str(value["schema_version"]),
            study_id=str(value["study_id"]),
            backend=str(value["backend"]),
            stage=str(value["stage"]),
            checkpoint=ArtifactBinding.from_dict(value["checkpoint"]),
            config=ArtifactBinding.from_dict(value["config"]),
            dataset=ArtifactBinding.from_dict(value["dataset"]),
            source=ArtifactBinding.from_dict(value["source"]),
            state_bank=(
                None if state_bank is None else ArtifactBinding.from_dict(state_bank)
            ),
            trainable_groups=_nonempty_tuple(
                value["trainable_groups"], "trainable groups", allow_empty=True
            ),
            latent_taps=_nonempty_tuple(
                value["latent_taps"], "latent taps", allow_empty=False
            ),
        )

