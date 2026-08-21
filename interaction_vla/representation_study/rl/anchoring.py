from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from interaction_vla.lerobot_bridge.provenance import sha256_file

from ..state_bank.io import write_json_atomic


ANCHOR_CACHE_SCHEMA = "nominal_anchor_cache_v1"
ANCHOR_SAMPLER_SCHEMA = "nominal_anchor_sampler_v1"


def nominal_residual_loss(residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 2 or not torch.isfinite(residual).all():
        raise ValueError("nominal residual must be a finite matrix")
    return residual.square().mean()


def latent_drift_loss(
    current: torch.Tensor,
    sft_target: torch.Tensor,
) -> torch.Tensor:
    if current.shape != sft_target.shape or current.ndim != 2:
        raise ValueError("current and SFT latents must be aligned matrices")
    if not torch.isfinite(current).all() or not torch.isfinite(sft_target).all():
        raise ValueError("latent anchor inputs must be finite")
    return (current - sft_target.detach()).square().mean()


def _payload_sha256(latents: np.ndarray, case_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(str(latents.dtype).encode("utf-8"))
    digest.update(json.dumps(list(latents.shape)).encode("utf-8"))
    digest.update(latents.tobytes(order="C"))
    digest.update(
        json.dumps(list(case_ids), separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorBatch:
    case_ids: tuple[str, ...]
    latents: np.ndarray


class NominalAnchorCache:
    def __init__(
        self,
        *,
        latents: np.ndarray,
        case_ids: tuple[str, ...],
        dataset_fingerprint: str,
        sft_checkpoint: str,
        tap_id: str,
        payload_sha256: str,
        seed: int,
    ) -> None:
        values = np.asarray(latents)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("anchor latents must be a non-empty finite float32 matrix")
        if len(case_ids) != values.shape[0] or len(set(case_ids)) != len(case_ids):
            raise ValueError("anchor case ids must be unique and align with latents")
        if len(dataset_fingerprint) != 64:
            raise ValueError("anchor dataset fingerprint must be a SHA-256 digest")
        if not sft_checkpoint.strip() or not tap_id.strip():
            raise ValueError("anchor checkpoint and tap id must be non-empty")
        expected = _payload_sha256(values, case_ids)
        if payload_sha256 != expected:
            raise ValueError("anchor payload SHA-256 differs")
        if seed < 0:
            raise ValueError("anchor sampler seed must be non-negative")
        self.latents = values.copy()
        self.case_ids = case_ids
        self.dataset_fingerprint = dataset_fingerprint
        self.sft_checkpoint = sft_checkpoint
        self.tap_id = tap_id
        self.payload_sha256 = payload_sha256
        self.rng = np.random.default_rng(seed)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        latents: object,
        case_ids: tuple[str, ...],
        dataset_fingerprint: str,
        sft_checkpoint: str,
        tap_id: str,
        seed: int,
    ) -> "NominalAnchorCache":
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        values = np.asarray(latents, dtype=np.float32)
        identifiers = tuple(str(value) for value in case_ids)
        payload_hash = _payload_sha256(values, identifiers)
        payload_path = destination / "anchors.npz"
        manifest_path = destination / "manifest.json"
        if payload_path.exists() or manifest_path.exists():
            raise FileExistsError(f"nominal anchor cache already exists: {destination}")
        temporary = payload_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                latents=values,
                case_ids=np.asarray(identifiers, dtype=np.str_),
            )
        temporary.replace(payload_path)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": ANCHOR_CACHE_SCHEMA,
                "dataset_fingerprint": dataset_fingerprint,
                "sft_checkpoint": sft_checkpoint,
                "tap_id": tap_id,
                "case_ids": list(identifiers),
                "latent_dtype": str(values.dtype),
                "latent_shape": list(values.shape),
                "payload_sha256": payload_hash,
                "archive_sha256": sha256_file(payload_path),
            },
        )
        return cls(
            latents=values,
            case_ids=identifiers,
            dataset_fingerprint=dataset_fingerprint,
            sft_checkpoint=sft_checkpoint,
            tap_id=tap_id,
            payload_sha256=payload_hash,
            seed=seed,
        )

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        dataset_fingerprint: str,
        sft_checkpoint: str,
        tap_id: str,
        seed: int,
    ) -> "NominalAnchorCache":
        source = Path(root)
        payload_path = source / "anchors.npz"
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != ANCHOR_CACHE_SCHEMA:
            raise ValueError("nominal anchor cache schema is incompatible")
        if manifest.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("nominal anchor dataset fingerprint differs")
        if manifest.get("sft_checkpoint") != sft_checkpoint:
            raise ValueError("nominal anchor SFT checkpoint differs")
        if manifest.get("tap_id") != tap_id:
            raise ValueError("nominal anchor tap id differs")
        if manifest.get("archive_sha256") != sha256_file(payload_path):
            raise ValueError("nominal anchor archive SHA-256 differs")
        with np.load(payload_path, allow_pickle=False) as archive:
            latents = np.asarray(archive["latents"])
            case_ids = tuple(str(value) for value in archive["case_ids"].tolist())
        if list(case_ids) != manifest.get("case_ids"):
            raise ValueError("nominal anchor case ids differ")
        if list(latents.shape) != manifest.get("latent_shape"):
            raise ValueError("nominal anchor latent shape differs")
        if str(latents.dtype) != manifest.get("latent_dtype"):
            raise ValueError("nominal anchor latent dtype differs")
        return cls(
            latents=latents,
            case_ids=case_ids,
            dataset_fingerprint=dataset_fingerprint,
            sft_checkpoint=sft_checkpoint,
            tap_id=tap_id,
            payload_sha256=str(manifest.get("payload_sha256", "")),
            seed=seed,
        )

    def sample(self, batch_size: int) -> AnchorBatch:
        if batch_size < 1:
            raise ValueError("anchor batch_size must be positive")
        positions = self.rng.choice(
            len(self.case_ids),
            size=batch_size,
            replace=batch_size > len(self.case_ids),
        )
        indices = np.asarray(positions, dtype=np.int64)
        return AnchorBatch(
            case_ids=tuple(self.case_ids[int(index)] for index in indices),
            latents=self.latents[indices].copy(),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": ANCHOR_SAMPLER_SCHEMA,
            "payload_sha256": self.payload_sha256,
            "rng_state": dict(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != ANCHOR_SAMPLER_SCHEMA:
            raise ValueError("nominal anchor sampler schema is incompatible")
        if state.get("payload_sha256") != self.payload_sha256:
            raise ValueError("nominal anchor sampler payload differs")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("nominal anchor sampler RNG state is invalid")
        self.rng.bit_generator.state = dict(rng_state)
