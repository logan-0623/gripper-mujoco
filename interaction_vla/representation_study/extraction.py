from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file

from .backends import make_backend
from .config import RepresentationStudyConfig
from .schemas.stages import ArtifactBinding, StageManifest
from .state_bank.io import load_manifest, load_records, load_split, write_json_atomic
from .state_bank.materialize import StateBankMaterializer, collate_observations
from .state_bank.validation import validate_state_bank
from .taps.registry import registered_taps


LATENT_SCHEMA_VERSION = "interaction_stage_latents_v1"


def _checkpoint_binding(checkpoint: str) -> tuple[ArtifactBinding, str | None]:
    path = Path(checkpoint)
    if path.is_dir():
        return ArtifactBinding(uri=path.as_posix(), sha256=fingerprint_tree(path)), None
    if path.is_file():
        return ArtifactBinding(uri=path.as_posix(), sha256=sha256_file(path)), None
    from huggingface_hub import model_info

    info = model_info(checkpoint)
    revision = str(info.sha)
    return (
        ArtifactBinding(
            uri=checkpoint,
            sha256=hashlib.sha256(revision.encode("utf-8")).hexdigest(),
        ),
        revision,
    )


def build_stage_manifest(
    config: RepresentationStudyConfig, *, backend: str, stage: str
) -> tuple[StageManifest, str | None]:
    stage_config = config.stage_config(backend, stage)
    checkpoint, revision = _checkpoint_binding(stage_config.checkpoint)
    bank_path = config.state_bank.output_dir / "manifest.json"
    bank = load_manifest(bank_path)
    manifest = StageManifest(
        study_id=config.study_id,
        backend=backend,
        stage=stage,
        checkpoint=checkpoint,
        config=ArtifactBinding(
            uri=config.config_path.as_posix(), sha256=sha256_file(config.config_path)
        ),
        dataset=bank.dataset,
        source=ArtifactBinding(
            uri="interaction_vla/representation_study",
            sha256=fingerprint_tree("interaction_vla/representation_study"),
        ),
        state_bank=ArtifactBinding(
            uri=bank_path.as_posix(), sha256=sha256_file(bank_path)
        ),
        trainable_groups=stage_config.trainable_groups,
        latent_taps=tuple(tap.tap_id for tap in registered_taps(backend)),
    )
    return manifest, revision


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_shard(path: Path, expected_ids: Sequence[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid latent shard: {path}") from error
    ids = arrays.get("state_id")
    if ids is None or ids.astype(str).tolist() != list(expected_ids):
        raise ValueError(f"latent shard State Bank rows changed: {path}")
    if any(
        np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError(f"latent shard contains non-finite values: {path}")
    return arrays


def _partition_records(
    records: Sequence[Any], split: Any, partition: str
) -> tuple[Any, ...]:
    if partition == "all":
        return tuple(sorted(records, key=lambda record: record.state_id))
    if partition not in {"train", "validation", "test"}:
        raise ValueError("latent partition must be all, train, validation, or test")
    ids = set(getattr(split, partition))
    return tuple(sorted((record for record in records if record.state_id in ids), key=lambda record: record.state_id))


def _destination(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    partition: str,
    limit: int | None,
) -> Path:
    suffix = partition if limit is None else f"{partition}_smoke_{limit}"
    return config.extraction.output_dir / backend / stage / suffix


def extract_latents(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    partition: str = "all",
    limit: int | None = None,
) -> dict[str, object]:
    if limit is not None and limit < 1:
        raise ValueError("latent extraction limit must be positive")
    records = load_records(config.state_bank.output_dir / "records.jsonl")
    split = load_split(config.state_bank.output_dir / "split.json")
    validate_state_bank(records, split)
    selected = _partition_records(records, split, partition)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("latent extraction selected no State Bank records")
    stage_manifest, hf_revision = build_stage_manifest(
        config, backend=backend, stage=stage
    )
    destination = _destination(
        config,
        backend=backend,
        stage=stage,
        partition=partition,
        limit=limit,
    )
    manifest_path = destination / "manifest.json"
    expected_header = {
        "schema_version": LATENT_SCHEMA_VERSION,
        "stage_manifest": stage_manifest.to_dict(),
        "state_ids": [record.state_id for record in selected],
        "partition": partition,
        "limit": limit,
        "batch_size": config.extraction.batch_size,
        "hf_revision": hf_revision,
    }
    if destination.exists():
        if not manifest_path.is_file():
            raise ValueError("latent extraction directory has no manifest")
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        differing = [key for key, value in expected_header.items() if actual.get(key) != value]
        if differing:
            raise ValueError("latent extraction manifest mismatch: " + ", ".join(differing))
        if actual.get("complete") is True:
            return inspect_latents(
                config,
                backend=backend,
                stage=stage,
                partition=partition,
                limit=limit,
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_json_atomic(manifest_path, {**expected_header, "complete": False})

    runtime = make_backend(backend, device=config.extraction.device)
    runtime.load_checkpoint_for_dataset(
        config.stage_config(backend, stage).checkpoint,
        repo_id=config.dataset.repo_id,
        dataset_root=config.dataset.root,
    )
    materializer = StateBankMaterializer(
        dataset_root=config.dataset.root,
        repo_id=config.dataset.repo_id,
        bridge_config=load_bridge_config(config.dataset.bridge_config),
        replay_position_tolerance=config.state_bank.replay_position_tolerance,
    )
    taps = tuple(tap.tap_id for tap in registered_taps(backend))
    batches = [
        selected[index : index + config.extraction.batch_size]
        for index in range(0, len(selected), config.extraction.batch_size)
    ]
    shard_paths: list[Path] = []
    progress = tqdm(total=len(batches), desc=f"{backend}/{stage} latents", unit="batch", dynamic_ncols=True)
    for batch_index, batch_records in enumerate(batches):
        shard = destination / "shards" / f"batch_{batch_index:06d}.npz"
        ids = [record.state_id for record in batch_records]
        if shard.is_file():
            _load_shard(shard, ids)
            shard_paths.append(shard)
            progress.update(1)
            continue
        materialized = {
            value.record.state_id: value
            for value in materializer.iter_records(batch_records)
        }
        if set(materialized) != set(ids):
            raise ValueError("State Bank materialization did not return the requested batch")
        ordered = [materialized[state_id] for state_id in ids]
        deterministic_seed = int.from_bytes(
            hashlib.sha256("\n".join(ids).encode("utf-8")).digest()[:8], "big"
        ) % (2**31)
        torch.manual_seed(deterministic_seed)
        values = runtime.get_latents(collate_observations(ordered), taps)
        arrays = {"state_id": np.asarray(ids)}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != len(ids):
                raise ValueError(f"backend output {name} is not aligned to the State Bank batch")
            arrays[name] = value.detach().to("cpu", torch.float32).numpy()
        _write_npz_atomic(shard, arrays)
        _load_shard(shard, ids)
        shard_paths.append(shard)
        progress.update(1)
    progress.close()

    loaded = [
        _load_shard(path, [record.state_id for record in batch])
        for path, batch in zip(shard_paths, batches, strict=True)
    ]
    keys = set(loaded[0])
    if any(set(shard) != keys for shard in loaded):
        raise ValueError("latent shard keys changed across batches")
    combined = {
        key: np.concatenate([shard[key] for shard in loaded], axis=0)
        for key in sorted(keys)
    }
    latent_path = destination / "latents.npz"
    _write_npz_atomic(latent_path, combined)
    report = {
        "passed": True,
        "schema_version": LATENT_SCHEMA_VERSION,
        "backend": backend,
        "stage": stage,
        "partition": partition,
        "records": len(selected),
        "taps": {name: list(combined[name].shape[1:]) for name in taps},
        "action_shape": list(combined["__action__"].shape[1:]),
        "latent_sha256": sha256_file(latent_path),
        "stage_manifest_sha256": hashlib.sha256(stage_manifest.to_json().encode("utf-8")).hexdigest(),
    }
    write_json_atomic(destination / "report.json", report)
    write_json_atomic(
        manifest_path,
        {
            **expected_header,
            "complete": True,
            "latent_path": latent_path.as_posix(),
            "latent_sha256": report["latent_sha256"],
            "report_path": (destination / "report.json").as_posix(),
        },
    )
    return report


def inspect_latents(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    partition: str = "all",
    limit: int | None = None,
) -> dict[str, object]:
    destination = _destination(
        config,
        backend=backend,
        stage=stage,
        partition=partition,
        limit=limit,
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("latent extraction is incomplete")
    latent_path = Path(str(manifest["latent_path"]))
    if sha256_file(latent_path) != manifest.get("latent_sha256"):
        raise ValueError("latent artifact SHA-256 mismatch")
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("latent_sha256") != manifest.get("latent_sha256"):
        raise ValueError("latent extraction report is incompatible")
    return report
