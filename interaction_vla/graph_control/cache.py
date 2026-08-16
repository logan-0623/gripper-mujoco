from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from interaction_vla.graph_finetune.schema import (
    GraphV2Normalization,
    GraphV2Targets,
)
from interaction_vla.lerobot_bridge.provenance import sha256_file

from .features import pack_oracle_current
from .schema import (
    CONDITIONS,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SCHEMA_VERSION,
    TOKEN_SLICES,
    validate_token,
)


CACHE_SCHEMA_VERSION = "graph_control_cache_v2"


def _hash_value(value: str | None, name: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CacheProvenance:
    condition: str
    dataset_fingerprint: str
    split_manifest_sha256: str
    graph_checkpoint_sha256: str | None
    graph_initialization: str | None
    graph_fraction: float | None
    graph_seed: int | None
    token_schema_version: str = TOKEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown graph control condition: {self.condition}")
        _hash_value(self.dataset_fingerprint, "dataset_fingerprint")
        _hash_value(self.split_manifest_sha256, "split_manifest_sha256")
        _hash_value(self.graph_checkpoint_sha256, "graph_checkpoint_sha256")
        if self.token_schema_version != TOKEN_SCHEMA_VERSION:
            raise ValueError("cache provenance token schema is incompatible")
        graph_values = (
            self.graph_checkpoint_sha256,
            self.graph_initialization,
            self.graph_fraction,
            self.graph_seed,
        )
        checkpoint_free = {"flat", "oracle_graph_v2"}
        if self.condition in checkpoint_free:
            if any(value is not None for value in graph_values):
                raise ValueError(
                    f"{self.condition} cache must not bind a Graph checkpoint"
                )
            return
        if any(value is None for value in graph_values):
            raise ValueError("predicted cache provenance must bind the Graph checkpoint")
        if self.graph_initialization not in {"random_init", "reflectvlm_init"}:
            raise ValueError("graph_initialization is incompatible")
        if not math.isclose(float(self.graph_fraction), 1.0):
            raise ValueError("continuous control requires a full-data Graph checkpoint")
        if int(self.graph_seed) < 0:
            raise ValueError("graph_seed must be non-negative")


@dataclass(frozen=True)
class TokenCache:
    path: Path
    row_indices: np.ndarray
    tokens: np.ndarray
    provenance: CacheProvenance
    sha256: str

    def __post_init__(self) -> None:
        rows, tokens = _validated_arrays(self.row_indices, self.tokens)
        _hash_value(self.sha256, "cache sha256")
        rows.setflags(write=False)
        tokens.setflags(write=False)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "tokens", tokens)

    @property
    def by_row(self) -> dict[int, np.ndarray]:
        return {int(row): self.tokens[index] for index, row in enumerate(self.row_indices)}


def _validated_arrays(
    row_indices: object, tokens: object
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_indices)
    values = np.asarray(tokens)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("cache row_indices must be a one-dimensional integer array")
    rows = rows.astype(np.int64, copy=True)
    if np.any(rows < 0) or len(set(rows.tolist())) != len(rows):
        raise ValueError("cache row_indices must be non-negative and unique")
    if values.shape != (len(rows), TOKEN_DIM):
        raise ValueError(f"cache tokens must have shape [{len(rows)}, {TOKEN_DIM}]")
    values = values.astype(np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("cache tokens must be finite")
    return rows, values


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _ordered_row_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(rows, dtype=np.int64).tobytes()
    ).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_token_cache(
    path: str | Path,
    row_indices: object,
    tokens: object,
    provenance: CacheProvenance,
) -> TokenCache:
    destination = Path(path)
    manifest_path = _manifest_path(destination)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite graph token cache: {destination}")
    rows, values = _validated_arrays(row_indices, tokens)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, row_indices=rows, tokens=values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(destination)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "token_dim": TOKEN_DIM,
        "token_feature_names": list(TOKEN_FEATURE_NAMES),
        "token_slices": {
            name: [value.start, value.stop] for name, value in TOKEN_SLICES.items()
        },
        "rows": len(rows),
        "ordered_row_sha256": _ordered_row_sha256(rows),
        "cache_sha256": digest,
        "provenance": asdict(provenance),
    }
    try:
        _write_json_atomic(manifest_path, manifest)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return TokenCache(destination, rows, values, provenance, digest)


def _provenance_from_manifest(value: object) -> CacheProvenance:
    if not isinstance(value, Mapping):
        raise ValueError("cache manifest provenance must be a mapping")
    try:
        return CacheProvenance(**dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("cache manifest provenance is invalid") from error


def load_token_cache(
    path: str | Path, *, expected: CacheProvenance | None = None
) -> TokenCache:
    source = Path(path)
    manifest_path = _manifest_path(source)
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"graph token cache is incomplete: {source}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid graph token cache manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("graph token cache manifest must be a mapping")
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("graph token cache schema is incompatible")
    if manifest.get("token_schema_version") != TOKEN_SCHEMA_VERSION:
        raise ValueError("graph token cache token schema is incompatible")
    if int(manifest.get("token_dim", -1)) != TOKEN_DIM:
        raise ValueError("graph token cache dimension is incompatible")
    expected_slices = {
        name: [value.start, value.stop] for name, value in TOKEN_SLICES.items()
    }
    if manifest.get("token_slices") != expected_slices:
        raise ValueError("graph token cache slices are incompatible")
    if manifest.get("token_feature_names") != list(TOKEN_FEATURE_NAMES):
        raise ValueError("graph token cache feature names are incompatible")
    digest = sha256_file(source)
    if manifest.get("cache_sha256") != digest:
        raise ValueError("graph token cache SHA-256 mismatch")
    provenance = _provenance_from_manifest(manifest.get("provenance"))
    if expected is not None:
        actual = asdict(provenance)
        wanted = asdict(expected)
        differing = [name for name in wanted if actual[name] != wanted[name]]
        if differing:
            raise ValueError(
                "graph token cache provenance mismatch: " + ", ".join(differing)
            )
    try:
        with np.load(source, allow_pickle=False) as loaded:
            if set(loaded.files) != {"row_indices", "tokens"}:
                raise ValueError("graph token cache arrays are incompatible")
            rows = loaded["row_indices"].copy()
            tokens = loaded["tokens"].copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid graph token cache: {source}") from error
    rows, tokens = _validated_arrays(rows, tokens)
    if int(manifest.get("rows", -1)) != len(rows):
        raise ValueError("graph token cache row count mismatch")
    if manifest.get("ordered_row_sha256") != _ordered_row_sha256(rows):
        raise ValueError("graph token cache ordered row hash mismatch")
    return TokenCache(source, rows, tokens, provenance, digest)


def _sample_scalar(sample: Mapping[str, object], name: str) -> int:
    return int(torch.as_tensor(sample.get(name, -1)).item())


def build_token_cache(
    path: str | Path,
    *,
    source: Any,
    episode_rows: Mapping[int, Sequence[int]],
    condition: str,
    runtime: Any | None,
    provenance: CacheProvenance,
    oracle_targets: Mapping[int, GraphV2Targets] | None = None,
    normalization: GraphV2Normalization | None = None,
) -> TokenCache:
    if condition != provenance.condition:
        raise ValueError("cache condition differs from provenance")
    if not episode_rows:
        raise ValueError("cache requires at least one episode")
    if condition == "flat":
        if runtime is not None or oracle_targets is not None or normalization is not None:
            raise ValueError("flat cache must not receive Graph runtime or oracle targets")
    elif condition == "oracle_graph_v2":
        if runtime is not None or oracle_targets is None or normalization is None:
            raise ValueError(
                "oracle_graph_v2 cache requires targets and normalization only"
            )
        if set(int(value) for value in oracle_targets) != set(
            int(value) for value in episode_rows
        ):
            raise ValueError("oracle targets must match cache episodes")
    else:
        if runtime is None or oracle_targets is not None or normalization is not None:
            raise ValueError("predicted Graph v2 cache requires only a frozen runtime")

    ordered_rows: list[int] = []
    tokens: list[np.ndarray] = []
    for episode in sorted(int(value) for value in episode_rows):
        rows = tuple(int(value) for value in episode_rows[episode])
        if not rows:
            raise ValueError(f"episode {episode} has no cache rows")
        if condition not in {"flat", "oracle_graph_v2"}:
            runtime.reset()
        for frame, row in enumerate(rows):
            sample = source[row]
            actual_index = _sample_scalar(sample, "index")
            actual_episode = _sample_scalar(sample, "episode_index")
            actual_frame = _sample_scalar(sample, "frame_index")
            if actual_index != row:
                raise ValueError(
                    f"source global row alignment changed: expected {row}, "
                    f"found {actual_index}"
                )
            if actual_episode != episode:
                raise ValueError(
                    f"source episode alignment changed: expected {episode}, "
                    f"found {actual_episode}"
                )
            if actual_frame != frame:
                raise ValueError(
                    f"source frame alignment changed: expected {frame}, "
                    f"found {actual_frame}"
                )
            if condition == "flat":
                token = np.zeros(TOKEN_DIM, dtype=np.float32)
            elif condition == "oracle_graph_v2":
                token = pack_oracle_current(
                    oracle_targets[episode],
                    frame_index=frame,
                    normalization=normalization,
                )
            else:
                token = runtime.predict_token(
                    agent_rgb=sample["observation.images.agent"],
                    wrist_rgb=sample["observation.images.wrist"],
                    state=sample["observation.state"],
                    task=str(sample["task"]),
                )
            ordered_rows.append(row)
            tokens.append(validate_token(token))
    return write_token_cache(
        path,
        np.asarray(ordered_rows, dtype=np.int64),
        np.stack(tokens, axis=0),
        provenance,
    )
