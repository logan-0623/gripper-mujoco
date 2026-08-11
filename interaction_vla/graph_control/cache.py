from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from interaction_vla.lerobot_bridge.provenance import sha256_file

from .features import CurrentGraphFields, pack_oracle_current
from .schema import CONDITIONS, TOKEN_DIM, TOKEN_SLICES


CACHE_SCHEMA_VERSION = "graph_control_cache_v1"


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

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown graph control condition: {self.condition}")
        _hash_value(self.dataset_fingerprint, "dataset_fingerprint")
        _hash_value(self.split_manifest_sha256, "split_manifest_sha256")
        _hash_value(self.graph_checkpoint_sha256, "graph_checkpoint_sha256")
        graph_values = (
            self.graph_checkpoint_sha256,
            self.graph_initialization,
            self.graph_fraction,
            self.graph_seed,
        )
        if self.condition == "flat":
            if any(value is not None for value in graph_values):
                raise ValueError("flat cache provenance must not bind a Graph checkpoint")
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
        "token_dim": TOKEN_DIM,
        "token_slices": {
            name: [value.start, value.stop] for name, value in TOKEN_SLICES.items()
        },
        "rows": len(rows),
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
    if int(manifest.get("token_dim", -1)) != TOKEN_DIM:
        raise ValueError("graph token cache dimension is incompatible")
    expected_slices = {
        name: [value.start, value.stop] for name, value in TOKEN_SLICES.items()
    }
    if manifest.get("token_slices") != expected_slices:
        raise ValueError("graph token cache slices are incompatible")
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
    return TokenCache(source, rows, tokens, provenance, digest)


def current_fields_from_teacher(
    arrays: Mapping[str, np.ndarray], *, frame_index: int
) -> CurrentGraphFields:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    try:
        relation_values = arrays["annotation.tc_tig.relation_values"]
        return CurrentGraphFields(
            entity_mask=np.asarray(arrays["annotation.tc_tig.entity_mask"])[frame_index],
            entity_visibility=np.asarray(
                arrays["annotation.tc_tig.entity_visibility"]
            )[frame_index],
            relation_mask=np.asarray(arrays["annotation.tc_tig.relation_mask"])[
                frame_index
            ],
            relation_semantics=np.asarray(relation_values)[frame_index, :, 12:22],
        )
    except (KeyError, IndexError) as error:
        raise ValueError("teacher current fields are incomplete or misaligned") from error


def _sample_index(sample: Mapping[str, object], expected: int) -> None:
    actual = int(torch.as_tensor(sample.get("index", -1)).item())
    if actual != expected:
        raise ValueError(
            f"source global row alignment changed: expected {expected}, found {actual}"
        )


def build_token_cache(
    path: str | Path,
    *,
    source: Any,
    row_indices: Sequence[int],
    condition: str,
    runtime: Any | None,
    batch_size: int,
    provenance: CacheProvenance,
    current_fields: Mapping[int, CurrentGraphFields] | None = None,
) -> TokenCache:
    rows = np.asarray(tuple(int(value) for value in row_indices), dtype=np.int64)
    if condition != provenance.condition:
        raise ValueError("cache condition differs from provenance")
    if batch_size < 1:
        raise ValueError("cache batch_size must be positive")
    if condition == "flat":
        if runtime is not None or current_fields is not None:
            raise ValueError("flat cache must not receive Graph runtime or teacher fields")
        return write_token_cache(
            path,
            rows,
            np.zeros((len(rows), TOKEN_DIM), dtype=np.float32),
            provenance,
        )
    if runtime is None:
        raise ValueError("predicted cache requires a frozen Graph runtime")
    if condition == "oracle_current" and current_fields is None:
        raise ValueError("oracle_current cache requires causal current teacher fields")
    if condition != "oracle_current" and current_fields is not None:
        raise ValueError("teacher current fields are exclusive to oracle_current")

    chunks: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        samples = [source[int(row)] for row in selected]
        for row, sample in zip(selected, samples, strict=True):
            _sample_index(sample, int(row))
        predicted = runtime.predict_tokens(
            agent_rgb=[sample["observation.images.agent"] for sample in samples],
            wrist_rgb=[sample["observation.images.wrist"] for sample in samples],
            state=[sample["observation.state"] for sample in samples],
            task=[str(sample["task"]) for sample in samples],
        )
        predicted = np.asarray(predicted, dtype=np.float32)
        if predicted.shape != (len(selected), TOKEN_DIM) or not np.isfinite(
            predicted
        ).all():
            raise ValueError("frozen Graph runtime returned invalid token batch")
        if condition == "oracle_current":
            predicted = np.stack(
                [
                    pack_oracle_current(
                        current_fields[int(row)], predicted[index], runtime.normalization
                    )
                    for index, row in enumerate(selected)
                ],
                axis=0,
            )
        chunks.append(predicted)
    tokens = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, TOKEN_DIM), dtype=np.float32)
    )
    return write_token_cache(path, rows, tokens, provenance)
