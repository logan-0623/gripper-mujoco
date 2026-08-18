from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from interaction_vla.lerobot_bridge.provenance import sha256_file

from .ablation import (
    representation_transform,
    resample_sequence_nearest,
    stratified_episode_permutation,
)
from .cache import (
    TokenCache,
    _manifest_path,
    _ordered_row_sha256,
    _validated_arrays,
    _write_json_atomic,
)
from .schema import (
    ABLATION_CONDITIONS,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SCHEMA_VERSION,
    TOKEN_SLICES,
)


ABLATION_CACHE_SCHEMA_VERSION = "control_alignment_ablation_cache_v1"
ABLATION_TRANSFORM_VERSION = "control_alignment_ablation_v1"
SHUFFLE_MANIFEST_SCHEMA_VERSION = "control_alignment_shuffle_v1"
_PARTITIONS = ("train", "validation", "test")


def _hash_value(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class AblationCacheProvenance:
    condition: str
    dataset_fingerprint: str
    split_manifest_sha256: str
    oracle_report_sha256: str
    source_condition: str
    source_cache_sha256: str
    graph_checkpoint_sha256: str
    graph_initialization: str
    graph_fraction: float
    graph_seed: int
    transform_version: str
    shuffle_manifest_sha256: str | None = None
    token_schema_version: str = TOKEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.condition not in ABLATION_CONDITIONS:
            raise ValueError(f"unknown ablation condition: {self.condition}")
        for name in (
            "dataset_fingerprint",
            "split_manifest_sha256",
            "oracle_report_sha256",
            "source_cache_sha256",
            "graph_checkpoint_sha256",
        ):
            _hash_value(str(getattr(self, name)), name)
        if self.shuffle_manifest_sha256 is not None:
            _hash_value(self.shuffle_manifest_sha256, "shuffle_manifest_sha256")
        if self.source_condition != "predicted_random_v2":
            raise ValueError("ablation source_condition must be predicted_random_v2")
        if self.graph_initialization != "random_init":
            raise ValueError("ablation Graph source must use random_init")
        if not math.isclose(float(self.graph_fraction), 1.0):
            raise ValueError("ablation Graph source must use the full-data checkpoint")
        if int(self.graph_seed) < 0:
            raise ValueError("ablation graph_seed must be non-negative")
        if self.transform_version != ABLATION_TRANSFORM_VERSION:
            raise ValueError("ablation transform version is incompatible")
        if self.token_schema_version != TOKEN_SCHEMA_VERSION:
            raise ValueError("ablation token schema is incompatible")
        if (self.condition == "shuffled_graph") != (
            self.shuffle_manifest_sha256 is not None
        ):
            raise ValueError(
                "only shuffled_graph must bind a shuffle manifest SHA-256"
            )


def _write_ablation_token_cache(
    path: Path,
    row_indices: object,
    tokens: object,
    provenance: AblationCacheProvenance,
) -> TokenCache:
    manifest_path = _manifest_path(path)
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite ablation token cache: {path}")
    rows, values = _validated_arrays(row_indices, tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, row_indices=rows, tokens=values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(path)
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": ABLATION_CACHE_SCHEMA_VERSION,
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "token_dim": TOKEN_DIM,
            "token_feature_names": list(TOKEN_FEATURE_NAMES),
            "token_slices": {
                name: [bounds.start, bounds.stop]
                for name, bounds in TOKEN_SLICES.items()
            },
            "rows": len(rows),
            "ordered_row_sha256": _ordered_row_sha256(rows),
            "cache_sha256": digest,
            "provenance": asdict(provenance),
        },
    )
    return TokenCache(path, rows, values, provenance, digest)  # type: ignore[arg-type]


def load_ablation_token_cache(
    path: str | Path,
    *,
    expected: AblationCacheProvenance | None = None,
) -> TokenCache:
    source = Path(path)
    manifest_path = _manifest_path(source)
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"ablation token cache is incomplete: {source}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ablation cache manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("ablation cache manifest must be a mapping")
    if manifest.get("schema_version") != ABLATION_CACHE_SCHEMA_VERSION:
        raise ValueError("ablation cache schema is incompatible")
    if manifest.get("token_schema_version") != TOKEN_SCHEMA_VERSION:
        raise ValueError("ablation cache token schema is incompatible")
    if manifest.get("token_dim") != TOKEN_DIM:
        raise ValueError("ablation cache token dimension is incompatible")
    if manifest.get("token_feature_names") != list(TOKEN_FEATURE_NAMES):
        raise ValueError("ablation cache token features are incompatible")
    expected_slices = {
        name: [bounds.start, bounds.stop] for name, bounds in TOKEN_SLICES.items()
    }
    if manifest.get("token_slices") != expected_slices:
        raise ValueError("ablation cache token slices are incompatible")
    digest = sha256_file(source)
    if manifest.get("cache_sha256") != digest:
        raise ValueError("ablation token cache SHA-256 mismatch")
    raw_provenance = manifest.get("provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("ablation cache provenance must be a mapping")
    try:
        provenance = AblationCacheProvenance(**dict(raw_provenance))
    except (TypeError, ValueError) as error:
        raise ValueError("ablation cache provenance is invalid") from error
    if expected is not None:
        actual_values = asdict(provenance)
        expected_values = asdict(expected)
        differing = [
            name for name in expected_values
            if actual_values.get(name) != expected_values[name]
        ]
        if differing:
            raise ValueError(
                "ablation cache provenance mismatch: " + ", ".join(differing)
            )
    try:
        with np.load(source, allow_pickle=False) as arrays:
            if set(arrays.files) != {"row_indices", "tokens"}:
                raise ValueError("ablation cache arrays are incompatible")
            rows, tokens = _validated_arrays(
                arrays["row_indices"].copy(), arrays["tokens"].copy()
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid ablation token cache: {source}") from error
    if manifest.get("rows") != len(rows):
        raise ValueError("ablation cache row count mismatch")
    if manifest.get("ordered_row_sha256") != _ordered_row_sha256(rows):
        raise ValueError("ablation cache ordered row hash mismatch")
    return TokenCache(source, rows, tokens, provenance, digest)  # type: ignore[arg-type]


def _validated_episode_rows(
    partition_episode_rows: Mapping[str, Mapping[int, Sequence[int]]],
    source_rows: np.ndarray,
) -> dict[str, dict[int, tuple[int, ...]]]:
    if set(partition_episode_rows) != set(_PARTITIONS):
        raise ValueError("ablation rows must contain train, validation, and test")
    result: dict[str, dict[int, tuple[int, ...]]] = {}
    combined: list[int] = []
    episodes: set[int] = set()
    for partition in _PARTITIONS:
        raw = partition_episode_rows[partition]
        if len(raw) < 2:
            raise ValueError(f"{partition} requires at least two episodes for shuffling")
        result[partition] = {}
        for episode, values in raw.items():
            episode_id = int(episode)
            rows = tuple(int(row) for row in values)
            if episode_id in episodes or not rows or len(set(rows)) != len(rows):
                raise ValueError("ablation episode rows are empty, duplicate, or overlapping")
            episodes.add(episode_id)
            result[partition][episode_id] = rows
            combined.extend(rows)
    if len(set(combined)) != len(combined):
        raise ValueError("ablation partition rows overlap")
    if set(combined) != set(int(row) for row in source_rows):
        raise ValueError("ablation partition rows do not exactly cover the source cache")
    return result


def _shuffle_manifest(
    rows: Mapping[str, Mapping[int, Sequence[int]]], *, shuffle_seed: int
) -> dict[str, Any]:
    permutations: dict[str, dict[str, int]] = {}
    strata: dict[str, dict[str, int]] = {}
    lengths: dict[str, dict[str, int]] = {}
    for index, partition in enumerate(_PARTITIONS):
        partition_lengths = {
            int(episode): len(tuple(values))
            for episode, values in rows[partition].items()
        }
        derived_seed = int(
            np.random.SeedSequence((int(shuffle_seed), index)).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        mapping, partition_strata = stratified_episode_permutation(
            partition_lengths, seed=derived_seed
        )
        permutations[partition] = {
            str(destination): int(source)
            for destination, source in sorted(mapping.items())
        }
        strata[partition] = {
            str(episode): int(value)
            for episode, value in sorted(partition_strata.items())
        }
        lengths[partition] = {
            str(episode): int(value)
            for episode, value in sorted(partition_lengths.items())
        }
    return {
        "schema_version": SHUFFLE_MANIFEST_SCHEMA_VERSION,
        "shuffle_seed": int(shuffle_seed),
        "partition_permutations": permutations,
        "partition_strata": strata,
        "episode_lengths": lengths,
        "resampling": "nearest_normalized_progress",
        "interpolation": False,
    }


def build_ablation_cache_matrix(
    directory: str | Path,
    *,
    source_cache: TokenCache,
    partition_episode_rows: Mapping[str, Mapping[int, Sequence[int]]],
    seed: int,
    shuffle_seed: int,
) -> tuple[dict[str, TokenCache], dict[str, Any]]:
    provenance = source_cache.provenance
    if (
        provenance.condition != "predicted_random_v2"
        or provenance.graph_initialization != "random_init"
        or not math.isclose(float(provenance.graph_fraction), 1.0)
        or int(provenance.graph_seed) != int(seed)
        or provenance.oracle_report_sha256 is None
        or provenance.graph_checkpoint_sha256 is None
    ):
        raise ValueError(
            "ablation source must be the matching full-data predicted_random_v2 cache"
        )
    rows = _validated_episode_rows(partition_episode_rows, source_cache.row_indices)
    destination = Path(directory)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"ablation cache destination must be empty: {destination}")
    if destination.exists():
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    manifest = _shuffle_manifest(rows, shuffle_seed=shuffle_seed)
    try:
        manifest_path = staging / "shuffle_manifest.json"
        _write_json_atomic(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        by_row = source_cache.by_row
        shuffled_by_row: dict[int, np.ndarray] = {}
        for partition in _PARTITIONS:
            mapping = manifest["partition_permutations"][partition]
            for destination_episode, destination_rows in rows[partition].items():
                source_episode = int(mapping[str(destination_episode)])
                source_rows = rows[partition][source_episode]
                source_tokens = np.stack([by_row[row] for row in source_rows])
                sampled = resample_sequence_nearest(
                    source_tokens, destination_length=len(destination_rows)
                )
                shuffled_by_row.update(
                    {
                        row: sampled[index]
                        for index, row in enumerate(destination_rows)
                    }
                )
        shuffled = np.stack(
            [shuffled_by_row[int(row)] for row in source_cache.row_indices]
        )
        transformed = {
            condition: (
                shuffled
                if condition == "shuffled_graph"
                else representation_transform(source_cache.tokens, condition)
            )
            for condition in ABLATION_CONDITIONS
        }
        for condition in ABLATION_CONDITIONS:
            cache_provenance = AblationCacheProvenance(
                condition=condition,
                dataset_fingerprint=provenance.dataset_fingerprint,
                split_manifest_sha256=provenance.split_manifest_sha256,
                oracle_report_sha256=provenance.oracle_report_sha256,
                source_condition="predicted_random_v2",
                source_cache_sha256=source_cache.sha256,
                graph_checkpoint_sha256=provenance.graph_checkpoint_sha256,
                graph_initialization="random_init",
                graph_fraction=1.0,
                graph_seed=int(seed),
                transform_version=ABLATION_TRANSFORM_VERSION,
                shuffle_manifest_sha256=(
                    manifest_sha if condition == "shuffled_graph" else None
                ),
            )
            _write_ablation_token_cache(
                staging / f"{condition}.npz",
                source_cache.row_indices,
                transformed[condition],
                cache_provenance,
            )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    caches = {
        condition: load_ablation_token_cache(destination / f"{condition}.npz")
        for condition in ABLATION_CONDITIONS
    }
    return caches, manifest
