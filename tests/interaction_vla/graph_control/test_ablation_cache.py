from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from interaction_vla.graph_control.ablation_cache import (
    ABLATION_CACHE_SCHEMA_VERSION,
    ABLATION_TRANSFORM_VERSION,
    AblationCacheProvenance,
    build_ablation_cache_matrix,
    load_ablation_token_cache,
)
from interaction_vla.graph_control.cache import CacheProvenance, write_token_cache
from interaction_vla.graph_control.schema import ABLATION_CONDITIONS, TOKEN_DIM


def _source_cache(tmp_path: Path):
    rows = np.arange(12, dtype=np.int64)
    tokens = np.arange(12 * TOKEN_DIM, dtype=np.float32).reshape(12, TOKEN_DIM)
    provenance = CacheProvenance(
        condition="predicted_random_v2",
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        graph_checkpoint_sha256="b" * 64,
        graph_initialization="random_init",
        graph_fraction=1.0,
        graph_seed=2,
        oracle_report_sha256="c" * 64,
    )
    return write_token_cache(tmp_path / "source.npz", rows, tokens, provenance)


def test_ablation_cache_matrix_binds_source_transform_and_shuffle_manifest(
    tmp_path: Path,
) -> None:
    source = _source_cache(tmp_path)
    partition_episode_rows = {
        "train": {0: (0, 1), 1: (2, 3, 4)},
        "validation": {2: (5, 6), 3: (7, 8)},
        "test": {4: (9,), 5: (10, 11)},
    }

    caches, shuffle_manifest = build_ablation_cache_matrix(
        tmp_path / "ablation",
        source_cache=source,
        partition_episode_rows=partition_episode_rows,
        seed=2,
        shuffle_seed=17,
    )

    assert tuple(caches) == ABLATION_CONDITIONS
    assert shuffle_manifest["partition_permutations"]["train"]
    assert all(cache.row_indices.tolist() == list(range(12)) for cache in caches.values())
    assert np.count_nonzero(caches["flat"].tokens) == 0
    assert np.array_equal(caches["full_graph"].tokens, source.tokens)
    assert not np.array_equal(caches["shuffled_graph"].tokens, source.tokens)
    for condition, cache in caches.items():
        assert cache.provenance.condition == condition
        assert cache.provenance.source_cache_sha256 == source.sha256
        assert cache.provenance.transform_version == ABLATION_TRANSFORM_VERSION
        loaded = load_ablation_token_cache(cache.path, expected=cache.provenance)
        np.testing.assert_array_equal(loaded.tokens, cache.tokens)
        manifest = json.loads(cache.path.with_suffix(".manifest.json").read_text())
        assert manifest["schema_version"] == ABLATION_CACHE_SCHEMA_VERSION
    assert caches["shuffled_graph"].provenance.shuffle_manifest_sha256 is not None
    assert caches["full_graph"].provenance.shuffle_manifest_sha256 is None


def test_ablation_cache_rejects_wrong_source_or_partition_overlap(tmp_path: Path) -> None:
    source = _source_cache(tmp_path)
    partitions = {
        "train": {0: (0, 1), 1: (2, 3)},
        "validation": {2: (4, 5), 3: (6, 7)},
        "test": {4: (8, 9), 5: (10, 11)},
    }
    caches, _ = build_ablation_cache_matrix(
        tmp_path / "ablation",
        source_cache=source,
        partition_episode_rows=partitions,
        seed=2,
        shuffle_seed=17,
    )
    wrong = replace(caches["full_graph"].provenance, source_cache_sha256="e" * 64)
    with pytest.raises(ValueError, match="source_cache_sha256"):
        load_ablation_token_cache(caches["full_graph"].path, expected=wrong)

    overlapping = {**partitions, "validation": {2: (3, 4), 3: (5, 6)}}
    with pytest.raises(ValueError, match="partition rows overlap"):
        build_ablation_cache_matrix(
            tmp_path / "bad",
            source_cache=source,
            partition_episode_rows=overlapping,
            seed=2,
            shuffle_seed=17,
        )


def test_ablation_provenance_requires_random_full_data_source() -> None:
    base = dict(
        condition="full_graph",
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        oracle_report_sha256="c" * 64,
        source_condition="predicted_random_v2",
        source_cache_sha256="e" * 64,
        graph_checkpoint_sha256="b" * 64,
        graph_initialization="reflectvlm_init",
        graph_fraction=1.0,
        graph_seed=0,
        transform_version=ABLATION_TRANSFORM_VERSION,
    )
    with pytest.raises(ValueError, match="random_init"):
        AblationCacheProvenance(**base)
