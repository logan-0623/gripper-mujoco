from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.cache import (
    CacheProvenance,
    build_token_cache,
    current_fields_from_teacher,
    load_token_cache,
    write_token_cache,
)
from interaction_vla.graph_control.features import GraphNormalization
from interaction_vla.graph_control.schema import TOKEN_DIM


def _provenance(condition: str = "predicted_reflect") -> CacheProvenance:
    return CacheProvenance(
        condition=condition,
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        graph_checkpoint_sha256=None if condition == "flat" else "b" * 64,
        graph_initialization=None if condition == "flat" else "reflectvlm_init",
        graph_fraction=None if condition == "flat" else 1.0,
        graph_seed=None if condition == "flat" else 0,
    )


def test_cache_round_trip_is_immutable_and_binds_provenance(tmp_path: Path) -> None:
    path = tmp_path / "tokens.npz"
    rows = np.array([2, 5, 9], dtype=np.int64)
    tokens = np.arange(3 * TOKEN_DIM, dtype=np.float32).reshape(3, TOKEN_DIM)

    written = write_token_cache(path, rows, tokens, _provenance())
    loaded = load_token_cache(path, expected=_provenance())

    assert written.sha256 == loaded.sha256
    np.testing.assert_array_equal(loaded.row_indices, rows)
    np.testing.assert_array_equal(loaded.tokens, tokens)
    with pytest.raises(FileExistsError, match="refusing"):
        write_token_cache(path, rows, tokens, _provenance())
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        load_token_cache(
            path,
            expected=replace(_provenance(), dataset_fingerprint="x" * 64),
        )


class _Source:
    def __init__(self, count: int) -> None:
        self.samples = [
            {
                "index": torch.tensor(index),
                "observation.images.agent": torch.zeros(3, 8, 8),
                "observation.images.wrist": torch.zeros(3, 8, 8),
                "observation.state": torch.full((10,), float(index)),
                "task": "place target",
            }
            for index in range(count)
        ]

    def __getitem__(self, index: int):
        return self.samples[index]


class _Runtime:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.normalization = GraphNormalization(
            state_mean=np.zeros(10, dtype=np.float32),
            state_std=np.ones(10, dtype=np.float32),
            relation_mean=np.zeros((8, 10), dtype=np.float32),
            relation_std=np.ones((8, 10), dtype=np.float32),
            residual_mean=0.0,
            residual_std=1.0,
        )

    def predict_tokens(self, *, agent_rgb, wrist_rgb, state, task):
        self.batch_sizes.append(len(state))
        result = np.zeros((len(state), TOKEN_DIM), dtype=np.float32)
        result[:, 0] = [float(np.asarray(value)[0]) for value in state]
        return result


def test_build_cache_preserves_global_rows_and_bounds_inference_batches(
    tmp_path: Path,
) -> None:
    runtime = _Runtime()
    path = tmp_path / "predicted.npz"

    cache = build_token_cache(
        path,
        source=_Source(6),
        row_indices=(5, 1, 3),
        condition="predicted_reflect",
        runtime=runtime,
        batch_size=2,
        provenance=_provenance(),
    )

    assert runtime.batch_sizes == [2, 1]
    assert cache.row_indices.tolist() == [5, 1, 3]
    assert cache.tokens[:, 0].tolist() == [5.0, 1.0, 3.0]


def test_flat_cache_never_invokes_graph_runtime(tmp_path: Path) -> None:
    cache = build_token_cache(
        tmp_path / "flat.npz",
        source=_Source(3),
        row_indices=(0, 2),
        condition="flat",
        runtime=None,
        batch_size=2,
        provenance=_provenance("flat"),
    )
    assert np.array_equal(cache.tokens, np.zeros((2, TOKEN_DIM), dtype=np.float32))


class _NoFutureGoal(dict[str, np.ndarray]):
    def __getitem__(self, key: str) -> np.ndarray:
        if key == "annotation.tc_tig.relation_goal":
            raise AssertionError("future relation_goal was accessed")
        return super().__getitem__(key)


def test_current_teacher_extraction_does_not_access_future_goal() -> None:
    arrays = _NoFutureGoal(
        {
            "annotation.tc_tig.entity_mask": np.ones((2, 6), dtype=np.bool_),
            "annotation.tc_tig.entity_visibility": np.ones((2, 6, 2), dtype=np.float32),
            "annotation.tc_tig.relation_mask": np.ones((2, 8), dtype=np.bool_),
            "annotation.tc_tig.relation_values": np.arange(
                2 * 8 * 24, dtype=np.float32
            ).reshape(2, 8, 24),
            "annotation.tc_tig.relation_goal": np.ones((2, 5), dtype=np.float32),
        }
    )

    current = current_fields_from_teacher(arrays, frame_index=1)

    np.testing.assert_array_equal(
        current.relation_semantics,
        arrays["annotation.tc_tig.relation_values"][1, :, 12:22],
    )


def test_cache_rejects_duplicate_rows_and_nonfinite_tokens(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        write_token_cache(
            tmp_path / "bad.npz",
            np.array([1, 1]),
            np.zeros((2, TOKEN_DIM), dtype=np.float32),
            _provenance(),
        )
    bad = np.zeros((1, TOKEN_DIM), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_token_cache(
            tmp_path / "nan.npz", np.array([0]), bad, _provenance()
        )
