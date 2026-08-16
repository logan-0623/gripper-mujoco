from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.cache import (
    CACHE_SCHEMA_VERSION,
    CacheProvenance,
    build_token_cache,
    load_token_cache,
    write_token_cache,
)
from interaction_vla.graph_control.schema import (
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SCHEMA_VERSION,
)
from interaction_vla.graph_finetune.schema import GraphV2Normalization

from .test_features import _targets


def _provenance(condition: str = "predicted_reflect_v2") -> CacheProvenance:
    checkpoint_free = condition in {"flat", "oracle_graph_v2"}
    return CacheProvenance(
        condition=condition,
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        graph_checkpoint_sha256=None if checkpoint_free else "b" * 64,
        graph_initialization=None if checkpoint_free else "reflectvlm_init",
        graph_fraction=None if checkpoint_free else 1.0,
        graph_seed=None if checkpoint_free else 0,
    )


def test_cache_round_trip_is_immutable_and_binds_v2_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tokens.npz"
    rows = np.array([2, 5, 9], dtype=np.int64)
    tokens = np.arange(3 * TOKEN_DIM, dtype=np.float32).reshape(3, TOKEN_DIM)

    written = write_token_cache(path, rows, tokens, _provenance())
    loaded = load_token_cache(path, expected=_provenance())
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())

    assert written.sha256 == loaded.sha256
    assert manifest["schema_version"] == CACHE_SCHEMA_VERSION
    assert manifest["token_schema_version"] == TOKEN_SCHEMA_VERSION
    assert manifest["token_feature_names"] == list(TOKEN_FEATURE_NAMES)
    assert len(manifest["ordered_row_sha256"]) == 64
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
    def __init__(self, episodes: int = 2, frames: int = 2) -> None:
        self.samples = []
        for episode in range(episodes):
            for frame in range(frames):
                index = len(self.samples)
                state = torch.zeros(10)
                state[0] = episode
                state[1] = frame
                self.samples.append(
                    {
                        "index": torch.tensor(index),
                        "episode_index": torch.tensor(episode),
                        "frame_index": torch.tensor(frame),
                        "observation.images.agent": torch.zeros(3, 8, 8),
                        "observation.images.wrist": torch.zeros(3, 8, 8),
                        "observation.state": state,
                        "task": "place target",
                    }
                )

    def __getitem__(self, index: int):
        return self.samples[index]


class _Runtime:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def reset(self) -> None:
        self.events.append(("reset",))

    def predict_token(self, *, agent_rgb, wrist_rgb, state, task):
        values = np.asarray(state, dtype=np.float32)
        episode = int(values[0])
        frame = int(values[1])
        self.events.append(("predict", episode, frame))
        token = np.zeros(TOKEN_DIM, dtype=np.float32)
        token[:2] = (episode, frame)
        return token


def test_build_cache_resets_runtime_and_predicts_in_episode_chronology(
    tmp_path: Path,
) -> None:
    runtime = _Runtime()

    cache = build_token_cache(
        tmp_path / "predicted.npz",
        source=_Source(),
        episode_rows={1: (2, 3), 0: (0, 1)},
        condition="predicted_reflect_v2",
        runtime=runtime,
        provenance=_provenance(),
    )

    assert runtime.events == [
        ("reset",),
        ("predict", 0, 0),
        ("predict", 0, 1),
        ("reset",),
        ("predict", 1, 0),
        ("predict", 1, 1),
    ]
    assert cache.row_indices.tolist() == [0, 1, 2, 3]
    assert cache.tokens[:, :2].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_flat_and_oracle_caches_do_not_require_graph_checkpoint(
    tmp_path: Path,
) -> None:
    source = _Source(episodes=1, frames=2)
    flat = build_token_cache(
        tmp_path / "flat.npz",
        source=source,
        episode_rows={0: (0, 1)},
        condition="flat",
        runtime=None,
        provenance=_provenance("flat"),
    )
    normalization = GraphV2Normalization(
        state_mean=np.zeros(10, dtype=np.float32),
        state_std=np.ones(10, dtype=np.float32),
        workspace_scale=1.0,
        velocity_scale=1.0,
    )
    oracle = build_token_cache(
        tmp_path / "oracle.npz",
        source=source,
        episode_rows={0: (0, 1)},
        condition="oracle_graph_v2",
        runtime=None,
        provenance=_provenance("oracle_graph_v2"),
        oracle_targets={0: _targets(frames=2)},
        normalization=normalization,
    )

    assert np.array_equal(flat.tokens, np.zeros((2, TOKEN_DIM), dtype=np.float32))
    assert np.count_nonzero(oracle.tokens) > 0


def test_cache_rejects_misaligned_episode_frame_and_global_rows(
    tmp_path: Path,
) -> None:
    source = _Source()
    source.samples[1]["frame_index"] = torch.tensor(9)

    with pytest.raises(ValueError, match="frame"):
        build_token_cache(
            tmp_path / "bad_alignment.npz",
            source=source,
            episode_rows={0: (0, 1)},
            condition="predicted_random_v2",
            runtime=_Runtime(),
            provenance=_provenance("predicted_random_v2"),
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
