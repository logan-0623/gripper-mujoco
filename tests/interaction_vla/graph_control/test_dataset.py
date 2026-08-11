from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.cache import CacheProvenance, write_token_cache
from interaction_vla.graph_control.dataset import GraphConditionedDataset
from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_FEATURE_NAMES


class _Meta:
    def __init__(self) -> None:
        self.features = {
            "observation.images.agent": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [10],
                "names": [f"state_{index}" for index in range(10)],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [f"action_{index}" for index in range(7)],
            },
        }
        self.stats = {"observation.state": {"mean": torch.zeros(10)}}
        self.fps = 20


class _Dataset:
    def __init__(self, *, forbidden: bool = False) -> None:
        self.meta = _Meta()
        self.features = self.meta.features
        self.samples = []
        for item, row in enumerate((8, 3)):
            sample = {
                "index": torch.tensor(row),
                "observation.images.agent": torch.full((3, 2, 2), float(item)),
                "observation.images.wrist": torch.full((3, 2, 2), float(item + 1)),
                "observation.state": torch.full((10,), float(item)),
                "action": torch.full((8, 7), float(item)),
                "task": "place target",
            }
            if forbidden:
                sample["annotation.tc_tig.relation_goal"] = torch.zeros(5)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int):
        return self.samples[item]


def _cache(tmp_path: Path):
    provenance = CacheProvenance(
        condition="flat",
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        graph_checkpoint_sha256=None,
        graph_initialization=None,
        graph_fraction=None,
        graph_seed=None,
    )
    tokens = np.zeros((2, TOKEN_DIM), dtype=np.float32)
    tokens[0, 0] = 8.0
    tokens[1, 0] = 3.0
    return write_token_cache(tmp_path / "cache.npz", [8, 3], tokens, provenance)


def test_dataset_adds_native_environment_state_by_global_index(tmp_path: Path) -> None:
    base = _Dataset()
    dataset = GraphConditionedDataset(base, _cache(tmp_path))

    first = dataset[0]
    second = dataset[1]

    assert first.keys() == base[0].keys() | {"observation.environment_state"}
    assert first["observation.environment_state"].shape == (TOKEN_DIM,)
    assert first["observation.environment_state"].dtype == torch.float32
    assert first["observation.environment_state"][0].item() == 8.0
    assert second["observation.environment_state"][0].item() == 3.0
    for name in base[0]:
        if isinstance(base[0][name], torch.Tensor):
            assert torch.equal(first[name], base[0][name])
        else:
            assert first[name] == base[0][name]


def test_metadata_proxy_declares_75d_environment_feature(tmp_path: Path) -> None:
    base = _Dataset()
    dataset = GraphConditionedDataset(base, _cache(tmp_path))

    assert dataset.meta.fps == 20
    assert dataset.meta.stats is base.meta.stats
    assert dataset.features == dataset.meta.features
    feature = dataset.meta.features["observation.environment_state"]
    assert feature == {
        "dtype": "float32",
        "shape": [TOKEN_DIM],
        "names": list(TOKEN_FEATURE_NAMES),
    }
    assert "observation.environment_state" not in base.meta.features


def test_dataset_refuses_teacher_annotations_in_policy_sample(tmp_path: Path) -> None:
    dataset = GraphConditionedDataset(_Dataset(forbidden=True), _cache(tmp_path))
    with pytest.raises(ValueError, match="forbidden"):
        dataset[0]


def test_dataset_refuses_missing_or_duplicate_cache_rows(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    base = _Dataset()
    base.samples[1]["index"] = torch.tensor(99)
    dataset = GraphConditionedDataset(base, cache)
    with pytest.raises(ValueError, match="row 99"):
        dataset[1]

