from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

from interaction_vla.graph_control.pipeline import _load_source


def test_downstream_graph_control_does_not_repin_collection_source(
    monkeypatch,
) -> None:
    received = {}

    def fake_validate(root, **kwargs):
        received.update(root=root, **kwargs)

    class FakeDataset:
        def __init__(self, repo_id, *, root):
            self.repo_id = repo_id
            self.root = root

    datasets = types.ModuleType("lerobot.datasets")
    datasets.LeRobotDataset = FakeDataset
    lerobot = types.ModuleType("lerobot")
    lerobot.datasets = datasets
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.validate_dataset_root", fake_validate
    )
    bridge = SimpleNamespace(
        dataset=SimpleNamespace(root=Path("dataset"), repo_id="local/data")
    )

    source = _load_source(bridge)

    assert source.repo_id == "local/data"
    assert received["bridge_config"] is None
    assert received["require_bridge_metadata"] is True
    assert received["replay"] is False
