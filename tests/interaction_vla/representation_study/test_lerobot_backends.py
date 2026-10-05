from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from interaction_vla.representation_study.backends.lerobot import (
    _TRAINABLE_PREFIXES,
    make_backend,
)
from interaction_vla.representation_study.taps.registry import registered_taps


@pytest.mark.parametrize("name", ["smolvla", "pi0"])
def test_backend_factory_exposes_fixed_tap_contract_without_loading_weights(name: str) -> None:
    backend = make_backend(name, device="cpu")
    assert backend.backend_name == name
    assert tuple(tap.tap_id for tap in registered_taps(name))


@pytest.mark.parametrize("name", ["unknown", "act"])
def test_backend_factory_rejects_unknown_policy(name: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        make_backend(name, device="cpu")


def test_modern_vla_fusion_group_does_not_unfreeze_entire_vlm() -> None:
    assert "model.vlm_with_expert" not in _TRAINABLE_PREFIXES["smolvla"]["fusion"]
    assert "model.paligemma_with_expert" not in _TRAINABLE_PREFIXES["pi0"]["fusion"]
    assert any("lm_expert" in value for value in _TRAINABLE_PREFIXES["smolvla"]["fusion"])


def test_dataset_bound_loader_preserves_checkpoint_camera_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("lerobot")
    import lerobot.configs.policies as policy_configs
    import lerobot.datasets.dataset_metadata as dataset_metadata
    import lerobot.policies as policies

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_inputs = {
        "observation.state": object(),
        "observation.images.camera1": object(),
        "observation.images.camera2": object(),
        "observation.images.camera3": object(),
    }
    policy_config = SimpleNamespace(
        device="cpu",
        pretrained_path=None,
        input_features=dict(checkpoint_inputs),
        output_features={"old.action": object()},
        normalization_mapping={},
    )
    metadata = SimpleNamespace(stats={"observation.state": {"mean": torch.zeros(1)}})
    rename_map = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    captured: dict[str, object] = {}

    class FakePreTrainedConfig:
        @classmethod
        def from_pretrained(cls, path: str) -> object:
            captured["checkpoint"] = path
            return policy_config

    class FakeMetadata:
        def __new__(cls, repo_id: str, *, root: Path) -> object:
            captured["repo_id"] = repo_id
            captured["dataset_root"] = root
            return metadata

    def fake_make_policy(
        config: object, *, ds_meta: object, rename_map: dict[str, str]
    ) -> object:
        captured["make_policy_config"] = config
        captured["make_policy_metadata"] = ds_meta
        captured["make_policy_rename_map"] = rename_map
        config.output_features = {"action": object()}
        return SimpleNamespace(config=config)

    def fake_make_processors(**kwargs: object) -> tuple[object, object]:
        captured["processor_kwargs"] = kwargs
        return object(), object()

    monkeypatch.setattr(policy_configs, "PreTrainedConfig", FakePreTrainedConfig)
    monkeypatch.setattr(dataset_metadata, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(policies, "make_policy", fake_make_policy)
    monkeypatch.setattr(policies, "make_pre_post_processors", fake_make_processors)

    backend = make_backend("smolvla", device="cpu")
    backend.load_checkpoint_for_dataset(
        checkpoint,
        repo_id="lerobot/libero",
        dataset_root=tmp_path / "dataset",
        rename_map=rename_map,
    )

    assert policy_config.input_features == checkpoint_inputs
    assert captured["make_policy_rename_map"] == rename_map
    processor_kwargs = captured["processor_kwargs"]
    assert isinstance(processor_kwargs, dict)
    preprocessor_overrides = processor_kwargs["preprocessor_overrides"]
    assert preprocessor_overrides["rename_observations_processor"] == {
        "rename_map": rename_map
    }
    assert preprocessor_overrides["normalizer_processor"]["features"] == {
        **policy_config.input_features,
        **policy_config.output_features,
    }
