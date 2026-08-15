import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import pytest
import torch

pytest.importorskip("lerobot")
from lerobot.processor import DeviceProcessorStep

from interaction_vla.lerobot_bridge.act_smoke import (
    ACTBundle,
    _act_config,
    _write_checkpoint_metadata,
    build_act_bundle_from_dataset,
    expected_smoke_report_contract,
    require_cached_backbone_weights,
    run_one_batch_check,
    validate_smoke_report_compatibility,
)
from interaction_vla.graph_control.cache import CacheProvenance, write_token_cache
from interaction_vla.graph_control.dataset import GraphConditionedDataset
from interaction_vla.graph_control.schema import TOKEN_DIM
from interaction_vla.lerobot_bridge.act_smoke import load_act_dataset
from interaction_vla.lerobot_bridge.rollout import load_act_runtime
from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.provenance import sha256_file


def test_one_batch_act_update_and_reload_are_finite(
    tiny_lerobot_dataset, tmp_path
) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    result = run_one_batch_check(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_dir=tmp_path / "checkpoint",
        device=torch.device("cpu"),
        batch_size=1,
        seed=0,
        architecture="test",
    )

    assert result.loss >= 0.0
    assert result.gradient_norm > 0.0
    assert result.reload_max_abs_error <= 1e-5
    assert (tmp_path / "checkpoint" / "model.safetensors").is_file()
    assert (tmp_path / "checkpoint" / "policy_preprocessor.json").is_file()
    assert (tmp_path / "checkpoint" / "policy_postprocessor.json").is_file()
    assert (tmp_path / "checkpoint" / "bridge_checkpoint.json").is_file()


def test_configured_act_uses_bridge_learning_rate() -> None:
    bridge = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")
    bridge = replace(bridge, act=replace(bridge.act, learning_rate=2e-5))
    config = _act_config(
        device=torch.device("cpu"),
        architecture="configured",
        bridge_config=bridge,
    )

    assert config.optimizer_lr == bridge.act.learning_rate


def test_configured_act_uses_recovery_backbone_and_horizon() -> None:
    bridge = load_bridge_config("configs/lerobot_act_recovery_macos.yaml")
    config = _act_config(
        device=torch.device("cpu"),
        architecture="configured",
        bridge_config=bridge,
    )

    assert config.chunk_size == 8
    assert config.n_action_steps == 1
    assert (
        config.pretrained_backbone_weights
        == bridge.act.pretrained_backbone_weights
    )


def test_checkpoint_records_cached_backbone_archive(
    tmp_path: Path, monkeypatch
) -> None:
    from torchvision.models import get_weight

    bridge = load_bridge_config("configs/lerobot_act_recovery_macos.yaml")
    identifier = bridge.act.pretrained_backbone_weights
    assert identifier is not None
    filename = Path(urlparse(get_weight(identifier).url).path).name
    hub = tmp_path / "hub"
    cached = hub / "checkpoints" / filename
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"official-weight-archive")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))

    resolved = require_cached_backbone_weights(identifier)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    class Dataset:
        features: dict[str, object] = {}

    bundle = ACTBundle(
        dataset=Dataset(),
        config=_act_config(
            device=torch.device("cpu"),
            architecture="configured",
            bridge_config=bridge,
        ),
        policy=None,
        preprocessor=None,
        postprocessor=None,
        backbone_weights_sha256=sha256_file(resolved),
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "data.bin").write_bytes(b"dataset")
    _write_checkpoint_metadata(
        checkpoint,
        bundle=bundle,
        dataset_root=dataset_root,
        device=torch.device("cpu"),
        extra=None,
    )

    payload = json.loads(
        (checkpoint / "bridge_checkpoint.json").read_text(encoding="utf-8")
    )
    assert payload["pretrained_backbone_weights"] == identifier
    assert payload["backbone_weights_sha256"] == sha256_file(cached)


def test_graph_conditioned_act_uses_separate_environment_token(
    tiny_lerobot_dataset, tmp_path
) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    base = load_act_dataset(dataset_root=dataset_root, repo_id=repo_id)
    rows = [int(base[index]["index"].item()) for index in range(len(base))]
    provenance = CacheProvenance(
        condition="flat",
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        graph_checkpoint_sha256=None,
        graph_initialization=None,
        graph_fraction=None,
        graph_seed=None,
    )
    cache = write_token_cache(
        tmp_path / "tokens.npz",
        rows,
        torch.zeros(len(rows), TOKEN_DIM).numpy(),
        provenance,
    )
    conditioned = GraphConditionedDataset(base, cache)

    torch.manual_seed(7)
    bundle = build_act_bundle_from_dataset(
        conditioned, device=torch.device("cpu"), architecture="test"
    )
    batch = next(iter(torch.utils.data.DataLoader(conditioned, batch_size=1)))
    processed = bundle.preprocessor(batch)
    loss, _ = bundle.policy.forward(processed)

    assert bundle.config.robot_state_feature.shape == (10,)
    assert tuple(bundle.config.env_state_feature.shape) == (TOKEN_DIM,)
    assert bundle.policy.model.encoder_env_state_input_proj.in_features == TOKEN_DIM
    assert torch.isfinite(loss)


def test_pilot_gate_rejects_a_smoke_report_missing_schema_contract(tmp_path) -> None:
    path = tmp_path / "smoke_report.json"
    report = {"passed": True, **expected_smoke_report_contract()}
    report.pop("schema_version")
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        validate_smoke_report_compatibility(path)

    report["schema_version"] = expected_smoke_report_contract()["schema_version"]
    path.write_text(json.dumps(report), encoding="utf-8")
    validate_smoke_report_compatibility(path)


def test_mps_saved_checkpoint_can_be_loaded_for_cpu(
    tiny_lerobot_dataset, tmp_path
) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    checkpoint = tmp_path / "mps-saved"
    run_one_batch_check(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_dir=checkpoint,
        device=torch.device("cpu"),
        batch_size=1,
        seed=0,
        architecture="test",
    )
    for name in ("config.json", "policy_preprocessor.json"):
        path = checkpoint / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "config.json":
            payload["device"] = "mps"
        else:
            for step in payload["steps"]:
                if step["registry_name"] == "device_processor":
                    step["config"]["device"] = "mps"
        path.write_text(json.dumps(payload), encoding="utf-8")

    policy, preprocessor, postprocessor = load_act_runtime(
        checkpoint, device=torch.device("cpu")
    )

    assert policy.config.device == "cpu"
    device_steps = [
        step
        for pipeline in (preprocessor, postprocessor)
        for step in pipeline.steps
        if isinstance(step, DeviceProcessorStep)
    ]
    assert device_steps
    assert all(step.device == "cpu" for step in device_steps)
