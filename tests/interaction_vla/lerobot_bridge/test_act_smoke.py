import json
from dataclasses import replace

import pytest
import torch

pytest.importorskip("lerobot")
from lerobot.processor import DeviceProcessorStep

from interaction_vla.lerobot_bridge.act_smoke import (
    _act_config,
    expected_smoke_report_contract,
    run_one_batch_check,
    validate_smoke_report_compatibility,
)
from interaction_vla.lerobot_bridge.rollout import load_act_runtime
from interaction_vla.lerobot_bridge.config import load_bridge_config


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
