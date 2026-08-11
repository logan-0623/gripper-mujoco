from dataclasses import replace
from pathlib import Path

import pytest

from interaction_vla.lerobot_bridge.config import load_bridge_config


def test_macos_environments_pin_the_validated_mujoco_runtime() -> None:
    base_requirements = Path("requirements-macos.txt").read_text(encoding="utf-8")
    requirements = Path("requirements-lerobot-macos.txt").read_text(encoding="utf-8")
    lock = Path("requirements-lerobot-macos.lock.txt").read_text(encoding="utf-8")

    assert "mujoco==3.3.4" in base_requirements.splitlines()
    assert "mujoco==3.3.4" in requirements.splitlines()
    assert "mujoco==3.3.4" in lock.splitlines()


def test_lerobot_environment_pairs_torch_2_10_with_torchcodec_0_10() -> None:
    requirements = Path("requirements-lerobot-macos.txt").read_text(encoding="utf-8")
    lock = Path("requirements-lerobot-macos.lock.txt").read_text(encoding="utf-8")

    assert "torch>=2.10,<2.11" in requirements.splitlines()
    assert "torchcodec>=0.10,<0.11" in requirements.splitlines()
    assert "torch==2.10.0" in lock.splitlines()
    assert "torchcodec==0.10.0" in lock.splitlines()


def test_lerobot_environment_pins_reflectvlm_dataset_runtime() -> None:
    requirements = Path("requirements-lerobot-macos.txt").read_text(encoding="utf-8")
    lock = Path("requirements-lerobot-macos.lock.txt").read_text(encoding="utf-8")

    assert "datasets==4.8.5" in requirements.splitlines()
    assert "datasets==4.8.5" in lock.splitlines()
    assert "socksio==1.0.0" in requirements.splitlines()
    assert "socksio==1.0.0" in lock.splitlines()


def test_smoke_config_locks_the_model_visible_contract() -> None:
    config = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")

    assert config.dataset.episodes == 5
    assert config.dataset.object_counts == (2, 3)
    assert config.dataset.fps == 20
    assert config.dataset.image_size == (256, 256)
    assert config.dataset.state_dim == 10
    assert config.dataset.action_dim == 7
    assert config.act.chunk_size == 8
    assert config.act.n_action_steps == 8
    assert config.act.steps == 500
    assert config.act.epochs is None
    assert config.source.backend == "franka_contact"


def test_pilot_config_uses_epochs_and_requires_smoke_report() -> None:
    config = load_bridge_config("configs/lerobot_act_pilot_macos.yaml")

    assert config.dataset.episodes == 50
    assert config.act.steps is None
    assert config.act.epochs == 5
    assert config.act.maximum_epochs == 10
    assert config.required_smoke_report is not None


def test_act_schedule_requires_exactly_one_stop_condition() -> None:
    config = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")

    with pytest.raises(ValueError, match="exactly one"):
        replace(config.act, steps=500, epochs=5)
    with pytest.raises(ValueError, match="exactly one"):
        replace(config.act, steps=None, epochs=None)
    with pytest.raises(ValueError, match="learning_rate"):
        replace(config.act, learning_rate=0.0)
