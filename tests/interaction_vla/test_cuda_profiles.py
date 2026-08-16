from __future__ import annotations

from pathlib import Path

from interaction_vla.graph_control.config import load_graph_control_config
from interaction_vla.graph_finetune.config import load_graph_finetune_config
from interaction_vla.graph_pretrain.config import load_graph_pretrain_config
from interaction_vla.lerobot_bridge.config import load_bridge_config


def test_linux_cuda_requirements_pin_the_validated_cuda_stack() -> None:
    requirements = Path("requirements-lerobot-linux-cuda.txt").read_text(
        encoding="utf-8"
    )

    assert "torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" in requirements
    assert (
        "torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
        in requirements
    )
    assert (
        "torchcodec-0.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
        in requirements
    )
    assert "lerobot[dataset,training]==0.6.1" in requirements.splitlines()
    assert "mujoco==3.3.4" in requirements.splitlines()


def test_linux_cuda_profiles_preserve_the_oracle_scientific_contract() -> None:
    bridge = load_bridge_config("configs/lerobot_act_recovery_linux_cuda.yaml")
    reflect = load_graph_pretrain_config(
        "configs/reflectvlm_graph_pretrain_linux_cuda.yaml"
    )
    finetune = load_graph_finetune_config(
        "configs/mujoco_graph_v2_finetune_linux_cuda.yaml"
    )
    oracle = load_graph_control_config(
        "configs/graph_v2_act_oracle_linux_cuda.yaml"
    )

    assert bridge.act.device == "cuda"
    assert bridge.act.batch_size == 2
    assert bridge.act.epochs == 10
    assert bridge.act.n_action_steps == 1
    assert bridge.dataset.root == Path("outputs/lerobot/franka_lerobot_act_pilot")
    assert bridge.act.output_dir == Path("outputs/graph_control/act_recovery_cuda/train")
    assert bridge.recovery is not None
    assert bridge.recovery.output_dir == Path(
        "outputs/graph_control/act_recovery_cuda/evaluation"
    )

    assert reflect.training.device == "cuda"
    assert reflect.training.batch_size == 16
    assert reflect.training.epochs == 20
    assert reflect.training.output_dir == Path(
        "outputs/graph_pretrain/reflectvlm_cuda"
    )

    assert finetune.training.device == "cuda"
    assert finetune.training.batch_size == 16
    assert finetune.training.epochs == 10
    assert finetune.training.seeds == (0, 1, 2)
    assert finetune.dataset.reflect_checkpoint == Path(
        "outputs/graph_pretrain/reflectvlm_cuda/checkpoint.pt"
    )
    assert finetune.training.output_dir == Path(
        "outputs/graph_finetune/mujoco_graph_v2_cuda"
    )

    assert oracle.bridge_config == Path(
        "configs/lerobot_act_recovery_linux_cuda.yaml"
    )
    assert oracle.conditions == ("flat", "oracle_graph_v2")
    assert oracle.seeds == (0,)
    assert oracle.training.formal_epochs == 10
    assert oracle.cache.directory == Path(
        "outputs/graph_control/graph_v2_oracle_cuda/cache"
    )
    assert oracle.training.output_dir == Path(
        "outputs/graph_control/graph_v2_oracle_cuda/runs"
    )


def test_linux_cuda_main_profile_keeps_three_seed_four_condition_matrix() -> None:
    config = load_graph_control_config("configs/graph_v2_act_pilot_linux_cuda.yaml")

    assert config.bridge_config == Path(
        "configs/lerobot_act_recovery_linux_cuda.yaml"
    )
    assert config.conditions == (
        "flat",
        "oracle_graph_v2",
        "predicted_random_v2",
        "predicted_reflect_v2",
    )
    assert config.seeds == (0, 1, 2)
    assert config.graph_runs_root == Path(
        "outputs/graph_finetune/mujoco_graph_v2_cuda"
    )
    assert config.training.formal_epochs == 10
