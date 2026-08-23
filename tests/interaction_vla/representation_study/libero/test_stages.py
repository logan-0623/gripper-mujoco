from dataclasses import replace
from pathlib import Path

from interaction_vla.representation_study.libero.stages import (
    EpisodeInfo,
    build_stage_manifests,
    plan_nested_subsets,
)
from interaction_vla.representation_study.libero.training import (
    _tree_sha256,
    build_stage_training_command,
)
import pytest
from interaction_vla.representation_study.libero.config import load_libero_study_config
from interaction_vla.representation_study.state_bank.io import write_json_atomic


def _episodes() -> tuple[EpisodeInfo, ...]:
    return tuple(
        EpisodeInfo(
            episode_index=task * 10 + episode,
            suite="libero_spatial",
            task_id=task,
            frames=100 + episode,
        )
        for task in range(4)
        for episode in range(8)
    )


def test_fraction_subsets_are_task_balanced_deterministic_and_nested() -> None:
    first = plan_nested_subsets(_episodes(), fractions=(0.25, 0.5, 1.0), seed=17)
    second = plan_nested_subsets(_episodes(), fractions=(0.25, 0.5, 1.0), seed=17)
    assert first == second
    assert set(first[0.25]).issubset(first[0.5])
    assert set(first[0.5]).issubset(first[1.0])
    assert len(first[0.25]) == 8
    assert len(first[0.5]) == 16
    assert len(first[1.0]) == 32
    for subset in first.values():
        tasks = {episode // 10 for episode in subset}
        assert tasks == {0, 1, 2, 3}


def test_absent_checkpoints_are_not_run_and_steps_scale_with_fraction(tmp_path: Path) -> None:
    subsets = plan_nested_subsets(_episodes(), fractions=(0.25, 0.5, 1.0), seed=17)
    manifests = build_stage_manifests(
        episodes=_episodes(),
        subsets=subsets,
        output_dir=tmp_path,
        base_model="lerobot/smolvla_base",
        base_revision="revision-a",
        dataset_repo_id="lerobot/libero",
        dataset_revision="revision-b",
        seed=17,
        epochs=2,
        batch_size=8,
        code_hash="a" * 64,
        config_hash="b" * 64,
    )
    assert tuple(manifests) == ("pretrained", "sft_25", "sft_50", "sft_100")
    assert manifests["pretrained"].status == "not_run"
    assert manifests["sft_25"].status == "not_run"
    assert manifests["sft_25"].training_steps < manifests["sft_50"].training_steps
    assert manifests["sft_50"].training_steps < manifests["sft_100"].training_steps
    assert manifests["sft_25"].subset_sha256 != manifests["sft_50"].subset_sha256


def test_training_command_is_bound_to_nested_manifest(tmp_path: Path) -> None:
    config = load_libero_study_config(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    )
    config = replace(config, output_dir=tmp_path)
    manifest = {
        "stage": "sft_25",
        "dataset_repo_id": "HuggingFaceVLA/libero",
        "dataset_revision": "a" * 40,
        "episode_indices": [1, 4, 7],
        "training_steps": 12,
    }
    write_json_atomic(tmp_path / "stages/sft_25/manifest.json", manifest)
    pretrained = tmp_path / "stages/pretrained/checkpoint"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text("{}")
    write_json_atomic(
        tmp_path / "stages/pretrained/manifest.json",
        {
            "status": "complete",
            "checkpoint": str(pretrained),
            "checkpoint_sha256": _tree_sha256(pretrained),
        },
    )
    command = build_stage_training_command(config, stage="sft_25")
    assert "--dataset.episodes=[1,4,7]" in command
    assert "--steps=12" in command
    assert "--cudnn_deterministic=true" in command

    (pretrained / "config.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="hash is stale"):
        build_stage_training_command(config, stage="sft_25")
