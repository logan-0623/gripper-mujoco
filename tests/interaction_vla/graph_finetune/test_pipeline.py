from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from interaction_vla.graph_finetune.config import load_graph_finetune_config
from interaction_vla.graph_finetune.pipeline import (
    compare_with_source,
    evaluate_with_source,
    inspect_with_source,
    load_finetune_checkpoint,
    require_finite_loss,
)

from .test_data import SyntheticSource, manifest_records, sidecars
from .test_model import save_reflect_checkpoint


def write_config(path: Path, output_dir: Path, reflect_checkpoint: Path) -> None:
    path.write_text(
        f"""
dataset:
  repo_id: test/mujoco
  root: {path.parent.as_posix()}/dataset
  reflect_checkpoint: {reflect_checkpoint.as_posix()}
  split_seed: 3
  split_ratios: [0.6, 0.2, 0.2]
model:
  image_size: 32
  max_language_tokens: 16
  image_embedding_dim: 32
  text_embedding_dim: 16
  graph_embedding_dim: 24
training:
  output_dir: {output_dir.as_posix()}
  device: cpu
  batch_size: 2
  num_workers: 0
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  fractions: [1.0]
  seeds: [0]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def configured(tmp_path: Path):
    checkpoint = tmp_path / "reflect.pt"
    save_reflect_checkpoint(checkpoint)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, tmp_path / "output", checkpoint)
    return load_graph_finetune_config(config_path)


def test_config_and_inspect_report_episode_isolation(tmp_path: Path) -> None:
    config = configured(tmp_path)
    records = manifest_records()
    source = SyntheticSource(records)

    report = inspect_with_source(config, source, records, sidecars(records))

    assert report["passed"] is True
    assert report["schema_version"] == "mujoco_semantic_graph_v1"
    assert report["episodes"] == 5
    assert report["frames"] == 15
    assert report["partition_episodes"] == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert report["partition_frames"] == {
        "train": 9,
        "validation": 3,
        "test": 3,
    }
    assert report["episode_overlap"] is False
    assert report["copied_token_count"] >= 2


def test_synthetic_paired_comparison_writes_reloadable_artifacts(
    tmp_path: Path,
) -> None:
    config = configured(tmp_path)
    records = manifest_records()
    source = SyntheticSource(records)

    report = compare_with_source(config, source, records, sidecars(records))

    assert report["passed"] is True
    assert report["paired_runs"] == 1
    assert set(report["conditions"]) == {"random_init", "reflectvlm_init"}
    run = report["runs"][0]
    assert run["fraction"] == 1.0
    assert run["seed"] == 0
    assert run["random_init"]["test_examples"] == 3
    assert run["reflectvlm_init"]["test_examples"] == 3
    assert run["random_init"]["test_row_indices"] == run["reflectvlm_init"][
        "test_row_indices"
    ]
    for value in run["delta"].values():
        assert math.isfinite(value)
    assert report["aggregate"]["random_init"]["goal_exact_accuracy"]["count"] == 1
    assert math.isfinite(
        report["aggregate"]["reflectvlm_init"]["semantic_relation_mae"]["mean"]
    )
    for condition in ("random_init", "reflectvlm_init"):
        checkpoint = Path(run[condition]["checkpoint"])
        model, vocabulary, payload = load_finetune_checkpoint(
            checkpoint, device="cpu"
        )
        assert checkpoint.is_file()
        assert payload["initialization"] == condition
        assert len(vocabulary.tokens) == model.token_embedding.num_embeddings
        assert payload["test_row_indices"] == run[condition]["test_row_indices"]
    output = config.training.output_dir
    assert json.loads((output / "comparison.json").read_text())["passed"] is True
    assert (output / "split_manifest.json").is_file()

    evaluation = evaluate_with_source(
        config,
        source,
        records,
        sidecars(records),
        Path(run["reflectvlm_init"]["checkpoint"]),
        partition="test",
    )
    assert evaluation["test_examples"] == 3
    assert evaluation["initialization"] == "reflectvlm_init"
    assert evaluation["test_row_indices"] == run["reflectvlm_init"][
        "test_row_indices"
    ]


def test_compare_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    config = configured(tmp_path)
    config.training.output_dir.mkdir(parents=True)
    (config.training.output_dir / "keep.txt").write_text("user data\n")
    records = manifest_records()

    with pytest.raises(FileExistsError, match="empty"):
        compare_with_source(
            config,
            SyntheticSource(records),
            records,
            sidecars(records),
        )


def test_require_finite_loss_rejects_nan() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_loss(torch.tensor(float("nan")))


def test_repository_configs_lock_smoke_and_pilot_matrices() -> None:
    smoke = load_graph_finetune_config(
        "configs/mujoco_graph_finetune_smoke_macos.yaml"
    )
    pilot = load_graph_finetune_config(
        "configs/mujoco_graph_finetune_pilot_macos.yaml"
    )

    assert smoke.dataset.split_ratios == (0.6, 0.2, 0.2)
    assert smoke.training.epochs == 1
    assert smoke.training.fractions == (1.0,)
    assert smoke.training.seeds == (0,)
    assert pilot.dataset.root == Path("outputs/lerobot/franka_lerobot_act_pilot")
    assert pilot.dataset.split_ratios == (0.8, 0.1, 0.1)
    assert pilot.training.fractions == (0.1, 0.25, 1.0)
    assert pilot.training.seeds == (0, 1, 2)
