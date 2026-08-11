from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from interaction_vla.graph_pretrain.config import load_graph_pretrain_config
from interaction_vla.graph_pretrain.pipeline import (
    evaluate_with_source,
    inspect_with_source,
    load_checkpoint,
    require_finite_loss,
    train_with_source,
)


def row(group: int, step: int) -> dict[str, object]:
    actions = ("pick up yellow", "insert orange", "reorient purple", "put down green")
    action = actions[(group + step) % len(actions)]
    return {
        "board_id": group,
        "env_seed": 1000 + group,
        "trajectory_id": group,
        "step_id": step,
        "history": str(list(actions[: step % len(actions)])),
        "oracle_action": action,
        "object_states": str(
            {
                "green block": "DONE",
                "orange block": "READY",
                "purple block": "BLOCKED (by predecessor)",
                "yellow nail": "BAD (is down)",
            }
        ),
        "object_in_hand": None,
        "object_is_upright": "{2: True, 3: True, 4: False, 5: False}",
        "object_descriptions": str(
            {
                "brick_1": "gray board",
                "brick_2": "green block",
                "brick_3": "orange block",
                "brick_4": "purple block",
                "brick_5": "yellow nail",
            }
        ),
        "object_dependencies": "{(2, 3), (3, 4), (4, 5)}",
        "image": Image.new(
            "RGB", (24, 18), color=(20 + group, 30 + step, 40)
        ),
    }


def source() -> list[dict[str, object]]:
    return [row(group, step) for group in range(8) for step in range(2)]


def write_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
dataset:
  repo_id: test/reflect
  source_split: train
  split_seed: 3
  split_ratios: [0.625, 0.1875, 0.1875]
  max_rows: null
model:
  image_size: 32
  max_history_tokens: 8
  image_embedding_dim: 16
  text_embedding_dim: 8
  graph_embedding_dim: 16
training:
  output_dir: {output_dir.as_posix()}
  device: cpu
  batch_size: 2
  num_workers: 0
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  seed: 4
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_config_and_inspect_report_grouped_partitions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, tmp_path / "output")
    config = load_graph_pretrain_config(config_path)

    report = inspect_with_source(config, source())

    assert report["passed"] is True
    assert report["schema_version"] == "reflect_semantic_graph_v1"
    assert report["rows"] == 16
    assert report["groups"] == 8
    assert sum(report["partition_rows"].values()) == 16
    assert all(count > 0 for count in report["partition_groups"].values())


def test_repository_smoke_config_is_bounded() -> None:
    config = load_graph_pretrain_config(
        "configs/reflectvlm_graph_pretrain_smoke_macos.yaml"
    )

    assert config.dataset.max_rows == 96
    assert config.training.epochs == 1
    assert config.training.output_dir == Path(
        "outputs/graph_pretrain/reflectvlm_smoke"
    )


def test_synthetic_train_checkpoint_reload_and_evaluate(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    write_config(config_path, output_dir)
    config = load_graph_pretrain_config(config_path)
    records = source()

    training = train_with_source(config, records)
    checkpoint = Path(training["checkpoint"])
    model, vocabulary, payload = load_checkpoint(checkpoint, device="cpu")
    evaluation = evaluate_with_source(
        config, records, checkpoint, partition="test"
    )

    assert training["steps"] > 0
    assert torch.isfinite(torch.tensor(training["best_validation_loss"]))
    assert checkpoint.is_file()
    assert (output_dir / "training_summary.json").is_file()
    assert (output_dir / "split_manifest.json").is_file()
    assert (output_dir / "evaluation.json").is_file()
    assert payload["schema_version"] == "reflect_semantic_graph_v1"
    assert len(vocabulary.tokens) == model.token_embedding.num_embeddings
    assert evaluation["partition"] == "test"
    assert evaluation["examples"] > 0
    assert torch.isfinite(torch.tensor(evaluation["mean_loss"]))
    for name in (
        "target_accuracy",
        "in_hand_accuracy",
        "state_accuracy",
        "upright_accuracy",
        "dependency_precision",
        "dependency_recall",
        "dependency_f1",
        "phase_accuracy",
        "goal_exact_accuracy",
    ):
        assert 0.0 <= evaluation[name] <= 1.0
    assert json.loads((output_dir / "evaluation.json").read_text())["partition"] == "test"


def test_checkpoint_rejects_incompatible_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": "wrong"}, path)

    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path, device="cpu")


def test_require_finite_loss_rejects_nan() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_loss(torch.tensor(float("nan")))
