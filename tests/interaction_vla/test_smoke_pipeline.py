from __future__ import annotations

import json

import pytest
import yaml

from interaction_vla.data import augment_recovery_from_config, collect_from_config
from interaction_vla.evaluate import evaluate_from_config
from interaction_vla.train import train_from_config


def test_cpu_collection_training_and_paired_evaluation_pipeline(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "smoke.yaml"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "test_smoke",
                "seed": 7,
                "device": "cpu",
                "max_objects": 5,
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "train": {
                    "object_counts": [2, 3],
                    "episodes": 6,
                    "batch_size": 64,
                    "epochs": 1,
                    "learning_rate": 0.003,
                    "model_seeds": [0],
                },
                "eval": {
                    "object_counts": [2, 4],
                    "ood_object_counts": [4],
                    "episodes_per_count": 1,
                    "max_steps": 1,
                },
                "model": {
                    "embedding_dim": 16,
                    "hidden_dim": 16,
                    "message_rounds": 1,
                    "action_dim": 4,
                },
                    "environment": {
                    "max_steps": 120,
                    "workspace_low": [-0.45, -0.35, 0.02],
                    "workspace_high": [0.45, 0.35, 0.55],
                        "min_object_distance": 0.12,
                    },
                    "recovery": {"enabled": True, "variants_per_episode": 1},
            }
        ),
        encoding="utf-8",
    )

    manifest = collect_from_config(config_path)
    recovery_manifest = augment_recovery_from_config(config_path)
    checkpoints = [
        train_from_config(config_path, representation)
        for representation in ("proprio", "flat", "graph")
    ]
    report_path = evaluate_from_config(config_path, checkpoints)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert manifest.exists()
    assert recovery_manifest.exists()
    assert (data_dir / "rejections.json").exists()
    assert all(checkpoint.exists() for checkpoint in checkpoints)
    assert set(report["by_policy"]) == {
        "proprio",
        "flat",
        "graph",
        "graph_edge_shuffled",
    }
    assert set(report["offline_physical_action_mse_by_policy"]) == set(report["by_policy"])
    assert set(report["offline_normalized_action_mse_by_policy"]) == set(report["by_policy"])
    for representation in ("proprio", "flat", "graph"):
        assert f"{representation} seed=0" in captured.err
    assert "mse=" in captured.err
