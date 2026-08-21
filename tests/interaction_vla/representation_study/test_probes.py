from __future__ import annotations

from pathlib import Path

import numpy as np

from interaction_vla.representation_study.config import ProbeConfig
from interaction_vla.representation_study.probes.targets import ProbeTarget
from interaction_vla.representation_study.probes.training import train_single_probe


def test_linear_probe_learns_accessible_binary_factor(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(90, 6)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64)
    target = ProbeTarget("contact", "binary", labels, 2)
    partitions = {
        "train": np.arange(0, 60),
        "validation": np.arange(60, 75),
        "test": np.arange(75, 90),
    }
    _, result, normalization = train_single_probe(
        features,
        target,
        partitions,
        model_kind="linear",
        config=ProbeConfig(
            output_dir=tmp_path,
            epochs=60,
            batch_size=16,
            learning_rate=0.03,
            weight_decays=(0.0, 1e-3),
            seed=3,
        ),
    )
    assert result["metrics"]["test"]["balanced_accuracy"] > 0.8
    assert result["metrics"]["test"]["brier_score"] >= 0.0
    assert result["metrics"]["test"]["expected_calibration_error"] >= 0.0
    assert result["baseline_metrics"]["test"]["balanced_accuracy"] <= 0.6
    assert result["sample_counts"] == {"train": 60, "validation": 15, "test": 15}
    assert normalization["mean"].shape == (6,)


def test_continuous_probe_reports_mae_and_r2(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(75, 4)).astype(np.float32)
    values = np.stack((features[:, 0] + features[:, 1], features[:, 2] - features[:, 3]), axis=1)
    target = ProbeTarget("geometry", "continuous", values.astype(np.float32), 2)
    partitions = {
        "train": np.arange(0, 50),
        "validation": np.arange(50, 62),
        "test": np.arange(62, 75),
    }
    _, result, _ = train_single_probe(
        features,
        target,
        partitions,
        model_kind="linear",
        config=ProbeConfig(tmp_path, 80, 16, 0.02, (0.0,), 5),
    )
    assert result["metrics"]["test"]["r2"] > 0.9
    assert result["metrics"]["test"]["r2"] > result["baseline_metrics"]["test"]["r2"]
