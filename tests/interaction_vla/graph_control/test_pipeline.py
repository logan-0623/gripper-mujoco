from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

from interaction_vla.graph_control.pipeline import (
    _atomic_output_directory,
    _load_source,
    _publish_evaluation,
    _train_seed_with_fallback,
    _validate_formal_epochs,
    evaluate_from_config,
)


def test_downstream_graph_control_does_not_repin_collection_source(
    monkeypatch,
) -> None:
    received = {}

    def fake_validate(root, **kwargs):
        received.update(root=root, **kwargs)

    class FakeDataset:
        def __init__(self, repo_id, *, root):
            self.repo_id = repo_id
            self.root = root

    datasets = types.ModuleType("lerobot.datasets")
    datasets.LeRobotDataset = FakeDataset
    lerobot = types.ModuleType("lerobot")
    lerobot.datasets = datasets
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.validate_dataset_root", fake_validate
    )
    bridge = SimpleNamespace(
        dataset=SimpleNamespace(root=Path("dataset"), repo_id="local/data")
    )

    source = _load_source(bridge)

    assert source.repo_id == "local/data"
    assert received["bridge_config"] is None
    assert received["require_bridge_metadata"] is True
    assert received["replay"] is False


def test_atomic_output_directory_removes_partial_run_and_publishes_success(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="training failed"):
        with _atomic_output_directory(destination) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("training failed")
    assert not destination.exists()
    assert not list(tmp_path.glob(".runs-*"))

    with _atomic_output_directory(destination) as staging:
        (staging / "complete.txt").write_text("complete", encoding="utf-8")
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "complete"


def test_evaluation_preflights_existing_outputs_before_loading_context(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "runs"
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "report.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(training=SimpleNamespace(output_dir=output_dir))
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_graph_control_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._context",
        lambda path: pytest.fail("context must not load for an existing evaluation"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_from_config(tmp_path / "config.yaml")


def test_evaluation_report_failure_does_not_publish_partial_records(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "evaluation"
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._write_json_atomic",
        lambda path, payload: (_ for _ in ()).throw(RuntimeError("report failed")),
    )

    with pytest.raises(RuntimeError, match="report failed"):
        _publish_evaluation(
            destination,
            records=[{"condition": "flat"}],
            report={"passed": True},
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".evaluation-*"))


def test_formal_training_rejects_bridge_epoch_drift() -> None:
    config = SimpleNamespace(training=SimpleNamespace(formal_epochs=5))
    _validate_formal_epochs(config, SimpleNamespace(act=SimpleNamespace(epochs=5)))
    with pytest.raises(ValueError, match="exactly 5"):
        _validate_formal_epochs(config, SimpleNamespace(act=SimpleNamespace(epochs=4)))


def test_seed_oom_discards_partial_matrix_and_restarts_all_conditions(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "seed_0"
    attempts: list[tuple[int, str]] = []
    cleared = []

    def train_attempt(batch_size: int, output_dir: Path):
        output_dir.mkdir()
        for condition in ("flat", "predicted_random", "predicted_reflect", "oracle_current"):
            attempts.append((batch_size, condition))
            (output_dir / condition).mkdir()
            if batch_size == 2 and condition == "predicted_reflect":
                raise RuntimeError("MPS backend out of memory")
        return {"passed": True}

    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._clear_mps_memory",
        lambda: cleared.append(True),
    )
    report = _train_seed_with_fallback(
        destination,
        batch_size=2,
        train_attempt=train_attempt,
    )

    assert attempts == [
        (2, "flat"),
        (2, "predicted_random"),
        (2, "predicted_reflect"),
        (1, "flat"),
        (1, "predicted_random"),
        (1, "predicted_reflect"),
        (1, "oracle_current"),
    ]
    assert cleared == [True]
    assert report["batch_size"] == 1
    assert report["fallback_from_batch_size"] == 2
    assert sorted(path.name for path in destination.iterdir()) == [
        "flat", "oracle_current", "predicted_random", "predicted_reflect"
    ]
