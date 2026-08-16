from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest
import torch

from interaction_vla.graph_control.pipeline import (
    _clear_accelerator_memory,
    _atomic_output_directory,
    _load_source,
    _publish_evaluation,
    _require_oracle_report,
    _require_recovery_report,
    _train_seed_with_fallback,
    _validate_formal_epochs,
    evaluate_from_config,
)


def test_accelerator_memory_cleanup_releases_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    _clear_accelerator_memory()

    assert calls == ["cuda"]


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
    config = SimpleNamespace(training=SimpleNamespace(formal_epochs=10))
    _validate_formal_epochs(config, SimpleNamespace(act=SimpleNamespace(epochs=10)))
    with pytest.raises(ValueError, match="exactly 10"):
        _validate_formal_epochs(config, SimpleNamespace(act=SimpleNamespace(epochs=4)))


def test_recovery_prerequisite_binds_exact_gate_and_returns_hash(tmp_path: Path) -> None:
    report_path = tmp_path / "recovery.json"
    report_path.write_text(
        '{"passed": true, "train_seen": {"success_rate": 0.9}, '
        '"heldout": {"success_rate": 0.7}}',
        encoding="utf-8",
    )
    config = SimpleNamespace(required_recovery_report=report_path)
    recovery = SimpleNamespace(
        train_success_threshold=0.8,
        heldout_success_threshold=0.3,
    )
    digest = _require_recovery_report(config, SimpleNamespace(recovery=recovery))

    assert len(digest) == 64

    recovery.train_success_threshold = 0.7
    with pytest.raises(ValueError, match="0.80/0.30"):
        _require_recovery_report(config, SimpleNamespace(recovery=recovery))

    recovery.train_success_threshold = 0.8
    report_path.write_text(
        '{"passed": true, "train_seen": {"success_rate": 0.9}, '
        '"heldout": {"success_rate": 0.2}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        _require_recovery_report(config, SimpleNamespace(recovery=recovery))


def test_predicted_matrix_requires_passing_oracle_gate_and_binds_hash(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "oracle.json"
    report_path.write_text(
        '{"passed": true, "oracle_gate": {"passed": true}}',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        conditions=(
            "flat",
            "oracle_graph_v2",
            "predicted_random_v2",
            "predicted_reflect_v2",
        ),
        required_oracle_report=report_path,
    )

    digest = _require_oracle_report(config)

    assert len(digest) == 64

    report_path.write_text(
        '{"passed": false, "oracle_gate": {"passed": false}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="oracle gate"):
        _require_oracle_report(config)


def test_oracle_matrix_forbids_oracle_report_binding() -> None:
    config = SimpleNamespace(
        conditions=("flat", "oracle_graph_v2"),
        required_oracle_report=Path("unexpected.json"),
    )
    with pytest.raises(ValueError, match="must be null"):
        _require_oracle_report(config)


def test_seed_oom_discards_partial_matrix_and_restarts_all_conditions(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "seed_0"
    attempts: list[tuple[int, str]] = []
    cleared = []

    def train_attempt(batch_size: int, output_dir: Path):
        output_dir.mkdir()
        for condition in ("flat", "oracle_graph_v2"):
            attempts.append((batch_size, condition))
            (output_dir / condition).mkdir()
            if batch_size == 2 and condition == "oracle_graph_v2":
                raise RuntimeError("MPS backend out of memory")
        return {"passed": True}

    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._clear_accelerator_memory",
        lambda: cleared.append(True),
    )
    report = _train_seed_with_fallback(
        destination,
        batch_size=2,
        train_attempt=train_attempt,
    )

    assert attempts == [
        (2, "flat"),
        (2, "oracle_graph_v2"),
        (1, "flat"),
        (1, "oracle_graph_v2"),
    ]
    assert cleared == [True]
    assert report["batch_size"] == 1
    assert report["fallback_from_batch_size"] == 2
    assert sorted(path.name for path in destination.iterdir()) == [
        "flat", "oracle_graph_v2"
    ]
