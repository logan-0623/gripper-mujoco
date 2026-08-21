from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import types

import numpy as np
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
    diagnose_from_config,
    evaluate_from_config,
    failure_analysis_from_config,
    sensitivity_from_config,
    trace_from_config,
)
from interaction_vla.graph_control.schema import ALL_CONDITIONS, TOKEN_DIM


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


def _diagnostic_fixture(tmp_path: Path):
    diagnostics = SimpleNamespace(
        output_dir=tmp_path / "diagnostics",
        bootstrap_samples=10,
        bootstrap_seed=17,
        max_lag=1,
        active_epsilon=1.0e-6,
        sensitivity_rows_per_episode=1,
        sensitivity_batch_size=2,
        sensitivity_scale=0.25,
    )
    config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        diagnostics=diagnostics,
        conditions=ALL_CONDITIONS,
        seeds=(0,),
    )
    split = SimpleNamespace(
        rows={
            "train": (10,),
            "validation": (11,),
            "test": (0, 1, 2, 3),
        },
        episodes={"train": (8,), "validation": (9,), "test": (0, 1)},
        sha256="1" * 64,
        path=tmp_path / "split.json",
    )
    source = SimpleNamespace(
        hf_dataset={
            "episode_index": np.array([0, 0, 1, 1]),
            "frame_index": np.array([0, 1, 0, 1]),
        }
    )
    teacher = np.linspace(0.1, 0.9, 4 * TOKEN_DIM).reshape(4, TOKEN_DIM)
    provenance = SimpleNamespace(dataset_fingerprint="2" * 64)
    caches = {
        "flat": SimpleNamespace(
            row_indices=np.arange(4),
            tokens=np.zeros_like(teacher),
            sha256="3" * 64,
            provenance=provenance,
        ),
        "oracle_graph_v2": SimpleNamespace(
            row_indices=np.arange(4),
            tokens=teacher.copy(),
            sha256="4" * 64,
            provenance=provenance,
        ),
        "predicted_random_v2": SimpleNamespace(
            row_indices=np.arange(4),
            tokens=teacher * 0.9,
            sha256="5" * 64,
            provenance=provenance,
        ),
        "predicted_reflect_v2": SimpleNamespace(
            row_indices=np.arange(4),
            tokens=teacher * 1.1,
            sha256="6" * 64,
            provenance=provenance,
        ),
    }
    context = (config, SimpleNamespace(), split, source, "7" * 64, "8" * 64)
    return config, context, caches


def _patch_diagnostic_context(monkeypatch, config, context, caches) -> None:
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_graph_control_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._context", lambda path: context
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._load_cache_matrix",
        lambda *args, **kwargs: caches,
    )


def test_diagnostics_requires_configured_output(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(diagnostics=None)
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_graph_control_config",
        lambda path: config,
    )

    with pytest.raises(ValueError, match="diagnostics config"):
        diagnose_from_config(tmp_path / "config.yaml", partition="test")


def test_diagnostics_preflights_output_before_loading_context(
    tmp_path: Path, monkeypatch
) -> None:
    config, _, _ = _diagnostic_fixture(tmp_path)
    destination = config.diagnostics.output_dir / "test"
    destination.mkdir(parents=True)
    (destination / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_graph_control_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._context",
        lambda path: pytest.fail("context must not load for an existing diagnostic"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        diagnose_from_config(tmp_path / "config.yaml", partition="test")


def test_diagnostics_rejects_cache_rows_before_computing_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _diagnostic_fixture(tmp_path)
    caches["predicted_random_v2"].row_indices = np.array([0, 1, 3, 2])
    _patch_diagnostic_context(monkeypatch, config, context, caches)
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.build_representation_diagnostics",
        lambda **kwargs: pytest.fail("metrics must not run for misaligned rows"),
    )

    with pytest.raises(ValueError, match="cache rows"):
        diagnose_from_config(tmp_path / "config.yaml", partition="test")


def test_diagnostics_publication_is_atomic_on_report_failure(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _diagnostic_fixture(tmp_path)
    _patch_diagnostic_context(monkeypatch, config, context, caches)
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._write_json_atomic",
        lambda path, payload: (_ for _ in ()).throw(RuntimeError("report failed")),
    )

    with pytest.raises(RuntimeError, match="report failed"):
        diagnose_from_config(tmp_path / "config.yaml", partition="test")

    assert not (config.diagnostics.output_dir / "test").exists()


def test_diagnostics_selects_partition_rows_and_publishes_jsonl(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _diagnostic_fixture(tmp_path)
    _patch_diagnostic_context(monkeypatch, config, context, caches)

    result = diagnose_from_config(tmp_path / "config.yaml", partition="test")

    assert result["passed"] is True
    assert result["partition"] == "test"
    assert result["rows"] == 4
    assert result["episodes"] == 2
    assert "by_seed_condition" not in result
    assert result["conditions"] == list(ALL_CONDITIONS)
    assert result["estimator_seeds"] == [0]
    assert result["report_path"].is_file()
    assert result["per_episode_path"].is_file()
    report = json.loads(result["report_path"].read_text(encoding="utf-8"))
    assert report["cache_sha256"]["seed_0/predicted_random_v2"] == "5" * 64
    records = [
        json.loads(line)
        for line in result["per_episode_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 8
    assert {record["episode_id"] for record in records} == {0, 1}


class _SensitivitySource:
    def __init__(self) -> None:
        self.hf_dataset = {
            "episode_index": np.asarray([0, 0, 1, 1, 8, 8]),
            "frame_index": np.asarray([0, 1, 0, 1, 0, 1]),
            "action": np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        }

    def __getitem__(self, row: int):
        return {
            "index": torch.tensor(row),
            "episode_index": torch.tensor(int(self.hf_dataset["episode_index"][row])),
            "frame_index": torch.tensor(int(self.hf_dataset["frame_index"][row])),
            "observation.state": torch.full((10,), float(row)),
        }


class _SensitivityPolicy:
    def reset(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def predict_action_chunk(self, batch):
        token = batch["observation.environment_state"]
        result = torch.zeros((len(token), 8, 7), dtype=torch.float32)
        result[:, 0, 0] = token.sum(dim=1)
        return result


def _sensitivity_fixture(tmp_path: Path):
    config, _, _ = _diagnostic_fixture(tmp_path)
    split = SimpleNamespace(
        rows={"train": (4, 5), "validation": (3,), "test": (0, 1, 2, 3)},
        episodes={"train": (8,), "validation": (1,), "test": (0, 1)},
        sha256="1" * 64,
        path=tmp_path / "split.json",
    )
    source = _SensitivitySource()
    teacher = np.linspace(0.1, 0.9, 6 * TOKEN_DIM).reshape(6, TOKEN_DIM)
    provenance = SimpleNamespace(dataset_fingerprint="2" * 64)
    caches = {
        condition: SimpleNamespace(
            row_indices=np.arange(6),
            tokens=(
                np.zeros_like(teacher)
                if condition == "flat"
                else teacher * (1.0 if condition == "oracle_graph_v2" else 0.9)
            ),
            sha256=str(index + 3) * 64,
            provenance=provenance,
        )
        for index, condition in enumerate(ALL_CONDITIONS)
    }
    context = (config, SimpleNamespace(), split, source, "7" * 64, "8" * 64)
    return config, context, caches


def _patch_sensitivity_context(monkeypatch, config, context, caches) -> None:
    _patch_diagnostic_context(monkeypatch, config, context, caches)
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._analysis_policy_runtime",
        lambda *args, seed, condition, **kwargs: (
            _SensitivityPolicy(),
            lambda batch: batch,
            lambda actions: actions,
            {
                "mode": "retrospective_analysis",
                "graph_source_fingerprint_match": False,
                "stored_graph_source_fingerprint": "a" * 64,
                "current_graph_source_fingerprint": "b" * 64,
            },
            Path(f"checkpoint/{seed}/{condition}"),
            "c" * 64,
        ),
    )


def test_sensitivity_preflights_output_before_loading_context(
    tmp_path: Path, monkeypatch
) -> None:
    config, _, _ = _sensitivity_fixture(tmp_path)
    destination = config.diagnostics.output_dir / "test" / "sensitivity_v3"
    destination.mkdir(parents=True)
    (destination / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_graph_control_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._context",
        lambda path: pytest.fail("context must not load for existing sensitivity"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        sensitivity_from_config(tmp_path / "config.yaml", partition="test")


def test_sensitivity_uses_balanced_rows_and_publishes_audited_report(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _sensitivity_fixture(tmp_path)
    _patch_sensitivity_context(monkeypatch, config, context, caches)

    result = sensitivity_from_config(tmp_path / "config.yaml", partition="test")

    assert result["passed"] is True
    assert result["partition"] == "test"
    assert result["observations"] == 2
    assert result["policy_seeds"] == [0]
    assert result["conditions"] == list(ALL_CONDITIONS)
    assert result["report_path"].is_file()
    assert result["records_path"].is_file()
    report = json.loads(result["report_path"].read_text(encoding="utf-8"))
    assert report["schema_version"] == "graph_policy_sensitivity_v3"
    assert report["action_statistics"]["minimum_scale"] == pytest.approx(1.0e-3)
    assert report["selected_rows"] == [0, 2]
    assert report["checkpoint_sha256"]["seed_0/flat"] == "c" * 64
    assert report["checkpoint_compatibility"]["seed_0/flat"][
        "graph_source_fingerprint_match"
    ] is False
    records = result["records_path"].read_text(encoding="utf-8").splitlines()
    assert len(records) == 2 * len(ALL_CONDITIONS) * (3 * 12 + 2)
    assert report["control_interventions"] == [
        "zero",
        "temporally_matched_random",
    ]
    assert report["control_provenance"]["seed_0"]["alignment"] == (
        "normalized_episode_progress_nearest"
    )
    parsed_records = [json.loads(line) for line in records]
    controls = [
        record for record in parsed_records if record["group"] == "all_tokens"
    ]
    assert {record["intervention"] for record in controls} == {
        "zero",
        "temporally_matched_random",
    }


def test_sensitivity_v3_reuses_compatible_v2_group_records(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _sensitivity_fixture(tmp_path)
    _patch_sensitivity_context(monkeypatch, config, context, caches)
    first = sensitivity_from_config(tmp_path / "config.yaml", partition="test")
    v3_report = json.loads(first["report_path"].read_text(encoding="utf-8"))
    v3_records = [
        json.loads(line)
        for line in first["records_path"].read_text(encoding="utf-8").splitlines()
    ]
    v2_records = [
        record for record in v3_records if record["group"] != "all_tokens"
    ]
    v2_dir = config.diagnostics.output_dir / "test" / "sensitivity_v2"
    v2_dir.mkdir()
    v2_report = {
        **v3_report,
        "schema_version": "graph_policy_sensitivity_v2",
        "rows": len(v2_records),
    }
    (v2_dir / "report.json").write_text(
        json.dumps(v2_report), encoding="utf-8"
    )
    (v2_dir / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in v2_records),
        encoding="utf-8",
    )
    shutil.rmtree(config.diagnostics.output_dir / "test" / "sensitivity_v3")
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.finite_difference_interventions",
        lambda *args, **kwargs: pytest.fail(
            "compatible v2 group interventions must be reused"
        ),
    )

    result = sensitivity_from_config(tmp_path / "config.yaml", partition="test")
    report = json.loads(result["report_path"].read_text(encoding="utf-8"))

    assert report["reused_sensitivity_v2"]["records"] == len(v2_records)
    assert len(result["records_path"].read_text(encoding="utf-8").splitlines()) == (
        len(v2_records) + 2 * len(config.conditions) * 2
    )


def _trace_fixture(tmp_path: Path):
    config, context, caches = _sensitivity_fixture(tmp_path)
    config.config_path.write_text("trace: test\n", encoding="utf-8")
    config.trace = SimpleNamespace(
        enabled=True,
        output_dir=tmp_path / "traced_evaluation",
        resume=True,
    )
    config.training = SimpleNamespace(output_dir=tmp_path / "runs")
    config.evaluation = SimpleNamespace(
        layouts=("normal",),
        object_counts=(2,),
        cases_per_cell=1,
        master_seed=17,
        max_steps=2,
    )
    bridge = SimpleNamespace(
        dataset=SimpleNamespace(root=tmp_path / "dataset", repo_id="local/data"),
        act=SimpleNamespace(device="cpu"),
        teacher=SimpleNamespace(),
    )
    context = (config, bridge, context[2], context[3], "7" * 64, "8" * 64)
    return config, context, caches


def _patch_trace_context(monkeypatch, config, context, caches, rollout_calls) -> None:
    _patch_diagnostic_context(monkeypatch, config, context, caches)
    case = SimpleNamespace(
        case_id="normal_n2_000", seed=17, layout="normal", object_count=2
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.paired_evaluation_cases",
        lambda **kwargs: (case,),
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._oracle_inputs",
        lambda *args: (None, None, SimpleNamespace()),
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.fingerprint_tree",
        lambda path: "9" * 64,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline._runtime_for_condition",
        lambda *args, seed, condition, **kwargs: SimpleNamespace(
            condition=condition,
            policy_seed=seed,
            checkpoint=Path(f"checkpoint/{seed}/{condition}"),
            checkpoint_compatibility={"mode": "retrospective_analysis"},
        ),
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.OracleGraphV2TokenProvider",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.TCTIGTeacherExtractor",
        lambda config: object(),
    )

    def fake_rollout(config, runtime, case, **kwargs):
        rollout_calls.append((runtime.policy_seed, runtime.condition, case.case_id))
        kwargs["trace_callback"](
            {
                "condition": runtime.condition,
                "policy_seed": runtime.policy_seed,
                "case_id": case.case_id,
                "environment_seed": case.seed,
                "layout": case.layout,
                "object_count": case.object_count,
            }
        )

    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.rollout_case", fake_rollout
    )

    def fake_write(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.write_trace_episode_atomic",
        fake_write,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_trace_episode",
        lambda path: [json.loads(path.read_text(encoding="utf-8"))],
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.trace_episode_summary",
        lambda records: {
            **records[0],
            "success": records[0]["condition"] != "flat",
            "wrong_object_interaction": False,
            "wrong_object_stable_grasp": False,
            "target_drop": False,
            "timeout": records[0]["condition"] == "flat",
            "termination_reason": (
                "timeout" if records[0]["condition"] == "flat" else "success"
            ),
            "steps": 1,
            "mean_ik_projection_scale": 1.0,
            "action_clipping_rate": 0.0,
            "gripper_switch_count": 0,
            "checkpoint": f"checkpoint/{records[0]['condition']}",
        },
    )


def test_trace_pipeline_is_resumable_and_manifest_bound(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _trace_fixture(tmp_path)
    rollout_calls = []
    _patch_trace_context(monkeypatch, config, context, caches, rollout_calls)

    result = trace_from_config(tmp_path / "config.yaml")

    assert result["passed"] is True
    assert result["episodes"] == len(ALL_CONDITIONS)
    assert len(rollout_calls) == len(ALL_CONDITIONS)
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["completed_episodes"] == len(ALL_CONDITIONS)
    assert result["report_path"].is_file()

    rollout_calls.clear()
    resumed = trace_from_config(tmp_path / "config.yaml")
    assert resumed["resumed"] is True
    assert rollout_calls == []

    config.evaluation.max_steps = 3
    with pytest.raises(ValueError, match="manifest.*max_steps"):
        trace_from_config(tmp_path / "config.yaml")


def test_failure_analysis_binds_complete_trace_and_train_cache_thresholds(
    tmp_path: Path, monkeypatch
) -> None:
    config, context, caches = _trace_fixture(tmp_path)
    _patch_diagnostic_context(monkeypatch, config, context, caches)
    trace_root = config.trace.output_dir
    trace_root.mkdir(parents=True)
    cases = [
        {
            "case_id": "normal_n2_000",
            "environment_seed": 17,
            "layout": "normal",
            "object_count": 2,
        }
    ]
    (trace_root / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "trace_schema_version": "graph_control_step_trace_v1",
                "config_sha256": __import__("hashlib").sha256(
                    config.config_path.read_bytes()
                ).hexdigest(),
                "split_manifest_sha256": context[2].sha256,
                "dataset_fingerprint": "2" * 64,
                "conditions": list(ALL_CONDITIONS),
                "policy_seeds": [0],
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    for condition in ALL_CONDITIONS:
        path = (
            trace_root
            / "traces"
            / "seed_0"
            / condition
            / "normal_n2_000.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    received = {}

    def fake_thresholds(*, condition_tokens, teacher_tokens_by_seed, quantile):
        received["condition_tokens"] = condition_tokens
        received["teacher"] = teacher_tokens_by_seed
        received["quantile"] = quantile
        return {
            f"seed_0/{condition}": {"goal_residual": 0.5}
            for condition in ALL_CONDITIONS
        }

    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.training_error_thresholds",
        fake_thresholds,
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.load_trace_episode",
        lambda path: [
            {
                "policy_seed": 0,
                "condition": path.parent.name,
                "case_id": "normal_n2_000",
                "environment_seed": 17,
                "layout": "normal",
                "object_count": 2,
            }
        ],
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.episode_error_exposure",
        lambda records, *, thresholds: {
            "policy_seed": 0,
            "condition": records[0]["condition"],
            "case_id": "normal_n2_000",
        },
    )
    monkeypatch.setattr(
        "interaction_vla.graph_control.pipeline.build_failure_analysis_report",
        lambda episodes, **kwargs: {
            "passed": True,
            "schema_version": "graph_failure_association_v1",
            "episodes": len(episodes),
            "policy_seeds": [0],
            "conditions": list(ALL_CONDITIONS),
        },
    )

    result = failure_analysis_from_config(
        tmp_path / "config.yaml", traces=trace_root
    )

    assert result["passed"] is True
    assert result["episodes"] == len(ALL_CONDITIONS)
    assert result["report_path"].is_file()
    assert result["exposures_path"].is_file()
    assert received["quantile"] == 0.75
    assert set(received["condition_tokens"]) == {
        (0, condition) for condition in ALL_CONDITIONS
    }
