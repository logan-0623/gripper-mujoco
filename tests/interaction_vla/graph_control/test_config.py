from __future__ import annotations

from pathlib import Path

import pytest

from interaction_vla.graph_control.config import load_graph_control_config
from interaction_vla.graph_control.schema import ALL_CONDITIONS, ORACLE_CONDITIONS


def _write_config(
    path: Path,
    *,
    conditions: str = "[flat, oracle_graph_v2]",
    graph_runs_root: str = "null",
) -> None:
    path.write_text(
        f"""
bridge_config: configs/lerobot_act_recovery_macos.yaml
required_recovery_report: outputs/graph_control/act_recovery/evaluation/recovery_report.json
required_oracle_report: null
split_manifest: outputs/graph_finetune/mujoco_graph_v2/split_manifest.json
graph_runs_root: {graph_runs_root}
conditions: {conditions}
seeds: [0]
cache:
  directory: outputs/graph_control/graph_v2_oracle/cache
  batch_size: 1
training:
  output_dir: outputs/graph_control/graph_v2_oracle/runs
  smoke_steps: 1
  formal_epochs: 10
evaluation:
  layouts: [normal]
  object_counts: [2]
  cases_per_cell: 20
  master_seed: 2057736129
  max_steps: 180
diagnostics:
  output_dir: outputs/graph_control/graph_v2_oracle/diagnostics
  bootstrap_samples: 2000
  bootstrap_seed: 2057736129
  max_lag: 3
  active_epsilon: 1.0e-6
  sensitivity_rows_per_episode: 4
  sensitivity_batch_size: 4
  sensitivity_scale: 0.25
""".lstrip(),
        encoding="utf-8",
    )


def test_oracle_config_locks_two_conditions_and_recovery_prerequisite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)

    config = load_graph_control_config(path)

    assert config.conditions == ORACLE_CONDITIONS
    assert config.required_recovery_report == Path(
        "outputs/graph_control/act_recovery/evaluation/recovery_report.json"
    )
    assert config.graph_runs_root is None
    assert config.training.formal_epochs == 10
    assert config.graph_checkpoint("flat", 0) is None
    assert config.graph_checkpoint("oracle_graph_v2", 0) is None
    assert config.evaluation.cells == (("normal", 2),)
    assert config.diagnostics is not None
    assert config.diagnostics.output_dir == Path(
        "outputs/graph_control/graph_v2_oracle/diagnostics"
    )
    assert config.diagnostics.bootstrap_samples == 2000
    assert config.diagnostics.bootstrap_seed == 2057736129
    assert config.diagnostics.max_lag == 3
    assert config.diagnostics.active_epsilon == 1.0e-6
    assert config.diagnostics.sensitivity_rows_per_episode == 4
    assert config.diagnostics.sensitivity_batch_size == 4
    assert config.diagnostics.sensitivity_scale == 0.25


def test_full_matrix_requires_graph_runs_root(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(
        path,
        conditions="[flat, oracle_graph_v2, predicted_random_v2, predicted_reflect_v2]",
        graph_runs_root="outputs/graph_finetune/mujoco_graph_v2",
    )
    text = path.read_text().replace("layouts: [normal]", "layouts: [normal, crowded]")
    text = text.replace("object_counts: [2]", "object_counts: [2, 3]")
    text = text.replace(
        "required_oracle_report: null",
        "required_oracle_report: outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json",
    )
    path.write_text(text)

    config = load_graph_control_config(path)

    assert config.conditions == ALL_CONDITIONS
    assert config.required_oracle_report == Path(
        "outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json"
    )
    assert config.graph_checkpoint("predicted_random_v2", 0) == Path(
        "outputs/graph_finetune/mujoco_graph_v2/random_init/fraction_1/seed_0/checkpoint.pt"
    )


def test_config_rejects_arbitrary_condition_matrix(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path, conditions="[flat, predicted_reflect_v2]")
    with pytest.raises(ValueError, match="oracle pair or full"):
        load_graph_control_config(path)


def test_full_matrix_requires_oracle_report_path(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(
        path,
        conditions="[flat, oracle_graph_v2, predicted_random_v2, predicted_reflect_v2]",
        graph_runs_root="outputs/graph_finetune/mujoco_graph_v2",
    )
    with pytest.raises(ValueError, match="required_oracle_report"):
        load_graph_control_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("batch_size: 1", "batch_size: 0", "batch_size"),
        ("seeds: [0]", "seeds: [0, 0]", "seeds"),
        ("cases_per_cell: 20", "cases_per_cell: 0", "cases_per_cell"),
        ("layouts: [normal]", "layouts: [crowded]", "layouts"),
        ("object_counts: [2]", "object_counts: [3]", "object_counts"),
        ("formal_epochs: 10", "formal_epochs: 4", "formal_epochs"),
    ],
)
def test_config_rejects_invalid_oracle_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_graph_control_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("bootstrap_samples: 2000", "bootstrap_samples: 0", "bootstrap_samples"),
        ("bootstrap_seed: 2057736129", "bootstrap_seed: -1", "bootstrap_seed"),
        ("max_lag: 3", "max_lag: -1", "max_lag"),
        ("active_epsilon: 1.0e-6", "active_epsilon: 0", "active_epsilon"),
        (
            "sensitivity_rows_per_episode: 4",
            "sensitivity_rows_per_episode: 0",
            "sensitivity_rows_per_episode",
        ),
        (
            "sensitivity_batch_size: 4",
            "sensitivity_batch_size: 0",
            "sensitivity_batch_size",
        ),
        ("sensitivity_scale: 0.25", "sensitivity_scale: 2", "sensitivity_scale"),
    ],
)
def test_config_rejects_invalid_diagnostics_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    path.write_text(path.read_text().replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_graph_control_config(path)


def test_config_rejects_unknown_diagnostics_field(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    path.write_text(
        path.read_text().replace(
            "  active_epsilon: 1.0e-6",
            "  active_epsilon: 1.0e-6\n  unexpected: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown diagnostics fields: unexpected"):
        load_graph_control_config(path)


def test_config_without_diagnostics_remains_compatible(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.index("diagnostics:\n")], encoding="utf-8")

    config = load_graph_control_config(path)

    assert config.diagnostics is None
