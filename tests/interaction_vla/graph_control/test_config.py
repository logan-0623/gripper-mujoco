from __future__ import annotations

from pathlib import Path

import pytest

from interaction_vla.graph_control.config import load_graph_control_config


def _write_config(path: Path, *, conditions: str | None = None) -> None:
    condition_yaml = conditions or """
  - flat
  - predicted_random
  - predicted_reflect
  - oracle_current"""
    path.write_text(
        f"""
bridge_config: configs/lerobot_act_pilot_macos.yaml
split_manifest: outputs/graph_finetune/mujoco_pilot/split_manifest.json
graph_runs_root: outputs/graph_finetune/mujoco_pilot
conditions:{condition_yaml}
seeds: [0, 1, 2]
cache:
  directory: outputs/graph_control/cache
  batch_size: 16
training:
  output_dir: outputs/graph_control/act_pilot
  smoke_steps: 1
evaluation:
  layouts: [normal, crowded]
  object_counts: [2, 3]
  cases_per_cell: 5
  master_seed: 2057736129
  max_steps: 500
""".lstrip(),
        encoding="utf-8",
    )


def test_config_locks_conditions_checkpoints_and_evaluation_cells(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    config = load_graph_control_config(path)

    assert config.conditions == (
        "flat",
        "predicted_random",
        "predicted_reflect",
        "oracle_current",
    )
    assert config.seeds == (0, 1, 2)
    assert config.graph_checkpoint("flat", 0) is None
    assert config.graph_checkpoint("predicted_random", 2) == Path(
        "outputs/graph_finetune/mujoco_pilot/random_init/fraction_1/seed_2/checkpoint.pt"
    )
    assert config.graph_checkpoint("predicted_reflect", 1) == Path(
        "outputs/graph_finetune/mujoco_pilot/reflectvlm_init/fraction_1/seed_1/checkpoint.pt"
    )
    assert config.graph_checkpoint("oracle_current", 1) == config.graph_checkpoint(
        "predicted_reflect", 1
    )
    assert config.evaluation.cells == (
        ("normal", 2),
        ("normal", 3),
        ("crowded", 2),
        ("crowded", 3),
    )


def test_config_rejects_incomplete_condition_matrix(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path, conditions=" [flat, predicted_reflect]")
    with pytest.raises(ValueError, match="exactly"):
        load_graph_control_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("batch_size: 16", "batch_size: 0", "batch_size"),
        ("seeds: [0, 1, 2]", "seeds: [0, 0]", "seeds"),
        ("cases_per_cell: 5", "cases_per_cell: 0", "cases_per_cell"),
        ("layouts: [normal, crowded]", "layouts: [normal]", "layouts"),
    ],
)
def test_config_rejects_invalid_pairing_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_graph_control_config(path)


def test_config_rejects_unknown_section(tmp_path: Path) -> None:
    path = tmp_path / "graph_control.yaml"
    _write_config(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("surprise: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_graph_control_config(path)
