from __future__ import annotations

from pathlib import Path

import pytest

from interaction_vla.graph_control.ablation_config import load_ablation_config
from interaction_vla.graph_control.schema import ABLATION_CONDITIONS


def _write_config(path: Path, *, conditions: str | None = None, seeds: str = "[0, 1, 2]") -> None:
    path.write_text(
        f"""
base_graph_control_config: configs/graph_v2_act_pilot_macos.yaml
conditions: {conditions or list(ABLATION_CONDITIONS)}
seeds: {seeds}
shuffle_seed: 2057736129
cache:
  directory: outputs/graph_control/control_alignment_ablation/cache
training:
  output_dir: outputs/graph_control/control_alignment_ablation/runs
  smoke_steps: 1
  formal_epochs: 10
""".lstrip(),
        encoding="utf-8",
    )


def test_ablation_config_is_strict_and_requires_three_seeds(tmp_path: Path) -> None:
    path = tmp_path / "ablation.yaml"
    _write_config(path)

    config = load_ablation_config(path)

    assert config.base_graph_control_config == Path(
        "configs/graph_v2_act_pilot_macos.yaml"
    )
    assert config.conditions == ABLATION_CONDITIONS
    assert config.seeds == (0, 1, 2)
    assert config.shuffle_seed == 2057736129
    assert config.cache_dir == Path(
        "outputs/graph_control/control_alignment_ablation/cache"
    )
    assert config.training_output_dir == Path(
        "outputs/graph_control/control_alignment_ablation/runs"
    )
    assert config.smoke_output_dir == Path(
        "outputs/graph_control/control_alignment_ablation/smoke"
    )


def test_ablation_config_rejects_partial_matrix_and_too_few_seeds(tmp_path: Path) -> None:
    path = tmp_path / "ablation.yaml"
    _write_config(path, conditions="[flat, full_graph]")
    with pytest.raises(ValueError, match="exact progressive"):
        load_ablation_config(path)

    _write_config(path, seeds="[0, 1]")
    with pytest.raises(ValueError, match="at least three"):
        load_ablation_config(path)


def test_repository_ablation_configs_isolate_macos_and_cuda_outputs() -> None:
    mac = load_ablation_config("configs/control_alignment_ablation_macos.yaml")
    cuda = load_ablation_config("configs/control_alignment_ablation_linux_cuda.yaml")

    assert mac.conditions == cuda.conditions == ABLATION_CONDITIONS
    assert mac.seeds == cuda.seeds == (0, 1, 2)
    assert mac.cache_dir != cuda.cache_dir
    assert mac.training_output_dir != cuda.training_output_dir
    assert "_cuda" not in mac.training_output_dir.as_posix()
    assert "_cuda" in cuda.training_output_dir.as_posix()

