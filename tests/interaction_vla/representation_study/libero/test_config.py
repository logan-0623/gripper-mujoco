from pathlib import Path

import pytest

from interaction_vla.representation_study.libero.config import (
    LIBERO_CONFIG_SCHEMA,
    load_libero_study_config,
)


def test_smoke_config_is_isolated_deterministic_and_rl_free() -> None:
    config = load_libero_study_config(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    )

    assert config.schema_version == LIBERO_CONFIG_SCHEMA
    assert config.seed == 2057736129
    assert config.output_dir == Path("outputs/representation_study/libero_smolvla_smoke")
    assert config.sources.lerobot_repo_id == "lerobot/libero"
    assert config.coverage.suites == ("libero_spatial", "libero_object")
    assert config.coverage.tasks_per_suite == 3
    assert config.coverage.fail_on_unsupported_task
    assert config.sources.raw_hdf5_root is not None
    assert config.state_bank.holdout_episodes_per_task == 3
    assert config.stages.fractions == (0.25, 0.5, 1.0)
    assert config.stages.epochs > 0
    assert config.stages.batch_size == 2
    assert config.taps.names == (
        "vision_output",
        "multimodal_fusion",
        "action_expert_input",
        "pre_action",
    )
    assert config.taps.pooling == "valid_token_mean"
    assert not hasattr(config, "rl")


def test_config_rejects_non_nested_fraction_order(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: libero_interaction_representation_v1
seed: 1
output_dir: outputs/test
sources:
  lerobot_repo_id: lerobot/libero
  lerobot_revision: main
  raw_hdf5_root: data/libero
coverage:
  suites: [libero_spatial]
  fail_on_unsupported_task: true
replay: {}
annotations: {}
state_bank: {}
splits: {}
stages:
  base_model: lerobot/smolvla_base
  base_revision: main
  fractions: [0.5, 0.25, 1.0]
  epochs: 1
taps: {}
probes: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_libero_study_config(path)


def test_config_rejects_absolute_output_path(tmp_path: Path) -> None:
    text = Path(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace(
        "outputs/representation_study/libero_smolvla_smoke", "/tmp/libero-output"
    ), encoding="utf-8")

    with pytest.raises(ValueError, match="repository-relative"):
        load_libero_study_config(path)


def test_config_rejects_empty_artifact_or_source_roots(tmp_path: Path) -> None:
    text = Path(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    ).read_text(encoding="utf-8")
    for original in (
        "outputs/representation_study/libero_smolvla_smoke",
        "data/libero/raw",
    ):
        path = tmp_path / f"bad-{original.replace('/', '-')}.yaml"
        path.write_text(text.replace(original, ""), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty repository-relative"):
            load_libero_study_config(path)


def test_config_fails_closed_for_unreviewed_suite(tmp_path: Path) -> None:
    text = Path(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "bad-suite.yaml"
    path.write_text(
        text.replace(
            "[libero_spatial, libero_object]",
            "[libero_spatial, libero_goal]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not reviewed"):
        load_libero_study_config(path)


def test_config_rejects_invalid_gate_and_probe_parameters(tmp_path: Path) -> None:
    text = Path(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    ).read_text(encoding="utf-8")
    replacements = (
        ("states_per_episode: 16", "states_per_episode: 0"),
        ("confidence_level: 0.95", "confidence_level: 1.0"),
        ("denoising_call: final", "denoising_call: first"),
    )
    for index, (original, changed) in enumerate(replacements):
        path = tmp_path / f"bad-parameter-{index}.yaml"
        path.write_text(text.replace(original, changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_libero_study_config(path)
