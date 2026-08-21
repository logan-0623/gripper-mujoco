from __future__ import annotations

from pathlib import Path

import pytest

from interaction_vla.representation_study.config import load_study_config


def _write(path: Path, extra: str = "") -> None:
    path.write_text(
        """schema_version: interaction_representation_runtime_v1
study_id: test
dataset:
  repo_id: local/test
  root: dataset
  split_manifest: split.json
  bridge_config: bridge.yaml
trace:
  root: traces
  condition: predicted_reflect_v2
state_bank:
  output_dir: bank
  selection_seed: 7
  split_ratios: [0.8, 0.1, 0.1]
  expert_per_phase: 1
  policy_per_stratum: 2
  replay_position_tolerance: 0.0001
stages:
  act:
    sft:
      checkpoint: checkpoint
      trainable_groups: [all]
extraction:
  output_dir: latents
  device: cpu
  batch_size: 2
probes:
  output_dir: probes
  epochs: 10
  batch_size: 8
  learning_rate: 0.001
  weight_decays: [0.0, 0.001]
  seed: 7
interventions:
  output_dir: interventions
  partition: test
  batch_size: 4
  max_states: 16
  modes: [zero, mean, matched_random]
sft:
  output_dir: sft
  steps: 20
  batch_size: 2
  learning_rate: 0.00001
  weight_decay: 0.0001
  grad_clip_norm: 10.0
  save_every: 10
  num_workers: 0
  seed: 7
rl:
  output_dir: rl
  device: cpu
  total_steps: 64
  rollout_steps: 16
  update_epochs: 2
  minibatch_size: 8
  learning_rate: 0.0003
  representation_learning_rate: 0.00001
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 1.0
  residual_scale: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
  reward_mode: sparse
  progress_reward_scale: 0.0
  max_episode_steps: 30
  object_counts: [2, 3]
  layouts: [normal, crowded]
  eval_interval: 32
  eval_episodes: 4
  success_threshold: 0.5
  seed: 7
analysis:
  output_dir: analysis
  bootstrap_samples: 100
  confidence_level: 0.95
  seed: 7
"""
        + extra,
        encoding="utf-8",
    )


def test_study_config_loads_strict_runtime_contract(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    _write(path)
    config = load_study_config(path)
    assert config.study_id == "test"
    assert config.state_bank.policy_per_stratum == 2
    assert config.sft.steps == 20
    assert config.rl.reward_mode == "sparse"
    assert config.rl.residual_scale[-1] == 0.2
    assert config.analysis.bootstrap_samples == 100
    first = config.state_bank_selection_sha256()
    assert len(first) == 64


def test_study_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    _write(path, "unknown: true\n")
    with pytest.raises(ValueError, match="unknown config fields"):
        load_study_config(path)
